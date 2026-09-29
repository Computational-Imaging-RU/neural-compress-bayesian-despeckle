from pathlib import Path
import argparse
import time

import numpy as np
import torch
from PIL import Image
from sewar.full_ref import msssim
from skimage.metrics import structural_similarity
from tqdm import tqdm


CONFIGS = {
    4: {
        "lambda": 0.04,
        "coefficients_path": Path(
            "bd-qmap-weights/"
            "epoch=2039--train_loss=309.2960205078125--denoise_psnr=27.3035--train_bpp=1.8831_"
            "k_4__step_4__noise_0__imagenet_redefined_coeffs_by_counting_uniques.npz"
        ),
    },
    8: {
        "lambda": 0.01,
        "coefficients_path": Path(
            "bd-qmap-weights/"
            "epoch=2074--train_loss=163.79995727539062--denoise_psnr=29.4267--train_bpp=0.8960_"
            "k_8__step_8__noise_0__imagenet_redefined_coeffs_by_counting_uniques.npz"
        ),
    },
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the neural-compression BD-QMAP despeckler directly on "
            "raw grayscale images. Gamma speckle noise is generated internally."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Set12"),
        help="Input grayscale image or directory of images (default: Set12).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated noisy images and reconstructions.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=sorted(CONFIGS),
        default=4,
        help="Patch size to use (default: 4).",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=None,
        help=(
            "Override the default lambda for the selected k. "
            "Defaults: k=4 -> 0.04, k=8 -> 0.01."
        ),
    )
    parser.add_argument(
        "--looks",
        type=float,
        default=1.0,
        help=(
            "Number of looks L for Gamma speckle. Noise is Gamma(shape=L, "
            "scale=1/L). Default: 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate the speckle noise (default: 42).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of image patches processed at once (default: 5).",
    )
    return parser.parse_args()


def collect_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_path}")
    return image_paths


def load_grayscale(path):
    """Load an image as grayscale float64 in [0, 1]."""
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float64) / 255.0


def add_gamma_speckle(clean, looks, rng=None):
    """
    Generate multiplicative Gamma speckle:

        Y = X * N,
        N ~ Gamma(shape=L, scale=1/L).

    The noisy array is intentionally NOT clipped before denoising.
    """
    if looks <= 0:
        raise ValueError("--looks must be positive.")

    speckle = np.random.gamma(shape=looks, scale=1.0 / looks, size=clean.shape)
    return clean * speckle


def save_grayscale(image, path):
    """Save a float image for visualization, clipping only for the PNG file."""
    image_uint8 = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


def mse(x, y):
    return np.mean((x - y) ** 2)


def psnr(x, y, peak=1.0):
    error = mse(x, y)
    if error == 0:
        return float("inf")
    return -10.0 * np.log10(error / (peak**2))


def compute_metrics(reference, estimate):
    """Return PSNR, SSIM, and MS-SSIM for images normalized to [0, 1]."""
    estimate_for_metrics = np.clip(estimate, 0.0, 1.0)
    psnr_value = psnr(reference, estimate_for_metrics, peak=1.0)
    ssim_value = structural_similarity(reference, estimate_for_metrics, data_range=1.0)
    ms_ssim_value = float(msssim(reference, estimate_for_metrics, MAX=1.0).real)
    return psnr_value, ssim_value, ms_ssim_value


def get_device():
    """Prefer the original cuda:3 device, with sensible fallbacks."""
    if torch.cuda.is_available():
        gpu_index = 3 if torch.cuda.device_count() > 3 else 0
        return torch.device(f"cuda:{gpu_index}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_overlap_count(height, width, patch_size, device):
    """Number of overlapping stride-1 patches contributing to each pixel."""
    ones = torch.ones(1, 1, height, width, device=device)
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=1)
    fold = torch.nn.Fold(
        output_size=(height, width),
        kernel_size=patch_size,
        stride=1,
    )
    return fold(unfold(ones))


def qmap_despeckle(
    coefficients_path, noisy, patch_size, lmbda, looks=1.0, batch_size=5
):
    height, width = noisy.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Image size {noisy.shape} is smaller than k={patch_size}."
        )

    device = get_device()
    print(f"device: {device}")

    codebook = np.load(coefficients_path)
    coeffs = codebook["unique_redefined_entropies"]
    centroids = codebook["unique_codes"].reshape(-1, patch_size**2)

    coeffs = torch.from_numpy(coeffs).float().to(device)
    centroids = torch.from_numpy(centroids).float().to(device)

    # Terms reused for every noisy patch.
    centroid_values = centroids.unsqueeze(0)  # (1, num_codes, k^2)
    log_centroid_values = torch.log(centroid_values + 1e-12)
    rate_cost = lmbda * coeffs.unsqueeze(0)  # (1, num_codes)

    noisy_tensor = (
        torch.from_numpy(noisy)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=1)
    patches = unfold(noisy_tensor).squeeze(0).T  # (num_patches, k^2)

    selected_patches = []

    with torch.no_grad():
        for start in tqdm(
            range(0, patches.shape[0], batch_size),
            desc="Q-MAP patch search",
        ):
            batch = patches[start : start + batch_size]

            # L-look Gamma-speckle negative log-likelihood, up to terms
            # independent of the candidate clean patch:
            #     -log p(y | x) = L * (log x + y/x) + const(y, L).
            likelihood_cost = looks * (
                log_centroid_values
                + batch.unsqueeze(1) / (centroid_values + 1e-12)
            ).sum(dim=2) / (patch_size**2)

            total_cost = likelihood_cost + rate_cost
            best_indices = torch.argmin(total_cost, dim=1)
            selected_patches.append(centroids[best_indices])

    selected_patches = torch.cat(selected_patches, dim=0)
    selected_patches = selected_patches.T.unsqueeze(0)

    fold = torch.nn.Fold(
        output_size=(height, width),
        kernel_size=patch_size,
        stride=1,
    )
    reconstruction_sum = fold(selected_patches)
    overlap_count = get_overlap_count(height, width, patch_size, device)

    reconstruction = reconstruction_sum / overlap_count
    return reconstruction.squeeze().cpu().numpy()


def main():
    args = parse_args()

    config = CONFIGS[args.k]
    lmbda = config["lambda"] if args.lmbda is None else args.lmbda
    coefficients_path = config["coefficients_path"]

    if not coefficients_path.exists():
        raise FileNotFoundError(
            f"Could not find the k={args.k} codebook:\n{coefficients_path}"
        )

    image_paths = collect_images(args.input)
    args.output_dir = args.output_dir / f"k={args.k}_lambda={lmbda}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration: k={args.k}, lambda={lmbda}")
    print(f"Gamma speckle: looks={args.looks}, seed={args.seed}")
    print(f"coefficients: {coefficients_path}")
    print(f"images: {len(image_paths)}")

    np.random.seed(args.seed)

    for image_path in image_paths:
        image_name = image_path.stem
        clean = load_grayscale(image_path)
        noisy = add_gamma_speckle(clean, looks=args.looks, rng=None)

        noisy_psnr = psnr(clean, noisy)

        print(f"\n{image_name} | shape={clean.shape}")
        print(f"noisy PSNR: {noisy_psnr:.4f} dB")

        start = time.perf_counter()
        reconstruction = qmap_despeckle(
            coefficients_path,
            noisy,
            patch_size=args.k,
            lmbda=lmbda,
            looks=args.looks,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - start

        psnr_value, ssim_value, ms_ssim_value = compute_metrics(
            clean,
            reconstruction.astype(np.float64),
        )

        print(f"runtime: {elapsed:.2f} s")
        print(
            f"PSNR={psnr_value:.4f} dB | "
            f"SSIM={ssim_value:.4f} | "
            f"MS-SSIM={ms_ssim_value:.4f}"
        )

        prefix = f"{image_name}_L={args.looks:g}_seed={args.seed}"

        clean_path = args.output_dir / f"{prefix}_clean.png"
        noisy_path = args.output_dir / f"{prefix}_noisy_psnr={noisy_psnr:.2f}.png"
        reconstruction_path = args.output_dir / (
            f"{prefix}_NCdespeckle_k={args.k}_lambda={lmbda:.3f}_"
            f"psnr={psnr_value:.4f}_ssim={ssim_value:.4f}_"
            f"ms_ssim={ms_ssim_value:.4f}.png"
        )

        save_grayscale(clean, clean_path)
        save_grayscale(noisy, noisy_path)
        save_grayscale(reconstruction, reconstruction_path)

        np.savez_compressed(
            args.output_dir / f"{prefix}_data.npz",
            clean=clean,
            noisy=noisy,
            reconstruction=reconstruction,
            looks=args.looks,
            seed=args.seed,
            k=args.k,
            lmbda=lmbda,
        )


if __name__ == "__main__":
    main()
