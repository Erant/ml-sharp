"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
@click.option(
    "--cull-transparent/--no-cull-transparent",
    "cull_transparent",
    is_flag=True,
    default=True,
    help="Remove splats with opacity below threshold (default: enabled).",
)
@click.option(
    "--opacity-threshold",
    type=float,
    default=1e-3,
    help="Minimum opacity threshold for culling transparent splats (default: 0.001).",
)
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
    cull_transparent: bool,
    opacity_threshold: float,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, alpha, _, f_px = io.load_rgba(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        gaussians = predict_image(
            gaussian_predictor,
            image,
            alpha,
            f_px,
            torch.device(device),
            cull_transparent=cull_transparent,
            opacity_threshold=opacity_threshold,
        )

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    alpha: np.ndarray | None,
    f_px: float,
    device: torch.device,
    cull_transparent: bool = True,
    opacity_threshold: float = 1e-3,
) -> Gaussians3D:
    """Predict Gaussians from an image.

    Args:
        predictor: The Gaussian predictor model
        image: RGB image as numpy array (H, W, 3)
        alpha: Optional alpha channel as numpy array (H, W, 1) normalized to [0, 1]
        f_px: Focal length in pixels
        device: Device to run on
        cull_transparent: Whether to remove splats with low opacity
        opacity_threshold: Minimum opacity threshold for culling

    Returns:
        Predicted 3D Gaussians
    """
    internal_shape = (1536, 1536)
    stride = 2  # Stride used by initializer to downsample to Gaussian grid

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Process alpha mask if present
    mask_pt = None
    if alpha is not None:
        LOGGER.info("Processing alpha mask.")
        # Convert alpha to torch tensor and resize to internal shape
        alpha_pt = torch.from_numpy(alpha.copy()).float().to(device).permute(2, 0, 1)
        alpha_resized_pt = F.interpolate(
            alpha_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        # Downsample to Gaussian grid using avg_pool2d (matches initializer stride)
        # This maps each 2x2 pixel block to one Gaussian splat
        mask_pt = F.avg_pool2d(alpha_resized_pt, kernel_size=stride, stride=stride)
        LOGGER.debug(f"\tMask shape: {mask_pt.shape}")
    else:
        LOGGER.info("No alpha mask present, all splats will be opaque.")

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor, mask=mask_pt)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    # Cull transparent splats if requested
    if cull_transparent:
        num_splats_before = gaussians.opacities.numel()
        gaussians = gaussians.filter_by_opacity(min_opacity=opacity_threshold)
        num_splats_after = gaussians.opacities.numel()
        num_culled = num_splats_before - num_splats_after
        if num_culled > 0:
            LOGGER.info(
                f"Culled {num_culled:,} transparent splats "
                f"({100.0 * num_culled / num_splats_before:.1f}%), "
                f"keeping {num_splats_after:,} visible splats."
            )

    return gaussians
