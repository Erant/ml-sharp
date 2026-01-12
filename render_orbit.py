#!/usr/bin/env python
"""Render gaussian splats from an orbital viewpoint.

This script loads a PLY file, computes the center of mass of the splats,
and renders from a camera position that orbits around that center point.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image

from sharp.utils.gaussians import load_ply
from sharp.utils.gsplat import GSplatRenderer

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def rotation_matrix_x(angle_deg: float, device: torch.device) -> torch.Tensor:
    """Create rotation matrix around X-axis (pitch - up/down tilt)."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def rotation_matrix_y(angle_deg: float, device: torch.device) -> torch.Tensor:
    """Create rotation matrix around Y-axis (yaw - horizontal rotation)."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def translation_matrix(tx: float, ty: float, tz: float, device: torch.device) -> torch.Tensor:
    """Create translation matrix."""
    return torch.tensor([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)


def look_at_matrix(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Create a look-at view matrix (OpenCV convention: x=right, y=down, z=forward).

    Args:
        eye: Camera position (3,)
        target: Point camera is looking at (3,)
        up: Up vector (3,)

    Returns:
        4x4 view matrix
    """
    # OpenCV convention: Z points forward (away from camera)
    forward = target - eye
    forward = forward / torch.norm(forward)

    # Right vector (X-axis)
    right = torch.cross(forward, up)
    right = right / torch.norm(right)

    # Recalculate up vector (Y-axis) - points down in OpenCV
    up = torch.cross(right, forward)

    # Build rotation matrix (camera to world)
    rotation = torch.stack([right, up, forward], dim=0)  # 3x3

    # Build view matrix (world to camera)
    view_matrix = torch.eye(4, dtype=torch.float32)
    view_matrix[:3, :3] = rotation
    view_matrix[:3, 3] = -rotation @ eye

    return view_matrix


@click.command()
@click.option(
    "-i",
    "--input-ply",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input PLY file",
)
@click.option(
    "-o",
    "--output-png",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output PNG file (will include alpha channel)",
)
@click.option(
    "--rotate-y",
    type=float,
    default=0.0,
    help="Rotation around Y-axis (horizontal) in degrees",
)
@click.option(
    "--rotate-x",
    type=float,
    default=0.0,
    help="Rotation around X-axis (vertical) in degrees",
)
@click.option(
    "--translate-z",
    type=float,
    default=0.0,
    help="Translation along camera-to-center vector (positive = move away, negative = move closer)",
)
@click.option(
    "--width",
    type=int,
    default=1024,
    help="Output image width in pixels",
)
@click.option(
    "--height",
    type=int,
    default=1024,
    help="Output image height in pixels",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for rendering (cuda/cpu)",
)
def main(
    input_ply: Path,
    output_png: Path,
    rotate_y: float,
    rotate_x: float,
    translate_z: float,
    width: int,
    height: int,
    device: str,
):
    """Render gaussian splats with orbital camera control."""

    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    device_obj = torch.device(device)

    # Load PLY file
    LOGGER.info(f"Loading PLY from {input_ply}")
    gaussians, metadata = load_ply(input_ply)
    gaussians = gaussians.to(device_obj)

    # Compute center of mass
    # mean_vectors shape: (B, N, 3) where B=1
    mean_positions = gaussians.mean_vectors[0]  # (N, 3)
    center_of_mass = mean_positions.mean(dim=0)  # (3,)

    LOGGER.info(f"Center of mass: [{center_of_mass[0]:.3f}, {center_of_mass[1]:.3f}, {center_of_mass[2]:.3f}]")
    LOGGER.info(f"Number of splats: {mean_positions.shape[0]:,}")

    # Original camera position is at origin looking down +Z
    original_camera_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device_obj)

    # Vector from center to camera
    camera_to_center = center_of_mass - original_camera_pos
    camera_distance = torch.norm(camera_to_center)

    LOGGER.info(f"Camera distance from center: {camera_distance:.3f}")

    # Apply rotations around center of mass
    # 1. Translate center to origin
    T_to_origin = translation_matrix(-center_of_mass[0].item(), -center_of_mass[1].item(), -center_of_mass[2].item(), device_obj)

    # 2. Apply rotations (Y first for horizontal orbit, then X for vertical)
    R_y = rotation_matrix_y(rotate_y, device_obj)
    R_x = rotation_matrix_x(rotate_x, device_obj)
    R = R_x @ R_y

    # 3. Translate back
    T_from_origin = translation_matrix(center_of_mass[0].item(), center_of_mass[1].item(), center_of_mass[2].item(), device_obj)

    # 4. Compute new camera position after rotation
    rotation_transform = T_from_origin @ R @ T_to_origin
    camera_pos_homogeneous = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=device_obj)
    rotated_camera_pos = (rotation_transform @ camera_pos_homogeneous)[:3]

    # 5. Apply Z translation (along camera-to-center vector)
    camera_to_center_vec = center_of_mass - rotated_camera_pos
    camera_to_center_normalized = camera_to_center_vec / torch.norm(camera_to_center_vec)
    final_camera_pos = rotated_camera_pos - camera_to_center_normalized * translate_z

    LOGGER.info(f"Rotations: Y={rotate_y}°, X={rotate_x}°")
    LOGGER.info(f"Z translation: {translate_z:.3f}")
    LOGGER.info(f"Final camera position: [{final_camera_pos[0]:.3f}, {final_camera_pos[1]:.3f}, {final_camera_pos[2]:.3f}]")

    # Build look-at matrix (camera always points at center of mass)
    up_vector = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device_obj)
    view_matrix = look_at_matrix(final_camera_pos, center_of_mass, up_vector)

    # Build intrinsics matrix
    # Use focal length from metadata if available, otherwise estimate
    f_px = metadata.focal_length_px
    intrinsics = torch.tensor([
        [f_px, 0, (width - 1) / 2.0, 0],
        [0, f_px, (height - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device_obj)

    LOGGER.info(f"Focal length: {f_px:.2f}px")
    LOGGER.info(f"Rendering at {width}x{height}...")

    # Render
    renderer = GSplatRenderer(color_space=metadata.color_space, background_color="black")
    rendering_output = renderer(
        gaussians,
        extrinsics=view_matrix[None],  # Add batch dimension
        intrinsics=intrinsics[None],   # Add batch dimension
        image_width=width,
        image_height=height,
    )

    # Extract RGBA
    # color: (1, 3, H, W) in range [0, 1]
    # alpha: (1, 1, H, W) in range [0, 1]
    color = rendering_output.color[0].permute(1, 2, 0)  # (H, W, 3)
    alpha = rendering_output.alpha[0].permute(1, 2, 0)  # (H, W, 1)

    # Combine into RGBA
    rgba = torch.cat([color, alpha], dim=2)  # (H, W, 4)

    # Convert to uint8
    rgba_uint8 = (rgba * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()

    # Save as PNG with alpha
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgba_uint8, mode="RGBA")
    image.save(output_png)

    LOGGER.info(f"Saved RGBA PNG to {output_png}")
    LOGGER.info(f"Image size: {width}x{height}")

    # Print statistics
    alpha_mean = alpha.mean().item()
    alpha_min = alpha.min().item()
    alpha_max = alpha.max().item()
    LOGGER.info(f"Alpha channel stats: min={alpha_min:.3f}, mean={alpha_mean:.3f}, max={alpha_max:.3f}")


if __name__ == "__main__":
    main()
