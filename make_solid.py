#!/usr/bin/env python3
"""Generate a watertight solid mesh using a signed-distance voxel grid.

This script approximates MeshMixer's "Make Solid" (accurate mode) by sampling
an input mesh into a dense signed-distance field, applying an optional offset,
and extracting a watertight surface via marching cubes.
"""
import argparse
import gc
import sys
from typing import Tuple

import numpy as np
import open3d as o3d
from skimage import measure


def _build_sampling_axes(min_bound: np.ndarray, max_bound: np.ndarray, voxel_size: float, padding: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero")

    expanded_min = min_bound - padding
    expanded_max = max_bound + padding

    xs = np.arange(expanded_min[0], expanded_max[0] + voxel_size, voxel_size, dtype=np.float32)
    ys = np.arange(expanded_min[1], expanded_max[1] + voxel_size, voxel_size, dtype=np.float32)
    zs = np.arange(expanded_min[2], expanded_max[2] + voxel_size, voxel_size, dtype=np.float32)

    if xs.size < 2 or ys.size < 2 or zs.size < 2:
        raise ValueError("Bounding box is too small for the given voxel size")

    return xs, ys, zs


def _estimate_grid_size(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> int:
    return int(xs.size) * int(ys.size) * int(zs.size)


def _compute_signed_distance(mesh: o3d.geometry.TriangleMesh, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, batch_size: int = 1_000_000) -> np.ndarray:
    xy_points = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    num_xy = xy_points.shape[0]

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    sdf_grid = np.empty((xs.size, ys.size, zs.size), dtype=np.float32)
    buffer = np.empty((num_xy, 3), dtype=np.float32)
    buffer[:, :2] = xy_points

    distances = np.empty(num_xy, dtype=np.float32)
    for zi, z in enumerate(zs):
        buffer[:, 2] = z
        for start in range(0, num_xy, batch_size):
            stop = min(start + batch_size, num_xy)
            query = o3d.core.Tensor(buffer[start:stop], dtype=o3d.core.Dtype.Float32)
            distances[start:stop] = scene.compute_signed_distance(query).numpy()
        sdf_grid[:, :, zi] = distances.reshape(xs.size, ys.size)

    return sdf_grid


def _extract_mesh_from_sdf(sdf_grid: np.ndarray, axes: Tuple[np.ndarray, np.ndarray, np.ndarray], voxel_size: float, level: float = 0.0) -> o3d.geometry.TriangleMesh:
    xs, ys, zs = axes

    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid,
        level=level,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float32)
    verts_world = verts + origin

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_world.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def make_solid(input_path: str, output_path: str, voxel_size: float, offset: float, padding: float, max_voxels: int, verbose: bool) -> None:
    if verbose:
        print(f"Loading mesh: {input_path}")

    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        raise ValueError("Input mesh is empty or could not be loaded")
    if not mesh.has_triangles():
        raise ValueError("Input mesh does not contain triangles")

    aabb = mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    effective_padding = max(padding, abs(offset) + 2.0 * voxel_size)
    axes = _build_sampling_axes(min_bound, max_bound, voxel_size, effective_padding)

    grid_voxels = _estimate_grid_size(*axes)
    if verbose:
        print(f"Sampling grid: {axes[0].size} x {axes[1].size} x {axes[2].size} ({grid_voxels:,} voxels)")
    if grid_voxels > max_voxels:
        raise ValueError(
            f"Grid would allocate {grid_voxels:,} voxels which exceeds the safety limit of {max_voxels:,}. "
            "Increase voxel size or reduce padding."
        )

    if verbose:
        print("Computing signed distance field...")
    sdf_grid = _compute_signed_distance(mesh, *axes)

    if verbose:
        print("Applying offset and extracting surface...")
    sdf_grid -= offset

    if np.all(sdf_grid > 0) or np.all(sdf_grid < 0):
        raise ValueError("Offset/voxel size combination does not intersect the mesh surface")

    solid_mesh = _extract_mesh_from_sdf(sdf_grid, axes, voxel_size)

    if verbose:
        print(f"Saving solid mesh: {output_path}")
    o3d.io.write_triangle_mesh(output_path, solid_mesh, write_ascii=False)

    del mesh, solid_mesh, sdf_grid
    gc.collect()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate MeshMixer's Make Solid (accurate mode) via voxelized SDF sampling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_mesh", help="Path to the input mesh (STL/OBJ/etc.)")
    parser.add_argument("output_mesh", help="Path to the output watertight mesh (STL)")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Linear size of voxels used for sampling")
    parser.add_argument("--offset", type=float, default=0.0, help="Surface offset distance (positive to dilate, negative to erode)")
    parser.add_argument("--padding", type=float, default=2.0, help="Extra space added around the mesh bounding box")
    parser.add_argument(
        "--max-voxels",
        type=int,
    default=50_000_000,
        help="Safety limit on total voxels to avoid excessive memory use",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        make_solid(
            input_path=args.input_mesh,
            output_path=args.output_mesh,
            voxel_size=args.voxel_size,
            offset=args.offset,
            padding=args.padding,
            max_voxels=args.max_voxels,
            verbose=args.verbose,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
