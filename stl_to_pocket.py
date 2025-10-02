#!/usr/bin/env python3
import sys
import os
import argparse
import gc

# Set environment variables for headless operation before importing Open3D
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.2'

import open3d as o3d
import numpy as np
from multiprocessing import Pool, cpu_count

# --- Core Rasterization Logic (Adapted from your provided script) ---
# This section contains the functions responsible for converting the 3D mesh
# into a 2D depth map.

def get_orientation_normal(mesh, orientation):
    """Get the normal vector for the specified orientation."""
    if orientation == "auto":
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        max_area = 0
        largest_face_normal = np.array([0, 0, 1])
        for tri in triangles:
            v0, v1, v2 = vertices[tri]
            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
            if area > max_area:
                max_area = area
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal) * -1
                largest_face_normal = normal
        return largest_face_normal
    
    orientation_map = {
        "top": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "bottom": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "front": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "back": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "left": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "right": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    }
    return orientation_map.get(orientation, np.array([0.0, 0.0, 1.0], dtype=np.float64))

def rasterize_triangle_batch(args):
    """Rasterizes a batch of triangles to a depth buffer."""
    triangle_batch, vertices, mesh_dims, min_bound, width, height, batch_idx = args
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float32)
    
    for triangle in triangle_batch:
        v0, v1, v2 = vertices[triangle]
        
        # Convert to screen coordinates, flipping Y for image space
        x0 = (v0[0] - min_bound[0]) / mesh_dims[0] * (width - 1)
        y0 = (height - 1) - (v0[1] - min_bound[1]) / mesh_dims[1] * (height - 1)
        x1 = (v1[0] - min_bound[0]) / mesh_dims[0] * (width - 1)
        y1 = (height - 1) - (v1[1] - min_bound[1]) / mesh_dims[1] * (height - 1)
        x2 = (v2[0] - min_bound[0]) / mesh_dims[0] * (width - 1)
        y2 = (height - 1) - (v2[1] - min_bound[1]) / mesh_dims[1] * (height - 1)
        
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        
        if min_x > max_x or min_y > max_y:
            continue
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < 1e-10: continue
                
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1 - w0 - w1
                
                if w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6:
                    interpolated_z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if interpolated_z > depth_buffer[py, px]:
                        depth_buffer[py, px] = interpolated_z
    return depth_buffer

def generate_depth_data(mesh, resolution):
    """Generates a high-resolution depth map from an STL mesh."""
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    mesh_dims = max_bound - min_bound

    x_len, y_len = mesh_dims[0], mesh_dims[1]
    if x_len > y_len:
        width = resolution
        height = int(resolution * y_len / x_len)
    else:
        height = resolution
        width = int(resolution * x_len / y_len)

    print(f"Rasterizing to a {width}x{height} depth map...")
    
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    num_cores = min(cpu_count(), 8)
    batch_size = max(100, len(triangles) // (num_cores * 4))
    
    triangle_batches = []
    for i in range(0, len(triangles), batch_size):
        batch = triangles[i:i + batch_size]
        batch_idx = i // batch_size
        triangle_batches.append((batch, vertices, mesh_dims, min_bound, width, height, batch_idx))

    if len(triangle_batches) > 1 and num_cores > 1:
        with Pool(processes=num_cores) as pool:
            results = pool.map(rasterize_triangle_batch, triangle_batches)
    else:
        results = [rasterize_triangle_batch(batch) for batch in triangle_batches]

    # Combine results by taking the maximum Z value for each pixel
    final_depth_buffer = np.full((height, width), -np.inf, dtype=np.float32)
    for batch_result in results:
        final_depth_buffer = np.maximum(final_depth_buffer, batch_result)

    return final_depth_buffer, mesh_dims

# --- Modified Functionality: Depth Map to Positive Tool Mesh ---

def build_positive_tool_mesh(depth_buffer, mesh_dims):
    """Construct a positive tool mesh from the depth map."""
    print("Building new 3D mesh from depth map...")
    height, width = depth_buffer.shape

    # Create grids of vertices for the top (model shape) and bottom (flat base) surfaces
    top_vertices = np.zeros((height, width, 3))
    bottom_vertices = np.zeros((height, width, 3))

    # Generate vertex coordinates
    y_coords = np.linspace(0, mesh_dims[1], height)
    x_coords = np.linspace(0, mesh_dims[0], width)
    xv, yv = np.meshgrid(x_coords, y_coords)

    # Populate top surface vertices
    top_vertices[:, :, 0] = xv
    # Flip Y-axis back from image coordinates to model coordinates
    top_vertices[:, :, 1] = yv[::-1]
    
    # Use the depth map directly to create the positive shape
    # Where there's no model (depth is -inf), the base is flat (z=0).
    # Where there is a model, use its actual height.
    valid_mask = depth_buffer != -np.inf
    positive_z = np.zeros_like(depth_buffer)
    positive_z[valid_mask] = depth_buffer[valid_mask]
    top_vertices[:, :, 2] = positive_z

    # Populate bottom surface vertices (a simple flat plane at z=0)
    bottom_vertices[:, :, 0] = xv
    bottom_vertices[:, :, 1] = yv[::-1]
    bottom_vertices[:, :, 2] = 0.0

    # Combine top and bottom vertices into a single list
    num_verts_per_surface = width * height
    vertices = np.vstack([top_vertices.reshape(-1, 3), bottom_vertices.reshape(-1, 3)])
    
    triangles = []
    vertices_np = vertices

    cell_mask = (
        valid_mask[:-1, :-1]
        | valid_mask[1:, :-1]
        | valid_mask[:-1, 1:]
        | valid_mask[1:, 1:]
    )

    cell_indices = np.arange(height * width).reshape(height, width)
    offset = num_verts_per_surface
    mesh_center_hint = vertices_np[:num_verts_per_surface].mean(axis=0)

    up_hint = np.array([0.0, 0.0, 1.0])
    down_hint = np.array([0.0, 0.0, -1.0])
    west_hint = np.array([-1.0, 0.0, 0.0])
    east_hint = np.array([1.0, 0.0, 0.0])
    north_hint = np.array([0.0, 1.0, 0.0])
    south_hint = np.array([0.0, -1.0, 0.0])

    def append_oriented(a, b, c, hint):
        tri = [a, b, c]
        normal = np.cross(vertices_np[b] - vertices_np[a], vertices_np[c] - vertices_np[a])
        if np.linalg.norm(hint) < 1e-8:
            hint = up_hint
        if np.dot(normal, hint) < 0:
            tri[1], tri[2] = tri[2], tri[1]
        triangles.append(tri)

    def add_quad(v00, v10, v01, v11):
        append_oriented(v00, v01, v10, up_hint)
        append_oriented(v10, v01, v11, up_hint)
        append_oriented(v00 + offset, v10 + offset, v01 + offset, down_hint)
        append_oriented(v10 + offset, v11 + offset, v01 + offset, down_hint)

    def add_wall(top_a, top_b, hint):
        bottom_a = top_a + offset
        bottom_b = top_b + offset
        append_oriented(top_a, top_b, bottom_a, hint)
        append_oriented(top_b, bottom_b, bottom_a, hint)

    def add_wall_auto(top_a, top_b):
        bottom_a = top_a + offset
        bottom_b = top_b + offset
        face_center = (
            vertices_np[top_a]
            + vertices_np[top_b]
            + vertices_np[bottom_a]
            + vertices_np[bottom_b]
        ) / 4.0
        hint = face_center - mesh_center_hint
        append_oriented(top_a, top_b, bottom_a, hint)
        append_oriented(top_b, bottom_b, bottom_a, hint)

    for r in range(height - 1):
        for c in range(width - 1):
            if not cell_mask[r, c]:
                continue

            v00 = cell_indices[r, c]
            v10 = cell_indices[r, c + 1]
            v01 = cell_indices[r + 1, c]
            v11 = cell_indices[r + 1, c + 1]
            add_quad(v00, v10, v01, v11)

            # West boundary
            if c == 0 or not cell_mask[r, c - 1]:
                add_wall(v00, v01, west_hint)
            # East boundary
            if c == cell_mask.shape[1] - 1 or not cell_mask[r, c + 1]:
                add_wall(v10, v11, east_hint)
            # North boundary
            if r == 0 or not cell_mask[r - 1, c]:
                add_wall(v00, v10, north_hint)
            # South boundary
            if r == cell_mask.shape[0] - 1 or not cell_mask[r + 1, c]:
                add_wall(v01, v11, south_hint)

    # Detect boundary edges missing a mate and close them with vertical walls.
    triangles_snapshot = list(triangles)
    edge_counts = {}
    for tri in triangles_snapshot:
        a, b, c = tri
        edges = ((a, b), (b, c), (c, a))
        for u, v in edges:
            key = (u, v) if u < v else (v, u)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for (u, v), count in edge_counts.items():
        if count != 1:
            continue
        if u >= offset and v >= offset:
            add_wall_auto(u - offset, v - offset)
        elif u < offset and v < offset:
            add_wall_auto(u, v)
        
    final_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles)
    )

    final_mesh.remove_duplicated_vertices()
    final_mesh.remove_degenerate_triangles()
    final_mesh.remove_duplicated_triangles()
    final_mesh.orient_triangles()
    final_mesh.compute_triangle_normals()

    triangles_np = np.asarray(final_mesh.triangles)
    vertices_np = np.asarray(final_mesh.vertices)
    centers = vertices_np[triangles_np].mean(axis=1)
    normals = np.asarray(final_mesh.triangle_normals)
    mesh_center = final_mesh.get_center()
    orientation_score = np.mean(np.sum((centers - mesh_center) * normals, axis=1))
    if orientation_score < 0:
        flipped = triangles_np[:, [0, 2, 1]]
        final_mesh.triangles = o3d.utility.Vector3iVector(flipped)
        final_mesh.compute_triangle_normals()

    final_mesh.compute_vertex_normals()

    return final_mesh

# --- Main Execution ---

def create_stl_positive_tool(input_path, output_path, resolution, orientation):
    """Main function to load, process, and save the positive tool STL."""
    print(f"Loading STL: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    if not mesh.has_triangles():
        raise ValueError("Input STL file has no triangles.")
    mesh.compute_vertex_normals()

    print(f"Orienting mesh using '{orientation}' method...")
    target_normal = get_orientation_normal(mesh, orientation)
    target_normal = target_normal.astype(np.float64)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.cross(target_normal, z_axis)
    angle = np.arccos(np.clip(np.dot(target_normal, z_axis), -1.0, 1.0))
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-6 and angle > 1e-6:
        axis = (axis / axis_norm).astype(np.float64)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh.rotate(R, center=mesh.get_center())

    min_bound = mesh.get_axis_aligned_bounding_box().get_min_bound()
    mesh.translate(-min_bound)

    depth_buffer, mesh_dims = generate_depth_data(mesh, resolution)
    tool_mesh = build_positive_tool_mesh(depth_buffer, mesh_dims)
    
    original_center = mesh.get_center()
    tool_center = tool_mesh.get_center()
    tool_mesh.translate([
        original_center[0] - tool_center[0], 
        original_center[1] - tool_center[1], 
        0
    ])

    print(f"Saving positive tool STL to: {output_path}")
    o3d.io.write_triangle_mesh(output_path, tool_mesh)
    
    del mesh, depth_buffer, tool_mesh
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an STL file into a positive 'tool' STL suitable for boolean subtraction in CAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_stl", help="Path to the input STL file.")
    parser.add_argument("output_stl", help="Path to save the output tool STL file.")
    parser.add_argument(
        "--resolution", "-r", type=int, default=512,
        help="The resolution of the internal depth map. Higher values create more detail but increase file size and processing time."
    )
    parser.add_argument(
        "--orientation", choices=["auto", "top", "bottom", "left", "right", "front", "back"],
        default="auto", help="Specify which face of the model should be oriented up to create the positive tool from."
    )
    args = parser.parse_args()

    try:
        create_stl_positive_tool(
            input_path=args.input_stl,
            output_path=args.output_stl,
            resolution=args.resolution,
            orientation=args.orientation,
        )
        print("\nConversion completed successfully! ✅")
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)