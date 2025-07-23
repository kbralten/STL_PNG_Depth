
#!/usr/bin/env python3
import sys
import os

os.environ['EGL_PLATFORM'] = 'surfaceless'
import open3d as o3d
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2

def stl_to_depthmap(stl_path, start_height=0.0,total_height=0.0):
    # Load STL mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    # Find the largest face (triangle) by area
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    max_area = 0
    largest_face_normal = np.array([0, 0, 1])
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        if area > max_area:
            max_area = area
            # Compute normal
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal) *-1
            largest_face_normal = normal

    # Compute rotation to align largest face normal to +Z
    z_axis = np.array([0, 0, 1])
    axis = np.cross(largest_face_normal, z_axis)
    angle = np.arccos(np.clip(np.dot(largest_face_normal, z_axis), -1.0, 1.0))
    if np.linalg.norm(axis) > 1e-6 and angle > 1e-6:
        axis = axis / np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh.rotate(R, center=mesh.get_center())

    # Move mesh so all coordinates are positive
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    translation = -min_bound
    mesh.translate(translation)

    # Use Open3D OffscreenRenderer for depth rendering
    import open3d.visualization.rendering as rendering
    bounds = mesh.get_axis_aligned_bounding_box()
    center = mesh.get_center()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    mesh_dims = max_bound - min_bound

    # Determine aspect ratio and set image size
    x_len, y_len = mesh_dims[0], mesh_dims[1]
    if x_len > y_len:
        width = 6000
        height = int(6000 * y_len / x_len)
    else:
        height = 6000
        width = int(6000 * x_len / y_len)
    mesh_height = mesh_dims[2]
    up = np.array([0, 1, 0])

    renderer = rendering.OffscreenRenderer(width, height)
    near = min_bound[2] - mesh_height
    far = max_bound[2] + mesh_height+mesh_height * 20.0
    mat = rendering.MaterialRecord()
    renderer.scene.add_geometry("mesh", mesh, mat)

    #print the mesh dimensions and bounds
    print(f"Mesh dimensions: {mesh_dims}")
    print(f"Mesh bounds: min {min_bound}, max {max_bound}")
    print(f"Mesh height: {mesh_height}")

    center = np.array([0,0,0])
    
    print (f"Center: {center}")
    
    eye = center + np.array([0, 0, mesh_height * 20.0])
    up = np.array([0, 1, 0])
    renderer.setup_camera(0, center, eye, up)
    renderer.scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        min_bound[0], max_bound[0], min_bound[1], max_bound[1],
        near, far
    )

    depth = renderer.render_to_depth_image()
    depth_np = np.asarray(depth)

    # Normalize and invert depth: closest is white, deepest is black
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    depth_inv = 1.0 - depth_norm
    depth_img = (depth_inv * 255).astype(np.uint8)

    scale_to_mm = 1/(mesh_height)
    print(f"Scale to mm: {scale_to_mm}, max in mm: {np.max(depth_inv)/ scale_to_mm}, min in mm: {np.min(depth_inv) / scale_to_mm}")

    #add start_height to clip depth values
    print(f"Start max value: {np.max(depth_img)}, min value: {np.min(depth_img)}")
    scaled_depth_img = (depth_img + (start_height * scale_to_mm)*255).clip(0, 255).astype(np.uint8)
    print(f"End max value: {np.max(scaled_depth_img)}, min value: {np.min(scaled_depth_img)}")

    #clip and scale to total_height
    if total_height > 0:
        scaled_depth_img = 255-((255-scaled_depth_img) * total_height / mesh_height).clip(0, 255).astype(np.uint8)
        print(f"Total height scaling: max value: {np.max(scaled_depth_img)}, min value: {np.min(scaled_depth_img)}")

    #update depth_img to scaled_depth_img where the original depth is not zero or not 255
    depth_img = np.where((depth_img != 0) & (depth_img != 255), scaled_depth_img, depth_img)    

    # Encode PNG for SVG embedding
    png_buffer = BytesIO()
    img_obj = Image.fromarray(depth_img)
    img_obj.save(png_buffer, format="PNG")
    png_base64 = base64.b64encode(png_buffer.getvalue()).decode("ascii")

    # Save PNG to disk if requested
    if getattr(stl_to_depthmap, 'write_png', True):
        png_path = os.path.splitext(stl_path)[0] + ".png"
        img_obj.save(png_path, format="PNG")
        print(f"PNG saved to {png_path}")

    # SVG dimensions in model units (mm)
    svg_width = mesh_dims[0]
    svg_height = mesh_dims[1]

    # Pad depth_img with a 1-pixel black border
    padded = np.pad(depth_img, pad_width=1, mode='constant', constant_values=0)
    mask = (padded == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    # Build SVG path elements
    path_elems = []
    # Remove the outermost contour (largest by area)
    contour_areas = [cv2.contourArea(c) for c in contours]
    if contour_areas:
        max_idx = np.argmax(contour_areas)
        contours = [c for i, c in enumerate(contours) if i != max_idx]

    for contour in contours:
        if len(contour) < 2:
            continue
        points = []
        for pt in contour.squeeze():
            if len(pt.shape) == 0:
                continue
            x_px, y_px = pt
            
            x_model = x_px / depth_img.shape[1] * mesh_dims[0] + min_bound[0]
            y_model = y_px / depth_img.shape[0] * mesh_dims[1]
            points.append(f"{x_model},{y_model}")
        if points:
            path_str = "M " + " L ".join(points) + " Z"
            path_elems.append(f'<path d="{path_str}" stroke="rgb(255,219,102)" fill="none" stroke-width="0.5" />')

    # Find islands of non-transparent pixels (connected components)
    mask = ((depth_img != 0) & (depth_img != 255)).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    png_layers = []
    for label in range(1, num_labels):  # label 0 is background
        island_mask = (labels == label)
        # Find bounding box of the island
        coords = np.argwhere(island_mask)
        if coords.size == 0:
            continue
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # Crop mask and depth image
        cropped_mask = island_mask[y0:y1+1, x0:x1+1]
        cropped_depth = depth_img[y0:y1+1, x0:x1+1]
        # Create RGBA image: only pixels in island are visible, rest transparent
        alpha = np.where(cropped_mask, 255, 0).astype(np.uint8)
        rgb = np.stack([cropped_depth]*3, axis=-1)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        # Set black/white pixels to transparent as well
        rgba[(cropped_depth==0)|(cropped_depth==255), 3] = 0
        img = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        # Calculate position and size in SVG units
        x_svg = x0 / depth_img.shape[1] * svg_width
        y_svg = y0 / depth_img.shape[0] * svg_height
        w_svg = (x1 - x0 + 1) / depth_img.shape[1] * svg_width
        h_svg = (y1 - y0 + 1) / depth_img.shape[0] * svg_height
        png_layers.append(f'<image x="{x_svg}" y="{y_svg}" width="{w_svg}" height="{h_svg}" xlink:href="data:image/png;base64,{png_b64}" />')

    # Output SVG with width/height in mm, viewBox in mesh units, and xlink:href for each PNG layer
    if getattr(stl_to_depthmap, 'write_svg', True):
        svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{svg_width}mm" height="{svg_height}mm" viewBox="0 0 {svg_width} {svg_height}">
            {''.join(png_layers)}
            {''.join(path_elems)}
        </svg>
        '''
        svg_path = os.path.splitext(stl_path)[0] + ".svg"
        with open(svg_path, "w") as f:
            f.write(svg_template)
        print(f"SVG saved to {svg_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert STL to depthmap SVG")
    parser.add_argument("stl_path", help="Input STL file")
    parser.add_argument("--start-height", type=float, default=0.0, help="Start height offset (mm)")
    parser.add_argument("--total-height", type=float, default=0.0, help="Total height for depth normalization (mm)")
    parser.add_argument("--only-png", action="store_true", help="Only write PNG, not SVG")
    parser.add_argument("--only-svg", action="store_true", help="Only write SVG, not PNG")
    args = parser.parse_args()

    # Feature switches
    stl_to_depthmap.write_png = not args.only_svg
    stl_to_depthmap.write_svg = not args.only_png

    stl_to_depthmap(args.stl_path, start_height=args.start_height, total_height=args.total_height)

