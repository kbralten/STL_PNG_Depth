
#!/usr/bin/env python3
import sys
import os

# Set environment variables for headless operation before importing Open3D
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.2'

import open3d as o3d
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2

# Configure Open3D for headless operation
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

def stl_to_depthmap(stl_path, start_height=0.0,total_height=0.0):
    def write_svg_contours(depth_img, mesh_dims, min_bound, stl_path, overall_contours):
        mask = ((depth_img > 1) & (depth_img < 254)).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        svg_width = mesh_dims[0]
        svg_height = mesh_dims[1]
        min_x = min_bound[0]
        min_y = min_bound[1]
        contour_paths = []
        # Add only the largest overall outline contour
        if overall_contours:
            contour_areas = [cv2.contourArea(c) for c in overall_contours]
            if contour_areas:
                max_idx = np.argmax(contour_areas)
                largest_contour = overall_contours[max_idx]
                points = []
                for pt in largest_contour.squeeze():
                    x_px, y_px = pt
                    x_model = x_px / depth_img.shape[1] * svg_width + min_x
                    y_model = y_px / depth_img.shape[0] * svg_height + min_y
                    points.append(f"{x_model},{y_model}")
                if points:
                    path_str = "M " + " L ".join(points) + " Z"
                    contour_paths.append(f'<path d="{path_str}" stroke="rgb(255,219,102)" fill="none" stroke-width="1.0" />')
        # Add island contours
        for label in range(1, num_labels):
            island_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(island_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 2:
                    continue
                points = []
                for pt in contour.squeeze():
                    x_px, y_px = pt
                    x_model = x_px / depth_img.shape[1] * svg_width + min_x
                    y_model = y_px / depth_img.shape[0] * svg_height + min_y
                    points.append(f"{x_model},{y_model}")
                if points:
                    path_str = "M " + " L ".join(points) + " Z"
                    contour_paths.append(f'<path d="{path_str}" stroke="rgb(255,219,102)" fill="none" stroke-width="0.1" />')
        # Write contours SVG file
        contours_svg_path = os.path.splitext(stl_path)[0] + "-contours.svg"
        contours_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}mm" height="{svg_height}mm" viewBox="0 0 {svg_width} {svg_height}">
            {''.join(contour_paths)}
        </svg>
        '''
        with open(contours_svg_path, "w") as f:
            f.write(contours_svg)
        print(f"Contours SVG saved to {contours_svg_path}")
    
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

    # Use proper triangle rasterization with anti-aliasing for smooth depth rendering
    import gc
    from scipy import ndimage
    
    bounds = mesh.get_axis_aligned_bounding_box()
    center = mesh.get_center()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    mesh_dims = max_bound - min_bound

    # Determine aspect ratio and set image size
    x_len, y_len = mesh_dims[0], mesh_dims[1]
    if x_len > y_len:
        width = 1000  # Reasonable size for performance
        height = int(1000 * y_len / x_len)
    else:
        height = 1000  
        width = int(1000 * x_len / y_len)
    mesh_height = mesh_dims[2]

    print(f"Mesh dimensions: {mesh_dims}")
    print(f"Mesh bounds: min {min_bound}, max {max_bound}")
    print(f"Mesh height: {mesh_height}")

    center = np.array([0,0,0])
    print(f"Center: {center}")

    # Use optimized triangle rasterization with supersampling for anti-aliasing
    try:
        print("Creating smooth depth image using optimized triangle rasterization...")
        
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        print(f"Rasterizing {len(triangles)} triangles with anti-aliasing...")
        
        # Use supersampling for anti-aliasing (render at 2x resolution)
        ss_factor = 2
        ss_width = width * ss_factor
        ss_height = height * ss_factor
        
        # Initialize with background depth
        depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
        
        # Rasterize triangles with subpixel precision
        for tri_idx, triangle in enumerate(triangles):
            if tri_idx % 2000 == 0:
                print(f"Processing triangle {tri_idx}/{len(triangles)}")
                
            v0, v1, v2 = vertices[triangle]
            
            # Convert to screen coordinates (supersampled)
            x0 = (v0[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
            y0 = (v0[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
            x1 = (v1[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
            y1 = (v1[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
            x2 = (v2[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
            y2 = (v2[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
            
            # Get bounding box
            min_x = max(0, int(np.floor(min(x0, x1, x2))))
            max_x = min(ss_width-1, int(np.ceil(max(x0, x1, x2))))
            min_y = max(0, int(np.floor(min(y0, y1, y2))))
            max_y = min(ss_height-1, int(np.ceil(max(y0, y1, y2))))
            
            if min_x > max_x or min_y > max_y:
                continue
            
            # Calculate triangle area for barycentric coordinates
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            if area < 1e-10:
                continue
            
            # Rasterize triangle using barycentric coordinates
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    # Calculate barycentric coordinates using the standard formula
                    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                    if abs(denom) < 1e-10:
                        continue
                    
                    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                    w2 = 1 - w0 - w1
                    
                    # Check if point is inside triangle (with small tolerance for edge cases)
                    tolerance = -1e-6
                    if w0 >= tolerance and w1 >= tolerance and w2 >= tolerance:
                        # Interpolate depth using barycentric coordinates
                        depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                        
                        # Z-buffer test (keep closest surface)
                        if depth > depth_buffer[py, px]:
                            depth_buffer[py, px] = depth
        
        print("Triangle rasterization completed, applying anti-aliasing...")
        
        # Replace -inf with minimum depth for background
        valid_mask = depth_buffer != -np.inf
        if np.any(valid_mask):
            min_depth = np.min(depth_buffer[valid_mask])
            depth_buffer[~valid_mask] = min_depth
        else:
            depth_buffer[:] = 0
        
        # Downsample with anti-aliasing (box filter)
        depth_downsampled = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                # Sample 2x2 region from supersampled image
                y_start = y * ss_factor
                x_start = x * ss_factor
                region = depth_buffer[y_start:y_start+ss_factor, x_start:x_start+ss_factor]
                depth_downsampled[y, x] = np.mean(region)
        
        # Apply light Gaussian smoothing for final polish
        depth_np = ndimage.gaussian_filter(depth_downsampled, sigma=0.5)
        
        print("Smooth triangle rasterization completed successfully")
        
    except Exception as e:
        print(f"Error during triangle rasterization: {e}")
        raise

    # Normalize depth: closest surfaces are white, deepest are black
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    # Keep depth_norm as is (higher Z = closer = whiter)
    depth_img = (depth_norm * 255).astype(np.uint8)

    scale_to_mm = 1/(mesh_height)
    print(f"Scale to mm: {scale_to_mm}, max in mm: {np.max(depth_norm)/ scale_to_mm}, min in mm: {np.min(depth_norm) / scale_to_mm}")

    #add start_height to clip depth values
    print(f"Start max value: {np.max(depth_img)}, min value: {np.min(depth_img)}")
    scaled_depth_img = (depth_img + (start_height * scale_to_mm)*255).clip(0, 255).astype(np.uint8)
    print(f"End max value: {np.max(scaled_depth_img)}, min value: {np.min(scaled_depth_img)}")

    #clip and scale to total_height
    if total_height > 0:
        scaled_depth_img = 255-((255-scaled_depth_img) * total_height / mesh_height).clip(0, 255).astype(np.uint8)
        print(f"Total height scaling: max value: {np.max(scaled_depth_img)}, min value: {np.min(scaled_depth_img)}")

    #update depth_img to scaled_depth_img where the original depth is not zero or not 255
    depth_img = np.where((depth_img >1) & (depth_img <254), scaled_depth_img, depth_img)    

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

    png_layers = []
    if getattr(stl_to_depthmap, 'segment', False):
        # Find islands of non-transparent pixels (connected components)
        mask = ((depth_img > 1) & (depth_img < 254)).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

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
            rgba[(cropped_depth<=1)|(cropped_depth>=254), 3] = 0
            img = Image.fromarray(rgba)
            buf = BytesIO()
            img.save(buf, format="PNG")
            png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            # Calculate position and size in SVG units
            x_svg = x0 / depth_img.shape[1] * svg_width
            y_svg = y0 / depth_img.shape[0] * svg_height
            w_svg = (x1 - x0 + 1) / depth_img.shape[1] * svg_width
            h_svg = (y1 - y0 + 1) / depth_img.shape[0] * svg_height
            png_layers.append(f'<image x="{x_svg}" y="{y_svg}" width="{w_svg}" height="{h_svg}" xlink:href="data:image/png;base64,{png_b64}" />')
    else:
        # Include the entire image as a single layer
        # Create RGBA image with transparency for black/white pixels
        mask = ((depth_img > 1) & (depth_img < 254)).astype(np.uint8)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgb = np.stack([depth_img]*3, axis=-1)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        img = Image.fromarray(rgba)
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        png_layers.append(f'<image x="0" y="0" width="{svg_width}" height="{svg_height}" xlink:href="data:image/png;base64,{png_b64}" />')

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

    # Write SVG contours if requested
    if getattr(stl_to_depthmap, 'write_svg_contours', False):
        write_svg_contours(depth_img, mesh_dims, min_bound, stl_path, contours)
        print("Contours SVG written with only island outlines.")
    
    # Clean up resources to prevent segmentation fault
    import gc
    del mesh
    if 'depth' in locals():
        del depth
    if 'depth_np' in locals():
        del depth_np
    if 'depth_img' in locals():
        del depth_img
    gc.collect()

if __name__ == "__main__":
    import argparse
    import gc
    import sys
    
    parser = argparse.ArgumentParser(description="Convert STL to depthmap SVG")
    parser.add_argument("stl_path", help="Input STL file")
    parser.add_argument("--start-height", type=float, default=0.0, help="Start height offset (mm)")
    parser.add_argument("--total-height", type=float, default=0.0, help="Total height for depth normalization (mm)")
    parser.add_argument("--only-png", action="store_true", help="Only write PNG, not SVG")
    parser.add_argument("--only-svg", action="store_true", help="Only write SVG, not PNG")
    parser.add_argument("--svg-contours", action="store_true", help="Write SVG file with only contours for each island")
    parser.add_argument("--segment", "-s", action="store_true", help="Split image into separate segments for each island")
    args = parser.parse_args()

    # Feature switches
    stl_to_depthmap.write_png = not args.only_svg
    stl_to_depthmap.write_svg = not args.only_png
    stl_to_depthmap.write_svg_contours = args.svg_contours
    stl_to_depthmap.segment = args.segment

    # Run main conversion with proper exception handling
    try:
        stl_to_depthmap(args.stl_path, start_height=args.start_height, total_height=args.total_height)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Force cleanup to prevent segmentation fault
        gc.collect()
        print("Conversion completed successfully.")

