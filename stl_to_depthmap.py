
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
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

def rasterize_triangle_batch(args):
    """Rasterize a batch of triangles in parallel"""
    triangle_batch, vertices, mesh_dims, min_bound, ss_width, ss_height, batch_idx = args
    
    # Initialize depth buffer for this batch
    depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
    
    for local_idx, triangle in enumerate(triangle_batch):
        v0, v1, v2 = vertices[triangle]
        
        # Convert to screen coordinates (supersampled)
        x0 = (v0[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y0 = (v0[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x1 = (v1[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y1 = (v1[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x2 = (v2[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y2 = (v2[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        
        # Bounding box for this triangle
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(ss_width-1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(ss_height-1, int(np.ceil(max(y0, y1, y2))))
        
        if min_x > max_x or min_y > max_y:
            continue
        
        # Rasterize triangle using barycentric coordinates
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                # Calculate barycentric coordinates
                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < 1e-10:
                    continue
                
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1 - w0 - w1
                
                # Check if point is inside triangle (with small tolerance)
                tolerance = -1e-6
                if w0 >= tolerance and w1 >= tolerance and w2 >= tolerance:
                    # Interpolate depth
                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    # Update depth buffer (z-buffer test)
                    if depth > depth_buffer[py, px]:
                        depth_buffer[py, px] = depth
    
    return depth_buffer, batch_idx

# Configure Open3D for headless operation
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

def stl_to_depthmap(stl_path):
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

    if getattr(stl_to_depthmap, 'verbose', False):
        print(f"Mesh dimensions: {mesh_dims}")
        print(f"Mesh bounds: min {min_bound}, max {max_bound}")
        print(f"Mesh height: {mesh_height}")

    center = np.array([0,0,0])
    if getattr(stl_to_depthmap, 'verbose', False):
        print(f"Center: {center}")

    # Use optimized triangle rasterization with supersampling for anti-aliasing
    try:
        if getattr(stl_to_depthmap, 'verbose', False):
            print("Creating smooth depth image using optimized triangle rasterization...")
        
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"Rasterizing {len(triangles)} triangles with anti-aliasing...")
        
        # Use supersampling for anti-aliasing (render at 2x resolution)
        ss_factor = 2
        ss_width = width * ss_factor
        ss_height = height * ss_factor
        
        # Parallel triangle rasterization
        num_cores = min(cpu_count(), 8)  # Limit to 8 cores to avoid overwhelming system
        batch_size = max(100, len(triangles) // (num_cores * 4))  # Adaptive batch size
        
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"Using {num_cores} cores with batch size {batch_size}")
        
        # Split triangles into batches for parallel processing
        triangle_batches = []
        for i in range(0, len(triangles), batch_size):
            batch = triangles[i:i + batch_size]
            batch_idx = i // batch_size
            triangle_batches.append((batch, vertices, mesh_dims, min_bound, ss_width, ss_height, batch_idx))
        
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"Processing {len(triangle_batches)} batches in parallel...")
        
        # Process batches in parallel
        if len(triangle_batches) > 1 and num_cores > 1:
            with Pool(processes=num_cores) as pool:
                results = pool.map(rasterize_triangle_batch, triangle_batches)
        else:
            # Fall back to serial processing for small datasets
            results = [rasterize_triangle_batch(batch) for batch in triangle_batches]
        
        # Combine results from all batches
        depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
        
        if getattr(stl_to_depthmap, 'verbose', False):
            print("Combining results from parallel processing...")
        for batch_result, batch_idx in results:
            # Merge depth buffers using z-buffer test
            mask = batch_result > depth_buffer
            depth_buffer[mask] = batch_result[mask]
        
        if getattr(stl_to_depthmap, 'verbose', False):
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
        
        if getattr(stl_to_depthmap, 'verbose', False):
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
        if getattr(stl_to_depthmap, 'verbose', False):
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

def stl_to_depthmap_sliced(stl_path, slice_height):
    """Create multiple depth map slices from an STL file"""
    import gc
    
    # Load STL mesh (same as original function)
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    # Find the largest face and align mesh (same as original function)
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

    # Align mesh to Z-axis
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

    # Get mesh dimensions after alignment
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    mesh_dims = max_bound - min_bound
    mesh_height = mesh_dims[2]
    
    if getattr(stl_to_depthmap, 'verbose', False):
        print(f"Mesh height: {mesh_height}mm")
        print(f"Slice height: {slice_height}mm")
    
    # Calculate number of slices needed
    num_slices = int(np.ceil(mesh_height / slice_height))
    if getattr(stl_to_depthmap, 'verbose', False):
        print(f"Creating {num_slices} slices of {slice_height}mm each")
    
    # Step 1: Generate full contour on complete model first
    if getattr(stl_to_depthmap, 'verbose', False):
        print("Generating overall contours from full model...")
    full_depth_img, full_contours, mesh_info = generate_depth_image(mesh)
    
    # Step 2: Create individual slices
    slice_data = []
    base_filename = os.path.splitext(stl_path)[0]
    
    for slice_idx in range(num_slices):
        slice_start = slice_idx * slice_height
        slice_end = min((slice_idx + 1) * slice_height, mesh_height)
        
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"\nSlice {slice_idx + 1} ({slice_start:.1f}mm to {slice_end:.1f}mm):")
        
        # Generate depth image for this slice
        slice_depth_img = generate_slice_depth_image(mesh, slice_start, slice_end, mesh_height, mesh_info)
        
        # Count non-transparent pixels and get value range
        non_transparent = (slice_depth_img > 0) & (slice_depth_img < 255)  # Gradients within slice
        white_pixels = (slice_depth_img == 255)  # Surfaces exceeding slice depth
        total_non_transparent = np.sum(non_transparent) + np.sum(white_pixels)
        
        if getattr(stl_to_depthmap, 'verbose', False):
            if total_non_transparent > 0:
                print(f"  Non-transparent pixels: {total_non_transparent}")
                if np.sum(white_pixels) > 0:
                    print(f"    (including {np.sum(white_pixels)} pure white pixels from surfaces exceeding slice depth)")
                if np.any(non_transparent):
                    print(f"  Value range: {np.min(slice_depth_img[non_transparent])} to {np.max(slice_depth_img[non_transparent | white_pixels])}")
                else:
                    print(f"  Value range: 255 to 255 (all pure white)")
            else:
                print(f"  Non-transparent pixels: 0")
                print(f"  Value range: N/A to N/A")
        
        slice_data.append({
            'depth_img': slice_depth_img,
            'start': slice_start,
            'end': slice_end,
            'index': slice_idx
        })
    
    # Step 3: Generate SVG with all slices
    if getattr(stl_to_depthmap, 'write_svg', True):
        generate_sliced_svg(slice_data, full_contours, mesh_info, base_filename, mesh_height)
    
    # Step 4: Generate individual PNG files if requested
    if getattr(stl_to_depthmap, 'write_png', True):
        png_path = base_filename + ".png"
        # Use the full depth image (not sliced) for PNG output
        img_obj = Image.fromarray(full_depth_img)
        img_obj.save(png_path, format="PNG")
        print(f"PNG saved to {png_path}")

def generate_depth_image(mesh):
    """Generate depth image and return mesh info for slicing"""
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.get_min_bound()
    max_bound = bounds.get_max_bound()
    mesh_dims = max_bound - min_bound

    # Use same rendering logic as main function
    x_len, y_len = mesh_dims[0], mesh_dims[1]
    if x_len > y_len:
        width = 1000
        height = int(1000 * y_len / x_len)
    else:
        height = 1000  
        width = int(1000 * x_len / y_len)

    # Use the triangle rasterization from main function
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    # Render full depth map
    ss_factor = 2
    ss_width = width * ss_factor
    ss_height = height * ss_factor
    depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
    
    for tri_idx, triangle in enumerate(triangles):
        v0, v1, v2 = vertices[triangle]
        
        # Convert to screen coordinates
        x0 = (v0[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y0 = (v0[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x1 = (v1[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y1 = (v1[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x2 = (v2[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y2 = (v2[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        
        # Rasterize triangle (same logic as main function)
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(ss_width-1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(ss_height-1, int(np.ceil(max(y0, y1, y2))))
        
        if min_x > max_x or min_y > max_y:
            continue
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < 1e-10:
                    continue
                
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1 - w0 - w1
                
                tolerance = -1e-6
                if w0 >= tolerance and w1 >= tolerance and w2 >= tolerance:
                    depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if depth > depth_buffer[py, px]:
                        depth_buffer[py, px] = depth
    
    # Downsample
    valid_mask = depth_buffer != -np.inf
    if np.any(valid_mask):
        min_depth = np.min(depth_buffer[valid_mask])
        depth_buffer[~valid_mask] = min_depth
    else:
        depth_buffer[:] = 0
    
    depth_downsampled = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            y_start = y * ss_factor
            x_start = x * ss_factor
            region = depth_buffer[y_start:y_start+ss_factor, x_start:x_start+ss_factor]
            depth_downsampled[y, x] = np.mean(region)
    
    # Apply smoothing
    from scipy import ndimage
    depth_np = ndimage.gaussian_filter(depth_downsampled, sigma=0.5)
    
    # Normalize and convert to image
    depth_min = np.min(depth_np)
    depth_max = np.max(depth_np)
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)
    
    # Generate contours from full image
    padded = np.pad(depth_img, pad_width=1, mode='constant', constant_values=0)
    mask = (padded == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove outermost contour
    contour_areas = [cv2.contourArea(c) for c in contours]
    if contour_areas:
        max_idx = np.argmax(contour_areas)
        contours = [c for i, c in enumerate(contours) if i != max_idx]
    
    # Create global transparency mask - areas that are pure black (0) or pure white (255) in full depth
    # These correspond to areas with no surface detail or extreme depth values
    global_transparency_mask = (depth_img == 0) | (depth_img == 255)
    
    # Create global segmentation mask for consistent island boundaries across all slices
    segment_mask = ((depth_img > 1) & (depth_img < 254)).astype(np.uint8)
    global_num_labels, global_labels, global_stats, global_centroids = cv2.connectedComponentsWithStats(segment_mask, connectivity=8)
    
    mesh_info = {
        'min_bound': min_bound,
        'max_bound': max_bound,
        'mesh_dims': mesh_dims,
        'width': width,
        'height': height,
        'depth_min': depth_min,
        'depth_max': depth_max,
        'global_transparency_mask': global_transparency_mask,
        'global_segmentation': {
            'num_labels': global_num_labels,
            'labels': global_labels,
            'stats': global_stats,
            'centroids': global_centroids
        }
    }
    
    return depth_img, contours, mesh_info

def rasterize_slice_triangle_batch(args):
    """Rasterize a batch of triangles for a slice in parallel"""
    triangle_batch, vertices, mesh_dims, min_bound, ss_width, ss_height, slice_start, slice_end, mesh_height, batch_idx = args
    
    depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
    
    for triangle in triangle_batch:
        v0, v1, v2 = vertices[triangle]
        
        # Check if triangle intersects with slice range or is above it
        tri_min_z = min(v0[2], v1[2], v2[2])
        tri_max_z = max(v0[2], v1[2], v2[2])
        
        # Skip only triangles that are completely below the slice start
        if tri_max_z < slice_start:
            continue  # Triangle completely below slice - transparent
        
        # Convert to screen coordinates
        x0 = (v0[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y0 = (v0[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x1 = (v1[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y1 = (v1[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        x2 = (v2[0] - min_bound[0]) / mesh_dims[0] * (ss_width - 1)
        y2 = (v2[1] - min_bound[1]) / mesh_dims[1] * (ss_height - 1)
        
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(ss_width-1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(ss_height-1, int(np.ceil(max(y0, y1, y2))))
        
        if min_x > max_x or min_y > max_y:
            continue
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                if abs(denom) < 1e-10:
                    continue
                
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
                w2 = 1 - w0 - w1
                
                tolerance = -1e-6
                if w0 >= tolerance and w1 >= tolerance and w2 >= tolerance:
                    interpolated_z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    
                    # For slicing, filter out surfaces that are below the slice
                    # (these should be transparent as they're nearer than slice start)
                    if interpolated_z < slice_start:
                        continue  # Skip - will be transparent (nearer than slice)
                    
                    # Store the actual Z coordinate for surfaces at or above slice start
                    final_depth = interpolated_z
                    
                    if final_depth > depth_buffer[py, px]:
                        depth_buffer[py, px] = final_depth
    
    return depth_buffer, batch_idx

def generate_slice_depth_image(mesh, slice_start, slice_end, mesh_height, mesh_info):
    """Generate depth image for a specific slice using parallel processing"""
    from scipy import ndimage
    
    # Extract mesh info
    min_bound = mesh_info['min_bound']
    mesh_dims = mesh_info['mesh_dims']
    width = mesh_info['width']
    height = mesh_info['height']
    
    # Render slice with parallel processing
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    ss_factor = 2
    ss_width = width * ss_factor
    ss_height = height * ss_factor
    
    # Parallel processing for slice (use fewer cores than main rendering)
    num_cores = min(cpu_count() // 2, 4)  # Use fewer cores for slices
    batch_size = max(100, len(triangles) // (num_cores * 2))
    
    # Split triangles into batches
    triangle_batches = []
    for i in range(0, len(triangles), batch_size):
        batch = triangles[i:i + batch_size]
        batch_idx = i // batch_size
        triangle_batches.append((batch, vertices, mesh_dims, min_bound, ss_width, ss_height, 
                               slice_start, slice_end, mesh_height, batch_idx))
    
    # Process batches in parallel
    if len(triangle_batches) > 1 and num_cores > 1:
        with Pool(processes=num_cores) as pool:
            results = pool.map(rasterize_slice_triangle_batch, triangle_batches)
    else:
        results = [rasterize_slice_triangle_batch(batch) for batch in triangle_batches]
    
    # Combine results
    depth_buffer = np.full((ss_height, ss_width), -np.inf, dtype=np.float32)
    for batch_result, batch_idx in results:
        mask = batch_result > depth_buffer
        depth_buffer[mask] = batch_result[mask]
    
    # Downsample with anti-aliasing
    depth_downsampled = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            y_start = y * ss_factor
            x_start = x * ss_factor
            region = depth_buffer[y_start:y_start+ss_factor, x_start:x_start+ss_factor]
            # Handle -inf values (transparent areas)
            valid_region = region[region != -np.inf]
            if len(valid_region) > 0:
                depth_downsampled[y, x] = np.mean(valid_region)
            else:
                depth_downsampled[y, x] = -np.inf
    
    # Apply smoothing only to valid areas
    smoothed = ndimage.gaussian_filter(depth_downsampled, sigma=0.5)
    depth_np = np.where(depth_downsampled != -np.inf, smoothed, depth_downsampled)
    
    # Convert to image with proper transparency and black handling
    depth_img = np.zeros((height, width), dtype=np.uint8)
    
    valid_mask = depth_np != -np.inf
    if np.any(valid_mask):
        # Separate pixels based on their relationship to the slice
        exceeds_slice_mask = depth_np > slice_end  # Surfaces above slice - pure black
        within_slice_mask = (depth_np >= slice_start) & (depth_np <= slice_end)  # Surfaces within slice - gradients
        # Note: surfaces below slice_start are already filtered out in rasterization (transparent)
        
        # Handle pixels within slice range - show as gradients
        if np.any(within_slice_mask):
            valid_depths = depth_np[within_slice_mask]
            # Normalize depths within slice range: slice_start=black(1), slice_end=light_gray(254)
            depth_norm = (valid_depths - slice_start) / (slice_end - slice_start + 1e-8)
            depth_norm = np.clip(depth_norm, 0, 1)
            # Map to 1-254 range (avoiding pure black 0 and pure white 255)
            depth_img[within_slice_mask] = (depth_norm * 253 + 1).astype(np.uint8)
        
        # Handle pixels that exceed slice depth - pure white (opaque)
        depth_img[exceeds_slice_mask] = 255  # Pure white for surfaces exceeding slice
        
        # All other pixels remain 0 (transparent):
        # - Areas with no surfaces (transparent in full model) 
        # - Surfaces nearer than slice_start (filtered out in rasterization)
        
        # Transparent areas remain 0
        depth_img[~valid_mask] = 0  # Transparent areas
    
    # Apply global transparency mask - areas that are pure black/white in full depth model
    # should be transparent in ALL slices (these are areas with no surface detail)
    global_transparency_mask = mesh_info['global_transparency_mask']
    depth_img[global_transparency_mask] = 255  # Override with transparency
    
    return depth_img

def generate_sliced_svg(slice_data, contours, mesh_info, base_filename, mesh_height):
    """Generate SVG with multiple depth image layers"""
    from io import BytesIO
    
    svg_width = mesh_info['mesh_dims'][0]
    svg_height = mesh_info['mesh_dims'][1]
    min_bound = mesh_info['min_bound']
    width = mesh_info['width']
    height = mesh_info['height']
    
    if getattr(stl_to_depthmap, 'verbose', False):
        print(f"\nGenerating SVG layers for {len(slice_data)} depth images")
    
    # Create PNG layers for each slice
    png_layers = []
    
    for i, slice_info in enumerate(slice_data):
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"\nProcessing layer {i+1}:")
        depth_img = slice_info['depth_img']
        
        # Count non-transparent pixels before processing
        non_transparent_before = np.sum((depth_img > 0) & (depth_img < 255))
        black_pixels_before = np.sum(depth_img == 0)
        total_before = non_transparent_before + black_pixels_before
        if getattr(stl_to_depthmap, 'verbose', False):
            print(f"  Non-transparent pixels before conversion: {total_before}")
            if black_pixels_before > 0:
                print(f"    (including {black_pixels_before} pure black pixels)")
        
        if total_before > 0:
            all_visible = (depth_img > 0)
            if getattr(stl_to_depthmap, 'verbose', False):
                print(f"  Value range before conversion: {np.min(depth_img[all_visible])} to {np.max(depth_img[all_visible])}")
        
        if getattr(stl_to_depthmap, 'segment', False):
            # Segmented mode - use global segmentation for consistent islands across all slices
            global_segmentation = mesh_info['global_segmentation']
            global_labels = global_segmentation['labels']
            global_num_labels = global_segmentation['num_labels']

            for label in range(1, global_num_labels):
                island_mask = (global_labels == label)
                coords = np.argwhere(island_mask)
                if coords.size == 0:
                    continue
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                
                # Crop depth data for this island from the current slice
                cropped_depth = depth_img[y0:y1+1, x0:x1+1]
                cropped_island_mask = island_mask[y0:y1+1, x0:x1+1]
                
                # Apply island mask - only show pixels that belong to this island
                masked_depth = np.where(cropped_island_mask, cropped_depth, 255)
                
                # Create RGBA image with transparency for value 0
                rgba = np.zeros((masked_depth.shape[0], masked_depth.shape[1], 4), dtype=np.uint8)
                rgba[:, :, 0] = masked_depth  # R
                rgba[:, :, 1] = masked_depth  # G  
                rgba[:, :, 2] = masked_depth  # B
                rgba[:, :, 3] = np.where(masked_depth == 255, 0, 255)  # A: transparent for 255, opaque otherwise
                
                # Debug: Count pixel types in this island
                if getattr(stl_to_depthmap, 'verbose', False):
                    total_pixels = rgba.shape[0] * rgba.shape[1]
                    white_pixels = np.sum(masked_depth == 255)  # Surfaces exceeding slice depth
                    transparent_pixels = np.sum(masked_depth == 0)  # Transparent areas (global mask + outside island)
                    gray_pixels = np.sum((masked_depth > 0) & (masked_depth < 255))  # Gradients within slice
                    print(f"    Island {label}: {total_pixels} total pixels ({white_pixels} white, {gray_pixels} gray, {transparent_pixels} transparent)")
                
                img = Image.fromarray(rgba)
                buf = BytesIO()
                img.save(buf, format="PNG")
                png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                
                x_svg = x0 / width * svg_width
                y_svg = y0 / height * svg_height
                w_svg = (x1 - x0 + 1) / width * svg_width
                h_svg = (y1 - y0 + 1) / height * svg_height
                png_layers.append(f'<image x="{x_svg}" y="{y_svg}" width="{w_svg}" height="{h_svg}" xlink:href="data:image/png;base64,{png_b64}" />')
        else:
            # Non-segmented mode - entire image as one layer
            # Create RGBA image with transparency for value 255
            rgba = np.zeros((depth_img.shape[0], depth_img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, 0] = depth_img  # R
            rgba[:, :, 1] = depth_img  # G
            rgba[:, :, 2] = depth_img  # B 
            rgba[:, :, 3] = np.where(depth_img == 255, 0, 255)  # A: transparent for 255, opaque otherwise
            
            # Debug: Count pixel types in the full image
            if getattr(stl_to_depthmap, 'verbose', False):
                total_pixels = rgba.shape[0] * rgba.shape[1]
                black_pixels = np.sum(depth_img == 0)
                transparent_pixels = np.sum(depth_img == 0)
                gray_pixels = np.sum((depth_img > 0) & (depth_img < 255))
                print(f"    Full image: {total_pixels} total pixels ({black_pixels} black, {gray_pixels} gray, {transparent_pixels} transparent)")
            
            img = Image.fromarray(rgba)
            buf = BytesIO()
            img.save(buf, format="PNG")
            png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            png_layers.append(f'<image x="0" y="0" width="{svg_width}" height="{svg_height}" xlink:href="data:image/png;base64,{png_b64}" />')
    
    # Build contour paths from full model
    path_elems = []
    for contour in contours:
        if len(contour) < 2:
            continue
        points = []
        for pt in contour.squeeze():
            if len(pt.shape) == 0:
                continue
            x_px, y_px = pt
            x_model = x_px / width * mesh_info['mesh_dims'][0] + min_bound[0]
            y_model = y_px / height * mesh_info['mesh_dims'][1]
            points.append(f"{x_model},{y_model}")
        if points:
            path_str = "M " + " L ".join(points) + " Z"
            path_elems.append(f'<path d="{path_str}" stroke="rgb(255,219,102)" fill="none" stroke-width="0.5" />')
    
    # Write SVG
    svg_template = f'''<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{svg_width}mm" height="{svg_height}mm" viewBox="0 0 {svg_width} {svg_height}">
        {''.join(png_layers)}
        {''.join(path_elems)}
    </svg>
    '''
    svg_path = base_filename + ".svg"
    with open(svg_path, "w") as f:
        f.write(svg_template)
    print(f"SVG saved to {svg_path}")

if __name__ == "__main__":
    import argparse
    import gc
    import sys
    
    parser = argparse.ArgumentParser(description="Convert STL to depthmap SVG")
    parser.add_argument("stl_path", help="Input STL file")
    parser.add_argument("--slice-height", type=float, help="Height of each slice in mm (enables slicing mode)")
    parser.add_argument("--only-png", action="store_true", help="Only write PNG, not SVG")
    parser.add_argument("--only-svg", action="store_true", help="Only write SVG, not PNG")
    parser.add_argument("--svg-contours", action="store_true", help="Write SVG file with only contours for each island")
    parser.add_argument("--segment", "-s", action="store_true", help="Split image into separate segments for each island")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Feature switches
    stl_to_depthmap.write_png = not args.only_svg
    stl_to_depthmap.write_svg = not args.only_png
    stl_to_depthmap.write_svg_contours = args.svg_contours
    stl_to_depthmap.segment = args.segment
    stl_to_depthmap.verbose = args.verbose

    # Run main conversion with proper exception handling
    try:
        if args.slice_height:
            # Slicing mode enabled
            stl_to_depthmap_sliced(args.stl_path, slice_height=args.slice_height)
        else:
            # Regular single depth map mode
            stl_to_depthmap(args.stl_path)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Force cleanup to prevent segmentation fault
        gc.collect()
        print("Conversion completed successfully.")

