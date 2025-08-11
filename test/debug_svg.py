#!/usr/bin/env python3
"""
SVG Debug Utility
Analyzes SVG files containing embedded PNG images and reports pixel statistics.
"""

import base64
import re
import sys
from io import BytesIO
import numpy as np
from PIL import Image

def analyze_svg_paths(svg_content):
    """Analyze all SVG paths in the content"""
    
    # Find all path elements
    path_pattern = r'<path[^>]*\bd=["\']([^"\']*)["\'][^>]*>'
    path_matches = re.findall(path_pattern, svg_content, re.IGNORECASE)
    
    # Also find paths without quotes around d attribute
    path_pattern_alt = r'<path[^>]*\bd=([^\s>]+)[^>]*>'
    path_matches_alt = re.findall(path_pattern_alt, svg_content, re.IGNORECASE)
    
    all_paths = path_matches + path_matches_alt
    
    print(f"\nSVG Path Analysis:")
    print(f"Found {len(all_paths)} path elements")
    
    if not all_paths:
        return
    
    print("=" * 40)
    
    # Analyze path commands
    command_counts = {}
    total_coordinates = 0
    x_coords = []
    y_coords = []
    
    for i, path_data in enumerate(all_paths, 1):
        print(f"\nPath {i}:")
        print(f"  Data length: {len(path_data)} characters")
        
        # Extract commands (letters)
        commands = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]', path_data)
        print(f"  Commands: {len(commands)} total")
        
        # Count each command type
        path_commands = {}
        for cmd in commands:
            cmd_upper = cmd.upper()
            path_commands[cmd_upper] = path_commands.get(cmd_upper, 0) + 1
            command_counts[cmd_upper] = command_counts.get(cmd_upper, 0) + 1
        
        if path_commands:
            cmd_summary = ', '.join([f"{cmd}:{count}" for cmd, count in sorted(path_commands.items())])
            print(f"  Command breakdown: {cmd_summary}")
        
        # Extract numeric coordinates
        coord_pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
        coordinates = re.findall(coord_pattern, path_data)
        
        if coordinates:
            coords = [float(c) for c in coordinates]
            print(f"  Coordinates: {len(coords)} numbers")
            
            # Assuming pairs of x,y coordinates for most commands
            if len(coords) >= 2:
                x_vals = coords[::2]  # Even indices
                y_vals = coords[1::2]  # Odd indices
                
                if x_vals and y_vals:
                    print(f"  X range: {min(x_vals):.2f} to {max(x_vals):.2f}")
                    print(f"  Y range: {min(y_vals):.2f} to {max(y_vals):.2f}")
                    
                    x_coords.extend(x_vals)
                    y_coords.extend(y_vals)
            
            total_coordinates += len(coords)
    
    # Overall statistics
    print(f"\nOverall Path Statistics:")
    print(f"  Total paths: {len(all_paths)}")
    print(f"  Total coordinates: {total_coordinates}")
    
    if command_counts:
        print(f"  Command summary:")
        for cmd, count in sorted(command_counts.items()):
            cmd_name = {
                'M': 'MoveTo', 'L': 'LineTo', 'H': 'HorizontalLineTo', 'V': 'VerticalLineTo',
                'C': 'CurveTo', 'S': 'SmoothCurveTo', 'Q': 'QuadraticCurveTo', 'T': 'SmoothQuadraticCurveTo',
                'A': 'EllipticalArc', 'Z': 'ClosePath'
            }.get(cmd, cmd)
            print(f"    {cmd} ({cmd_name}): {count}")
    
    if x_coords and y_coords:
        print(f"  Overall coordinate ranges:")
        print(f"    X: {min(x_coords):.2f} to {max(x_coords):.2f}")
        print(f"    Y: {min(y_coords):.2f} to {max(y_coords):.2f}")

def analyze_svg_images(svg_path):
    """Analyze all embedded PNG images in an SVG file"""
    
    with open(svg_path, 'r') as f:
        svg_content = f.read()
    
    print(f"SVG Analysis: {svg_path}")
    print("=" * 60)
    
    # Analyze SVG paths first
    analyze_svg_paths(svg_content)
    
    # Find all base64 encoded PNG images
    png_pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
    matches = re.findall(png_pattern, svg_content)
    
    if not matches:
        print(f"\nNo embedded PNG images found in {svg_path}")
        return
    
    print(f"\nPNG Image Analysis:")
    print(f"Found {len(matches)} embedded PNG images")
    print("=" * 40)
    
    for i, base64_data in enumerate(matches, 1):
        print(f"\nImage {i}:")
        analyze_png_from_base64(base64_data)

def analyze_png_from_base64(base64_data):
    """Analyze a PNG image from base64 data"""
    
    try:
        # Decode base64 to bytes
        png_bytes = base64.b64decode(base64_data)
        
        # Load image
        img = Image.open(BytesIO(png_bytes))
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract channels
        r = img_array[:, :, 0]
        g = img_array[:, :, 1] 
        b = img_array[:, :, 2]
        a = img_array[:, :, 3]
        
        # Check if it's grayscale (R=G=B)
        is_grayscale = np.all(r == g) and np.all(g == b)
        
        print(f"  Image size: {img.width}x{img.height}")
        print(f"  Mode: {img.mode}")
        print(f"  Is grayscale: {is_grayscale}")
        
        if is_grayscale:
            # Analyze grayscale values
            gray_values = r  # Since R=G=B, just use R channel
            
            # Count pixel types
            transparent_pixels = np.sum(a == 0)
            opaque_pixels = np.sum(a == 255)
            semi_transparent_pixels = np.sum((a > 0) & (a < 255))
            
            print(f"  Alpha channel:")
            print(f"    Transparent (alpha=0): {transparent_pixels:,} pixels")
            print(f"    Semi-transparent (alpha 1-254): {semi_transparent_pixels:,} pixels")
            print(f"    Opaque (alpha=255): {opaque_pixels:,} pixels")
            
            if opaque_pixels > 0:
                # Analyze opaque pixels only
                opaque_mask = a == 255
                opaque_gray_values = gray_values[opaque_mask]
                
                black_pixels = np.sum(opaque_gray_values == 0)
                white_pixels = np.sum(opaque_gray_values == 255)
                gray_pixels = np.sum((opaque_gray_values > 0) & (opaque_gray_values < 255))
                
                print(f"  Opaque pixel values:")
                print(f"    Black (value=0): {black_pixels:,} pixels")
                print(f"    Gray (values 1-254): {gray_pixels:,} pixels")
                print(f"    White (value=255): {white_pixels:,} pixels")
                
                if gray_pixels > 0:
                    print(f"    Gray value range: {opaque_gray_values[opaque_gray_values < 255].min()} to {opaque_gray_values[opaque_gray_values > 0].max()}")
            
            if transparent_pixels > 0:
                # Analyze what values the transparent pixels had
                transparent_mask = a == 0
                transparent_gray_values = gray_values[transparent_mask]
                
                unique_transparent_values = np.unique(transparent_gray_values)
                print(f"  Transparent pixel underlying values: {unique_transparent_values}")
        
        else:
            print(f"  Not grayscale - full color analysis not implemented")
            print(f"  R range: {r.min()}-{r.max()}")
            print(f"  G range: {g.min()}-{g.max()}")
            print(f"  B range: {b.min()}-{b.max()}")
            print(f"  A range: {a.min()}-{a.max()}")
    
    except Exception as e:
        print(f"  Error analyzing image: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_svg.py <svg_file>")
        sys.exit(1)
    
    svg_path = sys.argv[1]
    analyze_svg_images(svg_path)

if __name__ == "__main__":
    main()
