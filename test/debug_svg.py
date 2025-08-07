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

def analyze_svg_images(svg_path):
    """Analyze all embedded PNG images in an SVG file"""
    
    with open(svg_path, 'r') as f:
        svg_content = f.read()
    
    # Find all base64 encoded PNG images
    png_pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
    matches = re.findall(png_pattern, svg_content)
    
    if not matches:
        print(f"No embedded PNG images found in {svg_path}")
        return
    
    print(f"SVG Analysis: {svg_path}")
    print(f"Found {len(matches)} embedded PNG images")
    print("=" * 60)
    
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
