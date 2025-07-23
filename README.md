# 3DtoDepthmap

This project converts 3D STL mesh files into depth map images and SVGs suitable for laser engraving, CNC, or other fabrication workflows. It uses Open3D for mesh processing and rendering, and outputs both raster (PNG) and vector (SVG) formats.

## Features
- Converts STL files to depth map PNGs
- Embeds PNGs into SVGs with real-world dimensions (mm)
- Optionally outputs only PNG or only SVG
- Supports cropping and layering of islands in the depth map
- Allows custom start height and total height for depth normalization

## Requirements
- Python 3.8+
- open3d
- numpy
- Pillow
- opencv-python

Install dependencies with:
```bash
pip install open3d numpy Pillow opencv-python
```

## Usage

```bash
python stl_to_depthmap.py <input.stl> [--start-height HEIGHT] [--total-height HEIGHT] [--only-png] [--only-svg]
```

### Arguments
- `<input.stl>`: Path to the STL file to convert.
- `--start-height HEIGHT`: (Optional) Start height offset in mm (default: 0.0).
- `--total-height HEIGHT`: (Optional) Total height for depth normalization in mm (default: 0.0, uses mesh height).
- `--only-png`: Only write the PNG file, not the SVG.
- `--only-svg`: Only write the SVG file, not the PNG.

### Example
Convert an STL to both PNG and SVG:
```bash
python stl_to_depthmap.py foam.stl
```

Convert with a custom height range and only output SVG:
```bash
python stl_to_depthmap.py foam.stl --start-height 2 --total-height 10 --only-svg
```

## Output
- `<input>.png`: The depth map as a PNG image.
- `<input>.svg`: The SVG with embedded PNG layers and vector contours, sized in mm to match the mesh.

## License
MIT
