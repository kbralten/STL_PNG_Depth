# 3DtoDepthmap

This project converts 3D STL mesh files into depth map images and SVGs suitable for laser engraving, CNC, or other fabrication workflows. It uses Open3D for mesh processing and rendering, and outputs both raster (PNG) and vector (SVG) formats.

## Features
- Converts STL files to depth map PNGs
- Embeds PNGs into SVGs with real-world dimensions (mm)
- Replaces any full-depth islands and the perimeter with SVG curves
- Optionally outputs only PNG or only SVG
- The SVG output has cropping and layering of islands in the depth map to create unique regions instead of a global image
- Allows custom start height and total height for depth normalization and/or partial cuts
- Auto orients the mesh so the largest surface area is "down" away from the camera. On most models this puts the side to be engraved (which has a smaller surface area because of any recesses) up toward the camera.

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
python stl_to_depthmap.py <input.stl> [--start-height HEIGHT] [--total-height HEIGHT] [--only-png] [--only-svg] [--svg-contours]
```


### Arguments
- `<input.stl>`: Path to the STL file to convert.
- `--start-height HEIGHT`: (Optional) Start height offset in mm (default: 0.0).
- `--total-height HEIGHT`: (Optional) Total height for depth normalization in mm (default: 0.0, uses mesh height).
- `--only-png`: Only write the PNG file, not the SVG.
- `--only-svg`: Only write the SVG file, not the PNG.
- `--svg-contours`: Write a separate SVG file (`<input>-contours.svg`) containing only the vector contours for each island and the overall outline.


### Example
Convert an STL to both PNG and SVG:
```bash
python stl_to_depthmap.py foam.stl
```

Convert with a custom height range and only output SVG:
```bash
python stl_to_depthmap.py foam.stl --start-height 2 --total-height 10 --only-svg
```

Generate a separate SVG with only contours:
```bash
python stl_to_depthmap.py foam.stl --svg-contours
```

## Output
- `<input>.png`: The depth map as a PNG image.
- `<input>.svg`: The SVG with embedded PNG layers and vector contours, sized in mm to match the mesh.

## Notes
 - Throws a lot of warnings, but no functional impact.
   - throws a segfault when completing
   - the camera perspective may throw a warning.

## License
GPL v2