# 3DtoDepthmap

This project converts 3D STL mesh files into depth map images and SVGs suitable for laser engraving, CNC, or other fabrication workflows. It includes:

- **A full-featured Python program** for advanced mesh processing, PNG and SVG output, and vectorization.
- **A web-based PNG-only version** (see `stl_to_depthmap.html`) for quick STL-to-depthmap conversion in your browser, no install required.

## Features

### Python Program
- Converts STL files to depth map PNGs
- *Embeds PNGs into SVGs* with *real-world dimensions (mm)*
- Replaces any full-depth *islands and the perimeter with SVG curves*
- Optionally outputs *only PNG or only SVG*
- **Supports slicing**: Allows splitting the depth map into multiple layers for fabrication workflows. See below for details on slicing height behavior.
- **Optional segmentation**: Segmentation of islands and contours can be toggled to either create single images (per layer) or segment into islands (per layer).
  - The SVG output has cropping and layering of islands in the depth map to create unique regions instead of a global image
- *Auto orients the mesh* so the largest surface area is "down" away from the camera. On most models this puts the side to be engraved (which has a smaller surface area because of any recesses) up toward the camera.

### Web Version
- Converts *STL files to grayscale PNG depth maps* directly in your browser
- No installation or Python required
- Supports *interactive 3D preview* and depth map download
- No SVG/vector output (PNG only)

## Requirements

### Python Program
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


### Python Program

```bash
python stl_to_depthmap.py <input.stl> [--slice-height HEIGHT] [--only-png] [--only-svg] [--svg-contours] [--segment] [--verbose]
```

#### Arguments
- `<input.stl>`: Path to the STL file to convert.
- `--slice-height HEIGHT`: (Optional) Height of each slice in mm (enables slicing mode).
  - **Slicing explainer:** If the model height is not a multiple of the slice height, the bottom slice will be the smaller remainder, and all higher slices will be full height. For example, a 40mm model with `--slice-height 15` will produce:
    - Slice 1: 0–10mm (bottom, remainder)
    - Slice 2: 10–25mm
    - Slice 3: 25–40mm (top)
- `--only-png`: Only write the PNG file, not the SVG.
- `--only-svg`: Only write the SVG file, not the PNG.
- `--svg-contours`: Write a separate SVG file (`<input>-contours.svg`) containing only the vector contours for each island and the overall outline.
- `--segment`: Enable segmentation of islands and contours.
- `--verbose`: Print detailed progress and debug output during processing.

#### Example
Convert an STL to both PNG and SVG:
```bash
python stl_to_depthmap.py foam.stl
```

Convert with slicing into 5 mm layers and only output SVG:
```bash
python stl_to_depthmap.py foam.stl --slice-height 5 --only-svg
```

Generate a separate SVG with only contours and enable segmentation:
```bash
python stl_to_depthmap.py foam.stl --svg-contours --segment
```

## Output
- `<input>.png`: The depth map as a PNG image.
- `<input>.svg`: The SVG with embedded PNG layers and vector contours, sized in mm to match the mesh.  

## Web Version

Open `stl_to_depthmap.html` in your browser. Upload an STL file, select the desired view, and download the PNG depth map.

## License
GPL v2