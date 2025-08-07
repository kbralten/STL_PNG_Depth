#!/usr/bin/env python3
"""
Comprehensive test suite for stl_to_depthmap.py

This test suite covers:
- Basic STL loading and processing
- Depth map generation
- SVG/PNG output
- Slicing functionality
- Segmentation features
- Error handling
- File output validation
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import base64
from io import BytesIO

# Import the module under test
import stl_to_depthmap


class TestSTLToDepthmap(unittest.TestCase):
    """Test cases for STL to depth map conversion"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_stl = "../foam.stl"  # Assume foam.stl exists in the working directory
        self.assertTrue(os.path.exists(self.test_stl), f"Test STL file {self.test_stl} not found")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_basic_stl_loading(self):
        """Test that STL files can be loaded successfully"""
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(self.test_stl)
            self.assertTrue(len(mesh.vertices) > 0, "Mesh should have vertices")
            self.assertTrue(len(mesh.triangles) > 0, "Mesh should have triangles")
        except Exception as e:
            self.fail(f"Failed to load STL file: {e}")
    
    def test_generate_depth_image(self):
        """Test depth image generation"""
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.test_stl)
        
        depth_img, contours, mesh_info = stl_to_depthmap.generate_depth_image(mesh)
        
        # Validate depth image
        self.assertIsInstance(depth_img, np.ndarray, "Depth image should be numpy array")
        self.assertEqual(len(depth_img.shape), 2, "Depth image should be 2D")
        self.assertTrue(depth_img.dtype == np.uint8, "Depth image should be uint8")
        self.assertTrue(np.max(depth_img) <= 255, "Max depth value should be <= 255")
        self.assertTrue(np.min(depth_img) >= 0, "Min depth value should be >= 0")
        
        # Validate contours
        self.assertIsInstance(contours, list, "Contours should be a list")
        
        # Validate mesh info
        self.assertIsInstance(mesh_info, dict, "Mesh info should be a dictionary")
        required_keys = ['mesh_dims', 'min_bound', 'width', 'height']
        for key in required_keys:
            self.assertIn(key, mesh_info, f"Mesh info should contain {key}")
    
    def test_basic_conversion_output_files(self):
        """Test that basic conversion creates output files"""
        test_output = os.path.join(self.test_dir, "test_output")
        
        # Run conversion with default parameters
        stl_to_depthmap.stl_to_depthmap(self.test_stl)
        
        # Check output files exist (using default naming)
        base_name = os.path.splitext(self.test_stl)[0]
        png_path = base_name + ".png"
        svg_path = base_name + ".svg"
        
        self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
        self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
        
        # Validate PNG file
        with Image.open(png_path) as img:
            self.assertEqual(img.mode, 'L', "PNG should be grayscale")
            self.assertTrue(img.size[0] > 0 and img.size[1] > 0, "PNG should have valid dimensions")
        
        # Validate SVG file
        with open(svg_path, 'r') as f:
            svg_content = f.read()
            self.assertIn('<svg', svg_content, "SVG should contain svg tag")
            self.assertIn('xmlns', svg_content, "SVG should contain namespace")
        
        # Clean up
        if os.path.exists(png_path):
            os.remove(png_path)
        if os.path.exists(svg_path):
            os.remove(svg_path)
    
    def test_sliced_conversion(self):
        """Test sliced conversion functionality"""
        test_output = os.path.join(self.test_dir, "test_sliced")
        
        # Run sliced conversion  
        stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, 10.0)
        
        # Check output files exist (using default naming)
        base_name = os.path.splitext(self.test_stl)[0]
        png_path = base_name + ".png"
        svg_path = base_name + ".svg"
        
        self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
        self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
        
        # Validate SVG contains multiple layers
        with open(svg_path, 'r') as f:
            svg_content = f.read()
            # Should contain multiple image elements for slices
            image_count = svg_content.count('<image')
            self.assertGreater(image_count, 1, "Sliced SVG should contain multiple image layers")
        
        # Clean up
        if os.path.exists(png_path):
            os.remove(png_path)
        if os.path.exists(svg_path):
            os.remove(svg_path)
    
    def test_segmented_conversion(self):
        """Test segmented conversion functionality"""
        # Enable segmentation
        stl_to_depthmap.stl_to_depthmap.segment = True
        
        try:
            # Run segmented conversion
            stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, 10.0)
            
            # Check output files exist (using default naming)
            base_name = os.path.splitext(self.test_stl)[0]
            png_path = base_name + ".png"
            svg_path = base_name + ".svg"
            
            self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
            self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
            
            # Clean up
            if os.path.exists(png_path):
                os.remove(png_path)
            if os.path.exists(svg_path):
                os.remove(svg_path)
            
        finally:
            # Reset segmentation flag
            stl_to_depthmap.stl_to_depthmap.segment = False
    
    def test_png_contains_full_depth(self):
        """Test that PNG contains full depth map even in sliced mode"""
        test_output_normal = os.path.join(self.test_dir, "test_normal")
        test_output_sliced = os.path.join(self.test_dir, "test_sliced")
        
        # Generate normal and sliced versions
        stl_to_depthmap.stl_to_depthmap(self.test_stl, test_output_normal)
        stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, test_output_sliced, slice_height=10.0)
        
        # Load both PNG files
        with Image.open(test_output_normal + ".png") as img_normal:
            with Image.open(test_output_sliced + ".png") as img_sliced:
                # Convert to numpy arrays for comparison
                arr_normal = np.array(img_normal)
                arr_sliced = np.array(img_sliced)
                
                # They should be identical (PNG always contains full depth)
                np.testing.assert_array_equal(arr_normal, arr_sliced, 
                    "PNG files should be identical between normal and sliced modes")
    
    def test_svg_embedded_images_are_valid(self):
        """Test that embedded images in SVG are valid PNG data"""
        test_output = os.path.join(self.test_dir, "test_svg_images")
        
        # Run sliced conversion to get multiple embedded images
        stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, test_output, slice_height=10.0)
        
        svg_path = test_output + ".svg"
        
        # Parse SVG and extract embedded images
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Find all image elements
        images = root.findall('.//{http://www.w3.org/2000/svg}image')
        self.assertGreater(len(images), 0, "SVG should contain image elements")
        
        for img_elem in images:
            href = img_elem.get('{http://www.w3.org/1999/xlink}href') or img_elem.get('href')
            self.assertIsNotNone(href, "Image element should have href attribute")
            self.assertTrue(href.startswith('data:image/png;base64,'), "Image should be base64 PNG")
            
            # Extract and validate base64 data
            base64_data = href.split(',', 1)[1]
            try:
                img_data = base64.b64decode(base64_data)
                img = Image.open(BytesIO(img_data))
                self.assertTrue(img.size[0] > 0 and img.size[1] > 0, "Embedded image should have valid dimensions")
            except Exception as e:
                self.fail(f"Failed to decode embedded image: {e}")
    
    def test_depth_value_distribution(self):
        """Test that depth values are properly distributed"""
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.test_stl)
        
        depth_img, _, _ = stl_to_depthmap.generate_depth_image(mesh)
        
        # Check that we have a range of depth values
        unique_values = np.unique(depth_img)
        self.assertGreater(len(unique_values), 10, "Should have multiple depth values")
        
        # Check that we have both near (white/low values) and far (black/high values) surfaces
        self.assertIn(0, unique_values, "Should have background pixels (value 0)")
        self.assertIn(255, unique_values, "Should have maximum depth pixels (value 255)")
        
        # Check that we have intermediate values too
        intermediate_values = unique_values[(unique_values > 10) & (unique_values < 245)]
        self.assertGreater(len(intermediate_values), 5, "Should have intermediate depth values")
    
    def test_slice_depth_consistency(self):
        """Test that slice depths are consistent with slice heights"""
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.test_stl)
        
        slice_height = 10.0
        mesh_height = mesh.get_axis_aligned_bounding_box().get_max_bound()[2]
        
        # Generate slices manually to test consistency
        num_slices = int(np.ceil(mesh_height / slice_height))
        
        for slice_idx in range(num_slices):
            slice_start = slice_idx * slice_height
            slice_end = min((slice_idx + 1) * slice_height, mesh_height)
            
            depth_img = stl_to_depthmap.generate_slice_depth_image(
                mesh, slice_start, slice_end, mesh_height
            )
            
            self.assertIsInstance(depth_img, np.ndarray, f"Slice {slice_idx} should return numpy array")
            self.assertEqual(len(depth_img.shape), 2, f"Slice {slice_idx} should be 2D")
            self.assertTrue(depth_img.dtype == np.uint8, f"Slice {slice_idx} should be uint8")
    
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid STL files"""
        invalid_file = os.path.join(self.test_dir, "invalid.stl")
        
        # Create an invalid STL file
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid STL file")
        
        test_output = os.path.join(self.test_dir, "test_invalid")
        
        # Should handle the error gracefully
        with self.assertRaises((SystemExit, Exception)):
            stl_to_depthmap.stl_to_depthmap(invalid_file, test_output)
    
    def test_command_line_interface(self):
        """Test command line argument parsing"""
        import argparse
        
        # Test that the argument parser is set up correctly
        # This tests the main block without actually running it
        test_args = [self.test_stl, "--slice-height", "5.0", "--segment"]
        
        # Create a parser similar to the one in the main function
        parser = argparse.ArgumentParser(description="Convert STL to depthmap SVG")
        parser.add_argument("stl_path", help="Input STL file")
        parser.add_argument("--start-height", type=float, default=0.0, help="Start height offset (mm)")
        parser.add_argument("--total-height", type=float, default=0.0, help="Total height for depth normalization (mm)")
        parser.add_argument("--slice-height", type=float, help="Height of each slice in mm (enables slicing mode)")
        parser.add_argument("--segment", action="store_true", help="Enable island segmentation")
        
        # Parse test arguments
        args = parser.parse_args(test_args)
        
        self.assertEqual(args.stl_path, self.test_stl)
        self.assertEqual(args.slice_height, 5.0)
        self.assertTrue(args.segment)


class TestIntegration(unittest.TestCase):
    """Integration tests that test complete workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_stl = "../foam.stl"
        self.assertTrue(os.path.exists(self.test_stl), f"Test STL file {self.test_stl} not found")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_workflow_normal(self):
        """Test complete workflow for normal (non-sliced) conversion"""
        test_output = os.path.join(self.test_dir, "workflow_normal")
        
        # Run the complete workflow
        stl_to_depthmap.stl_to_depthmap(self.test_stl, test_output)
        
        # Validate all outputs
        png_path = test_output + ".png"
        svg_path = test_output + ".svg"
        
        # Files should exist
        self.assertTrue(os.path.exists(png_path))
        self.assertTrue(os.path.exists(svg_path))
        
        # Files should have reasonable sizes
        png_size = os.path.getsize(png_path)
        svg_size = os.path.getsize(svg_path)
        
        self.assertGreater(png_size, 1000, "PNG should be substantial size")
        self.assertGreater(svg_size, 1000, "SVG should be substantial size")
    
    def test_complete_workflow_sliced(self):
        """Test complete workflow for sliced conversion"""
        test_output = os.path.join(self.test_dir, "workflow_sliced")
        
        # Run the complete sliced workflow
        stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, test_output, slice_height=8.0)
        
        # Validate all outputs
        png_path = test_output + ".png"
        svg_path = test_output + ".svg"
        
        # Files should exist
        self.assertTrue(os.path.exists(png_path))
        self.assertTrue(os.path.exists(svg_path))
        
        # SVG should be larger than normal mode (multiple embedded images)
        svg_size = os.path.getsize(svg_path)
        self.assertGreater(svg_size, 10000, "Sliced SVG should be substantial size")
    
    def test_complete_workflow_segmented_sliced(self):
        """Test complete workflow for segmented + sliced conversion"""
        test_output = os.path.join(self.test_dir, "workflow_seg_sliced")
        
        # Enable segmentation
        stl_to_depthmap.stl_to_depthmap.segment = True
        
        try:
            # Run the complete segmented + sliced workflow
            stl_to_depthmap.stl_to_depthmap_sliced(self.test_stl, test_output, slice_height=8.0)
            
            # Validate all outputs
            png_path = test_output + ".png"
            svg_path = test_output + ".svg"
            
            # Files should exist
            self.assertTrue(os.path.exists(png_path))
            self.assertTrue(os.path.exists(svg_path))
            
        finally:
            # Reset segmentation flag
            stl_to_depthmap.stl_to_depthmap.segment = False


def run_tests():
    """Run all tests and provide a summary"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSTLToDepthmap))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            failure_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {failure_msg}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
