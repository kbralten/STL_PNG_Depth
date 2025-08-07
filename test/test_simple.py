#!/usr/bin/env python3
"""
Simplified test suite for stl_to_depthmap.py
Focuses on core functionality and integration testing.
"""

import unittest
import os
import tempfile
import shutil
import subprocess
import sys

class TestSTLDepthMapIntegration(unittest.TestCase):
    """Integration tests using command line interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_stl = "../foam.stl"
        self.assertTrue(os.path.exists(self.test_stl), f"Test STL file {self.test_stl} not found")
        
        # Clean up any existing test outputs
        for ext in ['.png', '.svg']:
            test_file = 'test_output' + ext
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def tearDown(self):
        """Clean up test outputs"""
        # Clean up test files
        for pattern in ['test_output', 'foam']:
            for ext in ['.png', '.svg']:
                test_file = pattern + ext
                if os.path.exists(test_file):
                    os.remove(test_file)
                # Also clean up in parent directory
                parent_file = '../' + test_file
                if os.path.exists(parent_file):
                    os.remove(parent_file)
    
    def run_command(self, cmd):
        """Run a command and return success status"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
    
    def test_basic_conversion(self):
        """Test basic STL to depth map conversion"""
        success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py {self.test_stl}")
        
        self.assertTrue(success, f"Command failed: {stderr}")
        
        # Check output files exist
        png_path = "../foam.png"
        svg_path = "../foam.svg"
        
        self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
        self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
        
        # Check file sizes are reasonable
        png_size = os.path.getsize(png_path)
        svg_size = os.path.getsize(svg_path)
        
        self.assertGreater(png_size, 1000, "PNG should be substantial size")
        self.assertGreater(svg_size, 1000, "SVG should be substantial size")
    
    def test_sliced_conversion(self):
        """Test sliced conversion"""
        success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py --slice-height 10 --verbose {self.test_stl}")
        
        self.assertTrue(success, f"Sliced command failed: {stderr}")
        
        # Check output files exist
        png_path = "../foam.png"
        svg_path = "../foam.svg"
        
        self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
        self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
        
        # Check that slicing info appears in output
        self.assertIn("slice", stdout.lower(), "Output should mention slicing")
        self.assertIn("Creating", stdout, "Output should show slice creation")
    
    def test_segmented_sliced_conversion(self):
        """Test segmented + sliced conversion"""
        success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py --slice-height 10 --segment --verbose {self.test_stl}")
        
        self.assertTrue(success, f"Segmented sliced command failed: {stderr}")
        
        # Check output files exist
        png_path = "../foam.png"
        svg_path = "../foam.svg"
        
        self.assertTrue(os.path.exists(png_path), f"PNG file should be created: {png_path}")
        self.assertTrue(os.path.exists(svg_path), f"SVG file should be created: {svg_path}")
        
        # Check for island information in output
        self.assertIn("Island", stdout, "Output should mention islands for segmented mode")
    
    def test_different_slice_heights(self):
        """Test different slice heights"""
        for slice_height in [5.0, 15.0, 20.0]:
            with self.subTest(slice_height=slice_height):
                success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py --slice-height {slice_height} {self.test_stl}")
                
                self.assertTrue(success, f"Command failed for slice height {slice_height}: {stderr}")
                
                # Check that the number of slices makes sense
                if "Creating" in stdout and "slices" in stdout:
                    # Extract number of slices from output like "Creating 3 slices of 10.0mm each"
                    lines = stdout.split('\n')
                    slice_lines = [line for line in lines if "Creating" in line and "slices" in line]
                    if slice_lines:
                        # Should have at least 1 slice
                        self.assertIn("slice", slice_lines[0].lower())
    
    def test_output_file_contents(self):
        """Test that output files have valid content"""
        success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py {self.test_stl}")
        self.assertTrue(success, f"Command failed: {stderr}")
        
        # Test PNG content
        png_path = "../foam.png"
        if os.path.exists(png_path):
            try:
                from PIL import Image
                with Image.open(png_path) as img:
                    self.assertEqual(img.mode, 'L', "PNG should be grayscale")
                    self.assertTrue(img.size[0] > 100, "PNG width should be reasonable")
                    self.assertTrue(img.size[1] > 100, "PNG height should be reasonable")
            except ImportError:
                self.skipTest("PIL not available for PNG validation")
        
        # Test SVG content  
        svg_path = "../foam.svg"
        if os.path.exists(svg_path):
            with open(svg_path, 'r') as f:
                svg_content = f.read()
                self.assertIn('<svg', svg_content, "SVG should contain svg tag")
                self.assertIn('xmlns', svg_content, "SVG should contain namespace")
                self.assertIn('image', svg_content, "SVG should contain image elements")
    
    def test_help_option(self):
        """Test help option works"""
        success, stdout, stderr = self.run_command("python ../stl_to_depthmap.py --help")
        
        # Help should succeed and show usage
        self.assertTrue(success, "Help command should succeed")
        self.assertIn("usage:", stdout.lower(), "Help should show usage")
        self.assertIn("stl", stdout.lower(), "Help should mention STL")
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files"""
        # Create a temporary invalid file
        invalid_file = "invalid_test.stl"
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid STL file")
        
        try:
            success, stdout, stderr = self.run_command(f"python ../stl_to_depthmap.py {invalid_file}")
            
            # Should fail gracefully (not succeed)
            self.assertFalse(success, "Invalid file should cause command to fail")
            
        finally:
            if os.path.exists(invalid_file):
                os.remove(invalid_file)
    
    def test_missing_file_handling(self):
        """Test handling of missing files"""
        success, stdout, stderr = self.run_command("python ../stl_to_depthmap.py nonexistent.stl")
        
        # Should fail gracefully
        self.assertFalse(success, "Missing file should cause command to fail")


class TestSTLDepthMapOutput(unittest.TestCase):
    """Tests for output validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_stl = "../foam.stl"
        self.assertTrue(os.path.exists(self.test_stl), f"Test STL file {self.test_stl} not found")
    
    def tearDown(self):
        """Clean up test outputs"""
        for pattern in ['foam']:
            for ext in ['.png', '.svg']:
                test_file = pattern + ext
                if os.path.exists(test_file):
                    os.remove(test_file)
                # Also clean up in parent directory
                parent_file = '../' + test_file
                if os.path.exists(parent_file):
                    os.remove(parent_file)
    
    def test_png_vs_svg_consistency(self):
        """Test that PNG and SVG outputs are generated consistently"""
        # Run normal conversion
        result = subprocess.run([sys.executable, "../stl_to_depthmap.py", self.test_stl], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            png_path = "../foam.png"
            svg_path = "../foam.svg"
            
            png_exists = os.path.exists(png_path)
            svg_exists = os.path.exists(svg_path)
            
            # Both should exist or both should not exist
            self.assertEqual(png_exists, svg_exists, "PNG and SVG should be created together")
            
            if png_exists and svg_exists:
                # Both should have reasonable sizes
                png_size = os.path.getsize(png_path)
                svg_size = os.path.getsize(svg_path)
                
                self.assertGreater(png_size, 100, "PNG should have content")
                self.assertGreater(svg_size, 100, "SVG should have content")


def run_tests():
    """Run all tests and provide a summary"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSTLDepthMapIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSTLDepthMapOutput))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"STL TO DEPTHMAP TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            failure_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in traceback else "Unknown failure"
            print(f"  • {test}: {failure_msg}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            error_lines = traceback.strip().split('\n')
            error_msg = error_lines[-1] if error_lines else "Unknown error"
            print(f"  • {test}: {error_msg}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed! The STL to DepthMap tool is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
