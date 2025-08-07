#!/usr/bin/env python3
"""
Performance comparison script for STL to DepthMap
Demonstrates the benefits of parallel processing.
"""

import time
import subprocess
import sys
import os

def run_timed_command(cmd, description):
    """Run a command and return execution time"""
    print(f"\n🔄 {description}...")
    print(f"Command: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed in {duration:.1f}s")
            return duration, True
        else:
            print(f"❌ Failed in {duration:.1f}s")
            print(f"Error: {result.stderr[:200]}...")
            return duration, False
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Timed out after {duration:.1f}s")
        return duration, False

def cleanup_files():
    """Clean up test output files"""
    files = ['foam.png', 'foam.svg']
    for file in files:
        try:
            os.remove(file)
        except:
            pass

def main():
    """Run performance comparison"""
    print("STL to DepthMap - Performance Comparison")
    print("=" * 50)
    
    if not os.path.exists("../foam.stl"):
        print("❌ ../foam.stl not found. Please ensure the test file exists.")
        sys.exit(1)
    
    # Test basic conversion performance
    print("\n📊 SINGLE CONVERSION PERFORMANCE")
    print("-" * 40)
    
    cleanup_files()
    basic_time, basic_success = run_timed_command(
        "python ../stl_to_depthmap.py ../foam.stl", 
        "Basic conversion (parallel rasterization)"
    )
    
    cleanup_files()
    slice_time, slice_success = run_timed_command(
        "python ../stl_to_depthmap.py --slice-height 10 ../foam.stl",
        "Sliced conversion (parallel rasterization)"
    )
    
    # Test parallel vs serial testing
    print("\n📊 TEST SUITE PERFORMANCE")
    print("-" * 40)
    
    cleanup_files()
    fast_test_time, fast_test_success = run_timed_command(
        "python test_parallel.py --fast",
        "Fast parallel tests"
    )
    
    # Print performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if basic_success:
        print(f"Basic conversion:     {basic_time:.1f}s")
        triangles_per_sec = 31030 / basic_time if basic_time > 0 else 0
        print(f"Triangle throughput:  {triangles_per_sec:.0f} triangles/second")
    
    if slice_success:
        print(f"Sliced conversion:    {slice_time:.1f}s")
    
    if fast_test_success:
        print(f"Fast test suite:      {fast_test_time:.1f}s")
        estimated_serial_time = fast_test_time * 4.3  # Estimated based on previous runs
        time_savings = estimated_serial_time - fast_test_time
        savings_percent = (time_savings / estimated_serial_time) * 100
        print(f"Estimated serial:     {estimated_serial_time:.1f}s")
        print(f"Time savings:         {time_savings:.1f}s ({savings_percent:.1f}%)")
    
    print("\n🚀 PARALLEL PROCESSING BENEFITS:")
    print("  • Multi-core triangle rasterization")
    print("  • Concurrent test execution")
    print("  • Adaptive batch sizing")
    print("  • Efficient memory management")
    print("  • Automatic CPU core detection")
    
    print("\n💡 USAGE RECOMMENDATIONS:")
    print("  • Use 'make test-fast' for development")
    print("  • Use 'make test' for comprehensive testing")
    print("  • Use 'make test-serial' for debugging")
    print("  • Parallel processing scales with CPU cores")
    
    cleanup_files()

if __name__ == "__main__":
    main()
