#!/usr/bin/env python3
"""
Parallel test runner for STL to Depthmap tests
Runs multiple test configurations in parallel to speed up testing.
"""

import unittest
import os
import tempfile
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_test_command(test_config):
    """Run a single test command and return results"""
    test_name, cmd, timeout = test_config
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        success = result.returncode == 0
        duration = time.time() - start_time
        
        return {
            'name': test_name,
            'success': success,
            'duration': duration,
            'stdout': result.stdout[-500:] if result.stdout else "",  # Last 500 chars
            'stderr': result.stderr[-500:] if result.stderr else "",
            'cmd': cmd
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'name': test_name,
            'success': False,
            'duration': duration,
            'stdout': "",
            'stderr': f"Test timed out after {timeout}s",
            'cmd': cmd
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'name': test_name,
            'success': False,
            'duration': duration,
            'stdout': "",
            'stderr': str(e),
            'cmd': cmd
        }

def cleanup_test_files():
    """Clean up test output files"""
    patterns = ['foam.png', 'foam.svg', 'test_*.png', 'test_*.svg', 'invalid_test.stl']
    for pattern in patterns:
        if '*' in pattern:
            import glob
            for file in glob.glob(pattern):
                try:
                    os.remove(file)
                except:
                    pass
        else:
            try:
                os.remove(pattern)
            except:
                pass

class ParallelTestRunner:
    """Test runner that executes tests in parallel"""
    
    def __init__(self, test_stl="../foam.stl"):
        self.test_stl = test_stl
        self.results = []
        
        # Ensure test file exists
        if not os.path.exists(self.test_stl):
            raise FileNotFoundError(f"Test STL file {self.test_stl} not found")
    
    def get_test_configurations(self):
        """Define test configurations to run in parallel"""
        tests = [
            # Basic functionality tests
            ("basic_conversion", f"python ../stl_to_depthmap.py {self.test_stl}", 120),
            ("help_command", "python ../stl_to_depthmap.py --help", 10),
            
            # Slicing tests with different heights
            ("slice_5mm", f"python ../stl_to_depthmap.py --slice-height 5 {self.test_stl}", 150),
            ("slice_10mm", f"python ../stl_to_depthmap.py --slice-height 10 {self.test_stl}", 120),
            ("slice_15mm", f"python ../stl_to_depthmap.py --slice-height 15 {self.test_stl}", 100),
            ("slice_20mm", f"python ../stl_to_depthmap.py --slice-height 20 {self.test_stl}", 90),
            
            # Segmentation tests
            ("segment_basic", f"python ../stl_to_depthmap.py --slice-height 10 --segment {self.test_stl}", 150),
            ("segment_fine", f"python ../stl_to_depthmap.py --slice-height 5 --segment {self.test_stl}", 180),
        ]
        
        return tests
    
    def run_parallel_tests(self, max_workers=None):
        """Run tests in parallel using ThreadPoolExecutor"""
        if max_workers is None:
            max_workers = min(cpu_count(), 4)  # Limit to 4 parallel tests
        
        print(f"Running tests in parallel with {max_workers} workers...")
        print("=" * 60)
        
        test_configs = self.get_test_configurations()
        total_tests = len(test_configs)
        
        cleanup_test_files()  # Clean up before starting
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_test = {executor.submit(run_test_command, config): config[0] 
                             for config in test_configs}
            
            completed = 0
            for future in as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    completed += 1
                    status = "✅ PASS" if result['success'] else "❌ FAIL"
                    duration = result['duration']
                    
                    print(f"[{completed}/{total_tests}] {status} {test_name} ({duration:.1f}s)")
                    
                    if not result['success'] and result['stderr']:
                        print(f"    Error: {result['stderr'][:100]}...")
                    
                    # Clean up after each test to avoid conflicts
                    cleanup_test_files()
                    
                except Exception as e:
                    print(f"[{completed}/{total_tests}] ❌ ERROR {test_name}: {e}")
                    completed += 1
        
        return self.results
    
    def print_summary(self):
        """Print test summary"""
        if not self.results:
            print("No test results to summarize")
            return
        
        print("\n" + "=" * 60)
        print("PARALLEL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        total_time = sum(r['duration'] for r in self.results)
        max_time = max(r['duration'] for r in self.results) if self.results else 0
        
        print(f"Tests run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"Total CPU time: {total_time:.1f}s")
        print(f"Wall clock time: ~{max_time:.1f}s (parallel execution)")
        print(f"Time savings: ~{(total_time - max_time):.1f}s ({((total_time - max_time)/total_time*100):.1f}%)")
        
        if failed_tests > 0:
            print(f"\nFAILED TESTS:")
            for result in self.results:
                if not result['success']:
                    print(f"  • {result['name']}: {result['stderr'][:100]}...")
        
        return passed_tests == total_tests

def run_fast_tests():
    """Quick test mode - run only essential tests"""
    print("Running FAST test mode...")
    
    quick_tests = [
        ("basic_conversion", f"python ../stl_to_depthmap.py ../foam.stl", 120),
        ("slice_test", f"python ../stl_to_depthmap.py --slice-height 10 ../foam.stl", 120),
        ("help_test", "python ../stl_to_depthmap.py --help", 10),
    ]
    
    runner = ParallelTestRunner()
    runner.run_parallel_tests(max_workers=3)
    return runner.print_summary()

def run_comprehensive_tests():
    """Full test mode - run all tests"""
    print("Running COMPREHENSIVE test mode...")
    
    runner = ParallelTestRunner()
    runner.run_parallel_tests(max_workers=4)
    return runner.print_summary()

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel STL to DepthMap test runner")
    parser.add_argument("--fast", action="store_true", help="Run only essential tests (faster)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    print("STL to DepthMap - Parallel Test Runner")
    print("=" * 60)
    
    if args.fast:
        success = run_fast_tests()
    else:
        success = run_comprehensive_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
