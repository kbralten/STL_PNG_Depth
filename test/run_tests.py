#!/usr/bin/env python3
"""
Simple test runner for STL to Depthmap tests
Usage: python run_tests.py
"""

import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from test_simple import run_tests
    
    print("Running STL to Depthmap Test Suite")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
        
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"Error running tests: {e}")
    sys.exit(1)
