# Makefile for STL to Depthmap project

.PHONY: test tests test-fast test-parallel clean help install benchmark

# Default target
help:
	@echo "Available targets:"
	@echo "  test/tests      - Run comprehensive tests (parallel)"
	@echo "  test-fast       - Run essential tests only (faster)"
	@echo "  test-parallel   - Run all tests in parallel (fastest)"
	@echo "  test-serial     - Run tests serially (slowest, most compatible)"
	@echo "  benchmark       - Run performance benchmarks"
	@echo "  clean           - Clean up test artifacts and temporary files"
	@echo "  install         - Install required dependencies"
	@echo "  help            - Show this help message"

# Run comprehensive tests in parallel (default)
test tests: test-parallel

# Run essential tests only (fastest)
test-fast:
	@echo "Running fast tests with parallel execution..."
	@if [ -f "pb/bin/activate" ]; then \
		. pb/bin/activate && python test_parallel.py --fast; \
	else \
		python test_parallel.py --fast; \
	fi

# Run all tests in parallel (comprehensive but fast)
test-parallel:
	@echo "Running comprehensive tests with parallel execution..."
	@if [ -f "pb/bin/activate" ]; then \
		. pb/bin/activate && python test_parallel.py; \
	else \
		python test_parallel.py; \
	fi

# Run tests serially (compatible with older systems)
test-serial:
	@echo "Running tests serially..."
	@if [ -f "pb/bin/activate" ]; then \
		. pb/bin/activate && python test_simple.py; \
	else \
		python test_simple.py; \
	fi

# Run performance benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	@if [ -f "pb/bin/activate" ]; then \
		. pb/bin/activate && python benchmark.py; \
	else \
		python benchmark.py; \
	fi

# Clean up test artifacts
clean:
	@echo "Cleaning up test artifacts..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "test_*.png" -delete 2>/dev/null || true
	@find . -name "test_*.svg" -delete 2>/dev/null || true
	@rm -rf /tmp/tmp* 2>/dev/null || true
	@echo "Cleanup complete."

# Install dependencies (if needed)
install:
	@echo "Installing/checking dependencies..."
	@if [ -f "pb/bin/activate" ]; then \
		. pb/bin/activate && pip install pillow numpy opencv-python scipy open3d; \
	else \
		pip install pillow numpy opencv-python scipy open3d; \
	fi
