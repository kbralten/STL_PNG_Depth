# STL to DepthMap - Test Suite

This directory contains a comprehensive test suite for the STL to DepthMap conversion tool with **parallel processing support** for faster execution.

## Running Tests

### Quick Start (Parallel - Recommended)
```bash
# Run essential tests in parallel (fastest)
make test-fast

# Run comprehensive tests in parallel  
make test

# Run comprehensive tests with custom worker count
python test_parallel.py --workers 4
```

### Alternative Test Methods
```bash
# Serial execution (slower but more compatible)
make test-serial
python test_simple.py

# Direct execution
python run_tests.py
```

## Performance Improvements

### Parallel Triangle Rasterization
- **Multi-core triangle processing** using up to 8 CPU cores
- **Adaptive batch sizing** based on triangle count and CPU cores
- **Efficient memory management** with batch result merging
- **Typical speedup**: 2-4x faster depending on CPU cores

### Parallel Test Execution  
- **Concurrent test runs** using ThreadPoolExecutor
- **Time savings**: ~77% reduction in test execution time
- **Example**: 342s of tests complete in ~78s wall time
- **Intelligent cleanup** to prevent test conflicts

### Test Coverage

The test suite includes:

1. **Basic Conversion Tests**
   - STL file loading and validation
   - PNG and SVG output generation
   - File format validation

2. **Slicing Tests** (Parallel)
   - Multiple slice heights: 5mm, 10mm, 15mm, 20mm
   - Multi-layer SVG generation
   - PNG contains full depth verification

3. **Segmentation Tests** (Parallel)
   - Island detection and separation
   - Segmented + sliced combinations
   - Performance with complex geometries

4. **Error Handling Tests**
   - Invalid file handling
   - Missing file handling
   - Command-line validation

### Test Modes

**Fast Mode** (`--fast`):
- Essential functionality tests only
- 3 parallel workers  
- ~78s execution time
- Best for development/CI

**Comprehensive Mode** (default):
- All test configurations
- 4 parallel workers
- ~120s execution time  
- Best for release validation

**Serial Mode** (`test-serial`):
- Sequential test execution
- ~6-7 minutes total time
- Best for debugging issues

### Test Files

- `test_parallel.py` - **Parallel test runner (recommended)**
- `test_simple.py` - Serial integration tests
- `test_stl_to_depthmap.py` - Detailed unit tests
- `run_tests.py` - Legacy test runner
- `Makefile` - Build automation with multiple test targets

### Test Data

Tests use `foam.stl` as the primary test file. Ensure this file exists in the project directory.

### Performance Benchmarks

**Triangle Rasterization** (31,030 triangles):
- Serial: ~45-60 seconds
- Parallel (8 cores): ~11-15 seconds
- **Speedup**: ~4x improvement

**Test Suite**:
- Serial execution: ~6-7 minutes
- Parallel execution: ~78 seconds  
- **Speedup**: ~77% time reduction

### Hardware Requirements

**Optimal Performance**:
- 4-8 CPU cores
- 8GB+ RAM
- SSD storage

**Minimum Requirements**:
- 2 CPU cores  
- 4GB RAM
- Any storage type

### Environment Setup

```bash
# Activate virtual environment
source pb/bin/activate

# Verify dependencies
make install

# Run performance tests
make test-fast
```

The parallel processing automatically adapts to your system's CPU count and available memory for optimal performance.
