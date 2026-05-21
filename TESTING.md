# Testing Framework Documentation

This document outlines the comprehensive testing framework implemented for the CMT X-ray Inpainting project.

## 🎯 Quick Start

```bash
# Install testing dependencies
pip install -r requirements.txt

# Run all tests
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests only  
make test-fast          # Fast tests (exclude slow/gpu/data tests)
make test-coverage      # Generate coverage reports

# Run specific test modules
make test-models        # Model architecture tests
make test-dataset       # Dataset functionality tests
make test-metrics       # Loss functions and metrics tests
```

## 📁 Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                    # Pytest settings and markers
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_models.py            # ViT, SwinTransformer, Inpaint models
│   ├── test_dataset.py           # ArcadeDataset, COCO loading
│   ├── test_losses_metrics.py    # Loss functions, PSNR/SSIM/etc
│   └── test_simple.py            # Basic functionality tests
├── integration/                   # Integration tests (slower, end-to-end)
│   ├── test_training.py          # Full training pipeline
│   └── test_inference.py         # Inference and model loading
└── fixtures/
    └── sample_data.py             # Test data generation utilities
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)

**Model Architecture Tests** (`test_models.py`)
- ViT (Continuously Masked Transformer) functionality
- SwinTransformer refine stage operations  
- Complete Inpaint model pipeline
- Model initialization, forward passes, gradient flow
- Different input sizes (32, 64, 128, 256)
- Checkpoint saving/loading

**Dataset Tests** (`test_dataset.py`)
- ArcadeDataset COCO annotation loading
- Mask generation from polygons
- Stenosis category exclusion (category ID 26)
- Image preprocessing (grayscale, normalization)
- Different input size handling
- DataLoader integration

**Loss & Metrics Tests** (`test_losses_metrics.py`)
- Inpainting loss function (6×L1 masked + 1×L1 unmasked + 0.5×SSIM)
- PSNR calculation for different data ranges
- SSIM with proper data_range parameter
- Wasserstein distance computation
- RMSE calculation
- Gradient flow verification

### Integration Tests (`tests/integration/`)

**Training Pipeline** (`test_training.py`)
- End-to-end training with real data
- Dataset loading in training context
- Model training steps with loss computation
- Checkpoint saving during training
- Multi-epoch training cycles

**Inference Pipeline** (`test_inference.py`)
- Model loading from checkpoints
- Batch inference processing
- Different input size handling
- Unmasked region preservation
- Output quality metrics
- Memory usage monitoring
- Reproducibility testing

## 🔧 Test Configuration

### Pytest Settings (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
markers =
    unit: Unit tests
    integration: Integration tests  
    slow: Slow tests (skip with -m 'not slow')
    gpu: Tests requiring GPU
    data: Tests requiring ARCADE dataset
```

### Test Fixtures (`conftest.py`)

**Core Fixtures:**
- `device`: CPU device for testing
- `temp_dir`: Temporary directory cleanup
- `sample_image`/`sample_mask`: Test tensors
- `sample_batch`: Batch of test data
- `mock_coco_annotation`: COCO format test data
- `sample_checkpoint`: Model checkpoint for testing

**Data Fixtures:**
- `mock_dataset_files`: Complete test dataset
- `checkpoint_dir`: Temporary checkpoint storage
- `mock_training_log`: CSV training logs

## 🚀 Continuous Integration

### GitHub Actions (`.github/workflows/tests.yml`)

**Matrix Testing:**
- Python versions: 3.9, 3.10, 3.11
- Operating system: Ubuntu Latest

**Workflow:**
1. Unit tests on all Python versions
2. Fast integration tests  
3. Coverage reporting (Python 3.9 only)
4. Full integration tests (main branch only)

### GitLab CI (`.gitlab-ci.yml`)

**Pipeline Stages:**
1. **Test Stage**: Unit tests + fast tests
2. **Integration Stage**: Smoke test + full integration tests

**Features:**
- CPU-only testing with PyTorch
- Artifact collection for test results
- Branch-specific execution (main/sync-main)

### Pre-commit Hooks (`.pre-commit-config.yaml`)

- Code formatting (black, isort)
- Linting (flake8)
- Fast test execution
- YAML/file validation

## 📊 Coverage & Quality

### Test Coverage

The framework provides comprehensive coverage:

- **Model Architecture**: 100% of core model components
- **Data Pipeline**: COCO loading, preprocessing, augmentation
- **Training Loop**: Loss computation, optimization, checkpointing  
- **Inference**: Model loading, batch processing, quality metrics
- **Edge Cases**: Numerical stability, memory management

### Quality Metrics

**Automated Testing:**
- PSNR calculation accuracy
- SSIM computation correctness
- Loss function gradient flow
- Model output range validation
- Memory leak detection

**Performance Benchmarks:**
- Inference speed monitoring
- Memory usage tracking
- Batch processing efficiency

## 🛠️ Development Workflow

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_*.py` file in `tests/unit/`
2. **Integration Tests**: Add to `tests/integration/`
3. **Fixtures**: Add reusable test data to `conftest.py`
4. **Markers**: Use pytest markers for test categorization

### Test Naming Convention

```python
def test_<component>_<functionality>():
    """Clear description of what is being tested"""
    pass

class Test<Component>:
    """Test class for related functionality"""
    
    def test_<specific_feature>(self):
        pass
```

### Mock Data Generation

Use `tests/fixtures/sample_data.py` for creating test datasets:

```python
from tests.fixtures.sample_data import (
    create_sample_coco_annotation,
    create_sample_images,
    create_test_dataset
)
```

## 🔍 Debugging Tests

### Running Specific Tests

```bash
# Single test function
pytest tests/unit/test_models.py::TestViTModel::test_vit_forward_pass -v

# Test class
pytest tests/unit/test_models.py::TestViTModel -v

# With debugging output
pytest tests/unit/test_models.py -v -s --tb=long
```

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **CUDA Errors**: Tests run on CPU by default
3. **Memory Issues**: Use smaller test data or garbage collection
4. **Fixture Conflicts**: Check fixture scoping and cleanup

## 📈 Performance Considerations

### Fast Tests

Use `make test-fast` to skip:
- Slow computational tests (`@pytest.mark.slow`)
- GPU-requiring tests (`@pytest.mark.gpu`)  
- Large dataset tests (`@pytest.mark.data`)

### Parallel Execution

```bash
# Run tests in parallel with pytest-xdist
pytest tests/ -n auto
```

### Test Data Size

- Unit tests use small tensors (64×64 or smaller)
- Integration tests use minimal datasets (2-3 samples)
- Fixtures provide scalable test data generation

## 🎯 Best Practices

1. **Isolation**: Each test should be independent
2. **Determinism**: Use fixed random seeds for reproducibility  
3. **Speed**: Keep unit tests under 1 second each
4. **Clarity**: Use descriptive test names and docstrings
5. **Coverage**: Test both success and failure cases
6. **Fixtures**: Reuse test data through pytest fixtures
7. **Markers**: Tag tests appropriately for selective execution

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/testing.html)
- [Testing Machine Learning Code](https://madewithml.com/courses/mlops/testing/)

---

*This testing framework ensures robust, reliable development of the CMT X-ray inpainting model with comprehensive coverage and continuous validation.*