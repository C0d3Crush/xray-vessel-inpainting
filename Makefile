.PHONY: help install smoke-test prep-phase1 phase1 train inference clean test test-unit test-integration test-coverage test-fast

# Default ARCADE paths
TRAIN_IMG = data/arcade/syntax/train/images
TRAIN_ANN = data/arcade/syntax/train/annotations/train.json
VAL_IMG = data/arcade/syntax/val/images
VAL_ANN = data/arcade/syntax/val/annotations/val.json
DATA_DIR = data

# Training params
EPOCHS = 100
BATCH_SIZE = 4
INPUT_SIZE = 64
DEVICE = cpu

help:
	@echo "Available targets:"
	@echo "  make install         - Install dependencies"
	@echo "  make smoke-test      - Quick pipeline verification (CPU, 1 epoch)"
	@echo "  make smoke-test-background - Train with random background masks (2 epochs)"
	@echo "  make cache-data      - Precompute masks & annotations (recommended)"
# Legacy prepare-samples removed
	@echo "  make prepare-patch-samples - Create 64×64 patches for proper visualization"
	@echo "  make prepare-background-samples - Create random background masks (vessel-free)"
	@echo "  make train           - Train CMT inpainting model"
	@echo "  make inference       - Run patch-based inference on samples/"
# Legacy inference-resize and visualize removed
	@echo "  make training-comparison - Create enhanced comparison visualization"
	@echo "  make plot            - Generate training plot from CSV"
	@echo "  make clean           - Remove checkpoints and logs"
	@echo ""
	@echo "Testing targets:"
	@echo "  make test            - Run full test suite"
	@echo "  make test-unit       - Run unit tests only"
	@echo "  make test-integration- Run integration tests only"
	@echo "  make test-fast       - Run fast tests (exclude slow tests)"
	@echo "  make test-coverage   - Run tests with coverage report"

install:
	pip install -r requirements.txt

smoke-test:
	python src/train.py --smoke_test --smoke_size 20 --epochs 1 --batch_size 1 --device cpu \
		--train_img $(TRAIN_IMG) --train_ann $(TRAIN_ANN) \
		--val_img $(VAL_IMG) --val_ann $(VAL_ANN)

smoke-test-background:
	@echo "Creating background training masks for smoke test..."
	python src/generate_background_masks.py \
		--input-img $(TRAIN_IMG) \
		--input-mask data/masks_cache/train \
		--output-img data/smoke_bg_img \
		--output-mask data/smoke_bg_mask \
		--variations 2 --safety-margin 5
	@echo "Training with random background masks (smoke test)..."
	python src/train.py --smoke_test --smoke_size 40 --epochs 2 --patch_mode --input_size 64 --patches_per_image 4 --batch_size 2 --device cpu \
		--train_img data/smoke_bg_img \
		--train_mask data/smoke_bg_mask \
		--val_img $(VAL_IMG) --val_ann $(VAL_ANN) \
		--output_dir checkpoints_bg

cache-data:
	@echo "Caching train masks..."
	python scripts/cache_masks.py --annotations $(TRAIN_ANN) --images $(TRAIN_IMG) --output $(DATA_DIR)/masks_cache/train
	@echo "Caching val masks..."
	python scripts/cache_masks.py --annotations $(VAL_ANN) --images $(VAL_IMG) --output $(DATA_DIR)/masks_cache/val
	@echo "Preprocessing annotations..."
	python scripts/preprocess_coco.py --annotations $(TRAIN_ANN) --output $(TRAIN_ANN:.json=.pkl)
	python scripts/preprocess_coco.py --annotations $(VAL_ANN) --output $(VAL_ANN:.json=.pkl)
	@echo "✓ Data caching complete. Training will be faster now."

# Legacy prepare-samples removed - use prepare-patch-samples or prepare-background-samples

prepare-patch-samples:
	python scripts/prepare_samples.py --annotations $(VAL_ANN) --images $(VAL_IMG) --num-samples 8 --overwrite --output-img outputs/samples/full_img --output-mask outputs/samples/full_mask
	python src/extract_patch_samples.py --img-dir outputs/samples/full_img --mask-dir outputs/samples/full_mask --output-img outputs/samples/patch_img --output-mask outputs/samples/patch_mask --patch-size $(INPUT_SIZE) --min-vessel-ratio 0.05

prepare-background-samples:
	python scripts/prepare_samples.py --annotations $(VAL_ANN) --images $(VAL_IMG) --num-samples 5 --overwrite --output-img outputs/samples/full_img --output-mask outputs/samples/full_mask
	python src/generate_background_masks.py --input-img outputs/samples/full_img --input-mask outputs/samples/full_mask --output-img outputs/samples/bg_img --output-mask outputs/samples/bg_mask --variations 3 --safety-margin 5
	python src/extract_patch_samples.py --img-dir outputs/samples/bg_img --mask-dir outputs/samples/bg_mask --output-img outputs/samples/bg_patch_img --output-mask outputs/samples/bg_patch_mask --patch-size $(INPUT_SIZE) --min-vessel-ratio 0.05

train:
	python src/train.py \
		--train_img $(TRAIN_IMG) \
		--train_ann $(TRAIN_ANN) \
		--val_img $(VAL_IMG) \
		--val_ann $(VAL_ANN) \
		--output_dir checkpoints \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--input_size $(INPUT_SIZE) \
		--device $(DEVICE)

inference:
	@if [ ! -f checkpoints/best.pth ]; then \
		echo "Error: checkpoints/best.pth not found. Train model first."; \
		exit 1; \
	fi
	python src/patch_inference_demo.py \
		--ckpt checkpoints/best.pth \
		--img_path outputs/samples/patch_img \
		--mask_path outputs/samples/patch_mask \
		--output_path outputs/samples/patch_results \
		--input_size $(INPUT_SIZE) \
		--device $(DEVICE)

# Legacy inference-resize removed - use inference (patch-based)

# Legacy visualize removed - use training-comparison

training-comparison:
	python scripts/create_training_comparison.py \
		--patch-img outputs/samples/patch_img \
		--patch-mask outputs/samples/patch_mask \
		--patch-result outputs/samples/patch_results \
		--output outputs/samples/patch_training_comparison.png \
		--title "Patch Training Results (64×64 patches)" \
		--images 104 162 66 9
	@echo "Removing old comparison files..."
	rm -f outputs/samples/comparisons/*.png
	@echo "✓ 64×64 patch comparison created, old comparisons removed"

plot:
	@if [ -f checkpoints/training_log.csv ]; then \
		python scripts/plot_training.py checkpoints/training_log.csv; \
	elif [ -f training_log.csv ]; then \
		python scripts/plot_training.py training_log.csv; \
	else \
		echo "Error: training_log.csv not found."; \
		exit 1; \
	fi

clean:
	rm -rf checkpoints/*.pth checkpoints/training_log.csv
	@echo "Checkpoints and logs removed."

# Testing targets
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "unit or not integration"

test-integration:
	pytest tests/integration/ -v -m "integration"

test-fast:
	pytest tests/ -v -m "not slow and not gpu and not data"

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-models:
	pytest tests/unit/test_models.py -v

test-dataset:
	pytest tests/unit/test_dataset.py -v

test-metrics:
	pytest tests/unit/test_losses_metrics.py -v
