.PHONY: help install smoke-test prep-phase1 phase1 train inference clean

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
	@echo "  make cache-data      - Precompute masks & annotations (recommended)"
	@echo "  make prepare-samples - Populate samples/ with test images from ARCADE"
	@echo "  make train           - Train CMT inpainting model"
	@echo "  make inference       - Run inference on samples/"
	@echo "  make visualize       - Create side-by-side comparisons (Input|Mask|Result)"
	@echo "  make plot            - Generate training plot from CSV"
	@echo "  make clean           - Remove checkpoints and logs"

install:
	pip install -r requirements.txt

smoke-test:
	python src/train.py --smoke_test --smoke_size 20 --epochs 1 --batch_size 1 --device cpu

cache-data:
	@echo "Caching train masks..."
	python scripts/cache_masks.py --annotations $(TRAIN_ANN) --images $(TRAIN_IMG) --output $(DATA_DIR)/masks_cache/train
	@echo "Caching val masks..."
	python scripts/cache_masks.py --annotations $(VAL_ANN) --images $(VAL_IMG) --output $(DATA_DIR)/masks_cache/val
	@echo "Preprocessing annotations..."
	python scripts/preprocess_coco.py --annotations $(TRAIN_ANN) --output $(TRAIN_ANN:.json=.pkl)
	python scripts/preprocess_coco.py --annotations $(VAL_ANN) --output $(VAL_ANN:.json=.pkl)
	@echo "✓ Data caching complete. Training will be faster now."

prepare-samples:
	python scripts/prepare_samples.py --annotations $(VAL_ANN) --images $(VAL_IMG) --num-samples 5 --overwrite --output-img outputs/samples/test_img --output-mask outputs/samples/test_mask

train:
	python src/train.py \
		--train_img $(TRAIN_IMG) \
		--train_ann $(TRAIN_ANN) \
		--val_img $(VAL_IMG) \
		--val_ann $(VAL_ANN) \
		--output_dir outputs/checkpoints \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--input_size $(INPUT_SIZE) \
		--device $(DEVICE)

inference:
	@if [ ! -f outputs/checkpoints/best.pth ]; then \
		echo "Error: outputs/checkpoints/best.pth not found. Train model first."; \
		exit 1; \
	fi
	python src/demo.py \
		--ckpt outputs/checkpoints/best.pth \
		--img_path outputs/samples/test_img \
		--mask_path outputs/samples/test_mask \
		--output_path outputs/samples/results \
		--input_size $(INPUT_SIZE) \
		--device $(DEVICE)

visualize:
	python scripts/visualize_results.py \
		--input outputs/samples/test_img \
		--mask outputs/samples/test_mask \
		--result outputs/samples/results \
		--output outputs/samples/comparisons

plot:
	@if [ -f outputs/checkpoints/training_log.csv ]; then \
		python scripts/plot_training.py outputs/checkpoints/training_log.csv; \
	elif [ -f training_log.csv ]; then \
		python scripts/plot_training.py training_log.csv; \
	else \
		echo "Error: training_log.csv not found."; \
		exit 1; \
	fi

clean:
	rm -rf outputs/checkpoints/*.pth outputs/checkpoints/training_log.csv
	@echo "Checkpoints and logs removed."
