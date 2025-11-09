#!/bin/bash
set -euo pipefail

# One-click incremental training demo using sample images.
# Usage: bash scripts/run_incremental_demo.sh [GPU_ID]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SAMPLES_DIR="$PROJECT_ROOT/assets/sora/Samples"
BASE_DATASET_DIR="$PROJECT_ROOT/assets/sora/incremental_demo/dataset"
MODEL_ROOT_BASE="$PROJECT_ROOT/output_incremental"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="$MODEL_ROOT_BASE/sora_incremental_$TIMESTAMP"
LOG_DIR="$MODEL_PATH/logs"
MASTR_CKPT="$PROJECT_ROOT/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

BASE_ITERS=${BASE_ITERS:-800}
INCREMENTAL_ITERS=${INCREMENTAL_ITERS:-1200}
STAGE1_ITERS=${STAGE1_ITERS:-400}
STAGE2_ITERS=${STAGE2_ITERS:-800}
N_BASE_VIEWS=${N_BASE_VIEWS:-2}

if [[ ! -d "$SAMPLES_DIR" ]]; then
  echo "[ERROR] Samples directory not found: $SAMPLES_DIR" >&2
  exit 1
fi

if [[ ! -f "$MASTR_CKPT" ]]; then
  echo "[ERROR] MASt3R checkpoint missing at $MASTR_CKPT" >&2
  echo "Download the pretrained model before running the demo." >&2
  exit 1
fi

NEW_IMAGE="$SAMPLES_DIR/2.jpg"
if [[ ! -f "$NEW_IMAGE" ]]; then
  echo "[ERROR] Expected new view image not found: $NEW_IMAGE" >&2
  exit 1
fi

# Prepare base dataset with the initial views
echo "[INFO] Preparing base dataset under $BASE_DATASET_DIR"
rm -rf "$BASE_DATASET_DIR"
mkdir -p "$BASE_DATASET_DIR/images"
for idx in $(seq 0 $((N_BASE_VIEWS-1))); do
  src="$SAMPLES_DIR/${idx}.jpg"
  dst="$BASE_DATASET_DIR/images/${idx}.jpg"
  if [[ ! -f "$src" ]]; then
    echo "[ERROR] Missing source image for base view: $src" >&2
    exit 1
  fi
  cp "$src" "$dst"
done

mkdir -p "$MODEL_PATH" "$LOG_DIR"

echo "[INFO] Running geometry initialization (base views)"
python "$PROJECT_ROOT/init_geo.py" \
  -s "$BASE_DATASET_DIR" \
  -m "$MODEL_PATH" \
  --n_views "$N_BASE_VIEWS" \
  --focal_avg \
  --conf_aware_ranking \
  --co_vis_dsp \
  > "$LOG_DIR/01_init_geo.log" 2>&1

echo "[INFO] Training base model with $N_BASE_VIEWS views"
python "$PROJECT_ROOT/train.py" \
  -s "$BASE_DATASET_DIR" \
  -m "$MODEL_PATH" \
  --n_views "$N_BASE_VIEWS" \
  --iterations "$BASE_ITERS" \
  --checkpoint_iterations "$BASE_ITERS" \
  --test_iterations "$BASE_ITERS" \
  --pp_optimizer \
  --optim_pose \
  > "$LOG_DIR/02_train_base.log" 2>&1

BASE_CKPT="$MODEL_PATH/chkpnt${BASE_ITERS}.pth"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "[ERROR] Base checkpoint not found at $BASE_CKPT" >&2
  exit 1
fi

echo "[INFO] Launching incremental training with new image $NEW_IMAGE"
python "$PROJECT_ROOT/train.py" \
  -s "$BASE_DATASET_DIR" \
  -m "$MODEL_PATH" \
  --n_views "$N_BASE_VIEWS" \
  --load_ckpt "$BASE_CKPT" \
  --new_image "$NEW_IMAGE" \
  --est_pose \
  --mast3r_ckpt "$MASTR_CKPT" \
  --est_max_refs "$N_BASE_VIEWS" \
  --pp_optimizer \
  --optim_pose \
  --incremental_iters "$INCREMENTAL_ITERS" \
  --stage1_iters "$STAGE1_ITERS" \
  --stage2_iters "$STAGE2_ITERS" \
  --save_diff "$MODEL_PATH/incremental_diff.ply" \
  > "$LOG_DIR/03_train_incremental.log" 2>&1

echo "[INFO] Incremental training complete. Key outputs:"
echo "  - Base data set       : $BASE_DATASET_DIR"
echo "  - Model directory     : $MODEL_PATH"
echo "  - Base checkpoint     : $BASE_CKPT"
echo "  - Incremental diff PLY: $MODEL_PATH/incremental_diff.ply"
echo "  - Incremental metrics : $MODEL_PATH/incremental_metrics/metrics.json"
echo "  - Logs                : $LOG_DIR"
