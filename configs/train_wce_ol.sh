#!/bin/bash
# Training script for WCE-OL benchmark (reproduces Table 2 results)

# This script trains the PCISM router on WCE-OL with risk calibration
# Expected results: Pair-F1: 0.4423, Action Accuracy: 0.7803

set -e

# Configuration
BENCHMARK_DIR="../../../data/output/benchmark_wikinews_ol_controlled_visual"
OUTPUT_DIR="./checkpoints/wce_ol"
DEVICE="cuda:0"

# Model hyperparameters (Section 5)
HIDDEN_DIM=160
DROPOUT=0.15
K_S=10  # Candidate set size
K_EX=3  # Exemplar buffer size

# Training hyperparameters (Section 5)
EPOCHS=80
BATCH_SIZE=512
LR=5e-4
WEIGHT_DECAY=1e-4

# Loss weights (Section 4.2)
CREATE_WEIGHT=0.35           # α_create
MARGIN_WEIGHT=0.2            # α_margin
GUARD_WEIGHT=0.0             # α_guard

# Risk calibration weights (WCE-OL only)
ENABLE_RISK_HEADS=true
POLLUTION_RISK_WEIGHT=0.1    # β_poll
FRAGMENTATION_RISK_WEIGHT=0.1  # β_frag
RISK_MARGIN_WEIGHT=0.0       # β_risk-m

# Inference risk scales (Equation 7)
POLLUTION_RISK_SCALE=0.1     # λ_poll
FRAGMENTATION_RISK_SCALE=0.1 # λ_frag

echo "=========================================="
echo "Training PCISM on WCE-OL (with risk calibration)"
echo "=========================================="
echo "Benchmark: $BENCHMARK_DIR"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""
echo "Model Config:"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Dropout: $DROPOUT"
echo "  K_s: $K_S, K_ex: $K_EX"
echo "  Risk heads: ENABLED"
echo ""
echo "Training Config:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo ""
echo "Loss Weights:"
echo "  α_create: $CREATE_WEIGHT"
echo "  α_margin: $MARGIN_WEIGHT"
echo "  α_guard: $GUARD_WEIGHT"
echo "  β_poll: $POLLUTION_RISK_WEIGHT"
echo "  β_frag: $FRAGMENTATION_RISK_WEIGHT"
echo ""
echo "Inference Risk Scales:"
echo "  λ_poll: $POLLUTION_RISK_SCALE"
echo "  λ_frag: $FRAGMENTATION_RISK_SCALE"
echo "=========================================="
echo ""

# Run training
python ../scripts/train_gpu_joint_memory_router.py \
    --train-records "${BENCHMARK_DIR}/train_stream_records.pkl" \
    --dev-records "${BENCHMARK_DIR}/dev_stream_records.pkl" \
    --test-records "${BENCHMARK_DIR}/test_stream_records.pkl" \
    --output-dir "$OUTPUT_DIR" \
    --hidden-dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --top-k $K_S \
    --history-size $K_EX \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --create-weight $CREATE_WEIGHT \
    --margin-weight $MARGIN_WEIGHT \
    --attachment-guard-weight $GUARD_WEIGHT \
    --enable-risk-heads \
    --pollution-risk-weight $POLLUTION_RISK_WEIGHT \
    --fragmentation-risk-weight $FRAGMENTATION_RISK_WEIGHT \
    --risk-margin-weight $RISK_MARGIN_WEIGHT \
    --pollution-risk-scale $POLLUTION_RISK_SCALE \
    --fragmentation-risk-scale $FRAGMENTATION_RISK_SCALE \
    --device $DEVICE \
    --retrieval-mode anchor \
    --evaluation-protocol action_f1

echo ""
echo "Training complete. Checkpoint saved to: $OUTPUT_DIR"
echo "Expected results: Pair-F1: 0.4423, Action Accuracy: 0.7803"
echo ""
echo "To evaluate without visual features (ablation):"
echo "  Add --disable-visual-signals flag"
echo "  Expected: Pair-F1: 0.1068 (-33.55 points)"
