#!/bin/bash
# Quick test run on sample data (should complete in ~5 minutes)
# This verifies the installation and basic training loop without requiring full benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "PSMEC Quick Test Run"
echo "========================================"
echo ""
echo "This test runs 2 epochs on sample data to verify:"
echo "  - Code can execute without errors"
echo "  - All dependencies are installed"
echo "  - Basic training loop works"
echo ""
echo "Note: This is NOT for reproducing paper results."
echo "Use configs/train_ecb_ol.sh for full training."
echo ""

# Check if sample data exists
if [ ! -f "$REPO_ROOT/data/ecb_ol/test_sample.pkl" ]; then
    echo "❌ Error: Sample data not found at $REPO_ROOT/data/ecb_ol/test_sample.pkl"
    exit 1
fi

echo "✅ Sample data found"
echo ""

# Create output directory
OUTPUT_DIR="/tmp/psmec_quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running quick test on ECB-OL sample..."
echo "Output directory: $OUTPUT_DIR"
echo ""

cd "$REPO_ROOT"

python scripts/train_gpu_joint_memory_router.py \
    --train-records data/ecb_ol/test_sample.pkl \
    --dev-records data/ecb_ol/test_sample.pkl \
    --test-records data/ecb_ol/test_sample.pkl \
    --output-dir "$OUTPUT_DIR" \
    --hidden-dim 160 \
    --dropout 0.15 \
    --top-k 10 \
    --history-size 3 \
    --epochs 2 \
    --batch-size 32 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --create-weight 0.35 \
    --margin-weight 0.2 \
    --attachment-guard-weight 0.0 \
    --retrieval-mode anchor \
    --device cpu \
    --seed 42

echo ""
echo "========================================"
echo "✅ Quick test completed successfully!"
echo "========================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  - To reproduce paper results, use: ./configs/train_ecb_ol.sh"
echo "  - Check README.md for full documentation"
echo ""
