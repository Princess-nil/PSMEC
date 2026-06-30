#!/usr/bin/env python
"""Quick test to verify installation and basic functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that core modules can be imported."""
    try:
        from new_task.streaming_memory import (
            MemorySlot,
            StreamingMentionRecord,
            new_memory_slot,
            update_memory_slot,
            memory_maturity_features,
            memory_candidate_features,
        )
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_sample_data():
    """Test that sample data can be loaded."""
    import pickle

    data_dir = Path(__file__).parent.parent / "data"

    try:
        ecb_sample = data_dir / "ecb_ol" / "test_sample.pkl"
        with open(ecb_sample, "rb") as f:
            ecb_data = pickle.load(f)
        print(f"✅ ECB-OL sample loaded: {len(ecb_data)} records")

        wce_sample = data_dir / "wce_ol" / "test_sample.pkl"
        with open(wce_sample, "rb") as f:
            wce_data = pickle.load(f)
        print(f"✅ WCE-OL sample loaded: {len(wce_data)} records")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def test_basic_functionality():
    """Test basic memory slot operations."""
    try:
        import numpy as np
        from new_task.streaming_memory import (
            MemorySlot,
            StreamingMentionRecord,
            new_memory_slot,
            update_memory_slot,
            memory_maturity_features,
        )

        # Create a mock record
        record = StreamingMentionRecord(
            mention_id="test_001",
            split="test",
            episode_id="ep_1",
            stream_index=0,
            doc_id="doc_1",
            sentence_id=0,
            mention_text="test event",
            bert_doc="Test document",
            topic="topic_1",
            predicted_topic="topic_1",
            gold_cluster="event_1",
            gold_action="CREATE",
            has_visual=True,
            mention_vector=np.random.randn(768).astype(np.float32),
            visual_vector=np.random.randn(512).astype(np.float32),
            lemma="test",
        )

        # Test CREATE operation
        slot = new_memory_slot("memory_001", record)
        assert len(slot.mentions) == 1
        assert slot.memory_id == "memory_001"
        print("✅ CREATE operation works")

        # Test maturity score computation
        maturity = memory_maturity_features(slot, record)
        assert "raw_maturity_features" in maturity
        assert maturity["raw_maturity_features"].shape == (11,)
        print(f"✅ Maturity features computed: {maturity['raw_maturity_features'].shape}")

        # Test UPDATE operation
        record2 = StreamingMentionRecord(
            mention_id="test_002",
            split="test",
            episode_id="ep_1",
            stream_index=1,
            doc_id="doc_2",
            sentence_id=1,
            mention_text="another test event",
            bert_doc="Another test document",
            topic="topic_1",
            predicted_topic="topic_1",
            gold_cluster="event_1",
            gold_action="LINK",
            has_visual=True,
            mention_vector=np.random.randn(768).astype(np.float32),
            visual_vector=np.random.randn(512).astype(np.float32),
            lemma="test",
        )
        update_memory_slot(slot, record2)
        assert len(slot.mentions) == 2
        print("✅ UPDATE operation works")

        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_script_exists():
    """Test that training script exists and has required functions."""
    try:
        scripts_dir = Path(__file__).parent
        train_script = scripts_dir / "train_gpu_joint_memory_router.py"

        if not train_script.exists():
            print("❌ Training script not found")
            return False

        # Check if key classes exist in the script
        content = train_script.read_text()
        required = ["JointMemoryRouter", "def train", "def main"]
        missing = [name for name in required if name not in content]

        if missing:
            print(f"❌ Training script missing: {missing}")
            return False

        print("✅ Training script verified")
        return True
    except Exception as e:
        print(f"❌ Training script check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PSMEC Anonymous Code Repository - Installation Test")
    print("="*60)

    tests = [
        ("Module Imports", test_imports),
        ("Sample Data Loading", test_sample_data),
        ("Basic Functionality", test_basic_functionality),
        ("Training Script", test_training_script_exists),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        results.append(test_func())

    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"✅ All {total} tests passed! Installation verified.")
        print("\nNext steps:")
        print("  1. Run full training: ./configs/train_ecb_ol.sh")
        print("  2. Or quick test: bash scripts/quick_test_run.sh")
        return 0
    else:
        print(f"❌ {total - passed}/{total} tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
