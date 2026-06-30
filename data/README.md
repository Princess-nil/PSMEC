# Data Preparation Instructions

This directory should contain the two benchmarks described in Section 3 of the paper:

## Directory Structure

```
data/
├── ecb_ol/                   # ECB-OL benchmark
│   ├── train_records.pkl     # Training observations
│   ├── dev_records.pkl       # Development observations
│   └── test_records.pkl      # Test observations
│
└── wce_ol/                   # WCE-OL benchmark
    ├── train_records.pkl     # Training observations
    ├── dev_records.pkl       # Development observations
    └── test_records.pkl      # Test observations
```

## Benchmark Descriptions

### ECB-OL: Text-dominant Streaming Benchmark (Section 3.2)

ECB-OL reorganizes ECB+ into temporal streams to test delayed textual support.

**Statistics (Table 1 in paper):**
- Test episodes: 1,770
- Unique source observations: 1,780
- Gold clusters: 805
- Visual coverage: 27.41%
- Late support rate: 33.28%
- Conflict rate: 14.18%

**Key Challenge:** Delayed evidence - critical information confirming or refuting a link arrives only in later observations.

### WCE-OL: Multimodal Streaming Benchmark (Section 3.3)

WCE-OL sources multimodal reports from Wikipedia Current Events with paired text-image evidence.

**Statistics (Table 1 in paper):**
- Test episodes: 776
- Unique source observations: 882
- Gold clusters: 473
- Visual coverage: 59.83%
- Late support rate: 13.40%
- Conflict rate: 96.65%

**Key Challenge:** High-conflict scenarios where multiple events share similar textual descriptions and visual evidence is needed to distinguish them.

## Data Format

Each `.pkl` file contains a list of observation records with the following fields:

```python
{
    'mention_id': str,           # Unique observation identifier
    'episode_id': str,           # Episode (topic) identifier
    'stream_index': int,         # Position in temporal stream
    'doc_id': str,              # Source document identifier
    'gold_cluster': str,         # Gold event cluster ID
    'gold_action': str,          # 'CREATE' or 'LINK'
    'mention_text': str,         # Event mention text
    'text_vector': np.ndarray,   # Text embedding (768-dim MPNet)
    'visual_vector': np.ndarray, # Visual embedding (512-dim CLIP ViT-B/32)
    'has_visual': bool,          # Whether visual evidence is available
    'entity_names': set,         # Entity names extracted from mention
}
```

## Episode Construction Protocol (Section 3.1)

Both benchmarks use a sliding window protocol:
- **Window size (W):** 16 observations
- **Stride (S):** 1 observation

Each episode consists of:
1. **Query observation (m_t):** Requires immediate CREATE or LINK decision
2. **Local window:** W-1 recent observations providing immediate context
3. **Prefix:** Historical observations beyond the window

**Candidate records** are constructed from all prefix observations sharing the same gold event label as the query.

## Streaming Constraints

**No-future-evidence constraint:** Models can only observe:
- Current observation m_t
- Historical event store S_{t-1}
- No access to future observations m_{t+1}, m_{t+2}, ...

**Irreversible decisions:** Once a CREATE or LINK action is committed, it cannot be revised.

## Encoders

### Text Encoder
- **Model:** MPNet (sentence-transformers/all-mpnet-base-v2)
- **Dimension:** 768
- **Frozen:** Yes (not fine-tuned)

### Visual Encoder
- **Model:** CLIP ViT-B/32
- **Dimension:** 512
- **Frozen:** Yes (not fine-tuned)

## Data Availability

Due to the anonymous review process, the actual benchmark data files are not included in this repository. They will be made publicly available upon paper acceptance.

For review purposes, the data format and statistics are provided above to enable understanding of the method and experimental setup.

## Citation

```
[Anonymous submission to WISE 2026]
Benchmarks constructed from:
- ECB+: Cybulska & Vossen (2014)
- Wikipedia Current Events: Gholipour Ghalandari et al. (2020)
```
