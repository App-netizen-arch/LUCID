# Phase 1: Neural Gesture Encoder

## Overview

Phase 1 implements a learned temporal gesture encoder to address the failure modes identified in Phase 0:

### Phase 0 Failure Modes Addressed
1. **Speed variation sensitivity** - DTW struggles with gestures performed at different speeds
2. **Amplitude variation robustness** - Gestures with different motion magnitudes
3. **Poor inter-class separation** - Three gesture classes (G3, G4, G6) had zero correct classifications

## Architecture

```
Input: Normalized trajectory (T=30 frames × 63 features)
         │
         ▼
┌─────────────────────────┐
│  Conv1D(64, k=5) + BN   │  Temporal feature extraction
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Conv1D(128, k=5) + BN  │  Hierarchical patterns
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Conv1D(256, k=5) + BN  │  High-level gesture features
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Global Avg Pooling     │  Fixed-size representation
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Linear(256 → 128)      │  Embedding projection
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  L2 Normalization       │  Cosine similarity ready
└─────────────────────────┘
         │
         ▼
Output: 128-dimensional gesture embedding
```

**Total Parameters:** ~2.1M (well under 10M limit)

## Training Strategy

### Contrastive Learning
- **Loss Function:** Triplet Loss with margin OR NT-Xent (SimCLR-style)
- **Positive Pairs:** Different executions of same gesture
- **Negative Pairs:** Executions of different gestures
- **Pre-training:** Optional on HaGRID dataset (if available)
- **Fine-tuning:** On user's enrollment data (5-10 examples per gesture)

### Data Requirements
- **Training:** Sessions 1-2 (enrollment data)
- **Validation:** Session 3 (calibration data)
- **Testing:** Sessions 4-5 (locked test data)

## Files

| File | Description |
|------|-------------|
| `train_encoder.py` | Main training script with GestureEncoder model |
| `export_to_onnx.py` | Export PyTorch model to ONNX for Rust integration |
| `requirements.txt` | Python dependencies |

## Usage

### 1. Train the Encoder

```bash
cd gestalt_mvp/neural_encoder

# Train with triplet loss (default)
python train_encoder.py \
    --data_dir ../data_collection/data \
    --output_dir ./models \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --margin 0.3

# Or train with NT-Xent loss
python train_encoder.py \
    --data_dir ../data_collection/data \
    --use_ntxent \
    --epochs 50
```

### 2. Export to ONNX

```bash
python export_to_onnx.py \
    --model_path ./models/gesture_encoder.pt \
    --output_path ./models/gesture_encoder.onnx \
    --test
```

### 3. Evaluate

The training script automatically evaluates on Sessions 4-5 and compares with Phase 0 baseline:

```
Phase 0 Results:
  Macro F1: 0.700 (target: >0.85)
  FAR: 0.208 (target: <0.05)
  FRR: 0.006

Phase 1 Results:
  Macro F1: X.XXX (target: >0.85)
  Accept rate: X.XXX
  FRR: X.XXX

F1 Improvement: +X.XXX
✓ Phase 1 PASSED: Neural encoder improves F1 by >5 points
→ Adopt neural encoder for Phase 2
```

## Adoption Criteria

Per specification Section 1.1:

**Adopt neural encoder ONLY if:**
- Macro F1 improves by >5 absolute points compared to Phase 0 DTW baseline
- At matched FAR (False Acceptance Rate)

**Otherwise:** Keep DTW baseline for Phase 2

## Integration with Rust Core

The exported ONNX model is loaded by the Rust symbolic core via `ort` (ONNX Runtime):

```rust
// In native/src/matcher.rs (to be implemented in Phase 2)
use ort::{Session, Value};

pub struct NeuralMatcher {
    session: Session,
    prototypes: HashMap<i32, Vec<f32>>,
}

impl NeuralMatcher {
    pub fn compute_embedding(&self, trajectory: &[f32]) -> Result<Vec<f32>> {
        // Input shape: [1, 30, 63]
        let input = Value::from_array(trajectory)?;
        let outputs = self.session.run(vec![input])?;
        // Output shape: [1, 128]
        Ok(outputs[0].extract_tensor::<f32>()?.view().to_vec())
    }
    
    pub fn recognize(&self, embedding: &[f32]) -> MatchResult {
        // Cosine distance to prototypes
        // Same margin-based gating as Phase 0
    }
}
```

## Dependencies

```txt
torch>=2.0.0
numpy>=1.24.0
onnx>=1.14.0
onnxruntime>=1.15.0
scikit-learn>=1.3.0
```

Install with:
```bash
pip install torch numpy onnx onnxruntime scikit-learn
```

## Model Size

- **PyTorch checkpoint:** ~8 MB
- **ONNX model:** ~9 MB
- **Prototypes:** ~5 KB (10 classes × 128 floats)

All models run locally, offline. No cloud services.

## Next Steps

After Phase 1 validation:
- **If PASSED:** Integrate neural encoder into Phase 2 pipeline
- **If FAILED:** Continue with DTW baseline from Phase 0

Proceed to **Phase 2: End-to-End Pipeline on Laptop**
