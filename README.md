# Gestalt-MVP — Neuro-Symbolic Gesture Recognition System

## Phase 0: Deterministic DTW + Symbolic Gating

A mobile-first neuro-symbolic cognitive prosthesis for non-verbal users that translates 
personalized hand gestures into spoken language using a deterministic DTW matcher gated 
by a symbolic confidence engine.

**This is Phase 0:** Single-user feasibility study with NO LLM, NO BDI, NO IMU, NO cloud services.

---

## 🎯 Primary Hypothesis

> A deterministic DTW matcher operating on Z-axis-normalized MediaPipe hand trajectories 
> with a 20% Sakoe-Chiba band can discriminate a 10-class dynamic gesture vocabulary for 
> a single enrolled user under held-out session variation, achieving **macro F1 > 0.85** 
> AND **open-set FAR < 5%** simultaneously.

---

## 📁 Project Structure

```
gestalt_mvp/
├── data_collection/          # Python data capture scripts
│   ├── collect.py            # MediaPipe capture with auto-segmentation
│   ├── normalize.py          # Translation/scale/Z-rotation normalization
│   ├── segment.py            # REST→MOTION→REST boundary detection
│   └── data/                 # Collected JSON segments (sessions 1-5)
│
├── evaluation/               # Locked test evaluation
│   ├── evaluate.py           # Threshold tuning + locked evaluation
│   ├── metrics.py            # FAR, FRR, F1, confusion matrix
│   ├── bootstrap.py          # Session-level bootstrap CIs
│   └── results/              # Output JSON/CSV
│
├── native/                   # Rust symbolic core + C++ MediaPipe glue
│   ├── Cargo.toml            # Rust dependencies
│   ├── src/
│   │   ├── lib.rs            # Main engine + FFI exports
│   │   ├── graph.rs          # Gesture prototypes + SQLite storage
│   │   ├── matcher.rs        # Weighted DTW + Sakoe-Chiba band
│   │   ├── fsm.rs            # 8-state deterministic FSM
│   │   ├── confidence.rs     # Multi-factor confidence calculation
│   │   ├── guards.rs         # Physics/Social/Epistemic guards
│   │   └── message_bus.rs    # Typed message system
│   └── mediapipe_cpp/
│       ├── hand_tracking.h   # C API header
│       └── hand_tracking.cpp # MediaPipe Hands implementation (~100 lines)
│
├── lucid_app/                # Flutter mobile UI
│   ├── pubspec.yaml          # Dart dependencies
│   ├── lib/
│   │   ├── main.dart         # App entry point
│   │   ├── bridge.dart       # FFI bindings to Rust
│   │   ├── camera_screen.dart # Camera + overlay + output panel
│   │   ├── gesture_overlay.dart # Hand landmark visualization
│   │   └── status_bar.dart   # FSM state indicator
│   └── assets/models/        # TTS models, hand_landmarker.task
│
├── PROTOCOL.md               # Pre-registration document (frozen spec)
└── README.md                 # This file
```

---

## 🔧 Tech Stack

| Layer | Language/Tool | Role |
|-------|--------------|------|
| UI + OS | Dart (Flutter) | Camera, audio, overlay |
| Symbolic Core | Rust | Graph, FSM, confidence, DTW |
| FFI Bridge | flutter_rust_bridge | Auto-generated Dart↔Rust |
| Hand Tracking | C++ (MediaPipe Tasks Vision) | ~100 lines glue only |
| Voice Output | Piper TTS / Sherpa-ONNX | Local, deterministic |
| Graph Storage | SQLite via rusqlite | Flat gesture→meaning table |

**Dropped languages:** Zig, Julia, Dafny, Lean 4, Swift, Kotlin, FAISS

---

## 🖐️ 10-Gesture Vocabulary

| ID | Name | Motion Description | Phrase Template |
|----|------|-------------------|-----------------|
| G1 | Water | Chest → sharp arc to mouth → return | "I would like some water, please." |
| G2 | Help | Both hands at waist, palms up, lift 6in → return | "Could you help me, please?" |
| G3 | Yes | Closed fist, sharp double-nod (up-down-up) → return | "Yes." |
| G4 | No | Index+middle extended, horizontal snip → return | "No." |
| G5 | Food | Bunched fingertips, tap chin twice → return to lap | "I would like some food, please." |
| G6 | Pain | Flat hand, palm in, tap chest center twice → return | "I am in pain. Please help me." |
| G7 | Stop | Flat hand, palm out, sharp forward push → return | "Stop. Please stop." |
| G8 | More | Both flat hands, palms facing, clap once → return | "I would like more, please." |
| G9 | Wait | Flat hand, palm down, slow horizontal circle → return | "Please wait a moment." |
| G10 | Bathroom | Flat hand "C" shape, tap lower abdomen twice → return | "I need to use the bathroom." |

**All gestures are PURELY DYNAMIC** (motion + return to rest). No static holds.

---

## 🧮 Normalization Math

Given raw 21 landmarks per frame:

1. **Translation:** `p'_i = p_i - p_wrist`
2. **Scale:** `s = ||p'_MCP_middle||_2`, then `p''_i = p'_i / s`
3. **Z-Rotation (azimuth only):** `θ = atan2(p''_MCP_middle.y, p''_MCP_middle.x)`
   - Apply R(-θ) to X and Y only
   - **Z-coordinates (depth) are PRESERVED**

Full transform: `p'''_i = R(-θ)(p_i - p_wrist) / s`

---

## 📊 Three-Way Data Split

| Sessions | Role | Rules |
|----------|------|-------|
| 1-2 | Enrollment | Build multi-exemplar library (5-10 per class) |
| 3 | Calibration | Tune ALL thresholds. FREEZE. Never touch. |
| 4-5 | Locked Test | Run ONCE. Report metrics. Never modify. |

**Split by SESSION, never by frame.**

---

## 🎯 Evaluation Metrics

### Primary Endpoint
**Macro F1 > 0.85 AND FAR < 5% simultaneously**

### Secondary Metrics
- 10×10 confusion matrix
- Per-class precision/recall
- FRR (False Rejection Rate)
- Near-miss rejection rate (NEG-3)
- p95 segmentation latency
- p95 DTW computation time
- Session-level bootstrap confidence intervals

### Cost Function
`Cost = 10 × FP + 2 × FR + 0.5 × Clarifications`

---

## 🏗️ Architecture

```
[ CAMERA (mounted, facing user) ]
         │
         ▼
[ MEDIAPIPE HANDS ]  →  21 landmarks × 30fps
         │
         ▼
[ PREPROCESSING & SEGMENTATION ]
    1. Translation / Scale / Z-Rotation normalization
    2. Velocity energy calculation
    3. REST → MOTION → REST boundary detection
         │
         ▼
[ DETERMINISTIC TRAJECTORY MATCHER ]
    DTW against enrolled exemplars (5-10 per class)
    Outputs: D1, D2, margin
         │
         ▼
[ COST-WEIGHTED THRESHOLD GATE ]
    ACCEPT / CLARIFY / REJECT
         │
         ▼
[ FINITE STATE MACHINE ]
    Gesture ID → Meaning Slot
    Handles CORRECTION state
    Handles SILENCE (hardwired, bypasses DTW)
         │
         ▼
[ DETERMINISTIC PHRASE TEMPLATES ]
    "I would like {OBJECT}, please."
         │
         ▼
[ TTS ACTUATOR ]  (Piper / Sherpa-ONNX)
```

### Sacred Boundaries
- ✅ Neural layer (MediaPipe) proposes evidence
- ✅ Symbolic core (Rust) resolves interpretations
- ✅ Phrasing layer (templates) never introduces meaning
- ❌ **NO LLM/Qwen/GPT in Phase 0**
- ❌ **NO VLM/SmolVLM2 in Phase 0**
- ❌ **NO IMU in Phase 0**
- ❌ **NO BDI — Use deterministic FSM**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Rust 1.70+
- Flutter 3.10+
- MediaPipe Tasks Vision library
- Camera with 30fps capability

### Step 1: Data Collection

```bash
cd data_collection
python collect.py --session 1 --gesture G1 --reps 10
```

Supported conditions: `normal`, `fast-small`, `slow-large`, `45-degree`, `occlusion`

### Step 2: Build Rust Core

```bash
cd native
cargo build --release
```

### Step 3: Generate FFI Bindings

```bash
cd lucid_app
flutter pub run build_runner build
```

### Step 4: Run Flutter App

```bash
flutter run
```

---

## 📋 Absolute Rules (Phase 0)

1. ❌ No LLMs (Qwen, GPT, LLaMA)
2. ❌ No VLMs (SmolVLM2)
3. ❌ No IMU data
4. ❌ No BDI — Use deterministic FSM
5. ❌ No sleep mode consolidation
6. ❌ No emotion detection
7. ❌ No FAISS — Brute-force nearest-neighbor in Rust
8. ❌ Never tune thresholds on Sessions 4-5
9. ❌ Never split data by frame — Split by session only
10. ❌ No static-hold gestures
11. ❌ No features beyond specification
12. ✅ All Session 3 thresholds FROZEN before Session 4
13. ✅ Soft-mute is dynamic swipe, not static hold
14. ✅ Corrections modify MAPPING, never overwrite recognizer
15. ✅ Every decision logged to append-only JSONL audit trail

---

## 📈 Success Criteria

Phase 0 succeeds if ALL criteria are met:

1. ✅ Macro F1 > 0.85 on Sessions 4-5
2. ✅ FAR < 5% on NEG-1 + NEG-2
3. ✅ Near-miss rejection > 90% on NEG-3
4. ✅ p95 end-to-end latency < 200ms
5. ✅ All code runs offline, no LLMs, no cloud
6. ✅ Full audit trail generated
7. ✅ Reproducible results (seeded randomness documented)

**If ANY criterion fails:** Analyze failure mode, do NOT proceed to Phase 1.

---

## 📄 Documentation

- **[PROTOCOL.md](PROTOCOL.md)** — Pre-registration document (frozen specification)
- **[evaluation/results/](evaluation/results/)** — Metrics output (after running evaluation)

---

## 🔒 License

MIT License — Open source for reproducibility verification

---

## 📝 Version

**Phase 0 v1.0** — Deterministic DTW + Symbolic Gating
