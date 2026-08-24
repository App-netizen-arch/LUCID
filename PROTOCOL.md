# PROTOCOL.md — Gestalt-MVP Pre-Registration Document

## Study Title
Neuro-Symbolic Gesture Recognition for Non-Verbal Communication: A Single-User Feasibility Study

## Principal Investigator
[Name Redacted]

## Study Phase
Phase 0 — Deterministic DTW + Symbolic Gating (No LLM, No BDI, No IMU)

---

## 1. PRIMARY HYPOTHESIS

> A deterministic DTW matcher operating on Z-axis-normalized MediaPipe hand trajectories with a 20% Sakoe-Chiba band can discriminate a 10-class dynamic gesture vocabulary for a single enrolled user under held-out session variation, achieving **macro F1 > 0.85** AND **open-set FAR < 5%** simultaneously.

---

## 2. SECONDARY HYPOTHESES

### H2.1: Segmentation Reliability
REST → MOTION → REST boundary detection using wrist velocity energy will achieve ≥95% temporal accuracy (±1 frame) compared to manual annotation.

### H2.2: Margin-Based Open-Set Rejection
The margin gating strategy (D2 - D1) will reject ≥95% of unknown gestures (NEG-1, NEG-2) while maintaining ≥85% acceptance rate for known classes.

### H2.3: Near-Miss Discrimination
The system will correctly reject ≥90% of near-miss gestures (NEG-3: wiping mouth, scratching cheek, adjusting glasses) that share kinematic similarity with target vocabulary.

### H2.4: Latency Budget
End-to-end latency (frame capture → phrase output) will have p95 < 200ms, enabling real-time communication.

### H2.5: Correction Efficacy
Caregiver-mediated corrections will reduce repeated errors by ≥80% within 3 correction cycles per gesture class.

---

## 3. EXPERIMENTAL DESIGN

### 3.1 Participant
- **N = 1** non-verbal adult (or proxy performer following standardized protocol)
- Right-hand dominant (or dominant hand if left-handed)
- No hand/upper limb motor impairments

### 3.2 Data Collection Protocol

#### Sessions (5 total, split by role):

| Session | Role | Purpose | Data Used For |
|---------|------|---------|---------------|
| 1 | Enrollment | Build exemplar library | Training |
| 2 | Enrollment | Expand exemplar diversity | Training |
| 3 | Calibration | Threshold tuning ONLY | Validation (tuning) |
| 4 | Locked Test | Final evaluation | Test (never tune) |
| 5 | Locked Test | Final evaluation | Test (never tune) |

**Critical:** All splits are by SESSION, never by frame or repetition.

#### Conditions per Session:
Each session includes all 10 gestures × 10 repetitions = 100 trials per condition:

1. **Normal**: Standard execution at comfortable speed
2. **Fast-Small**: Rapid, reduced amplitude (simulating urgency/fatigue)
3. **Slow-Large**: Deliberate, exaggerated motion (simulating motor impairment)
4. **45-Degree**: Camera angle shifted 45° from enrollment position
5. **Occlusion**: Partial hand occlusion (holding small object)

**Total dataset size:**
- 5 sessions × 5 conditions × 10 gestures × 10 reps = **2,500 gesture segments**
- Plus negative datasets:
  - NEG-1 (Random): 200 trials
  - NEG-2 (Everyday): 200 trials  
  - NEG-3 (Near-Miss): 300 trials
- **Grand total: 3,200 segments**

### 3.3 Gesture Vocabulary (10 Classes)

| ID | Name | Motion Description | Template Phrase |
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

### 3.4 Negative Datasets

#### NEG-1: Random Movements (200 trials)
- Waving, shaking hands, stretching
- Drumming fingers, random flailing

#### NEG-2: Everyday Activities (200 trials)
- Typing motions, drinking from cup
- Picking up phone, scratching arm
- Reaching for objects

#### NEG-3: Near-Miss Confusables (300 trials)
- Wiping mouth, scratching cheek
- Rubbing eyes, touching hair
- Adjusting glasses, folding arms
- Resting chin on hand

---

## 4. PREPROCESSING PIPELINE (FROZEN)

### 4.1 Normalization (per frame)

Given raw 21 landmarks $p_i \in \mathbb{R}^3$:

**Translation:**
$$p'_i = p_i - p_{\text{wrist}}$$

**Scale:**
$$s = \|p'_{\text{MCP\_middle}}\|_2$$
$$p''_i = p'_i / s$$

**Z-Rotation (azimuth alignment only):**
$$\theta = \text{atan2}(p''_{\text{MCP\_middle}.y}, p''_{\text{MCP\_middle}.x})$$
Apply rotation matrix $R(-\theta)$ to X and Y coordinates only.
**Z-coordinates (depth) are PRESERVED.**

Full transform:
$$p'''_i = R(-\theta)(p_i - p_{\text{wrist}}) / s$$

### 4.2 Segmentation

Interaction contract: **REST → GESTURE → REST**

- **Motion Start:** wrist velocity energy > τ_start for 3 consecutive frames
- **Motion End:** wrist velocity energy < τ_rest for ≥300ms (9 frames @30fps)
- **Duration bounds:** discard segments <0.3s or >2.5s
- **Sampling:** 30fps fixed

### 4.3 Landmark Weights (DTW cost function)

$$d(a,b) = \sqrt{\sum_{j=1}^{21} w_j \cdot \|a_j - b_j\|^2}$$

- **Fingertips (4, 8, 12, 16, 20):** $w_j = 2.0$
- **All other landmarks:** $w_j = 1.0$

### 4.4 Sakoe-Chiba Band

$$r = \max(3, \lfloor 0.20 \times \max(L_x, L_y) \rfloor)$$

Where $L_x, L_y$ are trajectory lengths.

---

## 5. RECOGNITION PIPELINE (FROZEN)

### 5.1 DTW Matching

For each gesture class $C$ with exemplars $\{p_1, ..., p_n\}$:

$$D(x, C) = \min_{i} \text{DTW}(x, p_i)$$

**Multi-exemplar support:** 5-10 exemplars per class from Sessions 1-2.

### 5.2 Margin-Based Gating

Let:
- $D_1$ = distance to nearest class
- $D_2$ = distance to second-nearest class
- $\text{margin} = D_2 - D_1$

| Condition | Decision | Action |
|-----------|----------|--------|
| $D_1 < \tau_{\text{accept}}$ AND $\text{margin} > \tau_{\text{margin}}$ | ACCEPT | Speak phrase |
| $D_1 < \tau_{\text{accept}}$ AND $\text{margin} \leq \tau_{\text{margin}}$ | CLARIFY | Request confirmation |
| $D_1 > \tau_{\text{accept}}$ | REJECT | Silent (no action) |

**Thresholds ($\tau_{\text{accept}}, \tau_{\text{margin}}$) tuned ONLY on Session 3.**

### 5.3 Finite State Machine

```
IDLE
  → LISTENING            (hand present, motion detected)
    → GESTURE_DETECTED   (motion + rest boundary found)
      → HIGH_CONF        → SPEAK
      → LOW_CONF         → CLARIFY or SILENT
  → CORRECTION           (user/caregiver signals "wrong")
  → SILENT               (soft-mute swipe OR physical volume button)
  → ERROR                (sensor failure, circuit breaker)
```

**Correction modifies MAPPING, never overwrites recognizer.**

---

## 6. EVALUATION METRICS

### 6.1 Primary Endpoint

**Success criterion:** Macro F1 > 0.85 AND FAR < 5% **simultaneously** on Sessions 4-5.

### 6.2 Secondary Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Macro F1 | Unweighted mean of per-class F1 scores | >0.85 |
| FAR (False Acceptance Rate) | Unknown trials accepted as any gesture / total unknown trials | <5% |
| FRR (False Rejection Rate) | Known gestures rejected / total known gestures | <15% |
| Near-Miss Rejection | NEG-3 correctly rejected / total NEG-3 | >90% |
| p95 Segmentation Latency | 95th percentile of segment boundary detection time | <50ms |
| p95 DTW Computation | 95th percentile of DTW matching time | <100ms |
| Clarification Rate | Trials requiring CLARIFY / total trials | <20% |

### 6.3 Confusion Matrix

10×10 matrix showing per-class precision/recall.

### 6.4 Cost Function

$$\text{Cost} = 10 \times \text{FP} + 2 \times \text{FR} + 0.5 \times \text{Clarifications}$$

Weighted to heavily penalize false positives (speaking wrong phrase).

### 6.5 Statistical Analysis

- **Session-level bootstrap confidence intervals** (10,000 resamples)
- Report 95% CI for all primary/secondary metrics
- No hypothesis testing (single-subject design)

---

## 7. THRESHOLD CALIBRATION PROTOCOL

### Session 3 ONLY (never touch Sessions 4-5):

1. **Grid search** over $\tau_{\text{accept}} \in [0.1, 0.5]$ (step 0.02)
2. **Grid search** over $\tau_{\text{margin}} \in [0.05, 0.3]$ (step 0.01)
3. Select pair maximizing: $(1 - \text{FAR}) \times \text{Macro F1}$
4. **FREEZE thresholds** before viewing Session 4 data
5. **Document** chosen values in audit trail

**Forbidden:** Any threshold adjustment after Session 3 calibration.

---

## 8. AUDIT TRAIL REQUIREMENTS

Every decision logged to append-only JSONL:

```json
{"timestamp": "...", "event": "GESTURE_DETECTED", "gesture_id": "G1", 
 "D1": 0.23, "D2": 0.45, "margin": 0.22, "decision": "ACCEPT"}
{"timestamp": "...", "event": "CORRECTION", "detected": "G1", 
 "intended": "G5", "mapping_updated": true}
{"timestamp": "...", "event": "THRESHOLD_CALIBRATION", "tau_accept": 0.32,
 "tau_margin": 0.15, "session": 3}
```

---

## 9. EXCLUSION CRITERIA

### Exclude segment if:
- Duration <0.3s or >2.5s
- Hand not visible for >30% of segment
- Segmentation boundary ambiguous (>3 frame uncertainty)
- Camera tracking lost (landmark confidence <0.5)

### Exclude session if:
- >20% of trials excluded
- Equipment malfunction documented

---

## 10. ETHICAL CONSIDERATIONS

- Single-user feasibility study (no IRB required for self-experimentation)
- All data stored locally, no cloud transmission
- Participant may withdraw at any time
- Data anonymized before publication

---

## 11. SUCCESS CRITERIA FOR PHASE 0

**Phase 0 is successful if:**

1. ✅ Macro F1 > 0.85 on Sessions 4-5
2. ✅ FAR < 5% on NEG-1 + NEG-2
3. ✅ Near-miss rejection > 90% on NEG-3
4. ✅ p95 end-to-end latency < 200ms
5. ✅ All code runs offline, no LLMs, no cloud services
6. ✅ Full audit trail generated
7. ✅ Reproducible results (seeded randomness documented)

**If ANY criterion fails:** Analyze failure mode, do NOT proceed to Phase 1.

---

## 12. TIMELINE

| Week | Milestone |
|------|-----------|
| 1-2 | Data collection (Sessions 1-5) |
| 3 | Threshold calibration (Session 3) |
| 4 | Locked evaluation (Sessions 4-5) |
| 5 | Analysis, report writing |
| 6 | Go/no-go decision for Phase 1 |

---

## 13. DECLARATIONS

**Pre-registered:** [Date]  
**Protocol version:** 1.0  
**Data availability:** Upon request for reproducibility verification  
**Code availability:** Open-source under MIT license  

**Conflicts of interest:** None declared  
**Funding:** Self-funded  

---

*This protocol is frozen. Any deviations must be documented in the audit trail with justification.*
