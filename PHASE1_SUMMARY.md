# Phase 0 Results Summary

## Primary Endpoints
- **Macro F1**: 0.700 (Target: > 0.85) ❌ FAIL
- **FAR**: 0.208 (Target: < 0.05) ❌ FAIL  
- **FRR**: 0.006 ✓ PASS

## Failure Mode Analysis

### Issue 1: Excessive Clarifications
- 95 out of 160 known samples (59%) went to CLARIFY state
- Root cause: tau_margin threshold (5.0) was set too high
- Mean margin for known gestures: 5.52, but 10th percentile only 1.09
- This means many genuine gestures have low confidence margins

### Issue 2: Poor Class Separation
Three gesture classes had ZERO correct classifications:
- G3 (Yes): 0 support - all rejected/misclassified
- G4 (No): 0 support - all rejected/misclassified  
- G6 (Pain): 0 support - all rejected/misclassified

These gestures likely have trajectories too similar to other classes under DTW.

### Issue 3: High False Acceptance Rate
- FAR = 20.8% (negative samples incorrectly accepted)
- Mean D1 for negatives: 14.44 vs mean D1 for positives: 5.05
- But significant overlap in distributions

## Phase 1 Recommendation

Per specification Section 1.1:
> IF Phase 0 failed any target: Identify WHICH failure mode caused it
> Build the neural encoder to address that SPECIFIC failure mode.

**Identified Failure Modes:**
1. Speed variation sensitivity (DTW doesn't handle temporal warping well enough)
2. Amplitude variation (some gestures performed larger/smaller than enrolled)
3. Insufficient inter-class margins in trajectory space

**Phase 1 Architecture Decision:**
Build a **1D-CNN temporal encoder** with contrastive learning to:
- Learn embeddings that maximize inter-class margins
- Be robust to speed/amplitude variations
- Preserve the symbolic core (FSM, gating, templates) unchanged

**Success Criteria for Phase 1:**
- Improve Macro F1 by > 5 absolute points (from 0.700 to > 0.750)
- Reduce FAR below 10% (from 20.8%)
- Maintain matched or better FRR

