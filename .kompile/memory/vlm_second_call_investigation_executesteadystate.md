---
name: vlm-second-call-investigation-executeSteadyState
description: executeSteadyState warm-up sequence RULED OUT — 4-way gate ensures correct fallback
type: project
---

## executeSteadyState — RULED OUT (2026-05-02)

### Gate (NativeDynamicShapePlan.cpp:2409-2416)
```cpp
if (!shapesFrozen_ || executeCount_ < 3 ||
    planPhase_ < PlanPhase::REPLAYING ||
    Environment::getInstance().tritonVerifyKernels()) {
  return execute(...);  // FALLBACK to full execute()
}
```

All four must be true simultaneously. After a fresh plan or invalidateForRebuild, `executeCount_=0` → immediate fallback.

### Warm-up timeline for second plan
| Step | executeCount_ | planPhase_ | Gate | What runs |
|------|---------------|------------|------|-----------|
| 1 | 0 | SHAPES_FROZEN | FALLBACK | phaseWarmup |
| 2 | 1 | SHAPES_FROZEN/POINTERS_STABLE | FALLBACK | phaseReplay → graph capture |
| 3 | 2 | POINTERS_STABLE/REPLAYING | FALLBACK | phaseReplay → CAPTURED→REPLAYING |
| 4+ | 3+ | REPLAYING | FAST PATH | platformTryFrozenFastPath |

### External input sync
- `executeSteadyState` delegates H2D sync to `platformTryFrozenFastPath` (same as full execute)
- Sync fires for any input where `db->isPrimaryActual()` is true (host newer than device)
- Variable-index cache (`variableExternalInputIndices_`) is per-plan, populated independently
- `argTableStable` gates Triton arg table refresh, NOT data sync — independent operations

**Why:** Investigating whether the hot path skips critical sync/setup for the second plan.
**How to apply:** The gate is correct. Second plan goes through 3 full execute() calls before entering steady state. Not the root cause.
