---
name: vlm-second-call-investigation-resetModelState
description: BenchmarkConfigApplier.resetModelState cleanup order + generateNative guard interaction — cleanup is CORRECT
type: project
---

## BenchmarkConfigApplier.resetModelState() — Cleanup Correct (2026-05-02)

### resetModelState (BenchmarkConfigApplier.java:52-86)
```java
model.resetSession();              // 1. removes+destroys session from map
model.clearPlaceholderOverrides(); // 2.
model.clearPlaceholders(true);     // 3.
model.clearOpInputs();             // 4.
model.clearDynamicShapePlanCache();// 5. closes Java plans, clears C++ cache
model.setGraphExecutionMode(AUTO); // 6.
```

### generateNative() guard interaction (GenerationPipeline.java:1322-1330)
```java
InferenceSession existingSession = decoder.getOrCreateSession(); // CREATES new session
if (existingExecutor != null && existingExecutor.isShapesFrozen()) { // FALSE on new
    decoder.clearDynamicShapePlanCache();  // SKIPPED
    decoder.resetSession();                // SKIPPED
}
```

The guard is bypassed because `resetModelState()` already removed the old session. `getOrCreateSession()` creates a BRAND NEW session with `shapesFrozen=false`. BUT this is CORRECT — `resetModelState()` already did the full cleanup:
- `resetSession()` destroyed the old session (freed session-owned buffers)
- `clearDynamicShapePlanCache()` destroyed ALL C++ native plans and CUDA graphs

No stale CUDA graphs survive. The C++ plan cache is completely empty.

### setShapesFrozen(false) — architecturally BANNED
`setShapesFrozen(false)` throws `IllegalArgumentException` unconditionally (DynamicShapePlanExecutor.java:794-833). shapesFrozen is only reset to false via internal direct field writes:
- Line 412: setPlan(null) (session teardown)
- Line 428: setPlan(newPlan) (plan change)
- Line 922: resetForNextPage()

### TestDspValidation flow
`testOutputAccuracy` → calls `runDecode()` twice → each call does `resetModelState(decoder)` then recompiles a fresh plan → creates fresh `GenerationPipeline` → calls `generate()`.

**Why:** Investigating whether cleanup race between resetModelState and generateNative guard leaves stale CUDA graphs.
**How to apply:** Cleanup is correct for the test path. resetModelState fully cleans up. The generateNative guard is a safety net for code paths that DON'T call resetModelState (e.g., direct multi-generation without benchmark harness).
