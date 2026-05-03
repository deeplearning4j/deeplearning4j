---
name: vlm-second-call-git-diff-analysis
description: "Git diff analysis: isDynamicShape BFS too conservative is NEW lead, expand timing is safe"
type: project
---

## Git Diff Analysis — New Leads (2026-05-02)

### KERNEL_FAILURE (50) Comment — Removed
KV max-allocation re-enabled in DynamicShapePlanExecutor.java. The auto-configure at line 3193 only fires when `maxKvCacheLength > 0`, which is 0 during warmup. The explicit configure at line 1608 fires BEFORE freeze. **Timing is safe** — expand happens before CUDA graph capture.

### isDynamicShape BFS — NEW LEAD (HIGH PRIORITY)
**File:** NativeDynamicShapePlan_slotexec.cpp:198-290

OLD code: simple one-hop check → many slots marked isDynamicShape=true
NEW code: BFS walks full upstream graph → external non-placeholder inputs NOT treated as dynamic

Comment at lines 209-212: "Treating all externals as dynamic caused nearly every slot to be isDynamicShape=true, preventing frozen-state stabilization."

**Potential bug:** If the BFS incorrectly marks a slot as non-dynamic (isDynamicShape=false) when it SHOULD be dynamic, the frozen fast-path at line 2323 (`if (slot.frozenContextReady() && !slot.flags.isDynamicShape)`) skips re-execution and returns a stale cached result. On the second plan, the stale result could be zeros (from prezero or from the previous session's teardown).

### Frozen Fast-Path Gate (slotexec.cpp:1564-1570)
```cpp
shapesFrozen_ && executeCount_ >= 2 && contextPool_ != nullptr && frozenContextReady() && !isDynamicShape
```
If isDynamicShape is incorrectly false for a slot whose output changes between decode steps, that slot's output is frozen at its warmup value — potentially zeros.

### Remaining Untraceable Changes in slotexec.cpp
- Warmup window `executeCount_ == 0` → `executeCount_ <= 1` — affects warmup pass count
- `discardCachedSlotArray` replacing raw delete — different cleanup behavior?
- `shapeOnlyMode_` guards — new mode that skips op execution entirely
- Fused chain intermediate separate allocation — reworked buffer management
- Max-size allocation rework with `db->expand()`

### prezero + frozen fast-path interaction
On second plan with same shapes (cache hit): `shapesFrozen_=true`, `executeCount_` high → prezero SKIPPED. If a slot is incorrectly `!isDynamicShape`, its output is never recomputed → stale zeros propagate.

**Why:** Understanding what changed between good/bad commits to narrow root cause.
**How to apply:** The isDynamicShape BFS change is the strongest remaining lead. A slot incorrectly marked non-dynamic would cause its frozen output to persist across decode steps, producing zeros if the frozen value was zero. Need to trace which specific slots the BFS affects in the SmolDocling/Qwen model and whether any logit-path slots are incorrectly classified.
