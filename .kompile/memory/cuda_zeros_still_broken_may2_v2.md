---
name: cuda-zeros-still-broken-may2-v2
description: CUDA SmolDocling still zeros after frozen fast-path fix — graph replay itself produces stale outputs, not just executeSlot
type: project
---

## CUDA SmolDocling Still Zeros After Frozen Fast-Path Fix (2026-05-02 13:30)

**Status:** `tokenIds{head=[216, 49229, 0, 0, 0, 0, 0, 0]}` — same as before the `!tl_dspReplayActive` fix.

**What was tried:**
1. Added `!tl_dspReplayActive` to frozen fast-path gate in `NativeDynamicShapePlan_slotexec.cpp:1574` — DID NOT FIX
2. Rebuilt CUDA, ran benchmark with OPTIMAL config — same zeros

**Key observation:** Decode steps show `exec=6ms copy=31ms` after warmup. The 6ms execution is graph replay (captured CUDA graph). The problem is in the graph replay path itself, not in the slot-by-slot executeSlot path. The `!tl_dspReplayActive` fix only helps during composite replay gap ops calling `executeSlot`, but the actual captured graph is replaying with stale data.

**Root cause hypothesis:** The captured CUDA graph records specific device pointer addresses. When the graph is replayed, the plan must update external input buffers (embeddings, attention_mask, position_ids, KV cache pointers) to their new values BEFORE launching the graph. If `refreshExternalInputs` or `markExternalInputVariable` isn't properly updating the captured graph's argument table, the graph replays with the warmup step's data.

**Next diagnostic step:** Run with `--config SLOT_BY_SLOT --debug` and `Nd4j.getEnvironment().setDebug(true); setVerbose(true)` to see if SLOT_BY_SLOT (no graph capture) also produces zeros. If SLOT_BY_SLOT works, the bug is in graph capture/replay. If SLOT_BY_SLOT also zeros, the bug is in the ops themselves.

**Validation step disabled:** Removed validation preflight from run-benchmark.sh — it was blocking runs without catching the real issue.

**How to apply:** Must isolate whether the bug is DSP graph replay vs op-level correctness before fixing.
