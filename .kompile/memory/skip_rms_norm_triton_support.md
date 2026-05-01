---
name: skip_rms_norm Triton support
description: "Adding Triton emitter for skip_rms_norm — the #1 perf blocker causing 0 Triton launches and no CUDA graph replay"
type: project
---

## skip_rms_norm Triton Support (2026-04-28)

**Problem:** `skip_rms_norm` (60 instances in SmolDocling decode) had no Triton emitter. Every occurrence invalidated the Triton module ("no SSA value for output slot"), resulting in 0 Triton launches and no CUDA graph replay. This caused 84% of decode time (~12.3ms/step) to be sync overhead.

**Expected impact:** Enabling CUDA graph replay should collapse 26 sync gaps, reducing per-step time from ~19.6ms to ~7ms → ~137 tok/s (from 50.98).

**Changes:**
1. `TritonIRBuilder.cpp` — registered skip_rms_norm in buildOpTable() as NORMALIZATION
2. `TritonIRBuilder_emitters.cpp` — added `skiprmsnorm` alias to `rmsnorm` case (the residual add is done in module dispatch)
3. `TritonIRBuilder_module.cpp` (buildModule) — loads skip tensor via getNormInput(1), computes hidden=input+skip, passes gamma=getNormInput(2), stores hidden to output[1]
4. `TritonIRBuilder_module.cpp` (buildSectionedModule) — same pattern using loadNormInput(src, rowWise=true)

**skip_rms_norm semantics:** input[0]=x, input[1]=skip, input[2]=gamma, input[3]=bias(opt). output[0]=rms_norm(x+skip)*gamma, output[1]=x+skip (hidden, optional)

**Build in progress.** If it compiles, benchmark with --clear-cache to verify Triton launches > 0.
