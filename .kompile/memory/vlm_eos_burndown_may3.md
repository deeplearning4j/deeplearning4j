---
name: vlm-eos-burndown-may3
description: "VLM EOS-on-step-2 burndown: eliminated hypotheses, remaining candidates, current state May 3 2026"
type: project
---

## VLM SmolDocling EOS-on-Second-Step Burndown (May 3 2026, updated 08:50 JST)

### CURRENT STATE
- Both SLOT_BY_SLOT and OPTIMAL produce: [216, 49229, 0] = [space, `<doctag>`, EOS]
- Native loop step 0 → produces token 0 (EOS) — WRONG
- CPU CRASHES with different bug before reaching this point

### ELIMINATED HYPOTHESES
1. ~~DSP-specific issue~~ — SLOT_BY_SLOT has same bug
2. ~~fusedGQADecodeKernel crash (error 701)~~ — fixed with thread cap 256
3. ~~Causal mask off-by-one (`i <= prefillSeqLen`)~~ — FIXED, didn't solve EOS
4. ~~Debug mode false positive~~ — Script bug fixed
5. ~~CUDA graph staging buffer stale data~~ — SLOT_BY_SLOT has NO graph capture, still fails
6. ~~Plan stale from warmup~~ — executeCount < 4, falls back to full execute(), not stale

### CRITICAL NEW FINDINGS (May 3 08:45 JST)

**Finding 1: CPU crashes with HALF×FLOAT matmul mismatch in native decode**
- Slot 430 (matmul): input [1,1,576] HALF × weight [576,192] FLOAT
- CPU `usualGemm` detects type mismatch and ABORTS
- CUDA silently handles mixed types (cuBLAS cublasGemmEx) but may produce WRONG results
- Hypothesis: CUDA produces garbage logits from silent type mismatch → argmax=0 → EOS
- **TEST: run with `--no-fp16` to disable weight pre-casting. If all FLOAT, no mismatch.**

**Finding 2: Java EOS handling is broken**
- `buildStopTokenIds()` at GenerationPipeline.java:1150 has guard:
  `if (eosTokenId >= 0 && eosTokenId < 100) { eosTokenId = -1; }`
- SmolDocling's EOS is token 0 → gets discarded → Java stopTokenIds = EMPTY
- Native C++ correctly stops on token 0, but Java reports finish=MAX_TOKENS
- Fix: change `>= 0` to `> 0` (token 0 is valid EOS for many models)

**Finding 3: Token 0 at i=0 not treated as zero-padding**
- Java line 1829: `if (tid == 0 && i > 0) break` — only skips token 0 when i > 0
- So native's single token 0 IS added to allTokens

### ROOT CAUSE HYPOTHESIS (STRONGEST)
The FP16 GraphOptimizer pre-casts weights to HALF. During native decode, activation at slot 430 is HALF (from FP16 ops) but weight [576,192] is still FLOAT (wasn't pre-cast). CUDA cuBLAS handles mixed types but may produce incorrect results for certain matrix dimensions. The garbage logits → argmax position 0 → EOS.

### NEXT ACTIONS (priority order)
1. **Run `--no-fp16` benchmark** — if this produces correct text, the FP16 mismatch IS the cause
2. Fix `buildStopTokenIds` EOS guard (reporting fix, independent of correctness)
3. If --no-fp16 works: fix the optimizer to ensure ALL weights in native decode path are pre-cast consistently
