---
name: dsp-regression-llm-ops-changes
description: Detailed llm_ops.cpp changes since 9bb2680e2b — silu, swish_mul, rope, rms_norm_linear
type: project
---

## llm_ops.cpp Changes Since 9bb2680e2b (May 2 2026)

File: `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp`

### silu op (line 312) — COMMITTED FIX
- Bug: when `output->buffer() == input->buffer()` (in-place), sigmoid(x) overwrites x before multiply
- Result: computes sigmoid(x) * sigmoid(x) instead of x * sigmoid(x)
- For large values → approaches 1.0, for negative → approaches 0.0
- After many layers → all-zero logits
- Fix: alias guard checks buffer pointer identity before choosing code path
- Gap: uses `->buffer()` not `->dataBuffer()` — could miss GPU-only aliasing where host and device buffers differ

### swish_mul op (lines 791, 798, 807) — COMMITTED FIX
- Three-branch alias guard for the three possible input/output aliasing patterns
- Same root cause as silu: in-place overwrites input before it's fully consumed

### rope op — COMMITTED FIX
- Delegates to `helpers::fusedRoPE()` instead of hardcoded float math
- Fixes FP16 precision loss in rotary position embeddings
- CPU impl: `helpers/cpu/fused_llm_ops.cpp` — full rewrite with typed dispatch, rank-3 support, precomputed invFreq table
- RISK: `fusedRoPECached` rewrite assumes cos/sin stride layout — non-contiguous tensors produce wrong embeddings

### rms_norm_linear op (line 1092-1131) — UNCOMMITTED FIX
- For rank > 2: reshapes x [B,S,K] → x2d [M,K] and output [B,S,N] → out2d [M,N]
- Old code: `reshape(order, shape, true)` — always copies. Wrote to copy, never copied back. Silent data loss.
- New code: `reshape(order, shape, false)` — zero-copy view when possible
- `directWrite` guard: checks if out2d shares buffer with output
- If not direct write: `output->assign(out2d)` copies results back
- CUDA stream sync: helper calls registerSpecialUse, assign's prepareUse handles sync

**Why:** These are the core LLM ops called on every token. Any bug here affects every layer of every model.
**How to apply:** The silu/swish_mul/rope fixes are committed and correct. The rms_norm_linear fix is uncommitted and must be kept.
