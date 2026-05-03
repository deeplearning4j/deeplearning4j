---
name: dsp-regression-commit-timeline
description: Commit timeline April 29 - May 2 2026 with risk classification for DSP accuracy regression
type: project
---

## Commit Timeline: DSP Accuracy Regression (Apr 29 - May 2 2026)

### Last Known-Good Commit
- **9bb2680e2b** — "fix: MKL SDPA prefill heap overrun + invalid batched GEMM strides"
- Fixed real bugs in MKL SDPA
- LATENT BUG: prefill bias (causal mask) still not applied

### Commits After Last-Good (chronological)

Each commit introduced performance optimizations that gated or removed safety checks.
The regression is cumulative — no single commit is "the" cause.

**Categories:**
- PERF: Performance optimization that removed a safety check
- FIX: Bug fix (positive)
- FEATURE: New functionality
- REFACTOR: Code restructuring

**Key commits and their risk:**

| Commit | Category | Risk | Description |
|--------|----------|------|-------------|
| silu/swish_mul alias guard | FIX | RESOLVED | Fixed sigmoid(x)^2 in-place bug |
| shapeFunctionOverride | PERF | MEDIUM | Skips validation at executeCount>=3 |
| validation gating step<3 | PERF | MEDIUM | All validation disabled after step 3 |
| prezero guard (in working tree) | PERF | CRITICAL | Skips prezero for frozen decode |
| BFS kMaxBfs=256 | FEATURE | CRITICAL | BFS too small for VLM models |
| backfillCachedOutputShapes guard | PERF | HIGH | Blocks shape correction after pre-pass |
| phaseShapeInferenceOnly | FEATURE | LOW | Shape pre-pass (safe — warmup clears) |
| OpenVINO before OneDNN | REFACTOR | MEDIUM | Changes CPU backend selection order |
| cascade failure demotion | PERF | MEDIUM | Silent fallback on compile failure |
| refreshStaleViewWrappers refactor | REFACTOR | MEDIUM | Changes view refresh iteration |
| validateIntegrity removal | PERF | LOW | Reduces corruption detection |
| NDArray canary debug-only | PERF | LOW | Release builds miss use-after-free |
| AttentionFusion permute absorption | FEATURE | HIGH | Wrong permute detection = wrong layout |
| DCE pass | FEATURE | HIGH | Could prune KV cache update ops |
| NormFusion stripTrivialOps fix | FIX | RESOLVED | Restricted to cast/identity |
| gather/concat DATADEP | FIX | POSITIVE | Correct needsZeroedOutput behavior |
| FP16 accumulator fixes | FIX | POSITIVE | Prevents overflow in rmsNorm |
| fusedRoPE rewrite | REFACTOR | MEDIUM | Stride assumptions for cos/sin cache |

### Working Tree (Uncommitted)
- BFS kMaxBfs=4096 — FIX for committed bug
- prezero skip — REGRESSION being introduced
- rms_norm_linear reshape — FIX for committed bug
- markExternalInputVariable — NEW FEATURE for CUDA graphs
- gpubackend markWarmupDone — FIX for state machine
- SameDiff.dup() DSP flags — FIX for committed bug
- GGMLModelImport forInference() — FEATURE
- Debug printfs in autoregressive_decode — CLEANUP needed

**Why:** Understanding the timeline helps with bisection and ensures we don't revert positive fixes while fixing regressions.
**How to apply:** When bisecting, test at 9bb2680e2b (last-good) and compare with HEAD. The regression is cumulative across multiple commits.
