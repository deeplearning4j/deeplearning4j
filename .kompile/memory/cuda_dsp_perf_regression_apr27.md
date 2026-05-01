---
name: cuda-dsp-perf-regression-apr27
description: CUDA GPU performance regression 86.7→53 tok/s, nsys-verified 22-island composite replay architecture, gap/island breakdown, failed attempts
type: project
---

# CUDA DSP Performance Regression: 86.7 → ~53 tok/s

**Last updated**: 2026-04-29
**Model**: SmolDocling-256M (30-layer decoder), RTX 4090, batch=1 seq=1 decode

## Baselines

| Configuration | tok/s | Architecture | Notes |
|---|---|---|---|
| **Pre-Triton peak** | 86.7 | Monolithic CUDA graph (cuBLAS+cuDNN) | Single `cudaGraphLaunch()` per step, commit `52ad5cf5d5` (03-19) |
| **Post-Triton nadir** | ~10 | Composite replay (unoptimized) | After Triton island refactor landed |
| **Current best** | ~53 | Composite replay (optimized) | 22 island graph replays + split gaps/step |
| **Target** | 100+ | TBD | Requires further island reduction or decode loop optimization |

## Architecture: nsys-Verified (2026-04-29)

**Per-step GPU timeline (nsys profile `/tmp/nsys_profile_20260429_051332.sqlite`):**

Each decode step = 22 island-gap cycles:
- Each cycle: H2D 8 bytes → `splitCuda` kernel (25us) → CUDA graph replay
- 9 tiny replays (~3us) — frozen/trivial islands
- 1 medium replay (~1ms)
- 12 full replays (~2.4ms each)

Then decode loop gap:
- `cudaStreamSynchronize` (waits for GPU compute)
- D2H token readback (8B)
- D2D KV scatter (3MB)
- Position/mask updates

**CRITICAL nsys insight:** CUPTI reports kernels INSIDE CUDA graph replays as
individual "regular" kernel events. The ~350 "native kernels per step" initially
seen were INSIDE the graph replays, NOT separate native executions. The only real
gap work is 22 split kernels + 22 H2D 8B copies.

**COMPOSITE_REPLAY_TIMING confirms:**
- `mergedGroups=1` — all islands in a single merged group
- `gapExec=0us` — ALL gap slots captured (mergedCaptureThroughViews=true)
- `mergedLaunch=~2.1ms` CPU-side graph launch time

## What Worked — Recovery from 10 to ~53 tok/s

### Tier 1: Major gains (>5 tok/s each)

| # | Change | Gain | Commit |
|---|---|---|---|
| 1 | Plan cache + batchedGemm + staging | 10→17.6 | `3075ec44ac` |
| 2 | Frozen fast path + composite replay | 6.8→23.8 | `3c6ed79824` |
| 3 | Triton island compile + precompile | 34.7→50.6 | `eac54da587` |
| 4 | freezeMergeSegments=true | 5.7→65.8* | `237c2fa5d3` |

### Tier 2: Moderate gains (1-5 tok/s each)

| # | Change | Commit |
|---|---|---|
| 5 | Active gap slot cache (skip 97% iterations) | `158f30a383` |
| 6 | scatter_nd_update FULLY_WRITING | `70d0b04d08` |
| 7 | Fused warp-shuffle softmax | `020d93aa26` |
| 8 | Skip frozen constants/identity in gap loop | `bc8695b7cc` |
| 9 | Eliminate redundant cudaStreamSynchronize | `cee34171d6` |
| 10 | checkIndices DSP gate (gather.cpp) | 51.44→53.05 (+3.1%) |

## What DIDN'T Work

| Approach | Result | Why |
|---|---|---|
| **Mega-graph** (monolithic CUDA graph) | 49.6% accuracy | Gap ops use pool addresses that go stale |
| **mergeViewGaps** | -5.4% | View/identity capture added more overhead than saved |
| **TILE in tritonIncludeTypes** | -6.4% (53→49.6) | Triton tile kernel slower than native splitCuda |
| **sizeAt replacing gather** | -4 tok/s | |
| **reshape_no_copy view bypass** | -29% | |
| **GQA forward4DDecode** | -31% | |
| **FuseGatedMLPPattern** | 51→48.2 | Chain stops at swish_mul boundary |
| **dspCastSinkMatmul (Pass 0.5)** | 50.04 tok/s, no change | Cast ops are tiny kernels; bottleneck is sync overhead not op count. Config flag works (visible in benchmark log) but FP16→FP32 casts before matmul are negligible cost. |
| **Graph-level DCE (Java GraphOptimizer)** | 0 ops removed | SmolDocling graph is fully connected — all 2742 ops reachable from 61 required outputs. No dead code exists in this model. DCE code is correct but this model has no dead branches to prune. |

### Details on dspCastSinkMatmul + DCE attempt (2026-04-29)

**dspCastSinkMatmul**: Added `.dspCastSinkMatmul(true)` to `BenchmarkConfig.optimal()`. This enables FusionPass Pass 0.5 in C++ (`FusionPass.cpp:358-423`) which marks FP16→FP32 cast ops as identity when ALL consumers are matmul ops (MmulHelper handles mixed precision internally via cublasSgemmEx). The flag propagated correctly (visible in benchmark config log), but the 62 Assign<float,float16> casts per step are tiny kernels (~microseconds each) — eliminating them doesn't move the needle when sync overhead is 84% of step time.

**Graph-level DCE**: Added backward reachability sweep in `GraphOptimizer.optimize()` (Java, lines 116-183). Walks from `requiredOutputs` backward through op graph, removes unreachable ops + their ARRAY output variables. Uses existing `OptimizationUtils.removeOp/removeVariable`. Result: decoder plan stays at 2742→2682 slots (the 60 reduction is from pre-existing constant folding, not DCE). Zero ops are unreachable — the model graph has no dead branches.

**Key lesson**: Op-count reduction optimizations (cast elimination, DCE) cannot bridge the 53→100 tok/s gap. The bottleneck is 22 island-gap cycles with H2D+split+graphLaunch overhead per cycle. The path to 100+ tok/s requires reducing the number of islands (eliminating split ops) or moving the decode loop control flow to GPU-side.

## Key Rules (hard-won)

1. **NEVER compile MATMUL via Triton for M=1 decode** — cuBLAS is 2.8x faster
2. **NEVER bake gap ops into CUDA graphs** — addresses go stale, accuracy regresses
3. **NEVER use monolithic CUDA graph capture** with Triton — always breaks accuracy
4. **NEVER add TILE to tritonIncludeTypes** — -6.4% regression proven 2026-04-29
5. **`p()` method does host write + syncToDevice** — never bypass
6. **toFloatVector() on CUDA views is EXTREMELY slow** — use `dup().data().asFloat()`
7. **Op-count reductions (DCE, cast-sink) don't help** — bottleneck is sync/launch overhead, not kernel count

## Remaining Gap: ~53 → 100+ tok/s

1. **Reduce 22 islands** — split ops force island boundaries; eliminating splits would merge islands
2. **GPU-side argmax** — eliminate 3MB D2H logits readback per step
3. **Reduce D2D KV scatter** — 3MB D2D per step
4. **Reduce per-island overhead** — each of 22 islands needs H2D + split + graph launch
5. **Move decode loop to GPU** — eliminate CPU-side sync entirely for steady-state decode