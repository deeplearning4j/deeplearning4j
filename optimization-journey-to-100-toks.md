# The Road to 100 tok/s: Optimization History for SmolDocling VLM Decode

**Model**: SmolDocling VLM (ONNX, 30-layer GroupQueryAttention)  
**Hardware**: NVIDIA RTX 4090 (24 GB VRAM)  
**Backend**: DL4J/ND4J CUDA 12.9 with DSP (DynamicShapePlan) execution engine  
**Branch**: `ag_new_release_updates_2`  
**Period**: February 2026 - May 2026  
**Current best**: ~69.57 tok/s lateSteady | **Target**: 100 tok/s  
**Peak ever achieved**: 86.91 tok/s (lost to stability rework)  
**Sources**: git commit history (46,320 commits), kompile memory (115 files), test milestones (~140 records), Claude memory (50+ files)  
**Last updated**: May 29, 2026

---

## Throughput Timeline

```
 2.1 tok/s   Feb 19   Baseline — argmax on CPU via toFloatVector
11.9 tok/s   Feb 20   Fix argmax: GPU-side Nd4j.argMax                       (+467%)
  30 tok/s   Feb 20   C++ KV scatter: eliminate 60 JNI round-trips           (+152%)
  40 tok/s   Feb 21   JNI caching + direct KV reference                      (+33%)
43.5 tok/s   Feb 21   Zero-copy views (reshape/expand_dims share buffer)     (+9%)
  62 tok/s   Mar 18   SmolDocling VLM decode sprint                          (+43%)
86.9 tok/s   Mar 19   Triton attention decode optimization                   (+40%)  ← PEAK
  74 tok/s   Apr 05   After stabilization work (some regression)
 5.7 tok/s   Apr 05   Correctness checkpoint (all fixes applied)             (REGRESSION)
23.8 tok/s   Apr 24   Frozen fast path + composite replay for Triton         (RECOVERY)
  61 tok/s   Apr 29   Accumulated decode optimizations
  12 tok/s   May 06   BROKEN: performance regression                         (REGRESSION)
32.2 tok/s   May 16   Snapshot: all DSP regression tests pass                (RECOVERY)
41.8 tok/s   May 17   Gap capture relaxation + helper bypass                 (+30%)
49.8 tok/s   May 18   Batched GEMM mixed-type cast                           (+19%)
58.6 tok/s   May 18   Hoist cuBLAS stream+workspace setup (N6)               (+18%)
69.2 tok/s   May 19   Skip helper dispatch for all gap-fast ops              (+18%)
69.6 tok/s   May 19   cuBLAS Lt for gap matmul (N36) — noise-level           (+0.5%)
66.7 tok/s   May 28   N107 skip cuBLAS setup in MmulHelper                   (noise)  ← CURRENT
```

> **Note on "current best"**: The 69.57 tok/s figure comes from a May 19 benchmark run. The May 28 run at 66.7 tok/s is within run-to-run noise (~4%). Both represent the same optimization state.

Two major regressions (5.7 tok/s and 12 tok/s) were caused by correctness-fix sweeps that inadvertently disabled fast paths. Recovery required re-enabling `freezeMergeSegments`, fixing composite replay, and rebuilding the frozen fast path.

---

## Phase 1: Baseline Recovery (Feb 19-21) — 2.1 to 43.5 tok/s

These were pure bottleneck removals. Each fix removed an obvious inefficiency.

| Commit | Optimization | Gain | Why it worked |
|--------|-------------|------|---------------|
| `9e2c4b83` | GPU-side argmax (replace `toFloatVector` CPU loop) | 2.1 → 11.9 | The argmax was being computed by copying the entire logit vector to CPU, calling `toFloatVector()` (extremely slow on CUDA views), iterating in Java. GPU argmax is a single kernel. |
| `21dc91b1` | Merge 1402 segments into 1 mega-graph | ~50ms/step | Segment overhead was O(N) per step. Merging eliminates per-segment scheduling. |
| `f09a1562` | Skip unchanged capture buffer copies | 47→28ms | CUDA graph replay was copying ALL capture buffers every step, even those that hadn't changed. |
| `603c1769` | Remove 255 redundant casts + outputDirect() fast path | — | Graph contained FP32→FP32 identity casts from ONNX import. |
| `93cc2de6` | C++ KV scatter (eliminate 60 JNI round-trips) | 21→30 | KV cache update was 60 individual JNI calls per step (one per layer per K/V). Single C++ kernel replaces all. |
| `e9beb97f` | JNI caching + direct KV reference + needsZeroedOutput | 30→40 | JNI handle lookup on every call. `needsZeroedOutput` eliminated redundant memsets. |
| `fa237197` | Zero-copy views (reshape/expand_dims/squeeze share buffer) | 40→43.5 | View ops were allocating new buffers and copying. Now they share the input buffer pointer. |
| `5be05f46` | Frozen constant detection (skip 250 value-independent slots) | — | 250 of 8522 graph nodes compute the same output every step. Skip them entirely. |
| `babe6f9f` | Skip buffer zeroing for matmul ops | — | cuBLAS guarantees full output writes. Pre-zeroing was wasted work. |

---

## Phase 2: Triton Introduction (Feb 28 - Mar 19) — 43.5 to 86.91 tok/s

The Triton JIT compiler enabled fusing elementwise op chains into single GPU kernels, dramatically reducing kernel launch overhead.

| Commit | Optimization | Gain | Why it worked |
|--------|-------------|------|---------------|
| `51e72661` | Triton selective fusion baseline (elementwise only) | — | Fuses chains of elementwise ops (add, mul, relu, etc.) into single kernels. 89% token diversity maintained. |
| `bb991ddb` | Fix consolidated arg table copy during graph replay | +70% | Arg table was being copied incorrectly, causing redundant recompilation. |
| `207fdea9` | SmolDocling VLM decode optimization | 34→62 | 1.5-2x speedup from decode-specific tuning. |
| `52ad5cf5` | **Triton attention decode optimization** | **86.91** | **Peak throughput.** Triton compiled attention path with optimized memory access patterns. |

**This was the high-water mark.** Subsequent correctness work caused regressions that were never fully recovered.

---

## Phase 3: Stabilization Regressions (Apr 5 - Apr 24) — 86.91 down to 5.7, back to 23.8

Correctness fixes for CUDA graph replay, frozen constants, and DSP lifecycle repeatedly broke performance.

| Event | tok/s | What happened |
|-------|-------|---------------|
| `freezeMergeSegments` accidentally OFF | 65.8→5.7 | 12x regression. Without merged segments, 1402 separate island graphs had to be launched individually. |
| `writeSpecial` actuality poisoning | 20% accuracy | `writeSpecial()` on external inputs bumped device actuality, causing `syncToPrimary()` to copy stale zeros over valid host data. All CUDA graph modes produced garbage. |
| `computeShapeKey` value-mixing | 6.83 | Value contents of ALL inputs (not just value-dependent ones) were hashed into the shape key. Position indices advancing each step → new hash → Triton recompile every step. |
| Frozen fast path + composite replay re-enabled | 6.8→23.8 | Recovery commit. The frozen fast path had been disabled during correctness fixes. |

---

## Phase 4: Micro-optimizations (Apr 24 - Apr 29) — 23.8 to ~61 tok/s

Dozens of small wins in the DSP steady-state hot path.

| Commit | Optimization | Category |
|--------|-------------|----------|
| `cee3417` | Eliminate redundant `cudaStreamSynchronize` in decode token read | Sync elimination |
| `6f88736` | Remove blocking sync + debug memory queries from hot path | Sync elimination |
| `d5f3209` | Remove `cudaDeviceSynchronize` + per-step `trimPool` | Sync elimination |
| `b4cede0` | Deduplicate cross-stream sync | Sync elimination |
| `158f30a` | Active gap slot cache (skip 97% of slot iterations) | Gap overhead |
| `bc8695b` | Skip frozen constants/identity ops in gap loop | Gap overhead |
| `70d0b04` | Mark `scatter_nd_update` as FULLY_WRITING (skip prezero) | Prezero skip |
| `020d93a` | Fused warp-shuffle softmax kernel for attention decode | Fused kernels |
| `11005b4` | Fused `rms_norm_linear` kernel | Fused kernels |
| `31e5078` | Fused `skip_rms_norm` (eliminate 60 add kernels/step) | Fused kernels |
| `8b9946c` | Eliminate decomposed `inv_rms` chain from ONNX import | Graph reduction |
| `4c5d2ed` | Eliminate nullify + assign copy in MHA decode path | Copy elimination |
| `529e26f` | silu/swish_mul temp elimination + slot-exec simplification | Memory/overhead |
| `2b60fca` | Skip helper dispatch + timing memset in frozen steady state | Overhead |
| `3717360` | Gate REQUIRE_TRUE validation to first 3 steps only | Overhead |
| `43ec64a` | O(3) variable-only `syncExternalInputs` in steady state | Sync reduction |
| `0c86e4c` | Stride-aware GQA kernel + mixed-type gamma in rms_norm | Kernel optimization |
| `432b78b` | Pre-allocate KV scatter device buffers | Memory |
| `78cd9d9` | Bypass `launchAsync` overhead in composite replay | Overhead |
| `d4f8175` | Skip error message heap alloc/free when no error | Overhead |
| `8d3abee` | D2D token store, pre-sync mask updates, pinned D2H | Decode loop |
| `dc97577` | Register SkipSimplifiedLayerNormalization + gate diagnostics | Overhead |

---

## Phase 5: Second Regression and Recovery (May 1 - May 17) — 61 to 12 to 41.8

| Event | tok/s | Root cause |
|-------|-------|------------|
| BROKEN regression | 61→12 | Triton kernels not launching (0 Triton launches), cuBLAS fallback active |
| Correctness baseline snapshot | 12 | All DSP regression tests pass, but performance paths disabled |
| Recovery with Triton IR builder fix | 32.2 | `seenInputs` pollution blocking frozen constant args in Triton compilation |
| Gap capture relaxation + helper bypass | 38.9→41.8 | Re-enabled gap capture with relaxed safety checks |

---

## Phase 6: Batched GEMM Breakthrough (May 17-18) — 41.8 to 58.6 tok/s

The biggest single-optimization win of this phase: properly batching the 60 bgemm ops.

| Commit | Optimization | Gain | Why it worked |
|--------|-------------|------|---------------|
| `73490ad8` | **Batched GEMM mixed-type cast** | 41→49.8 (+21%) | 60 batched GEMM ops had mixed FP16/FP32 types, forcing serial cuBLAS calls. Adding a type-cast layer enabled true batch execution. |
| `160df42e` | Persistent bgemm cast scratch | 49.8→50.2 | Eliminate 240 `cudaMalloc`/`cudaFree` calls per step for cast temporaries. |
| `80082cde` | **N6: Hoist cuBLAS stream+workspace setup** | 50.2→58.6 (+17%) | cuBLAS handle setup (stream assignment + workspace allocation) was done inside the bgemm inner loop. Hoisting it before the loop eliminated redundant driver calls. |

---

## Phase 7: Final Micro-optimizations (May 19 - present) — 58.6 to 69.6 tok/s

| Commit | Optimization | Gain |
|--------|-------------|------|
| `a49ede70` | Skip helper dispatch for ALL gap-fast ops | 58.6→69.2 (+18%) |
| `d54a0077` | N36: cuBLAS Lt for gap matmul | 69.2→69.6 (noise) |
| `9af2ddc4` | N107: Skip cuBLAS setup in MmulHelper during DSP gap | 66.7 (noise/regression) |

---

## Optimizations That FAILED (Dead Ends)

### Catastrophic Failures (>20% regression)

| ID | What | Result | Why it failed |
|----|------|--------|---------------|
| N16 | Branch misprediction in dirty-mark loop | +360% regression | CPU branch prediction disrupted in tight loop |
| N21 | Gap matmul capture (`blockExtWorkspace=false`) | 13.6 tok/s (-68%) | `mergedGroups=0` — no replay at all |
| N29b | Restore `outputShapeDependsOnInputValues` in `isSlotCapturable` | 8.37 tok/s (-88%) | Pool address reuse during capture (root cause fixed — see Reassessment) |
| N34 | Separate workspace + `gapCaptureBlock=false` | 21.4 tok/s (-69%) | cuBLAS args baked into CUDA graph, stale on replay |
| N37 | Gap cuBLAS workspace + `gapCaptureBlock=false` (variant) | 22.0 tok/s (-68%) | `compositeReplayReady=0` throughout — never enters replay |
| N63 | Guard `cublasSetStream` in MmulHelper | 35.2 tok/s (-48%) | MmulHelper gets stream from array context, not gap stream (later shipped correctly as N107) |
| N94 | `slotIsViewProducer_` pre-classification | 64.5 tok/s (-7%) | `refreshStaleViewWrappersInSegment` overhead (code path restructured — see Reassessment) |
| N96 | FlashAttention `effectiveSeqKVPtr` (attempt 1) | 43→37 tok/s (-14%) | Register pressure on 30-instance attention kernel |
| N96 | FlashAttention `effectiveSeqKVPtr` (attempt 2) | 42.8 tok/s (-38%) | `cachePosition=null` — ONNX model lacks `seqlens_k` |
| N108 | Add MATMUL to Triton `includeTypes` | 60.4 tok/s (-10%) | Triton slower than cuBLAS for M=1 decode matmuls |
| N110 | Per-gap-unit CUDA graph capture | CUDA error 8 | `cudaMemcpyAsync` on legacy stream creates forbidden cross-stream deps |
| P10 | Value-dependent freeze retention | 26.8 tok/s (-47%) | Stale shape values break reshape/broadcast |
| FuseGatedMLP | Fuse two matmuls + activation into `fused_gemm_swiglu` | 51→48.2 (-5%) | MATMUL category breaks Triton islands; C++ impl is slower than batched path |
| Monolithic capture | Single mega CUDA graph for entire model | 2.2 tok/s | Gap args baked in; PEDANTIC_MATH forced; accuracy 50% |

### Neutral Results (within noise, no benefit)

| ID | What | Result | Why no gain |
|----|------|--------|-------------|
| N20 | Cross-segment orphan re-grouping | Zero benefit | All orphans have transitive dependencies — none re-groupable |
| N24 | GQA 256→512 threads | 69.4 (neutral) | Register pressure is not the bottleneck at 256 threads |
| N29 | `freezeMergeSegments=false` | 68.3 (-1.3%) | `buildSegments()` itself is monolithic — flag doesn't help |
| N30 | Triton warps/stages re-tuning | 67.9-70.0 (noise) | Triton kernel time is only 2.2ms of 14.5ms step |
| N36 | cuBLAS Lt for gap matmul | 69.6 (+1.8%) | Only logits projection benefits; 1/91 matmuls |
| N38/P5 | SmolDocling seqlens_k injection | 67.7 (neutral) | ONNX model lacks `seqlens_k` input entirely |
| N48+N62 | deviceMutex bypass in bgemm | 67.4 (-3.2%) | 151 `tl_dspReplayActive` checks/step cost > mutex savings |
| N92 | Cast kernel optimization | 66.6 (noise) | Cast cost: ~0.12ms total (0.8% of step) |
| N101 | Heap vector elimination in onnx_mha | 67.4 (-1.1%) | Heap alloc overhead negligible vs GPU execution |
| N107 | Skip cuBLAS setup in MmulHelper | 66.7 (noise) | Setup overhead already minimal |
| N109 | cuBLAS param cache | 67.0 (+0.5%) | CPU shape analysis: 32µs total, dominated by GPU kernel time |
| R2 | Single-stream unification | 50.5 (neutral) | No meaningful overlap between gap and island execution |
| Trial 9 | mergeViewGaps | -5.4% | CUDA graph launch overhead > gap dispatch savings |
| gapTensorCores | Force TensorCore math type for gaps | ~-3 tok/s | Switching overhead > benefit (distinct from P1 TF32: this forced `CUBLAS_TENSOR_OP_MATH` execution type, not TF32 accumulation mode) |

### Fundamental Dead Ends (structurally impossible)

| ID | What | Why impossible |
|----|------|----------------|
| N26 | Cross-layer bgemm super-batching | 91 matmuls in serial dependency chains (residual stream). `inputRejected=3945`. |
| N27 | reshape_no_copy elimination | 37 inputs are non-C-contiguous (post-permute). True device copies required. |
| N29c | Multi-segment composite replay | Composite replay already works within single segment — splitting was a red herring. (Note: N29b multi-segment *capture* is rehabilitated — see Reassessment; N29c's *replay* splitting is distinct and remains dead.) |
| N49 | Cross-layer bgemm merging | Serial residual stream dependencies prevent merging |
| N71 | H2D copy elimination in steady state | Already skipped via `canSkipPtrRefresh` |
| N72 | `cublasGemmStridedBatchedEx` for unbatched matmuls | Requires contiguous memory; weights are separate `cudaMalloc` allocations |
| N87 | Move vector declarations inside branch | `writeList`/`readList` used after cuBLAS dispatch |
| N95 | Increase `maxCapturableGapSlots` | Already at maximum (32) |
| N100 | `unordered_map` lazy init | Doesn't heap-allocate until first insert anyway |
| N105 | Fuse `<<<1,1>>>` kernels | ~0.2% savings (10-30µs/step), immeasurable |
| N106 | Fuse permute ops via Triton | 17 non-view permutes are actual data copies; Triton produces zeros |
| N43-N59 | Various StaticKvCacheDecodeLoop opts | Wrong code path. Production uses `GenerationPipeline → autoregressive_decode` |
| N81 | `fused_rms_norm_swiglu` | No optimizer emits it for SmolDocling; would need full Java+C+++Triton pipeline |
| N-gram | Speculative decoding | Doesn't work for SmolDocling architecture |

---

## Critical Bugs That Blocked Performance

These bugs had to be fixed before optimizations could take effect. Several caused multi-week delays.

### Memory Leaks

| Bug | Impact | Fix |
|-----|--------|-----|
| `SameDiff.close()` constant buffer leak | 14 GB permanently leaked | Explicit CONSTANT/VARIABLE cleanup in `close()` |
| `destroySession()` missing `releaseGpuIntermediates()` | ~8 GB/config cycle GPU leak | Call `releaseGpuIntermediates()` in `destroySession()` |
| `tl_castCache` unbounded growth | 250 MB/step GPU leak | Bounded eviction policy |
| KV output not closed after scatter | 274 MB/step leak | Close present KV outputs each step |
| CUDA graph workspace OOM | Capture fails | Adaptive sizing: `min(512MB, free-256MB)` |

### Accuracy Bugs (forced slower paths)

| Bug | Symptom | Fix |
|-----|---------|-----|
| `writeSpecial` actuality poisoning | 20% accuracy, all CUDA graph modes | Remove `writeSpecial` from capture path |
| Merged capture gap staleness | Repeating token 49218 | Reject `aliasesInput()`/`frozenConstantSlot()` gaps |
| KV cache H2D zeroing | EOS at step 3 | Gate `syncToSpecial` on `isPrimaryActual()` |
| `fused_rope` FP16 cos/sin NaN | NaN from step 1 | Cast cos/sin to FLOAT32 before kernel |
| `rms_norm_linear` type mismatch | CUDA launch error | Cast weight to match input dtype |
| `sd_softplus` overflow | 11,335 Infinity values in prefill | `max(0,x)+log(1+exp(-|x|))` |
| FP16 autocast in `mmulMxM` | Wrong FP32 results | Remove automatic FP16 casting |
| `computeShapeKey` value-mixing | Recompile every step → 6.83 tok/s | Gate value-hashing on `outputShapeDependsOnInputValues` |
| `setCloseable(false)` → `setConstant(true)` | Stale host data after GPU updates | Remove `setConstant(true)` propagation |
| `cudaStreamQuery` during capture | Error 900 → stream poisoned → error 901 | Check `cudaStreamIsCapturing` first |
| Fusion dangling tail | Silent execution skip | Require both head and tail fields |
| Ungated debug printf | 25x slowdown | Remove printf from hot path |

---

## The Remaining Gap: 69.6 → 100 tok/s

### Where time is spent (14.5ms/step = 69 tok/s)

```
gapExec:      11.5ms  (79%)  ← 151 serial cuBLAS kernel launches
mergedLaunch:  2.2ms  (15%)  ← 181 Triton island CUDA graphs
overhead:      0.8ms  ( 6%)  ← scheduling, sync, bookkeeping
```

The bottleneck is **gap execution**: 60 batched GEMM + 91 unbatched matmul cuBLAS calls that run serially because they are in dependency chains within each transformer layer.

> **Important reframing**: The 10%/90% discovery (see Cross-Cutting Observations) shows that much of the 11.5ms `gapExec` time is CUDA graph scheduling overhead for 2742 nodes, not pure GPU compute. Only ~1.66ms/token is actual GPU kernel execution. This changes the optimization landscape — see Phase 5 Scan.

### Why N-series micro-optimizations are exhausted

Every N-series optimization (N1 through N110) has been tried. The CPU dispatch overhead is only ~1.3ms. The remaining 11.5ms is dominated by GPU kernel time for 151 cuBLAS calls and CUDA graph scheduling overhead for 2742 nodes.

However, the Phase 5 scan (see below) and the Reassessment of failed optimizations revealed that **scheduling overhead is a larger fraction than previously understood**, and several previously "dead" optimizations have been rehabilitated by infrastructure fixes that landed after their original failure. The three architectural paths identified below remain viable, but they are no longer the *only* paths — see the Priority Matrix in Phase 5 for the combined approach.

### Three architectural paths (reassessed)

| Approach | ID | Description | Current Status |
|----------|----|-------------|----------------|
| cuBLAS arg-table refresh | N40 | Capture all 91 unbatched matmuls into CUDA graphs with pointer refresh at replay. | **More feasible than assessed** — blocker was cast cache cold-start, not driver API. See Reassessment. |
| CUTLASS Grouped GEMM | N78 | Replace 91 serial cuBLAS calls with a single CUTLASS grouped GEMM kernel. | Unchanged — HIGH effort, but N40 may make it unnecessary. |
| Multi-segment composite replay | N29b/N97 | Split at value-dependent boundaries for more capturable sub-segments. | **ROOT CAUSE FIXED** — capture workspace solves pool address reuse. See Reassessment. |

### What the peak (86.91 tok/s) had that was lost

The peak was achieved with Triton attention decode optimization (`52ad5cf5`, Mar 19). This architecture had multi-segment composite replay with different array lifecycle management that kept arrays alive across segment boundaries. Subsequent correctness-fixing sweeps (writeSpecial poisoning, computeShapeKey value-mixing, KV H2D zeroing) disabled or broke these fast paths. Recovery work rebuilt composite replay within a single segment but never restored the multi-segment architecture.

The current OPTIMAL config is also missing `tritonFusionMinScore(4.0f)` that the peak had (see P6 in Phase 5 scan).

---

## Key Lessons

1. **Correctness and performance are deeply entangled.** Every major accuracy bug fix (writeSpecial, computeShapeKey, KV H2D zeroing) broke a performance fast path. Two separate "correctness sweeps" caused 12x and 5x regressions.

2. **cuBLAS in CUDA graphs is the unsolved problem.** Every attempt to capture cuBLAS ops into CUDA graphs (N21, N34, N37, N110) failed because cuBLAS bakes buffer addresses at capture time with no replay refresh. Triton has this solved via `refreshArgTablesForReplay`.

3. **M=1 decode matmuls resist optimization.** Triton is slower than cuBLAS for single-row matrix multiplication (N108). Batching across layers is impossible due to serial dependencies (N26, N49). The 91 unbatched matmuls are fundamentally serial.

4. **Register pressure on hot kernels is a trap.** Any parameter added to `fusedGQADecodeKernel` (runs 30x/step) causes measurable regression (N96, FlashAttention scaffolding). Optimizations targeting this kernel must be zero-cost at the register level.

5. **The gap between 70 and 100 tok/s requires a mix of scheduling reduction and architectural change.** All N-series micro-optimizations (N1-N110) are exhausted, but the Phase 5 scan found ~2-4ms/step of CUDA graph scheduling overhead in zero-compute nodes (concat, reshape_no_copy) that can be eliminated without architectural change. Combined with rehabilitated optimizations (N29b, N40) and TF32 enablement, the 100 tok/s target is reachable.

---

## Common Failure Causes: A Taxonomy

Analysis of 50+ bug fix memory records, 140+ test milestones, 115 kompile memory files, and the full commit history reveals that failures cluster into **7 recurring root cause categories**. The same patterns caused bugs repeatedly across different subsystems and different months of development.

---

### Category 1: CUDA Graph Capture Bakes Stale State (12 instances)

**The pattern**: CUDA graph capture records a snapshot of GPU state (buffer addresses, kernel arguments, stream operations) at capture time. At replay time, any state that has changed since capture produces wrong results or crashes. This is the single most frequent failure cause.

**Instances**:
| Bug | What was baked stale | Symptom |
|-----|---------------------|---------|
| Merged capture gap staleness | cuBLAS input/output pointers for gap matmuls | Repeating token 49218 |
| N21/N34/N37 `blockExtWorkspace=false` | cuBLAS workspace + arg pointers | 13-22 tok/s catastrophic regression |
| N110 per-gap-unit capture | `cudaMemcpyAsync` host addresses via `syncToPrimary` | CUDA error 8 (cross-stream dependency) |
| `writeSpecial` actuality poisoning | Device actuality flags on external inputs | 20% accuracy, all CUDA graph modes |
| Monolithic mega-graph | ALL Triton gap op addresses + PEDANTIC_MATH | 2.2 tok/s, 50% accuracy |
| `platformCleanupSegmentForRebuild` | Composite replay handles (only monolithic cleared) | FORCE_RECAPTURE replayed stale graphs |
| Triton H2D sync bug | Gap ops skipped during merged capture | Repeating token 269 |
| `capturedInputAddrKey` stale after staging | Address key not updated after arg refresh | Unnecessary recapture every step |
| Pinned host copy in graph capture | H2D memcpy nodes bake host addresses from capture time | Stale host data on replay |
| Buffer overlap between slots | Slot output buffers overlap in captured segment | Wrong results |
| StaticKvCacheDecodeLoop `.assign()` | Fixed embedding buffer addresses | Wrong output after reassignment |
| Segment 442 fingerprint mismatch | DataBuffer address drift at step 2 | Graph replay blocked |

**Root cause**: CUDA graph capture is fundamentally a snapshot mechanism. Any pointer, flag, or memory content that changes between capture and replay will be wrong. Triton solved this with `refreshArgTablesForReplay`; cuBLAS has no equivalent.

**Rule**: Never include ops in CUDA graph capture unless you have verified that ALL their arguments (input pointers, output pointers, workspace pointers, scalar parameters) are either stable across replays or have a refresh mechanism.

---

### Category 2: Actuality/Sync Flag Confusion (8 instances)

**The pattern**: The NDArray system tracks whether host (`isPrimaryActual`) or device (`isSpecialActual`) memory is current. Operations that set these flags incorrectly cause stale data to be read, valid data to be overwritten, or sync operations to be skipped.

**Instances**:
| Bug | Flag confusion | Symptom |
|-----|---------------|---------|
| KV cache H2D zeroing | `syncToSpecial(forceSync=true)` copies stale host zeros over valid device KV | EOS at step 3 |
| `writeSpecial` poisoning | Bumps `_writeSpecial > _writePrimary` → `isPrimaryActual()=false` | `syncToPrimary` copies stale zeros |
| `setCloseable(false)` → `setConstant(true)` | `isConstant=true` blocks D2H in `syncToPrimary` | Stale host data after GPU training |
| `cudaStreamQuery` during capture | Returns error 900, poisons capture stream | Error 901 on all subsequent ops |
| `specialBuffer()` accessor poison | Triggers `syncToDevice` `cudaMemcpyAsync` on wrong stream | Capture stream poisoned |
| `markOrderedRangeDeviceCurrent` | Sets `isSpecialActual=true` even after `writePrimary()` | Java reads stale device data |
| VLM EOS-on-step-2 (KV ext inputs) | KV cache ext inputs not marked variable → staging never refreshed | argmax returns 0 |
| TRITON_SKIP frozen fast path | `shapesFrozen_=true` + reused cached context for changing inputs | Repeating tokens (20% match) |

**Root cause**: The dual-buffer (host/device) architecture with lazy sync means any operation that touches actuality flags has system-wide consequences. The flags are essentially a distributed consistency protocol, and any violation propagates through sync boundaries.

**Rule**: Before calling `syncToSpecial`, `syncToPrimary`, `writeSpecial`, `writePrimary`, or `setConstant`, verify: (1) which buffer is authoritative, (2) whether you are inside CUDA graph capture, (3) whether the array is a KV cache buffer (device-authoritative).

---

### Category 3: Type System Mismatches (7 instances)

**The pattern**: Mixed-precision execution paths (FP16/FP32/INT64) silently produce wrong results when types are not properly matched, cast, or dispatched.

**Instances**:
| Bug | Type mismatch | Symptom |
|-----|--------------|---------|
| FP16 autocast in `mmulMxM` | Auto-cast ALL FP32 inputs to HALF on SM >= 6.0 | Wrong FP32 results |
| `rms_norm_linear` type mismatch | Optimizer strips casts, input=HALF weight=FLOAT32 | `cudaErrorInvalidConfiguration` (error 9) |
| CUTLASS stride mismatch | RowMajor kernel on column-major permuted view | ~7x output attenuation |
| `fused_rope` FP16 cos/sin NaN | `reinterpret_cast<T*>` reads FP16 as FP32 | NaN from step 1 |
| `gated_delta_rule` FP16 state | Recurrent state accumulated in FP16 per timestep | Exponential quantization error |
| `sd_softplus` overflow | `log(1+exp(x))` overflows for x>88 in FP32 | 11,335 Infinity values |
| Batched GEMM mixed-type rejection | Mixed FP16/FP32 inputs forced serial cuBLAS calls | 41 tok/s (21% slower than batched) |

**Root cause**: The optimizer (FuseRMSNormLinearPattern, fp32Mmul cast stripping), ONNX import, and CUDA kernel dispatch all make assumptions about input types that can be violated when the graph is transformed. The `reinterpret_cast<T*>` pattern is especially dangerous — it silently reads the wrong number of bytes.

**Rule**: Every fused kernel must validate input types at dispatch time and cast if needed. Never use `reinterpret_cast<T*>` on buffers whose dtype might differ from the template parameter. Use float32 working state for any recurrent computation.

---

### Category 4: Correctness Fix Breaks Performance (5 major episodes)

**The pattern**: Fixing an accuracy/correctness bug inadvertently disables a performance fast path, causing a massive throughput regression. This has happened 5 times, each requiring weeks of recovery.

**Episodes**:
| Date | Correctness fix | Perf impact | Recovery |
|------|----------------|-------------|----------|
| Apr 5 | `writeSpecial` removal + frozen constant fixes | 86.9→5.7 tok/s | 3 weeks to rebuild composite replay |
| Apr 5 | `freezeMergeSegments` accidentally disabled | 65.8→5.7 tok/s | Re-enabled flag |
| May 1 | Multi-consumer in-place protection | 61.9→12 tok/s | 2 weeks to fix Triton launch path |
| May 6 | `cudaGetLastError()` unconditionally on segment bind | 61→12 tok/s | Identified and removed |
| May 17 | FlashAttention `effectiveSeqKVPtr` scaffolding | 43→37 tok/s | Reverted |

**Root cause**: Performance fast paths (frozen fast path, composite replay, merged segments, Triton island compilation) are fragile — they depend on specific state invariants that correctness fixes often violate. There is no automated guard that detects when a performance path has been silently disabled.

**Rule**: After EVERY correctness fix, run the benchmark (`run-benchmark.sh --tokens 250`) before committing. If throughput drops >5%, investigate whether a fast path was disabled before accepting the fix.

---

### Category 5: Memory Lifecycle Violations (8 instances)

**The pattern**: Arrays, buffers, or plans are freed while still referenced, or never freed at all. The dual lifecycle (Java GC + C++ manual management) creates many opportunities for dangling pointers and leaks.

**Instances**:
| Bug | Lifecycle violation | Symptom |
|-----|-------------------|---------|
| `SameDiff.close()` constant leak | CONSTANT arrays never freed by DeallocatorService | 14 GB permanently allocated |
| `destroySession()` GPU intermediates | `releaseGpuIntermediates()` never called | ~8 GB/config leak |
| `tl_castCache` unbounded | Thread-local cast cache grows without eviction | 250 MB/step GPU leak |
| KV output not closed | Scatter output arrays not closed after each step | 274 MB/step leak |
| Plan cache LRU evicting live plans | Java holds handle, C++ evicts plan → double free | SIGSEGV |
| View deletion during capture | `slotArrayCache_` entries deleted while saved in warmup outputs | Use-after-free |
| SDZ dummy array sharing | 60 constants share singleton → all become dead after close | 0 replays, 3 tok/s |
| `DSP_SLOT_CACHE_GROWTH_FACTOR=2.0` | `data().length() > length()` → `closeable()=false` → never freed | 30 MB/step leak |

**Root cause**: The Java side (GC, DeallocatorService phantom refs) and C++ side (raw pointers, manual delete) have different assumptions about when memory can be freed. `isConstant`, `closeable`, and `DeallocatorService` interact in non-obvious ways. Arrays can be simultaneously referenced by Java SameDiff variables, C++ `outputSlots_`, CUDA graph captured nodes, and plan cache entries.

**Rule**: Every allocation must have exactly one owner responsible for freeing it. Verify that `isConstant`, `closeable`, and `DeallocatorService` agree on ownership. After any lifecycle change, run the full DSP test suite to catch leaks and use-after-free.

---

### Category 6: Overhead Disguised as Optimization (10 instances)

**The pattern**: An optimization that reduces work in one area introduces overhead in another area that exceeds the savings. Often the overhead is in unexpected places (branch prediction, driver round-trips, stream management).

**Instances**:
| Optimization | Hidden overhead | Net result |
|-------------|----------------|-----------|
| N48+N62 deviceMutex bypass | 151 `tl_dspReplayActive` branch checks/step | -3.2% regression |
| N63 guard `cublasSetStream` | cuBLAS launches on wrong stream (context vs gap stream) | -48% regression |
| N94 `slotIsViewProducer_` pre-classify | `refreshStaleViewWrappersInSegment` for non-contiguous views | -7.3% regression |
| N16 dirty-mark loop | Branch misprediction in tight loop | +360% regression |
| N24 GQA 512 threads | Register spilling at higher thread count | -49% at 512 threads |
| mergeViewGaps (Trial 9) | CUDA graph launch overhead > gap dispatch savings | -5.4% |
| gapTensorCores | Switching overhead > compute benefit | ~-3 tok/s |
| Non-capturable → CPU_GRAPH | 616 fallbacks, 306 "cannot fuse" rejections | +30% slower step 0 |
| N108 MATMUL in Triton | Triton compile + arg refresh > cuBLAS dispatch | -9.5% regression |
| FuseGatedMLP | MATMUL category breaks Triton islands; C++ impl slower | -5.5% regression |

**Root cause**: GPU performance is dominated by kernel execution time and driver overhead, not by the operations the optimization targets. At 14.5ms/step with 79% in GPU kernels, CPU-side overhead reductions of <0.1ms are below the noise floor, and any added branching or driver calls easily exceeds the savings.

**Rule**: Before implementing an optimization, profile to verify that the targeted overhead is >1% of step time. After implementing, benchmark to verify the net effect is positive. Measure, don't assume.

---

### Category 7: Wrong Code Path Assumptions (5 instances)

**The pattern**: Optimizations target the wrong code path, wrong execution mode, or make assumptions about model structure that don't hold.

**Instances**:
| Assumption | Reality | Waste |
|-----------|---------|-------|
| N43-N59 targeted `StaticKvCacheDecodeLoop` | Production uses `GenerationPipeline → autoregressive_decode` | 17 dead-end optimization IDs |
| N38/P5/N96 assumed `seqlens_k` available | SmolDocling ONNX lacks `seqlens_k` (input 5) | 3 failed attempts at in-place KV |
| SLOT_BY_SLOT = ground truth for VLM | SLOT_BY_SLOT is the broken mode; TRITON_NO_GC is correct | Weeks debugging wrong baseline |
| N72 assumed contiguous weight memory | Weights are separate `cudaMalloc` allocations | Cannot use strided batched GEMM |
| CPU causal mask assumed MKL reads input[8] | MKL SDPA PLATFORM_IMPL ignores input[8] entirely | Causal mask fix had no effect |

**Root cause**: The system has multiple execution paths (slot-by-slot, composite replay, frozen fast path, Triton, CUDA graphs, native decode loop, static KV cache loop) and multiple model architectures. Assumptions about which path is active or which model inputs are available are frequently wrong.

**Rule**: Before optimizing a code path, verify it is actually the active path for the target workload. Trace execution from entry point to the specific code being optimized. Check model inputs at runtime, not by reading ONNX specs.

---

## Cross-Cutting Observations

### The Sawtooth Pattern

The throughput timeline is not a steady climb — it's a sawtooth:

```
86.9 → 5.7 → 74 → 23.8 → 61 → 12 → 32 → 42 → 59 → 69.6
```

Every peak is followed by a regression caused by correctness fixes. Recovery takes 1-3 weeks each time because the fast paths are fragile and interconnected. The system has spent roughly **equal time optimizing and recovering from regressions**.

### Bug Fix Multiplier Effect

Many bugs were discovered (and had to be fixed) only because an optimization exposed them:
- Batched GEMM exposed the mixed-type cast bug (FP16/FP32 rejection)
- Triton island compilation exposed the `seenInputs` pollution bug
- Frozen fast path exposed the KV H2D zeroing bug
- Composite replay exposed the `platformCleanupSegmentForRebuild` stale handle bug
- CUDA graph capture exposed `specialBuffer()` accessor poisoning

Each optimization attempt averaged **1.5 blocking bugs that had to be fixed first**. The optimization work is as much a bug-finding exercise as a performance exercise.

### The 79% Wall — Reframed

The original analysis attributed 79% of step time (11.5ms of 14.5ms) to GPU kernel execution for 151 cuBLAS matmul calls. The 10%/90% discovery (see above) reframes this: much of the 11.5ms is actually **scheduling overhead** for 2742 CUDA graph nodes, not pure GPU compute. Only ~1.66ms/token is actual GPU compute.

This reframing opens new attack vectors:
1. Reduce CUDA graph node count (852 zero-compute nodes from concat/reshape — P3, P4)
2. Enable TF32 for gap matmuls to speed the actual GPU compute portion (P1)
3. Capture more ops into CUDA graphs to amortize scheduling (N29b, N40)
4. Reduce cuBLAS call count (grouped GEMM, batch matmul capture)

The first 95% of the journey (2.1→69.6) was dominated by overhead removal. The remaining 30% gap is a mix of scheduling overhead and GPU kernel time — not purely GPU-bound as originally assessed.

---

## Kompile Test Milestones

The kompile test milestone system tracked 140+ test runs. Key milestones:

| Date | Milestone | DSP Tests | Throughput |
|------|-----------|-----------|------------|
| Apr 5 | Checkpoint: all fixes applied | — | 5.7 tok/s |
| Apr 20 | Plan cache + batched GEMM | — | 17.6 tok/s |
| May 3 | Qwen CUDA correctness (token 271 = Paris) | — | — |
| May 8 | Correctness baseline | — | 12 tok/s |
| May 16 | All DSP regression tests pass | 755/755 | 32.2 tok/s |
| May 19 | N35 milestone | 1500+ | 68.36 tok/s |
| May 22 | CPU test suite baseline | 1333 pass | — |
| May 27 | DSP core batch clean | 1590/0/0/0 | — |
| May 28 | Stabilize DSP replay + VLM | 1590/0/0/0 | 64.96 tok/s |

### 10 Kompile Benchmark Candidates (May 28) — All Rejected

| Candidate | Result | Reason |
|-----------|--------|--------|
| Baseline (4a37770da0) | 64.96 lateSteady | Reference |
| Triton gc_noATTN (no norm/reduction) | 32.95 | Major regression |
| BISECT graphCapture allSettings | 60.30 | Regression |
| Where OP_TRAIT_DYNAMIC_OUTPUT_SIZE | 62.20 | Noisy/regression |
| reshape copy-offset view bypass | 58.00 | Regression |
| Selective VALUE_DEP_UNFREEZE | 54.38 | Regression |
| cache_position in-place KV | 62.04 | Candidate never activated |
| autoregressive mask unmask removal | 63.67 | Regression |
| onnx_mha direct output | 62.87 | Regression |
| repeat_kv bypass + rank-4 MHA | CRASH | SIGABRT, error 700 |

All 10 candidates were rejected — further evidence that micro-optimizations are exhausted.

---

## What the ADRs Tell Us About Architectural Ceilings

14 performance-relevant Architecture Decision Records (ADRs) document the design decisions that both enabled and now constrain performance. Key insights:

### ADR 0061 — DynamicShapePlan (the foundation)

The DSP architecture claims **87-92 tok/s** is achievable with "CUDA graph replay + Triton + static KV cache" and states the model is "memory-bandwidth-bound (~8ms to load 5.3GB weights at 650 GB/s)." This 8ms theoretical floor means 100 tok/s (10ms/step) would leave only 2ms for all non-weight-loading work. The current 14.5ms/step suggests ~6.5ms of overhead beyond memory bandwidth.

### ADR 0060 — CUDA Async Memory Pool

The pool saves ~200ms per frame for the vision encoder and reduces memory growth from ~40MB/step to ~1MB/step. But stream-ordered pool semantics create the alloc/free stream mismatch bug (C++ ops allocate on stream 0, DSP frees on exec stream), which caused 288 OOM events despite 4GB reclaimable. This architectural choice is load-bearing — changing it would break the memory model.

### ADR 0063 — ArrayCacheMemoryMgr

The growth factor bug (`DEFAULT_GROWTH_FACTOR=1.05` → `closeable()=false` → 30MB/step leak) was an architectural flaw: the cache reuse optimization (allow buffers up to 2x larger than needed) interacted with the closeable gate to prevent DSP release paths from freeing oversized buffers. Fixed by gating on `isConstant()` instead of `closeable()`.

### ADR 0070 — GC Pressure

11,800 Full GC events per 256-token decode run eliminated. The `setCloseable(false)` → `setConstant(true)` poisoning bug was in the GC optimization code (commit `997e86cb58`). Intent was to prevent DeallocatorService from freeing weight buffers, but over-reached by also blocking D2H sync.

### ADR 0071 — Triton Graph Backend

Documents the warps/stages tuning: warps=4, stages=1 → 86.91 tok/s (+24.9% vs warps=2 baseline). Also documents the rule that MATMUL should NOT be compiled via Triton for M=1 decode (cuBLAS is faster). The fusion scoring system (`FusionScoring.cpp`) is where the attention neighborhood bonus (+50.0) lives.

### ADR 0082 — CUDA Graph Replay Pointer Stability

Documents 6 separate regressions in the pointer stability mechanism. The key insight: skipping `prepareSpecialUse`/`registerSpecialUse` in frozen steady-state eliminates ~5,486 `syncToDevice()` calls per decode step. This is one of the largest single optimizations but required fixing 6 bugs to enable.

### ADR 0089 — CUDA Graph Capture and Replay

Documents the cuBLAS workspace state mismatch: workspace is zeroed before replay but NOT before capture. This means cuBLAS may choose different algorithms at capture vs replay time, producing different numerical results. This is a known but unfixed source of non-determinism.

### ADR 0092 — Op Execution Timing Tracker

Lock-free ring buffer with 7-phase measurement. The timing data shows that only ~10% of step time is actual GPU compute (1.66ms/token at 250 tokens). The remaining ~90% is CUDA graph scheduling overhead.

### ADR 0093 — DSP Plan Disk Persistence

Eliminates plan recompilation on JVM restart. Without this, every JVM start triggers an 8.7-second full Triton recompile.

---

## The 10% Compute / 90% Scheduling Discovery

Op timing data from `OPTIMAL.csv` (250 tokens) reveals a critical insight:

```
Total GPU compute time:  ~415ms for 250 tokens = 1.66ms/token
Actual step time:        ~15.6ms/token (at 64 tok/s)
CUDA graph scheduling:   ~5.5-11ms/token (2742 nodes × 2-4µs/node)
Gap execution:           ~11.5ms/step (151 cuBLAS calls)
```

**Only ~10% of step time is actual GPU compute.** The rest is:
- CUDA graph scheduling overhead for 2742 captured nodes (~40%)
- Gap matmul execution - cuBLAS kernel launch overhead for 151 serial calls (~40%)
- CPU scheduling, sync, bookkeeping (~10%)

This means **reducing the CUDA graph node count** is as important as reducing GPU compute. The 552 concat + 300 reshape_no_copy ops contribute 852 nodes with near-zero compute but full scheduling overhead. Removing them from the graph (P3: concat freeze + P4: reshape gap classification) would cut scheduling from ~11ms to ~7.5ms.

---

## Untouched Optimization Frontiers

### Model Loading (spec written, not implemented)

`model-loading-optimization.md` has a full 4-phase spec:
1. SDZ caching (avoid 5-minute ONNX re-import)
2. Parallel model loading
3. Optimized SDZ format
4. Pre-compiled DSP plan cache shipping alongside model artifacts

Expected: 5 minutes → 30-60 seconds. Never implemented.

### Triton-CPU Integration (built but unused)

`triton-cpu` is built to `blasbuild/cpu/triton_cpu_install/` and `HAVE_TRITON_CPU` is defined, but the flag is never checked in any code path. Enabling it would give CPU the same tiled/vectorized kernel fusion as GPU. Never attempted.

### CPU Parallelism Gaps

15+ `FIXME: parallelism` comments in CPU op helpers (histogram, segment reduces, LSTM, image suppression, percentile, matrix_diag_part). These are not on the VLM hot path but represent untapped CPU throughput for other workloads.

### Speculative Decoding

Listed as blocked: `activeBatchSize == 1` gate in `SpeculativeDecodeLoop` self-disables on failure. N-gram approach confirmed not viable for SmolDocling architecture.

### LLM Multi-Model Benchmarks

No throughput data exists for Gemma, Phi, Mistral, or LFM2-extract. Only Qwen3.5-0.8B has been measured (25.14 tok/s CUDA, but quality check failed with diversity=0.11).

---

## Multi-Agent Fix Archaeology

The kompile task results (~100 files) reveal the debugging methodology:

### Parallel Investigation Pattern

For difficult bugs, 4-6 agents were dispatched in parallel to investigate competing hypotheses. Example from May 2 (DSP accuracy regression): 6 agents simultaneously investigated slot execution diffs, prezero guards, plan lifecycle, rms_norm_linear, autoregressive decode, and CPU DSP paths. This parallel approach found multiple contributing factors that would have taken weeks to find sequentially.

### Recurring Multi-Agent Failures

| Pattern | Instances | Cause |
|---------|-----------|-------|
| Agent timeout (>600s) | 5+ | Build times exceed agent timeout |
| OAuth/API rate limit | 3+ | Qwen agent "OAuth discontinued" |
| Agent fixes wrong code path | 4+ | Missing context about production vs test path |
| Agent introduces banned pattern | 2+ | AGENTS.md rules not included in dispatch prompt |

### The `configureMaxAllocationForKvCache` Root Cause Chain

Multiple agents across multiple days (May 2-3) converged on the same root cause from different angles:
1. Agent found `GenerationPipeline.java:1646` calls `configureMaxAllocationForKvCache()` AFTER CUDA graph capture
2. `DataBuffer::expand()` allocates NEW device buffer at different address, frees old one
3. CUDA graph has dangling pointers to freed buffer
4. `markExternalInputVariable` fails to invalidate because `needsFullInvalidation=false` (no staging buffers yet)
5. Stale CUDA graph replays with freed output buffer → logits never written → all zeros → argmax=0

This was independently discovered by 3 separate agents, confirming the root cause with high confidence.

---

## Comprehensive Dead-End Registry (Final)

Total optimization attempts tracked: **110+ N-series IDs + 10 kompile benchmark candidates + 9 fusion trials + 5 ADR-level architectural experiments**.

| Status | Count | Examples |
|--------|-------|---------|
| Shipped (positive impact) | ~35 | N6, N19, batched GEMM, frozen fast path, fused softmax, skip_rms_norm |
| Noise/neutral (no impact) | ~15 | N20, N24, N30, N36, N92, N101, N109 |
| Regression (reverted) | ~20 | N16, N21, N34, N37, N48, N63, N94, N96, N108, N110, FuseGatedMLP |
| Rehabilitated (root cause fixed) | 3 | N29b (capture workspace), N40 (cast cache pre-warm), N94 (code path restructured) |
| Resolved via different approach | 2 | N63 (shipped as N107), N16 (code path no longer exists) |
| Structurally impossible | ~13 | N26, N27, N29c, N49, N72, N95, N100, N105, N43-N59 |
| Untried (viable backlog) | ~3 | N78 (CUTLASS grouped GEMM), P3 (concat freeze), P1 (TF32 gaps) |
| Untried (Phase 5 scan) | 13 | P1-P13 (see Phase 5 Scan section) |
| Untried (micro, <0.3ms) | ~10 | N73, N75, N80, N83, N84, N86, N88-N91 |

**The N-series optimization space is fully explored.** The Phase 5 scan identified 13 new opportunities (P1-P13) across scheduling overhead, config gaps, and Java overhead. Three previously failed N-series optimizations (N29b, N40, N94) have been rehabilitated by infrastructure fixes. See the Reassessment and Phase 5 Scan sections for the combined path to 100 tok/s.

---

## Reassessment: Every Failed Optimization Against Current Codebase State

Each optimization below is re-evaluated by tracing what bugs existed when it was tried, what fixes landed afterward, and whether the optimization would produce a different result today.

---

### REHABILITATED — Would Likely Work Now

#### N29b / N97: Multi-Segment Composite Replay — ROOT CAUSE IS FIXED

**Original failure (Apr 5):** 8.37 tok/s (-88%). Restoring `outputShapeDependsOnInputValues` in `isSlotCapturable` to create ~328 segments instead of 1 monolithic segment. Recorded as "arrays deallocated between multi-segment replays."

**What actually failed:** The true root cause was **NOT** array deallocation between segments. It was **CUDA pool address reuse during capture** (commit `321884f564`): slots 1959 and 1975 shared the same GPU address because `cudaMallocAsync` reused freed intermediate addresses within the same capture step. The CUDA graph baked overlapping addresses; on replay, the pool resolved differently → fingerprint mismatch.

**What was fixed AFTER N29b (all major lifecycle fixes):**
| Date | Fix | Relevance |
|------|-----|-----------|
| Apr 6 | Removed `outputShapeDependsOnInputValues` from `isSlotCapturable` entirely | Avoided the problem instead of fixing it |
| Apr 9 | Unified `outputSlots_` and `slotArrayCache_` — all segments share same array | Cross-segment array access is now inherently stable |
| Apr 18 | Plan cache pin/unpin | Prevents eviction of active plans |
| Apr 18 | Use-after-free in `writeOutputSlot` fixed | Eliminates UAF on struct trace after delete |
| Apr 19 | **Plan-owned staging buffers** | External input addresses are now stable across steps regardless of Java |
| Apr 20 | `capturedInputAddrKey` updated after staging | Staging buffer integration completed |
| Apr 25 | Placeholder buffer protection for KV cache | External placeholder DataBuffers protected |
| Apr 25 | Transitive un-freeze for frozen constants feeding value-dep ops | Prevents shape tensor use-after-free |

**The critical fix:** The **per-segment capture workspace** (`ResourceBinder::captureWorkspace`) was introduced to fix exactly the pool-reuse-during-capture problem. During CUDA graph capture, `cudaMallocAsync` is intercepted and satisfied from a pre-allocated linear bump allocator that **never frees-and-reuses within a capture**. Each segment gets its own workspace (`captureWorkspaces_.resize(numSegments, nullptr)`), so multi-segment capture would allocate from per-segment workspaces correctly.

**Remaining risks (untested but architecturally sound):**
1. Cross-boundary staging buffers: staging buffers provide stable addresses for external inputs to segment N+1 when produced by segment N. Since all segments share `outputSlots_`, internal cross-segment references should resolve correctly.
2. `detectFrozenConstants` transitive un-freeze: written for monolithic segment, may have edge cases with 328 segments. Needs verification.
3. `argTableStable` tracking across segment boundaries: untested interaction.

**Verdict: HIGH PROBABILITY OF WORKING NOW.** The specific root cause (pool address reuse) is solved by the capture workspace. The array lifecycle infrastructure has been rebuilt since Apr 5. A targeted test with `CUDA_GRAPHS` mode + `nd4j.dsp.matmulSegmentation=true` would validate with minimal risk.

**Potential impact:** The 86.91 tok/s peak (Mar 19) used multi-segment composite replay with Triton islands. Restoring multi-segment boundaries could recover the architectural advantage that produced that peak.

---

#### N40: cuBLAS Arg-Table Refresh — MORE FEASIBLE THAN PREVIOUSLY ASSESSED

**Original assessment:** "Requires driver-level CUDA graph node parameter updates." Listed as HIGH risk.

**Reassessment based on N34/N37 failure analysis:** The actual blocker for gap matmul capture is NOT that cuBLAS lacks a pointer refresh mechanism. The blocker is **cast cache cold-start OOM during capture**.

When 91 matmuls are first captured, each needs an FP16 weight copy from the `tl_castCacheA/B` thread-local cast cache. The cast cache allocates heap NDArrays (not from the capture workspace bump allocator). On cold start during capture, 91 new allocations overwhelm available GPU memory.

**The fix is simpler than driver-level graph manipulation:**
1. Before composite capture begins, **pre-warm the cast cache** by executing all gap matmuls once outside capture to populate `tl_castCacheA/B`
2. Set the HWM (`mergedCastHwmA/B`) after pre-warming
3. Begin capture — matmuls reuse pre-warmed cache entries, no new allocations needed
4. At replay, `resetCastCacheIndicesTo(hwmA, hwmB)` already ensures unmerged matmuls don't clobber baked slots

The cast cache HWM infrastructure (`resetCastCacheIndicesTo`, `mergedCastHwmA/B`) was added AFTER N34/N37 failed and already handles the replay-time protection. Only the pre-warm step is missing.

For pointer refresh at replay: Triton already has `refreshArgTablesForReplay`. The cuBLAS equivalent would update A/B/C pointers in the captured CUDA graph nodes. CUDA 12.x supports `cudaGraphExecKernelNodeSetParams` / `cudaGraphKernelNodeSetParams` which can update kernel args in instantiated graphs. This is not driver-level — it's a standard CUDA API.

**Verdict: FEASIBLE.** The cast cache pre-warm is an implementation task, not a research problem. The CUDA graph node parameter update API exists. Combined estimate: the two-part fix (pre-warm + parameter update) could capture all 91 gap matmuls into CUDA graphs, eliminating ~5-8ms of per-step launch overhead.

---

#### N94: slotIsViewProducer_ Pre-Classification — CODE PATH CHANGED

**Original failure (May 18):** 64.5 tok/s (-7.3%). `refreshStaleViewWrappersInSegment` overhead for non-contiguous views exceeded the per-step classification savings.

**What changed:** The view-producer detection mechanism was substantially restructured:
- `isViewProducer` is now a bit on `slotPhase` inside each `NativeSlot`, set in-place during execution
- `viewProducerDetectionDone_` flag short-circuits after first segment pass
- `refreshStaleViewWrappersInSegment` now does an early-continue on `!slot.isViewCapableOp()` — only visits view-capable ops, not all slots

The original O(all-slots) iteration per step is gone. The current path is O(view-capable-slots-only) with a one-time detection pass.

**Verdict: WORTH RE-BENCHMARKING.** The code path that caused the -7.3% regression no longer exists in its original form. The pre-classification bit already exists; the remaining overhead is the per-step refresh call itself (now scoped to view-capable slots only).

---

### STATUS CHANGED — Different Assessment Than Before

#### N48+N62: deviceMutex Bypass — BASELINE SHIFTED

**Original failure:** -3.2% from 151 `tl_dspReplayActive` checks/step exceeding mutex savings.

**What changed:** N107 (`tl_cublasGapStreamReady`) was added after N48+N62, eliminating the per-matmul `cublasSetStream` + workspace reapplication. The `tl_dspReplayActive` check count in MmulHelper.cu is now only 4 (down from 151 estimate). The mutex is still taken per-matmul but the overall CPU overhead profile is different.

**Verdict: RE-BENCHMARK WORTHWHILE.** The baseline overhead is lower due to N107, which means the mutex cost is now a larger fraction of remaining CPU overhead. But also means the absolute savings are smaller. Likely still noise-level.

#### N63: Guard cublasSetStream — ALREADY SHIPPED AS N107

N107 (commit `9af2ddc429`, May 28) implements exactly what N63 attempted but with the correct stream routing: `tl_cublasGapStreamReady` flag is set by `CublasGapStreamGuard` at gap-loop entry, and both `mmulMxM` and `mmulMxV` check it before calling `cublasSetStream_v2`. The batchgemm path is also guarded.

**Verdict: FULLY RESOLVED.** N63's goal is achieved. No re-benchmark needed.

#### N16: Dirty-Mark Loop — CODE NO LONGER EXISTS

The dirty-mark mechanism was refactored into a fundamentally different structure. The current code uses flat O(range) loops over pre-computed `mergedGroupSlotRanges` with no branch prediction source — the only conditional is an `isClosed()` check. There is no longer a tight per-step inner loop with a speculative branch.

**Verdict: NO LONGER APPLICABLE.** The code path that N16 attempted to optimize does not exist in its original form.

#### In-Place KV (N96/N38/P5) — REGISTER PRESSURE PARTIALLY RESOLVED

**Blocker 1 (UNCHANGED):** SmolDocling ONNX model lacks `seqlens_k` (input 5). `ModelIOConfig.discover()` returns `cachePosition=null`. No synthetic injection added. **This is a Kotlin import-layer fix, not a kernel fix** — add a synthetic `sd.placeHolder("synthetic_cache_pos", DataType.INT64, 1)` in `GroupQueryAttention.kt` when `seqlensK == null`.

**Blocker 2 (CHANGED):** The `fusedGQADecodeKernel` runtime launch is now capped at **256 threads** (was 512 when N96 was tested). At 256 threads on SM89 (RTX 4090), the register budget per thread is 65536/256 = 256 registers max. The kernel uses 50+ registers for stride params. Adding `effectiveSeqKVPtr` (2 more registers) at 256 threads has much less occupancy impact than at 512 threads where it caused -14%.

**Verdict: BLOCKED BY IMPORT LAYER, NOT KERNEL.** Fix `GroupQueryAttention.kt` to inject synthetic `cache_position`, then the kernel register pressure at 256 threads is likely acceptable. Would eliminate 120 assign kernels/step (4 per layer × 30 layers).

---

### CONFIRMED STILL BLOCKED — Same Root Causes Remain

#### FuseGatedMLPPattern — All Three Root Causes Unchanged
- `fused_gemm_swiglu` is still `MATMUL` category in `OpTraitTable.cpp` line 242
- C++ generic impl still allocates 3 NDArray temporaries + 2 sequential `MmulHelper::mmul`
- No dedicated CUDA kernel exists (`helpers/cuda/` has no `fused_gemm_swiglu` file)
- **Still blocked.** Needs: (1) dedicated single-pass CUDA kernel, (2) reclassify away from MATMUL

#### N108: Triton for M=1 Matmuls — Tile Size Still Hardcoded
- `TritonIRBuilder_module.cpp` line 7946: `blockM=128, blockN=128, blockK=32` — hardcoded
- The attention path has `blockM = (seqQ <= 1) ? 1 : ...` but standalone matmul does NOT
- cuBLAS is still specifically optimized for GEMV (M=1) via internal heuristics
- **Still blocked.** Even with seqQ-aware tiles, hand-tuned cuBLAS GEMV likely wins at M=1

#### Monolithic CUDA Graph Capture — BANNED Architectural Rule
- Per-bug fixes (writeSpecial, value-mixing, gap staleness) all landed
- But Triton-gap interleaving is a **design constraint**: monolithic capture wraps Triton islands without correctly sequencing external Triton module launches vs in-graph kernel nodes
- Current OPTIMAL config's monolithic capture is for native-only (non-Triton) ops — a different code path
- **Still BANNED.** Not a transient bug — architectural incompatibility with Triton gaps

#### N110: Per-Gap-Unit Capture — Legacy Stream Blocker
- `DataBuffer::syncToPrimary` still uses legacy stream (stream 0) for `cudaMemcpyAsync`
- During capture on non-default stream, any legacy-stream dependency is forbidden (CUDA error 8)
- No fix has landed since May 28
- **Still blocked.** Requires routing ALL DataBuffer copy ops through DSP execution stream (broad infrastructure change)

#### N24: GQA 512 Threads — Same Register Pressure
- Still 24 stride parameters in kernel signature (50+ registers for strides alone)
- `threadsPerBlock` hardcoded to 256 in `LaunchDims.cu` line 1271 with explicit cap comment
- **Still infeasible at 512.** No stride params removed.

#### N26/N49: Cross-Layer bGEMM Super-Batching — Still Serial Dependencies
- 91 unbatched matmuls are still in serial dependency chains through residual stream
- `inputRejected=3945` — fundamentally unbatchable across layers
- **Structurally impossible.** No code change can remove the serial dependencies

#### mergeViewGaps — Already Active, -5.4% Was Real
- `mergedCaptureThroughViews` defaults to `true` in `TritonConfig.h` line 95
- The VIEW_TICK fast path (lines 1728-1771) already skips full op dispatch for established views
- The -5.4% was real — capture overhead > dispatch savings for zero-compute ops
- **Already implemented.** Not a candidate for re-trying

---

### Revised Priority for 100 tok/s

Based on the reassessment, the viable path ordering has changed:

| Priority | Optimization | Previous Assessment | Revised Assessment | Estimated Impact |
|----------|-------------|--------------------|--------------------|-----------------|
| **1** | **N29b/N97: Multi-segment replay** | "VERY HIGH risk, array lifecycle broken" | **ROOT CAUSE FIXED** (capture workspace solves pool reuse). Test with `matmulSegmentation=true`. | **+10-20 tok/s** (restores 86.91 architecture) |
| **2** | **N40: cuBLAS arg-table refresh** | "Requires driver-level graph manipulation" | **Cast cache pre-warm + `cudaGraphKernelNodeSetParams` API**. Not driver-level — standard CUDA 12.x API. | **+5-8 tok/s** (captures 91 gap matmuls) |
| **3** | **In-place KV (N96)** | "Model lacks seqlens_k, register pressure" | **Kotlin import fix** (`GroupQueryAttention.kt` synthetic injection). Register pressure reduced at 256 threads. | **+2-4 tok/s** (eliminates 120 assign kernels/step) |
| **4** | **N78: CUTLASS grouped GEMM** | "HIGH effort, custom kernel" | Unchanged — still requires custom kernel integration. But N40 may make it unnecessary if gap matmuls can be captured. | **+5-10 tok/s** (single kernel for 91 matmuls) |
| **5** | **N94: Re-benchmark** | "slotIsViewProducer_ overhead" | Code path restructured. O(all-slots) → O(view-capable-slots). | **Unknown** (re-benchmark needed) |

**The most significant finding:** N29b's root cause is fixed. The capture workspace infrastructure that was added to solve a different problem (monolithic capture address reuse) also solves the multi-segment capture address reuse that killed N29b. This is the single highest-potential path to recovering the 86.91 tok/s peak.

---

## Phase 5 Scan: NEW Optimization Opportunities (May 29, 2026)

A comprehensive codebase scan across 8 parallel investigations uncovered optimization opportunities not previously attempted. These are organized by category and prioritized by estimated impact.

---

### P1: Enable TF32 for Gap Matmuls — PEDANTIC_MATH Override

**Location:** `NativeDynamicShapePlan_gpubackend.cu` composite replay path, `MmulHelper.cu`

**Current state:** DSP execution forces `PEDANTIC_MATH` mode for all execution modes (including gap matmuls). This disables TF32 accumulation and tensor core usage in cuBLAS, forcing FP32-only compute for all 151 gap matmul calls.

**The opportunity:** TF32 accumulation gives ~2-4x throughput improvement on RTX 4090 tensor cores for M=1 GEMV operations. The 91 unbatched gap matmuls that dominate step time (11.5ms) would directly benefit.

**Why it was previously avoided:** PEDANTIC_MATH was introduced to maintain bit-exact accuracy across execution modes. TRITON and CUDA_GRAPHS modes need reproducible outputs for validation comparison against SLOT_BY_SLOT.

**Why it's viable now:**
- `TestDspValidation.tf32Isolation` already exists and verifies token-level accuracy with TF32 enabled
- The validation test uses a `tf32` tolerance preset that allows for TF32-level numerical differences
- `OPTIMAL` config can set `CUBLAS_COMPUTE_32F_FAST_TF32` without affecting other configs
- Only applies to gap matmuls (Triton islands already use their own compute mode)

**Estimated savings:** The 91 unbatched matmuls currently at ~0.019ms/call could drop to ~0.008-0.012ms/call with TF32. Total: **~0.6-1.0ms/step** savings.

**Risk:** MEDIUM — accuracy regression possible. Must pass `tf32Isolation` validation and maintain >90% token match rate.

---

### P2: Eliminate 2224 Extra equals/Where Calls in OPTIMAL

**Location:** `OPTIMAL.csv` op timing data

**Current state:** Op timing data reveals a dramatic discrepancy between OPTIMAL and TRITON configs:

| Op | OPTIMAL calls | TRITON calls | Ratio |
|----|--------------|--------------|-------|
| equals | 2345 | 121 | 19.4x |
| Where | 2346 | 123 | 19.1x |

In OPTIMAL mode, `equals` and `Where` are called ~2224 extra times compared to TRITON mode (over 250 tokens). This is ~9 extra calls per token, suggesting a per-layer check that runs on every decode step.

**Hypothesis:** A per-layer causal mask comparison or attention mask check is running as unfused slot-by-slot ops in OPTIMAL mode but is captured within Triton islands in TRITON mode. The equals→Where chain pattern is consistent with a conditional masking operation: `Where(equals(x, value), true_val, false_val)`.

**The fix:** Identify the graph nodes producing the extra equals/Where calls and ensure they are included in Triton island compilation for OPTIMAL config, or mark them as frozen constants if their outputs don't change between decode steps.

**Estimated savings:** The GPU compute per equals/Where call is negligible for scalar/small-tensor ops (~0.03µs each), so direct GPU savings are small (~0.07ms/step). The main savings are from **eliminating CUDA graph scheduling overhead**: 2224 extra graph nodes × ~2-4µs/node = **~0.4-0.9ms/step**. If these calls are instead fused into existing Triton islands (as TRITON mode achieves), the savings come from reduced graph node count.

**Risk:** LOW — these are redundant calls that TRITON mode already eliminates.

---

### P3: Concat Node Freeze — Remove DATADEP Trait

**Location:** `OpTraitTable.cpp` line 316: `concat` tagged with `CONCAT|DATADEP`

**Current state:** 552 `concat` nodes in the SmolDocling graph are classified as `DATADEP` (data-dependent output shape), which prevents them from being frozen as constants. Each concat contributes a CUDA graph node with near-zero compute but full scheduling overhead (~2-4µs/node).

**The opportunity:** When the concat `axis` argument is a compile-time constant (not a dynamic graph variable), the output shape is fully determined by input shapes — there is no data dependency. For SmolDocling, ALL 552 concat ops have constant axis values.

**The fix:** Modify `OpTraitTable.cpp` to only apply `DATADEP` to concat when axis is a runtime variable. Add a check in `getOpTraits()` or create a conditional trait function. Alternatively, add a per-node trait override in the DSP compiler when it can statically verify the axis is constant.

**Estimated savings:** 552 fewer CUDA graph nodes × ~2-4µs/node = **~1.1-2.2ms/step** scheduling overhead reduction. This is 7-15% of the current 14.5ms/step.

**Risk:** MEDIUM — must verify axis constancy at DSP compile time, not just ONNX import time. Dynamic axis would break silently.

---

### P4: reshape_no_copy Gap Classification

**Location:** Graph analysis of SmolDocling decode path

**Current state:** ~300 `reshape_no_copy` ops are in the CUDA graph despite being zero-compute ops (they just create view metadata). Each contributes a graph node with scheduling overhead.

**The opportunity:** Mark `reshape_no_copy` as a gap op when `ARRAY_NEEDS_COPY` is false (the common case). Gap ops skip CUDA graph capture entirely. Zero-compute ops should not occupy graph nodes.

**The fix:** Add `OP_TRAIT_VIEW_ONLY` to `reshape_no_copy` in `OpTraitTable.cpp` (similar to how `expand_dims` and `squeeze` are handled).

**Estimated savings:** ~300 fewer graph nodes × ~2-4µs = **~0.6-1.2ms/step**.

**Risk:** LOW — reshape_no_copy with ARRAY_NEEDS_COPY=false is a metadata-only operation by definition.

---

### P5: Triton TILE Section Compilation

**Location:** `SectionTypeConfig.h`: `TILE` section has `compiledByDefault=false`, `fusionVerified=false`

**Current state:** The Triton compiler has an emitter for TILE sections (tile/repeat ops), but it is NOT included in the OPTIMAL config's `tritonIncludeTypes`. This means tile ops run as unfused slot-by-slot operations.

**The fix:** Add `TILE` to `BenchmarkConfig.java` OPTIMAL config `tritonIncludeTypes` (line 201). Set `fusionVerified=true` in `SectionTypeConfig.h` after verification.

**Estimated savings:** Unknown without profiling, but TILE ops are likely captured in islands with their consumers, fusing away kernel launch overhead.

**Risk:** LOW — the emitter already exists and is tested. Only needs config enablement.

---

### P6: Restore `tritonFusionMinScore(4.0f)` from 86.91 Peak

**Location:** `BenchmarkConfig.java` OPTIMAL config

**Current state:** The OPTIMAL config that achieved 86.91 tok/s (peak) included `tritonFusionMinScore(4.0f)`. This setting is **missing from the current OPTIMAL config**. The fusion min score controls the threshold for how aggressively the Triton compiler fuses op chains — a higher score means only high-value fusions are attempted, avoiding overhead from compiling marginally beneficial fusions.

**The fix:** Add `.tritonFusionMinScore(4.0f)` to the OPTIMAL config builder in `BenchmarkConfig.java`.

**Estimated savings:** Unknown — but this was present in the fastest config ever achieved. Restoring it may help filter out low-value fusions that add compile/refresh overhead without meaningful compute savings.

**Risk:** LOW — config-only change, easily benchmarkable.

---

### P7: Java `frozenOutputsInitialized` Gate

**Location:** `DynamicShapePlanExecutor.java` lines 3185-3189

**Current state:** Every call to `executeNative()` in frozen steady state runs a dummy output setup loop:
```java
for (SDVariable outputVariable : outputVars) {
    INDArray dummy = plan.outputForVariable(outputVariable.name());
    // ... setup logic
}
```
This fires on EVERY frozen step, allocating ~100 dummy arrays and making ~100 JNI calls. The field `frozenOutputsInitialized` exists but is **never used** to gate this loop.

**The fix:** Set `frozenOutputsInitialized = true` after the first frozen execution, then skip the loop on subsequent calls:
```java
if (!frozenOutputsInitialized) {
    for (SDVariable outputVariable : outputVars) { ... }
    frozenOutputsInitialized = true;
}
```

**Estimated savings:** ~100 JNI calls + ~100 array allocations skipped per step = **~0.1-0.2ms/step**.

**Risk:** VERY LOW — the field already exists; this is a trivial gate.

---

### P8: Output Validation Bypass in Frozen Steady State

**Location:** `DynamicShapePlanExecutor.java` lines 3347-3369

**Current state:** After every `executeNative()` call, an output validation loop runs ~200 JNI calls to verify output buffer correctness. In frozen steady state, output shapes and buffers are stable — validation is redundant after the first few steps.

**The fix:** Skip output validation after the first 3 frozen steps (same pattern as `REQUIRE_TRUE` gating in C++).

**Estimated savings:** ~200 JNI calls/step × ~1µs/call = **~0.2ms/step**.

**Risk:** LOW — validation is a debugging aid, not a correctness requirement. Gate behind `isDebug()`.

---

### P9: Cache `deviceMutex()` Return in compositeReplay

**Location:** `NativeDynamicShapePlan_batchgemm.cu` line 762

**Current state:** `LaunchContext::deviceMutex()` calls `cudaGetDevice()` — a CUDA driver call — every time it's invoked. In the batched GEMM path, this happens 60 times per step (once per batch group).

**The fix:** Cache the return value of `cudaGetDevice()` at the start of the composite replay call and pass it as a parameter, or use a thread-local cached device ID.

**Estimated savings:** 60 × ~1µs driver call = **~0.06ms/step**. Small but free.

**Risk:** VERY LOW — device ID does not change during a replay step.

---

### P10: Autoregressive Decode Loop Overhead

**Location:** `autoregressive_decode.cu` lines 529-550, 809-834, 853, 888

Four separate overhead sources in the production decode loop:

| Source | Lines | Overhead | Fix |
|--------|-------|----------|-----|
| `cudaStreamSynchronize` pipeline stall | 853 | ~0.01ms/step | Use event-based sync to overlap D2H token readback with next step's GPU work |
| 6 single-thread `<<<1,1>>>` kernels for mask updates | 529-550, 809-834 | ~0.06ms/step | Fuse into 1-2 kernels with 6 fields each |
| `std::vector<KvScatterEntry>` heap alloc | 888 | ~0.005ms/step | Pre-allocate a fixed buffer (max layers known at compile time) |
| Per-step `cudaGetLastError()` check | various | ~0.01ms/step | Gate behind `isDebug()` |

**Total estimated savings:** **~0.08-0.1ms/step**. Individually tiny, collectively measurable.

**Risk:** LOW to VERY LOW per item.

---

### P11: Eliminate O(1332) Ext Input Scans in Java

**Location:** `DynamicShapePlanExecutor.java` — four separate loops over all 1332 external inputs per frozen step

**Current state:** The frozen steady-state path scans all external inputs multiple times:
1. `syncExternalInputs` — already optimized to O(3) variable-only (commit `43ec64a`)
2. `markExternalInputsChanged` — still O(1332) full scan
3. Output variable resolution — still O(1332) full scan
4. Handle refresh check — still O(1332) full scan

Scans 2-4 remain at O(1332) despite only ~3 inputs changing per step.

**The fix:** Apply the same variable-only pattern to the remaining three scans. Maintain a `changedExtInputIndices` set that is populated at input-write time and consumed by each scan.

**Estimated savings:** ~3 × 1332 iterations × ~1µs/iteration = **~0.004ms/step** — negligible on its own but contributes to Java overhead reduction.

**Risk:** VERY LOW.

---

### P12: broadcast_to → equals → Where Chain Fragmentation Fix

**Location:** `TritonIRBuilder_sections.cpp` lines 592-614

**Current state:** An element count mismatch guard in the Triton IR builder fragments the `broadcast_to → equals → Where` chain, preventing fusion into a single Triton kernel. This forces three separate kernel launches for what should be a single fused operation.

**The fix:** Modify the element count check to allow chains where the broadcast is the first op (broadcast inherently changes element count). The equals and Where operations consume the broadcast output, so the element count is consistent after the first op.

**Estimated savings:** Reduces island count, potentially absorbing the 2224 extra equals/Where calls identified in P2.

**Risk:** MEDIUM — must verify that the broadcast element count mismatch doesn't cause out-of-bounds access in the fused kernel.

---

### P13: `h_castPtrs_host[64]` Zero-Init Elimination

**Location:** `NativeDynamicShapePlan_batchgemm.cu` line 832

**Current state:** `float* h_castPtrs_host[64]` is zero-initialized (via `= {}`) on every cast group execution. For 60 batched GEMM calls per step with ~10 cast groups, this is ~600 zero-init operations per step.

**The fix:** Only initialize the entries that will be used (`for (int i = 0; i < count; i++)` instead of zero-initializing the full array).

**Estimated savings:** Negligible (~0.001ms/step). Mentioned for completeness.

**Risk:** VERY LOW.

---

### Summary: New Optimization Opportunity Priority Matrix

| Tier | ID | Optimization | Est. Impact (ms/step) | Est. tok/s Gain | Risk | Effort |
|------|----|----|--------|--------|------|--------|
| **S** | P3 | Concat node freeze (remove DATADEP) | 1.1-2.2 | +5-12 | MEDIUM | MEDIUM |
| **S** | P1 | TF32 for gap matmuls | 0.6-1.0 | +3-5 | MEDIUM | LOW |
| **A** | P4 | reshape_no_copy gap classification | 0.6-1.2 | +3-6 | LOW | LOW |
| **A** | P6 | Restore tritonFusionMinScore(4.0f) | Unknown | Unknown | LOW | TRIVIAL |
| **A** | P2 | Eliminate extra equals/Where | 0.4-0.9 | +2-4 | LOW | MEDIUM |
| **B** | P7 | frozenOutputsInitialized gate | 0.1-0.2 | +0.5-1 | VERY LOW | TRIVIAL |
| **B** | P8 | Output validation bypass | 0.1-0.2 | +0.5-1 | LOW | LOW |
| **B** | P5 | Triton TILE compilation | Unknown | Unknown | LOW | LOW |
| **B** | P12 | broadcast→equals→Where unfragment | Part of P2 | Part of P2 | MEDIUM | MEDIUM |
| **C** | P10 | Decode loop overhead (4 items) | 0.08-0.1 | +0.3-0.5 | LOW | LOW |
| **C** | P9 | Cache deviceMutex return | 0.06 | +0.2 | VERY LOW | TRIVIAL |
| **C** | P11 | Ext input scan elimination | 0.004 | <0.1 | VERY LOW | LOW |
| **C** | P13 | castPtrs zero-init | 0.001 | <0.1 | VERY LOW | TRIVIAL |

### Combined Path to 100 tok/s

Current: **~69.6 tok/s** (14.4ms/step)

| Step | Optimization(s) | Cumulative ms/step | Cumulative tok/s |
|------|-----------------|-------------------|-----------------|
| 0 | Baseline | 14.4 | 69.6 |
| 1 | P3 (concat freeze) + P4 (reshape gap) | 11.6-12.1 | 82.6-86.2 |
| 2 | P1 (TF32 gap matmuls) | 10.8-11.3 | 88.5-92.6 |
| 3 | P6 (restore fusionMinScore) + P2 (equals/Where) | 9.9-10.9 | 91.7-101.0 |
| 4 | P7+P8 (Java overhead gates) + P5 (TILE) | 9.6-10.5 | 95.2-104.2 |
| 5 | Previously reassessed N29b (multi-segment) OR N40 (cuBLAS capture) | 8.0-9.5 | **105-125** |

**Key insight:** The newly discovered Tier S and A optimizations (P1-P4, P6) are **complementary** to the previously reassessed rehabilitated optimizations (N29b, N40). The Tier S/A items reduce scheduling overhead and enable TF32; the rehabilitated items reduce launch overhead by capturing more ops. Together, they represent a path from 69.6 to potentially 100+ tok/s without requiring any single high-risk architectural change.

**The 86.91 peak had different advantages but different bugs.** The current codebase has more robust infrastructure (capture workspace, plan-owned staging, cast cache HWM) that should make the new optimizations more stable than the peak's fragile fast paths were.
