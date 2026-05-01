---
name: cuda-dsp-optimization-plan
description: "VLM decode perf plan — 51 tok/s target 100+, accuracy gate, prioritized checklist, dead ends, one-at-a-time regression tracking"
type: project
---

# CUDA DSP Decode Optimization Plan (updated 2026-04-29)

**Current**: ~51 tok/s (~15ms GPU compute, 509 kernels/step)
**Target**: 100+ tok/s (<10ms/step)
**Model**: SmolDocling-256M (30-layer decoder), RTX 4090, batch=1 seq=1 decode

---

## ACCURACY GATE — MANDATORY FOR EVERY CHANGE

**Test**: pathfinder-mythic.pdf page 10 (text about mythic heroes)

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Dtest=TestSmolDoclingOptimizedPipeline#testOptimizedDoclingPipeline \
  -Dvlm.test.maxTokens=250 \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/accuracy-check.log
```

**PASS criteria**:
- Output contains SmolDocling **document layout tags** (`<doctag>`, `<text>`, `<section_header>`, `<page_header>`, `<list_item>`, `<location>`)
- Contains **actual English words** about mythic heroes
- **ANY repeating garbage** ("UserT", "UserTUserT", etc.) = model COMPLETELY BROKEN = REVERT IMMEDIATELY

**Performance gate**:
- `run-benchmark.sh --tokens 250` — must beat or match 51 tok/s baseline
- Any regression >2% from baseline = investigate before merging

**Process**: ONE change at a time. Benchmark + accuracy check after EACH change. Log result in this file.

---

## Per-Step GPU Time Budget (nsys-verified 04-29)

| Component | ms/step | Kernels/step | Source |
|-----------|---------|-------------|--------|
| cuBLAS GEMV (matmuls) | ~10-11 | ~180 | MmulHelper → cublasHgemm/GemmEx |
| Auto-cast (FP32→HALF) | ~1.4 | 98 | 49 gap matmuls × 2 casts each |
| Attention (GQA decode) | ~1.2 | ~30 | fusedGQADecodeCuda |
| Tile (broadcast_to) | ~0.3 | ~10 | ONNX Expand → NDArray::tile |
| Reduce | ~0.3 | ~5 | simpleReduce |
| KV copy (assign) | ~0.5 | ~120 | present→static KV scatter |
| Other (concat/gather/rope/norm) | ~1.3 | ~66 | misc |
| **Total** | **~15** | **~509** | |

To reach 100 tok/s: total step < 10ms → GPU compute must drop to ~7ms.

---

## OPTIMIZATION CHECKLIST (try one at a time, track regressions)

### OPT-01: Selective fused-matmul category enablement
- **What**: Create new Triton category (e.g. `FUSED_MATMUL`) for `rms_norm_linear`, `fused_gemm_swiglu`, `fused_two_layer_mlp`. Add to `tritonIncludeTypes` in OPTIMAL without enabling generic `matmul`. Dedicated Triton emitters exist but are unreachable because they share `MATMUL` category with generic matmul.
- **Expected impact**: HIGH — pulls fused ops out of gap execution, eliminates their auto-cast kernels, replaces multi-kernel sequences with single fused kernels.
- **Risk**: MEDIUM — emitters exist (`TritonIRBuilder_kernels.cpp:440,673,870`) but never exercised in OPTIMAL config.
- **Files**: `OpCategoryTable.h`, `SectionTypeConfig.h`, `TritonGraphBackend_compile.cu`, `BenchmarkConfig.java:200`
- **Prerequisite**: OPT-03 (fusion patterns must create these ops in graph first)
- **Result**: _not yet tested_

### OPT-02: FuseSkipRMSNormLinearPattern (new fusion)
- **What**: New pattern matching `skip_rms_norm → [cast] → matmul` → fused `skip_rms_norm_linear` op. SmolDocling has 30 such chains (1/layer). Current `FuseRMSNormLinearPattern` only matches `RmsNorm` not `SkipRmsNorm` (line 631).
- **Expected impact**: HIGH — 30 skip_rms_norm + 30 matmul → 30 fused ops per step.
- **Risk**: HIGH — requires new C++ op, shape fn, Triton emitter. Large surface area.
- **Files**: `NormalizationFusionOptimizations.java`, new op in `generic/nn/`, `OpCategoryTable.h`, `TritonIRBuilder_kernels.cpp`
- **Result**: _not yet tested_

### OPT-03: Re-enable FuseGatedMLPPattern (investigate regression first)
- **What**: `FuseGatedMLPPattern` disabled at `ActivationFusionOptimizations.java:277-282`. Previously -5.5% (51→48.2). Investigate WHY — pattern itself, `fused_gemm_swiglu` op, or Triton emitter? Emitter exists at `TritonIRBuilder_kernels.cpp:673`.
- **Expected impact**: HIGH if fixable — fuses gate+up matmuls + SiLU → single kernel (eliminates 30 matmuls + 30 swish_mul/step).
- **Risk**: MEDIUM — need root cause before re-enabling.
- **Files**: `ActivationFusionOptimizations.java`, `fused_gemm_swiglu.cpp`, `TritonIRBuilder_kernels.cpp:673`
- **Result**: _investigate first, no code change_

### OPT-04: Eliminate auto-cast via HALF activation propagation
- **What**: 49 gap matmuls see FP32×FP32 inputs triggering 98 `transformAnySimpleCached` casts despite HALF weights. ~161 Triton-island matmuls bypass MmulHelper entirely (zero casts). Trace what makes gap matmul inputs FP32 — if activations can stay HALF, all 98 casts vanish.
- **Expected impact**: HIGH — 98 fewer kernels, ~1.4ms/step saved.
- **Risk**: LOW-MEDIUM — precision validation needed.
- **Key finding**: Cast is in `MmulHelper.cu:873-905` (mmulMxM) and `:1176-1198` (mmulMxV). Gate: `aType==FLOAT32 && bType==FLOAT32 && cType==FLOAT32 && major>=6`. No same-dtype short-circuit in `cast()`.
- **Files**: `MmulHelper.cu`, graph type propagation, gap op dtype settings
- **Result**: _investigate first_

### OPT-05: QKV projection fusion (3 GEMVs → 1)
- **What**: Each attention layer: Q=X×Wq, K=X×Wk, V=X×Wv (3 GEMV on same input). Fuse to single `[1,D] × [D, 3*D_out]` GEMM with concatenated weights.
- **Expected impact**: MEDIUM — saves 60 cuBLAS kernel launches, ~0.3ms.
- **Risk**: MEDIUM — graph-level pattern detection + weight concat at import.
- **Files**: New fusion pattern, modified matmul dispatch
- **Result**: _not yet tested_

### OPT-06: MLP gate/up projection fusion (2 GEMVs → 1)
- **What**: Each MLP: `gate=X×Wgate, up=X×Wup` then `SiLU(gate)*up`. Fuse 2 matmuls to single `[1,D] × [D, 2*D_ff]` GEMM. Lighter version of OPT-03 (no SiLU fusion).
- **Expected impact**: MEDIUM — saves 30 cuBLAS launches, ~0.15ms.
- **Risk**: MEDIUM — same pattern detection as OPT-05.
- **Files**: New fusion pattern, weight concatenation
- **Result**: _not yet tested_

### OPT-07: cublasLt FAST_TF32 for logits projection
- **What**: `tryLtMatmul()` at `MmulHelper.cu:295,375` hardcodes `CUBLAS_COMPUTE_32F`. Change to `CUBLAS_COMPUTE_32F_FAST_TF32` when `cublasTf32Enabled() && smMajor>=8`.
- **Expected impact**: LOW — only fires for vocab projection (N>=16384, M==1), once/step.
- **Risk**: VERY LOW — TF32 already active on standard cuBLAS handle via `cublasSetMathMode`, this extends to cublasLt.
- **Key finding**: `cublasTf32` flag IS wired and enabled in optimal(). Standard handle gets `CUBLAS_TF32_TENSOR_OP_MATH`. cublasLt uses separate handle, doesn't inherit math mode.
- **Files**: `MmulHelper.cu:295,375`
- **Result**: _not yet tested_

### OPT-08: broadcast_to stride-0 view (eliminate tile kernels)
- **What**: Replace `broadcast_to`'s physical `tile()` with stride-0 view for dims expanding from 1.
- **Expected impact**: LOW-MEDIUM — eliminates ~10 tile kernels, ~0.3ms.
- **Risk**: HIGH — same risk as reshape_no_copy bypass (-29%). cuBLAS needs contiguous inputs.
- **Key finding**: ~10-11 `tileKernelDouble`/step from 62 `broadcast_to` ops (ONNX Expand import). KV-repeat tiles already eliminated by `fusedGQADecodeCuda`.
- **Files**: `broadcast_to.cpp:65`, consumer ops
- **Result**: _not yet tested — approach with extreme caution_

### OPT-09: In-place KV write (eliminate 120 assign kernels)
- **What**: Wire `seqlens_k` from ONNX through decode loop to activate `useInPlaceKv` in `onnx_multi_head_attention.cpp:170-218`. C++ impl is DONE and dormant.
- **Expected impact**: HIGH — 120 fewer assign kernels (~23% kernel count), ~0.5ms.
- **Risk**: MEDIUM — coordinated 4-file change. SmolDocling may not provide `seqlens_k`.
- **Files**: `AutoregressiveDecode.java`, `GenerationPipeline.java`, `ModelIOConfig.java`, `GroupQueryAttention.kt`, `autoregressive_decode.cu`
- **Result**: _not yet tested_

### OPT-10: Attention-neighborhood fusion (rope+attention+writeback)
- **What**: Fuse `fused_rope → permute → attention → permute` into single kernel.
- **Expected impact**: MEDIUM — eliminates launch gaps between tightly coupled ops.
- **Risk**: HIGH — complex kernel, large surface area.
- **Result**: _not yet tested_

### OPT-11: Narrow view-boundary stitching
- **What**: Targeted version of mergeViewGaps (which was -5.4%). Only stitch view ops between two captured Triton islands to merge them into 1 island.
- **Expected impact**: LOW-MEDIUM — reduces CUDA graph node count.
- **Risk**: MEDIUM — broad version failed.
- **Files**: `NativeDynamicShapePlan_gpubackend.cu`
- **Result**: _not yet tested_

### OPT-12: Gather/concat ladder coalescing
- **What**: Diagnose ~30 concat + ~20 gather ops/step. May be eliminable via shape manipulation or fused scatter/gather.
- **Expected impact**: LOW — ~0.2ms combined.
- **Risk**: LOW — diagnostic first.
- **Result**: _not yet tested_

### Recommended execution order

| Order | Item | Rationale |
|-------|------|-----------|
| 1 | OPT-04 investigate | No code change, identifies FP32 source. Informs OPT-01/03. |
| 2 | OPT-03 investigate | No code change, diagnose fused_gemm_swiglu regression root cause. |
| 3 | OPT-01 | Selective fused-matmul category — most self-contained HIGH impact. |
| 4 | OPT-03 implement | Re-enable gated MLP fusion if root cause found. |
| 5 | OPT-09 | In-place KV write — C++ done, just wiring. |
| 6 | OPT-04 implement | HALF propagation fix based on investigation. |
| 7 | OPT-02 | skip_rms_norm_linear — biggest new-code item. |
| 8 | OPT-07 | cublasLt TF32 — trivial, low risk. |
| 9 | OPT-05 | QKV fusion. |
| 10 | OPT-06 | MLP gate/up fusion. |
| 11 | OPT-12 | Gather/concat diagnosis. |
| 12 | OPT-11 | Narrow view stitching. |
| 13 | OPT-08 | broadcast_to views — last due to high risk. |
| 14 | OPT-10 | Attention neighborhood — most complex. |

---

## COMPLETED Optimizations

### Tier 1 (autoregressive_decode.cu)
- [x] **1a. D2D token copy** — cudaMemcpyAsync(D2D) replaces p() H2D+sync
- [x] **1b. Mask/position updates before sync** — overlap with graph execution
- [x] **1c. Pinned memory for D2H** — true async DMA

### Fused Ops
- [x] **skip_rms_norm** — residual add + RMS norm, +2.5% (52→53.28)
- [x] **rms_norm_linear** — single-kernel fused, 51.88 tok/s
- [x] **fused_gelu, fused_rope, fused_layer_norm, fused_rms_norm_swiglu**
- [x] **fusedElementwiseChain** — up to 8 consecutive elementwise ops
- [x] **fused warp-shuffle softmax** — fusedCausalMaskSoftmaxCuda
- [x] **fusedGQADecodeKernel** — stride-aware single-kernel GQA, perf-neutral but correct

### Infrastructure
- [x] **cublasLt epilogue pipeline** — bias/relu/gelu epilogues
- [x] **FusionPass Pass 5** — matmul→add(bias)→activation detection
- [x] **checkIndices elimination** — +3.1% (51.44→53.05)
- [x] **Bypass launchAsync overhead** in composite replay fast path
- [x] **Pre-allocate KV scatter buffers** + fast-path staging sync
- [x] **Skip error message heap alloc/free** when no error set
- [x] **Deduplicate cross-stream sync**

---

## DEAD ENDS (never attempt again)

| Attempt | Result | Why |
|---------|--------|-----|
| TILE in tritonIncludeTypes | -6.4% (53→49.6) | Triton tile slower than CUDA tile kernel |
| mergeViewGaps | -5.4% | Extra overhead from island merging |
| forward4DDecode GQA via .tile() | -31% | tile() slower than workspace+strided-batch |
| reshape_no_copy view bypass | -29% | Non-contiguous views kill cuBLAS GEMV |
| sizeAt replacing gather | -4 tok/s | sizeAt less optimized in DSP |
| Mega-graph (gap ops in capture) | 49.6% accuracy | Stale buffer addresses |
| FuseGatedMLPPattern | -5.5% (51→48.2) | Root cause TBD (OPT-03 investigation) |
| Pre-sync KV scatter move | 0% | GPU compute is bottleneck, not CPU |
| GQA forward4DDecode stride-aware | 0% (51.23) | Correct but perf-neutral vs cuBLAS |
| Mixed-type gamma skip_rms_norm | 0% (50.95) | FP16 pre-cast means no mismatch |
| silu/swish_mul temp elimination | 0% (51.42) | batch=1 tensors too small to matter |
| dspCastSinkMatmul | 0% | Cast ops are µs each |
| Graph-level DCE | 0 ops removed | SmolDocling graph fully connected |
| causal mask putScalar→assign | -4% (48.43) | Op dispatch overhead |
| Dirty-generation counter | slight regression | |
| Single-stream reordering | 0% | FIFO bound, need second stream |

### Key facts (verified, not speculation)
- GPU argmax IS implemented — only 8B D2H
- KV scatter IS efficient — batched ~135KB
- mergedGroups=1 IS working — 1 cudaGraphLaunch
- argTableStable=true — arg refresh skipped in steady state
- reshape_no_copy copies are LOAD-BEARING — cuBLAS needs C-contiguous
- VLM benchmark uses native C++ autoregressive_decode, NOT Java decode loop
- FP16 weight pre-cast INCLUDES gamma — no type mismatch in norm ops
- cublasTf32 flag IS wired and enabled in optimal() — standard handle has TF32
- cublasLt uses separate handle, does NOT inherit TF32 math mode
- 509 kernels/step: ~180 matmul + ~120 KV copy + ~98 auto-cast + ~30 attention + ~30 norm + ~30 rope + ~21 misc
- 49 gap matmuls × 2 casts = 98 transformAnySimpleCached; ~161 Triton-island matmuls bypass casting
- ~10 tile/step from 62 broadcast_to ops (ONNX Expand), NOT from KV repeat (eliminated by fusedGQA)

---

## Key Files

| File | Purpose |
|---|---|
| `autoregressive_decode.cu` | Decode loop |
| `FlashAttentionHelper.cu` | fusedGQADecodeKernel |
| `FlashAttentionHelper.cpp` | GQA dispatch (line 290) |
| `MmulHelper.cu` | cuBLAS/cublasLt, auto-cast logic |
| `OpCategoryTable.h` | Triton op→category mapping |
| `SectionTypeConfig.h` | Category compile/native gating |
| `TritonGraphBackend_compile.cu` | tritonIncludeTypes parsing |
| `TritonIRBuilder_kernels.cpp` | Fused emitters (rms_norm_linear:440, gatedMLP:673, twoLayerMLP:870) |
| `TritonIRBuilder_module.cpp` | MATMUL case dispatching to emitters (line 5926+) |
| `NormalizationFusionOptimizations.java` | FuseRMSNormLinearPattern (line 631 — RmsNorm only, not SkipRmsNorm) |
| `ActivationFusionOptimizations.java` | FuseGatedMLPPattern (disabled, line 277-282) |
| `BenchmarkConfig.java` | OPTIMAL config, tritonIncludeTypes (line 200) |
| `platform-tests/run-benchmark.sh` | Benchmark runner |
| `platform-tests/op-timing/OPTIMAL.csv` | Op timing data |


## 2026-04-29 19:12


---

## OPT-04 Investigation Results (2026-04-29)

### Definitive Finding: SmolDocling is natively FP32, NOT FP16

**Test**: `TestMatmulDtypeInspection` — inspects SameDiff graph pre- and post-GraphOptimizer.

**Pre-optimizer**: ALL 211 matmuls are `FLOAT × FLOAT`. 422 FP32 matmul inputs, 0 HALF.
Weights are FP32 constants. Activations are FP32 (from rms_norm, skip_rms_norm, onnx_mha, multiply).
133 cast ops exist but only 3 are non-trivial (FLOAT→BOOL, LONG→FLOAT).

**Post-optimizer**: `QuantizeConstantsToFP16` fires — converts 213 constant arrays to HALF.
BUT: SDVariable metadata is NOT updated → shape functions still see FLOAT→FLOAT.
GraphOptimizer removed 191 ops (2742→2551), reduced casts (133→2), but matmul dtype combos unchanged at graph level.

**At runtime**: MmulHelper receives FP32 activation × HALF weight → mixed-type path (`MmulHelper.cu:907-951`).
This casts only the FP32 activation to HALF (1 cast per matmul, not 2).
98 `transformAnySimpleCached`/step from nsys = 49 unique activation tensors cast (shared across Q/K/V projections via `tl_captureCastReuseA`).

### Root Cause Chain
1. ONNX model exports weights as FP32
2. ONNX import hooks (MatMul.kt, etc.) pass through FP32 as-is
3. `QuantizeConstantsToFP16` pre-casts weight ARRAYS to HALF at optimizer time
4. But does NOT update SDVariable.dataType() → shape functions compute FP32 output shapes
5. At runtime: activation (FP32 from shape fn) × weight (HALF from array) → mixed-type cast
6. MmulHelper casts activation FP32→HALF on every matmul call

### Fix Options for OPT-04
**A. Update SDVariable metadata after quantization** — make `quantizeAllToType` also call `sd.getVariable(name).setDataType(targetType)`. This propagates HALF through shape functions → matmul output becomes HALF → downstream ops receive HALF → cascade eliminates ALL runtime casts.
- Risk: HIGH — changes all downstream op dtypes. Must verify accuracy.
- Impact: Eliminates ALL 98 auto-cast kernels/step (~1.4ms).

**B. Add dtype-aware shape function to matmul** — check actual input array dtype at shape time, not just shape info dtype.
- Risk: MEDIUM — more targeted, only affects matmul output type.

**C. Skip cast when already in capture/replay** — if the cast result is cached and shape-stable, skip the cast kernel entirely.
- Risk: LOW — the cast cache already handles this during capture, but the assign kernel still fires.

### Corrected Perf Memory Facts
- SmolDocling weights are FP32 in ONNX model, pre-cast to HALF by QuantizeConstantsToFP16
- SDVariable metadata remains FLOAT after pre-cast → shape functions compute FP32 outputs
- ALL 211 matmuls see mixed-type (FP32 activation × HALF weight) at runtime
- 98 auto-cast kernels/step from 49 unique activation tensors (shared across Q/K/V)
- Previous claim "FP16 weight pre-cast INCLUDES gamma" was correct for the array, wrong for the metadata
