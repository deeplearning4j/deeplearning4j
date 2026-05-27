# PR07: C++ Op Implementations — Sub-Split Plan

**Total files:** 829 (excluding platform backends in PR08)
**Merge layer:** 2
**Recommendation:** Split into 5 sub-PRs by functional domain

## Sub-PR Summary

| Sub-PR | Name | Files | Description |
|---|---|---:|---|
| PR07a | NN & Domain-Specific Generic Ops | ~155 | nn/, decoder/, audio/, signal/, kernels/, images/, nlp/, strings/ |
| PR07b | Math & Data Generic Ops | ~147 | broadcastable/, reduce/, transforms/, boolean/, linalg/, loss/, random/ |
| PR07c | Structural & Utility Generic Ops | ~103 | shape/, parity_ops/, list/, tensor/, datatypes/, bitwise/, blas/, compat/, updaters/, tsne/, util/, grad/ |
| PR07d | Op Helpers (Headers + CPU + CUDA) | ~332 | helpers/*.h, helpers/cpu/*.cpp, helpers/cuda/*.cu, helpers/impl/*.cpp |
| PR07e | Op Infrastructure & Registry | ~92 | declarable/*.h, declarable/headers/, declarable/impl/, ops/*.cpp, ops/*.h |

**Total: 829 files**

---

## PR07a: NN & Domain-Specific Generic Ops (~155 files)

The largest and most important op category — neural network layers, attention,
audio processing, signal processing, and specialized domain ops.

### Files (from `libnd4j/include/ops/declarable/generic/`)

| Directory | Files | Contents |
|---|---:|---|
| `nn/` | 117 | Conv, pool, batch norm, attention, SDPA, LoRA, RoPE, RMSNorm, GeLU, etc. |
| `audio/` | 14 | MFCC, spectrogram, audio processing ops |
| `images/` | 18 | Image resize, crop, adjust, draw, HSV/RGB conversion |
| `signal/` | 3 | FFT, IFFT, signal processing |
| `decoder/` | 1 | Autoregressive decoder ops |
| `kernels/` | 1 | Custom kernel ops |
| `nlp/` | 2 | NLP-specific ops |
| `strings/` | 1 | String ops |

### Merge order
Can merge independently from PR07b-e (no cross-references between generic/ subdirs).

### Review focus
- Attention/SDPA ops — verify Triton mappability in OpTraitTable
- LoRA fused matmul — check type handling
- RoPE — verify FP16 path

### ADRs
- `ADRs/0067 - Scaled Dot-Product Attention Optimization.md`
- `ADRs/0068 - LoRA Fused MatMul.md`
- `ADRs/0069 - OCR Operations.md`

---

## PR07b: Math & Data Generic Ops (~147 files)

Element-wise, reduction, transform, and mathematical ops — the core
numerical computation primitives.

### Files (from `libnd4j/include/ops/declarable/generic/`)

| Directory | Files | Contents |
|---|---:|---|
| `transforms/` | 47 | Abs, clip, log, exp, sigmoid, softmax, gather, scatter, etc. |
| `broadcastable/` | 34 | Add, sub, mul, div, max, min, pow, boolean comparisons |
| `linalg/` | 30 | MatMul, SVD, Cholesky, QR, tri, eye, cross, det, etc. |
| `reduce/` | 18 | ReduceSum, ReduceMean, ReduceMax, ReduceMin, argmax, etc. |
| `loss/` | 18 | Cross-entropy, MSE, hinge, huber, cosine distance, etc. |

### Merge order
Independent from other PR07 sub-PRs.

### Review focus
- Broadcastable ops — stride/shape handling for views
- Reduce ops — verify ALL rank combinations
- Loss ops — gradient correctness

---

## PR07c: Structural & Utility Generic Ops (~103 files)

Shape manipulation, type conversion, data structure ops, and miscellaneous
utility operations.

### Files (from `libnd4j/include/ops/declarable/generic/`)

| Directory | Files | Contents |
|---|---:|---|
| `parity_ops/` | 31 | Where, unique, top_k, segment_*, stack, unstack, etc. |
| `shape/` | 16 | Reshape, transpose, permute, expand_dims, squeeze, tile, etc. |
| `boolean/` | 14 | Boolean conditional ops |
| `list/` | 11 | ArrayList ops (read, write, size, gather, scatter) |
| `datatypes/` | 10 | Type cast ops |
| `bitwise/` | 9 | Bitwise and, or, xor, shift ops |
| `random/` | 9 | RandomNormal, RandomUniform, Bernoulli, etc. |
| `tensor/` | 9 | Tensor manipulation ops |
| `updaters/` | 10 | SGD, Adam, Nesterov, RMSProp, AdaGrad, etc. |
| `tsne/` | 4 | t-SNE embedding ops |
| `blas/` | 4 | BLAS wrapper ops |
| `compat/` | 2 | Compatibility wrapper ops |
| `util/` | 2 | Utility ops |
| `grad/` | 1 | Gradient ops |

### Merge order
Independent from other PR07 sub-PRs.

### Review focus
- Shape ops — verify rank/ordering correctness
- Random ops — verify reproducibility with seeds

---

## PR07d: Op Helpers (Headers + CPU + CUDA) (~332 files)

All op helper implementations — the actual kernel code behind the generic
op definitions. Headers, CPU, CUDA, and shared implementations.

### Files

| Path | Files | Contents |
|---|---:|---|
| `helpers/*.h` | 63 | Helper function declarations |
| `helpers/cpu/*.cpp` | 122 | CPU kernel implementations |
| `helpers/cuda/*.cu` | 139 | CUDA kernel implementations |
| `helpers/impl/*.cpp` | 8 | Shared (platform-agnostic) helper implementations |

### Merge order
**Should merge WITH or AFTER PR07a-c** — the generic ops reference these helpers.
Helper headers define the interfaces that generic ops call.

### Review focus
- Every CPU helper must have a corresponding CUDA helper (and vice versa)
- Template instantiation: BUILD_SINGLE_TEMPLATE + BUILD_SINGLE_SELECTOR
- CUDA kernels: getLaunchDims(), SD_KERNEL/SD_HOST_DEVICE macros
- No raw __host__/__device__/__global__ annotations

---

## PR07e: Op Infrastructure & Registry (~92 files)

Op framework infrastructure: DeclarableOp base classes, op registration,
CustomOperations.h (the master op registry), declaration headers,
OpTraitTable (Triton dispatch SSOT), gemm, and op macros.

### Files

| Path | Files | Contents |
|---|---:|---|
| `declarable/*.h` | 16 | DeclarableOp.h, CustomOperations.h, OpRegistrator.h, PlatformHelper.h, KernelManager.h, MultiPlatformDispatcher.h, Legacy*Op.h |
| `declarable/headers/*.h` | 10 | audio.h, broadcastable.h, images.h, llm.h, etc. — forward declarations |
| `declarable/impl/*.cpp` | 21 | DeclarableOp.cpp, OpRegistrator.cpp, etc. — implementations |
| `ops/*.h` | ~20 | ops.h, op_macros.h, op_types.h, OpTraitTable.h, gemm.h, special_random_ops.h |
| `ops/*.cpp` | ~25 | OpTraitTable.cpp, gemm.cpp, BroadcastOpsTuple.cpp, etc. |

### Merge order
**Must merge BEFORE PR07a-d** — all ops depend on this infrastructure.
This is the foundation layer.

### Review focus
- **OpTraitTable.cpp** — this is the SSOT for Triton dispatch. Any change affects which ops can be Triton-compiled.
- CustomOperations.h — master op registry. Missing entries = ops silently unavailable.
- Op macros — any change cascades to all ops.

---

## Recommended Merge Sequence

```
PR07e (infra)  ──→  PR07a (nn/domain)  ──→  PR07d (helpers)
               ──→  PR07b (math/data)  ──→
               ──→  PR07c (structural) ──→
```

PR07e must merge first (infrastructure). PR07a/b/c can merge in parallel.
PR07d (helpers) should merge last since helpers implement what the generic ops declare.
