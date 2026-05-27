# PR07: C++ Op Implementations

**Estimated files:** ~829
**Merge layer:** 2
**Complexity:** Medium (high volume, but most changes are mechanical)
**Reviewers:** Core C++ team

## Description

All op implementations in libnd4j: generic op definitions, op helper CPU/CUDA
kernels, op headers, OpTraitTable, gemm, and op macros. This is the largest
single PR and **may need sub-splitting** by op category.

## Sub-Split Plan → [`PR07-sub-split.md`](PR07-sub-split.md)

Recommended 5-way split (verified against actual diff):

| Sub-PR | Name | Files |
|---|---|---:|
| PR07e | Op Infrastructure & Registry | ~92 |
| PR07a | NN & Domain-Specific Generic Ops | ~155 |
| PR07b | Math & Data Generic Ops | ~147 |
| PR07c | Structural & Utility Generic Ops | ~103 |
| PR07d | Op Helpers (Headers + CPU + CUDA) | ~332 |

Merge order: PR07e (infra) → PR07a/b/c (parallel) → PR07d (helpers)

## File Categories

### Generic op definitions (~448 files)
All files under `libnd4j/include/ops/declarable/generic/`:
- `audio/` — audio ops
- `bitwise/` — bitwise ops
- `blas/` — BLAS wrappers
- `boolean/` — boolean ops
- `broadcastable/` — element-wise broadcast ops
- `compat/` — compatibility ops
- `datatypes/` — type conversion ops
- `decoder/` — decoder/attention ops
- `grad/` — gradient ops
- `images/` — image processing ops
- `kernels/` — kernel ops
- `linalg/` — linear algebra ops
- `list/` — list ops
- `loss/` — loss function ops
- `nlp/` — NLP ops
- `nn/` — neural network ops (largest category)
- `parity_ops/` — parity/misc ops
- `random/` — random ops
- `reduce/` — reduction ops
- `shape/` — shape manipulation ops
- `signal/` — signal processing ops
- `strings/` — string ops
- `tensor/` — tensor ops
- `transforms/` — transform ops
- `tsne/` — t-SNE ops
- `updaters/` — optimizer updater ops
- `util/` — utility ops

### Op helper headers (~60+)
- `libnd4j/include/ops/declarable/helpers/*.h`

### Op helper CPU implementations (~130)
- `libnd4j/include/ops/declarable/helpers/cpu/*.cpp`

### Op helper CUDA implementations (~140)
- `libnd4j/include/ops/declarable/helpers/cuda/*.cu`

### Op helper shared implementations (~8)
- `libnd4j/include/ops/declarable/helpers/impl/*.cpp`

### Op declaration headers (~9)
- `libnd4j/include/ops/declarable/headers/*.h`

### Op infrastructure (~21)
- `libnd4j/include/ops/declarable/impl/*.cpp`

### Top-level op headers (~12)
- `libnd4j/include/ops/declarable/*.h` (DeclarableOp, CustomOperations, etc.)

### Op registry & macros (~14)
- `libnd4j/include/ops/impl/BroadcastOpsTuple.cpp`
- `libnd4j/include/ops/impl/OpTraitTable.cpp`
- `libnd4j/include/ops/OpTraitTable.h`
- `libnd4j/include/ops/impl/gemm.cpp`
- `libnd4j/include/ops/gemm.h`
- `libnd4j/include/ops/op_macros*.h` (includes op_macros_binary.h, op_macros_index_reduce.h, op_macros_special.h)
- `libnd4j/include/ops/op_types.h`
- `libnd4j/include/ops/ops.h`
- `libnd4j/include/ops/special_random_ops.h`
- `libnd4j/include/ops/impl/specials_double.hpp`
- `libnd4j/include/ops/impl/specials_single.hpp`

### Performance include (1)
- `libnd4j/include/performance/generated/include_ops.h`

### ADRs (3)
- `ADRs/0067 - Scaled Dot-Product Attention Optimization.md` — Fused Q@K^T, softmax, attn@V into single kernel via oneDNN/cuDNN
- `ADRs/0068 - LoRA Fused MatMul.md` — Fused four-step LoRA computation into single op
- `ADRs/0069 - OCR Operations.md` — Native OCR engine using SameDiff-executed ONNX model

## Review Focus

- OpTraitTable changes affect Triton dispatch — verify mappability
- New op registrations must have both CPU and CUDA implementations
- Template instantiation patterns must follow BUILD_SINGLE_TEMPLATE convention
