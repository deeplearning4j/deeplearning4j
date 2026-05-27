# PR Split Master Plan

**Branch:** `ag_new_release_updates_2`
**Total changed files:** ~4814
**Date:** 2026-05-23

## Existing Open PRs (STALE — from old snapshot b5893454)

These PRs cover only ~179 files from Feb 26 and are severely outdated.
They should be **closed and replaced** by this new split.

| PR | Title | Files | Status |
|---|---|---|---|
| #10418 | ADR/Docs Snapshot | 5 | STALE |
| #10419 | Build/Packaging Snapshot | 37 | STALE |
| #10420 | Triton Backend Snapshot | 43 | STALE |
| #10421 | DSP Runtime Snapshot | 57 | STALE |
| #10422 | Java Op API Snapshot | 23 | STALE |
| #10423 | ONNX Runtime/VLM Snapshot | 4 | STALE |
| #10424 | Platform Tests Snapshot | 10 | STALE |
| #10373 | Update loop compilation units | ? | STALE |

## New PR Split (22 PRs)

### Merge Order & Dependencies

PRs should merge roughly in this order (earlier PRs are dependencies):

```
Layer 0 (no deps):     PR01, PR02, PR20
Layer 1 (build/infra): PR03, PR04
Layer 2 (native core): PR05, PR06, PR07
Layer 3 (native feat): PR08, PR09, PR10, PR11
Layer 4 (java core):   PR12, PR13, PR14, PR15
Layer 5 (java feat):   PR16
Layer 6 (import/gen):  PR17, PR18, PR19, PR21
Layer 7 (validation):  PR22
```

### Summary Table

ADRs are distributed to their feature PRs. Only ADRs **actually changed in the diff**
are counted (many pre-existing ADRs on master are unchanged).

| PR | Name | Files | ADRs | Layer | Complexity |
|---|---|---:|---:|---|---|
| PR01 | Build System & CI | ~155 | 6 | 0 | Low |
| PR02 | System/Environment & Platform Macros | ~32 | 1 | 0 | Medium |
| PR03 | FlatBuffers Schema & Generated Code | ~324 | 1 | 1 | Low (mostly generated) |
| PR04 | Memory Management & Array Infrastructure | ~64 | 6 | 1 | High |
| PR05 | Legacy/Loops/NativeOps | ~176 | 0 | 2 | Medium |
| PR06 | Helpers & Utilities (C++) | ~87 | 2 | 2 | Medium |
| PR07 | C++ Op Implementations | ~829 | 3 | 2 | Medium (volume) |
| PR08 | C++ Platform Backends | ~467 | 7 | 3 | Medium (volume) |
| PR09 | DSP/Graph Execution (C++) | ~125 | 10 | 3 | High |
| PR10 | Triton/NVRTC/PTX Backend (C++) | ~47 | 1 | 3 | High |
| PR11 | DSP Runtime SDK | ~45 | 2 | 3 | Low |
| PR12 | ND4J Java API & Core Infrastructure | ~380 | 0 | 4 | High |
| PR13 | Java Op Definitions | ~309 | 0 | 4 | Medium |
| PR14 | Java Backend Impls (CUDA + CPU) | ~65 | 0 | 4 | High |
| PR15 | SameDiff Core & Training | ~114 | 3 | 4 | High |
| PR16 | DSP Runtime & Graph Optimizer (Java) | ~65 | 2 | 5 | High |
| PR17 | ONNX Import | ~175 | 0 | 6 | Medium |
| PR18 | GGML Import | ~86 | 3 | 6 | Medium |
| PR19 | LLM/VLM Generation Pipeline | ~217 | 1 | 6 | High |
| PR20 | Cross-Cutting ADRs & Documentation | ~65 | 5 | 0 | Low |
| PR21 | Miscellaneous Modules | ~172 | 1 | 6 | Low-Medium |
| PR22 | Platform Tests & Benchmark Scripts | ~537 | 2 | 7 | Medium |

**Total ADRs in the diff:** 56 (52 in ADRs/ + 4 root-level) distributed across 17 PRs.
PRs with no changed ADRs: PR05, PR12, PR13, PR14, PR17.

### ADR Distribution Summary

| PR | ADR Count | ADR Topics |
|---|---:|---|
| PR01 | 6 | Build: classifiers, type promotion, CUDA arch, Android, template instantiation |
| PR02 | 1 | CUDA macro standardization |
| PR03 | 1 | FlatBuffers upgrade (SDNB/SDZ format) |
| PR04 | 6 | Memory: offset, shape trie, CUDA pool, array cache, multi-GPU, GC |
| PR06 | 2 | Kernel selection, op timing tracker |
| PR07 | 3 | SDPA optimization, LoRA fused matmul, OCR ops |
| PR08 | 7 | Multi-backend kernel/op selection, TPU, MLIR, ZLUDA, Hexagon, LlamaCpp classifiers |
| PR09 | 10 | DSP execution, diagnostics, refactoring, 5 correctness fixes, CUDA graph replay, device transfer |
| PR10 | 1 | Triton graph backend |
| PR11 | 2 | DSP SDK, SDX serving protocol |
| PR15 | 3 | SameDiff mixed precision, execution framework (duplicate 0057), PEFT |
| PR16 | 2 | Java-side shape inference, InferenceSession optimization |
| PR18 | 3 | GGML import, quantization, architecture detection |
| PR19 | 1 | VLM inference pipeline |
| PR20 | 5 | Debugging/profiling, workspaces, namespace migration, ADR index |
| PR21 | 1 | OmniHub model repository |
| PR22 | 2 | Test architecture, test consolidation |

### Large PR Sub-Splits

Three PRs are large enough to warrant further splitting. Detailed sub-split
plans are in separate files.

#### PR07 (829 files) → 5 sub-PRs — [`PR07-sub-split.md`](PR07-sub-split.md)

| Sub-PR | Name | Files |
|---|---|---:|
| PR07e | Op Infrastructure & Registry | ~92 |
| PR07a | NN & Domain-Specific Generic Ops | ~155 |
| PR07b | Math & Data Generic Ops | ~147 |
| PR07c | Structural & Utility Generic Ops | ~103 |
| PR07d | Op Helpers (Headers + CPU + CUDA) | ~332 |

Merge order: PR07e → PR07a/b/c (parallel) → PR07d

#### PR08 (467 files) → 4 sub-PRs — [`PR08-sub-split.md`](PR08-sub-split.md)

| Sub-PR | Name | Files |
|---|---|---:|
| PR08a | ARM + Apple Backends | ~173 |
| PR08b | Intel/x86 Backends | ~72 |
| PR08c | CUDA Ecosystem Backends | ~96 |
| PR08d | Experimental Backends + Graph Backends | ~86 |

Merge order: PR08a/b/c (parallel) → PR08d

#### PR22 (537 files) → 7 sub-PRs — [`PR22-sub-split.md`](PR22-sub-split.md)

| Sub-PR | Name | Files |
|---|---|---:|
| PR22f | Benchmark Scripts & Test Infrastructure | ~40 |
| PR22e | ND4J Core & DL4J Tests | ~146 |
| PR22c | Op Validation & Optimizer Tests | ~69 |
| PR22a | DSP & SameDiff Tests | ~84 |
| PR22b | LLM & VLM Tests | ~78 |
| PR22d | Import Tests (ONNX/GGML/Keras/TF) | ~55 |
| PR22g | Data Artifacts & Op Traits | ~65 |

Merge order: PR22f → PR22e → PR22c → PR22a/b/d/g (parallel)

### Full PR Count

With sub-splitting: **22 top-level PRs** expanding to **30 leaf PRs** (16 sub-PRs replace 3 parents).

See individual `pr-plans/PRNN-*.md` files for complete file lists, ADR assignments, and descriptions.
