# ADR 0100 - SameDiff Graph Optimizer

## Status
Implemented

Proposed by: Adam Gibson (February 2026)

## Context

SameDiff graphs imported from ONNX or GGML contain many redundant operations: identity casts, decomposed normalization chains, unused variables, algebraically simplifiable expressions, and unfused patterns that could be combined into single ops. These redundancies increase execution time (more kernel launches, more memory round-trips) and prevent DSP from effectively capturing CUDA graphs or compiling Triton segments.

No prior ADR documents the graph optimization framework.

## Decision

Implement a multi-pass `GraphOptimizer` that rewrites SameDiff graphs before DSP compilation.

### Architecture

`GraphOptimizer` runs an `OptimizerSet` of named passes in a fixed-point loop (up to `nd4j.optimizer.maxIterations`, default 3), stopping early if no pass applied changes in an iteration. Graph-level dead code elimination (backward reachability from outputs) runs before any passes.

**Location**: `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/optimize/`

Individual passes can be skipped via `nd4j.optimizer.skip=PassName1,PassName2`. The optimizer is enabled by default (`nd4j.optimizer.enabled=true`).

### Pass Categories and Ordering

Passes execute in the order listed. Ordering matters — fusion passes depend on earlier simplification passes having canonicalized the graph.

#### Simplification Passes

| Pass | What it does |
|---|---|
| `UnusedFunctionOptimizations` | Dead code elimination (runs first and last) |
| `ConstantFunctionOptimizations` | Constant folding |
| `BroadcastEliminationOptimizations` | Redundant broadcasts, double negation, commutative canonicalization |
| `ReorderingOptimizations` | Constant reassociation, double transpose elimination |
| `AlgebraicOptimizations` | x+0→x, x\*1→x, x\*0→0 |
| `PeepholeOptimizations` | Idempotent ops, inverse pairs, negation propagation |
| `ArithmeticChainOptimizations` | Fold add/mul chains with constants |
| `StrengthReductionOptimizations` | pow(x,2)→square, div(x,c)→mul(x,1/c) |
| `IdentityFunctionOptimizations` | Remove identity/no-op functions |
| `ConcatSplitOptimizations` | Flatten nested concat, eliminate concat-split pairs |
| `SelectWhereOptimizations` | Simplify select/where with constant conditions |
| `RedundancyEliminationOptimizations` | Single-input concat, full-extent slice, identity gather |
| `ShapeFunctionOptimizations` | Static shape resolution |
| `CommonSubexpressionElimination` | Deduplicate identical ops |

#### Fusion Passes

| Pass | What it does |
|---|---|
| `AttentionFusionOptimizations` | Fuse Q/K/V attention patterns into fused SDPA (must precede HorizontalFusion) |
| `HorizontalFusionOptimizations` | Fuse parallel matmuls sharing the same input |
| `MatMulChainOptimizations` | Fold constant matmul chains, absorb transposes |
| `ActivationFusionOptimizations` | sigmoid(x)\*x → swish, SwiGLU detection |
| `NormalizationFusionOptimizations` | RMSNorm decomposition → single `rms_norm` op |
| `GatedDeltaNetFusionOptimizations` | Gated Delta Network pattern fusion |
| `LinearFusionOptimizations` | Linear layer pattern fusion |

#### Backend and Quantization Passes

| Pass | What it does |
|---|---|
| `RematerializationOptimizations` | Duplicate cheap ops to shorten live ranges |
| `QuantizationOptimizations` | Redundant cast removal and configurable low-precision weight storage (FP16 by default) |
| `CuDNNFunctionOptimizations` | Route patterns to cuDNN-backed ops |

### Integration with DSP

The optimizer runs during `DynamicShapePlanCompiler.compile()` before the DAG is converted to a DSP plan. Optimized graphs produce:
- Fewer slots (fewer kernel launches)
- More fusible segments (better Triton coverage)
- Fewer gap ops (more of the plan captured in CUDA graphs)

### Configuration

| Property | Default | Purpose |
|---|---|---|
| `nd4j.optimizer.enabled` | `true` | Enable/disable optimizer |
| `nd4j.optimizer.maxIterations` | `3` | Fixed-point iteration limit |
| `nd4j.optimizer.skip` | (none) | Comma-separated pass names to skip |
| `nd4j.optimizer.weightDtype` | `fp16` | Weight storage policy: `fp32`, `fp16`, `bf16`, `fp8`, `fp8_e5m2`, `int8`, or `int4` |
| `nd4j.optimizer.fp16` | `true` | Legacy compatibility flag used only when `weightDtype` is unset; `false` selects FP32 |
| `nd4j.optimizer.bf16` | `false` | Legacy compatibility flag used only when `weightDtype` is unset |

INT8 and INT4 are packed execution policies rather than dense integer casts. They require an importer such as GGUF to preserve compatible quantized weights and attach quantized-matmul metadata. The optimizer rejects eligible dense FP32 weights under these policies instead of silently changing representation or precision.

## Consequences

- Imported ONNX/GGML graphs are significantly simplified before execution (e.g., LLaMA casts reduced from 668 to 108)
- Fusion passes expose larger fusible segments to Triton and CUDA graph capture
- Fixed-point iteration catches cross-pass optimization opportunities
- Per-pass skip mechanism enables debugging which pass causes a regression

## Related ADRs

- [0061](0061%20-%20DynamicShapePlan%20Execution.md) — DSP compilation consumes the optimized graph
- [0071](0071%20-%20Triton%20Graph%20Backend.md) — Triton benefits from larger fusible segments
- [0097](0097%20-%20Decode%20Path%20Performance%20Optimizations.md) — fused ops produced by normalization/activation fusion passes
