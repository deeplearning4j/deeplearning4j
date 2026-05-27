# PR22: Platform Tests & Benchmark Scripts — Sub-Split Plan

**Total files:** 537
**Merge layer:** 7 (last — depends on all production code PRs)
**Recommendation:** Split into 7 sub-PRs by test domain

## Sub-PR Summary

| Sub-PR | Name | Files | Description |
|---|---|---:|---|
| PR22a | DSP & SameDiff Tests | ~84 | DSP lifecycle/regression/execution tests + SameDiff autodiff |
| PR22b | LLM & VLM Tests | ~78 | Generation pipeline, decode loop, benchmark, eval, VLM tests |
| PR22c | Op Validation & Optimizer Tests | ~69 | Op validation suite, graph optimizer passes |
| PR22d | Import Tests (ONNX/GGML/Keras/TF) | ~55 | All framework import tests |
| PR22e | ND4J Core & DL4J Tests | ~146 | Linalg, buffer, shape, workspace, ops, DL4J layer tests |
| PR22f | Benchmark Scripts & Test Infrastructure | ~40 | Shell scripts, Python helpers, bin/, config, pom.xml |
| PR22g | Data Artifacts & Op Traits | ~65 | Binary shards, CSV files, op traits tests, resources |

**Total: 537 files**

---

## PR22a: DSP & SameDiff Tests (~84 files)

All DSP execution mode tests, SameDiff autodiff tests, and DSP framework tests.

### Files

| Path | Files | Contents |
|---|---:|---|
| `nd4j/autodiff/samediff/` | 68 | SameDiffTests, SameDiffTrainingTest, ControlFlowTest, Dsp*Test, serialization, pipeline, peft, training, execution/ |
| `frameworks/samediff/dsp/` | 16 | DspLifecycleValidationTest, DspSlotLifecycleAuditTest, frozen/, regression/, lifecycle/ |

### Key test classes
- `SameDiffTests` — core autodiff verification
- `DspLifecycleValidationTest` — DSP phase progression
- `DspSlotLifecycleAuditTest` — slot lifecycle audit
- `TestDspValidation` — outputAccuracy, perOpSlot, decodeStep
- `TestDspConfigurationMatrix` — 8-config sweep
- `TestDspPipelineFacets` — pipeline facet integration

### Review focus
- DSP tests exercise ALL execution modes (SLOT_BY_SLOT, AUTO, CUDA_GRAPHS, TRITON)
- No test disables DSP or uses workarounds

### ADRs
- `ADRs/0006 - Test architecture.md`

---

## PR22b: LLM & VLM Tests (~78 files)

Generation pipeline tests, decode loop regression tests, benchmark suites,
eval framework, and VLM (SmolDocling) tests.

### Files

| Path | Files | Contents |
|---|---:|---|
| `llm/` | 59 | TestGenerationPipeline*, TestNativeDecodeLoopRegression, TestDspValidation, TestMythicPdfRegression, TestPage10*, TestQuantizedKVCache, TestTokenEviction, benchmark/, eval/, pipeline/ |
| `vlm/` | 19 | TestSmolDoclingOptimizedPipeline, TestVLMDecodeQuality, TestVLMGenerationPipeline, model/, preprocessing/ |

### Key test classes
- `TestGenerationPipeline*` — main decode loop correctness
- `TestNativeDecodeLoopRegression` — native decode regression
- `TestSmolDoclingOptimizedPipeline` — VLM end-to-end
- `TestLLMBenchmarkSuite` — multi-model benchmarks
- `EvalFrameworkTest` — eval harness
- `TestMultiModelPipeline*` — multi-model pipeline

### Review focus
- Token-level accuracy validation (match rate > 90%)
- KV cache lifecycle — no stale/leaked caches
- VLM vision encoder + text decoder integration

---

## PR22c: Op Validation & Optimizer Tests (~69 files)

Op-level validation (shape, gradient, output) and graph optimizer pass tests.

### Files

| Path | Files | Contents |
|---|---:|---|
| `nd4j/autodiff/opvalidation/` | 36 | BaseOpValidation, TestAttentionOpValidation, TestLayerOpValidation, TestMiscOpValidation, TestShapeOpValidation, TestTransformOpValidation, TestPeftOpValidation, TestMixedPrecisionGradientChecks |
| `nd4j/autodiff/optimization/` | 33 | TestGraphOptimizerPasses, TestAttentionFusionOptimization, TestHorizontalFusion, TestStrengthReduction, + util |

### Key test classes
- `BaseOpValidation` — op output correctness across types
- `TestGraphOptimizerPasses` — all optimizer passes
- `TestAttentionFusionOptimization` — SDPA fusion
- `TestHorizontalFusion` — horizontal op fusion

### Review focus
- Gradient checks — finite differences vs. autodiff
- Optimizer tests must exercise DSP execution (not just structural checks)
- Mixed precision gradient checks — FP16/FP32 boundary

---

## PR22d: Import Tests (ONNX/GGML/Keras/TF) (~55 files)

All model import test suites across frameworks.

### Files

| Path | Files | Contents |
|---|---:|---|
| `frameworkimport/` | 35 | Keras (e2e/, layers/), TensorFlow, ONNX (Java + Kotlin) |
| `ggml/` | 20 | GGMLModelImportTest, GGMLModelExportTest, RoundTripTest, TestGGMLDequantize, architecture, format, quantization |

### Key test classes
- `GGMLModelImportTest` — GGML/GGUF round-trip
- `KerasModelImportTest` — Keras HDF5 import
- `TestOnnxConverter` — ONNX model conversion
- `TestMicrosoftOnnxOps` — Microsoft ONNX extensions

### Review focus
- ONNX Gather with 2D constant indices (known issue)
- GGML quantization accuracy (Q4_K_M, IQ4_XS)
- Keras shape handling for all layer types

---

## PR22e: ND4J Core & DL4J Tests (~146 files)

ND4J core tests (linalg, buffer, shape, workspace, ops, mixed precision)
and DL4J layer tests.

### Files

| Path | Files | Contents |
|---|---:|---|
| `nd4j/linalg/` | 126 | Nd4jTestsC, buffer tests, shape tests, workspace tests, ops tests, broadcast, RNG, mixed precision, factory, framework, dataset, custom |
| `dl4jcore/` | 20 | Constraint, graph, convolution, embedding, normalization, multilayer, layers |

### Key test classes
- `Nd4jTestsC` — core ND4J ops and array operations
- `CudaMemoryAllocationFailoverTest` — CUDA OOM failover
- `TestWorkspaces` — workspace allocation/deallocation
- `ComputationGraphTest` — DL4J graph model

### Review focus
- Buffer tests — no buffer overruns
- Workspace lifecycle — proper open/close/spill
- CUDA view tests — stale device buffer detection

---

## PR22f: Benchmark Scripts & Test Infrastructure (~40 files)

Shell scripts, Python helpers, test runner infrastructure, and build config.

### Files

| Path | Files | Contents |
|---|---:|---|
| Shell scripts | 25 | run-benchmark.sh, run-validation.sh, run-dsp-matrix.sh, run-llm-benchmarks.sh, run-all-tests.sh, etc. |
| Python helpers | 10 | compare_autoreg.py, compare_hf_*.py, extract_intermediates.py, inspect_onnx_model.py, etc. |
| `bin/` | 4 | java wrapper, nvprof-java.sh, valgrind suppression |
| `pom.xml` | 1 | Test module build config (surefire, env var wiring) |

### Review focus
- Benchmark scripts use correct flags
- Python helpers produce comparable output formats
- pom.xml surefire configuration — env var wiring for DSP properties

### ADRs
- `ADRs/0010 - Test module consolidation.md`

---

## PR22g: Data Artifacts & Op Traits (~65 files)

Binary test data (model shards, FlatBuffer inputs), CSV timing data,
op trait frozen shape tests, test resources, and metadata.

### Files

| Path | Files | Contents |
|---|---:|---|
| `*.sdnb` | 6 | large-samediff-model.shard0-5.sdnb (model data shards) |
| `op-timing/*.csv` | 6+ | Op timing measurement data |
| `*.fb` | 1 | tmp-bert-input.fb (FlatBuffer test input) |
| `optraits/` | 4 | AllRegisteredOpsFrozenShapeTest, OpCategoryFrozenShapeTest, OpTraitTableComprehensiveTest, OpTraitTableEnumerationTest |
| `src/test/resources/` | 3 | log4j, logback configs |
| `nd4j/libnd4j/` | 6 | Native GTest wrapper classes |
| `nd4j/linalg/mixed/` | 6 | Mixed DSP/linalg tests |
| `META-INF/` | 1 | maven plugin.xml |
| Other data/config | ~30 | datavecTests, triton journal, remaining resources |

### Review focus
- Binary shards should be in Git LFS (check file sizes)
- Op traits tests — verify frozen shape baselines are up to date
- Native test wrappers — GTest suite discovery

### ADRs
- `ADRs/0056 - Libnd4j Native Test Integration.md`

---

## Recommended Merge Sequence

```
PR22f (scripts/infra) ──→  PR22e (ND4J/DL4J core)  ──→  PR22a (DSP/SameDiff)
                      ──→  PR22c (op validation)    ──→  PR22b (LLM/VLM)
                      ──→  PR22d (import)
                      ──→  PR22g (data/traits)
```

PR22f (infrastructure) should merge first — scripts and pom.xml are needed by all tests.
PR22e (core tests) next — establishes baseline correctness.
PR22c (op validation) provides op-level guarantees needed by higher-level tests.
PR22a/b/d/g can merge in any order after that.
