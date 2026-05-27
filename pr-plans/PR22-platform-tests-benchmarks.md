# PR22: Platform Tests & Benchmark Scripts

**Estimated files:** ~537
**Merge layer:** 7 (last — depends on all production code PRs)
**Complexity:** Medium
**Reviewers:** QA/test team

## Description

All platform tests and benchmark scripts. This is the validation layer
for everything in the other PRs. Should merge last.

## Sub-Split Plan → [`PR22-sub-split.md`](PR22-sub-split.md)

Recommended 7-way split (verified against actual diff):

| Sub-PR | Name | Files |
|---|---|---:|
| PR22a | DSP & SameDiff Tests | ~84 |
| PR22b | LLM & VLM Tests | ~78 |
| PR22c | Op Validation & Optimizer Tests | ~69 |
| PR22d | Import Tests (ONNX/GGML/Keras/TF) | ~55 |
| PR22e | ND4J Core & DL4J Tests | ~146 |
| PR22f | Benchmark Scripts & Test Infrastructure | ~40 |
| PR22g | Data Artifacts & Op Traits | ~65 |

Merge order: PR22f → PR22e → PR22c → PR22a/b/d/g (parallel)

## File Categories

### DSP tests (~47)
- `frameworks/samediff/dsp/` — 5 test classes
- `frameworks/samediff/dsp/frozen/` — 6 test classes
- `frameworks/samediff/dsp/lifecycle/` — 1 test class
- `frameworks/samediff/dsp/regression/` — 4 test classes
- `nd4j/autodiff/samediff/Dsp*` — 17 test classes
- `nd4j/autodiff/samediff/execution/` — 3 test classes
- `nd4j/autodiff/samediff/dsp/` — 6 test classes
- `nd4j/linalg/mixed/Dsp*` — 3 test classes
- `nd4j/linalg/api/GraphExecutionModeTest`

### Op validation tests (~35)
- `nd4j/autodiff/opvalidation/` — 35 test classes
  - BaseOpValidation, TestAttentionOpValidation, TestLayerOpValidation,
  - TestMiscOpValidation, TestShapeOpValidation, TestTransformOpValidation,
  - TestPeftOpValidation, TestMixedPrecisionGradientChecks, etc.

### Graph optimizer tests (~34)
- `nd4j/autodiff/optimization/` — 32 test classes + util
  - TestGraphOptimizerPasses, TestAttentionFusionOptimization,
  - TestHorizontalFusion, TestStrengthReduction, etc.
- `nd4j/autodiff/samediff/TestGraphOptimizerFusions.java`
- `nd4j/autodiff/samediff/TestGraphOptimizerOnSmolDocling.java`

### LLM tests (~60)
- `llm/generation/` — 40+ test classes
  - TestGenerationPipeline*, TestNativeDecodeLoopRegression,
  - TestDspValidation, TestMythicPdfRegression, TestPage10* series,
  - TestQuantizedKVCache, TestTokenEviction, etc.
- `llm/benchmark/` — TestLLMBenchmarkSuite, TestQwenLayerDiagnostics
- `llm/eval/` — EvalFrameworkTest
- `llm/pipeline/` — TestMultiModelPipeline*

### VLM tests (~18)
- `vlm/` — TestSmolDoclingOptimizedPipeline, TestVLMDecodeQuality,
  TestVLMGenerationPipeline, TestVLMModelImportPipeline, etc.
- `vlm/model/` — EmbeddingMergerTest, PipelinedVisionEncoderTest
- `vlm/preprocessing/` — ImageTilerTest, ImagePromptBuilderTest

### GGML tests (~18)
- `ggml/` — GGMLModelImportTest, GGMLModelExportTest, RoundTripTest,
  TestGGMLDequantize, architecture tests, format tests, quantization tests

### ONNX tests (~14)
- Java: MhaEmptyPastKvIsolationTest, OnnxOpTests, TestOnnxConverter, etc.
- Kotlin: TestOnnxFrameworkImporter, TestMicrosoftOnnxOps, etc.

### Keras import tests (~25)
- `frameworkimport/keras/` — KerasModelImportTest, e2e tests, layer tests
- `frameworkimport/keras/e2e/` — BatchNorm, ConvLSTM, ResNetImport, etc.

### TensorFlow import tests (~10)
- `frameworkimport/tensorflow/` — Java + Kotlin tests

### DL4J core tests (~18)
- `dl4jcore/` — constraint, graph, convolution, embedding, normalization tests

### ND4J core tests (~75)
- `nd4j/linalg/` — Nd4jTestsC, buffer tests, shape tests, workspace tests,
  ops tests, broadcast, RNG, mixed precision, etc.
- `nd4j/linalg/api/buffer/CudaMemoryAllocationFailoverTest.java`
- `nd4j/libnd4j/` — 6 native test wrappers

### Op traits tests (~4)
- `optraits/` — AllRegisteredOpsFrozenShapeTest, OpCategoryFrozenShapeTest,
  OpTraitTableComprehensiveTest, OpTraitTableEnumerationTest

### SameDiff general tests (~20)
- SameDiffTests, SameDiffTrainingTest, ControlFlowTest, serialization,
  pipeline, peft, training, TTS, VLM subpackage tests

### Benchmark scripts (~24)
- `run-all-tests.sh` through `run-zoo-tests.sh`
- `run-benchmark.sh`, `run-benchmark-cpu.sh`
- `run-dsp-matrix.sh`, `run-validation.sh`
- `run-llm-benchmarks.sh`
- `bootstrap-onnx.sh`

### Test infrastructure (~20)
- `extensions/` — BackendCheckerExtension, BackendTest, MultiBackendTest*
- `bin/` — java wrapper, nvprof-java.sh, valgrind suppression
- `src/test/resources/` — log4j, logback configs
- `META-INF/maven/plugin.xml`

### Python helper scripts (~7)
- `compare_autoreg.py`, `compare_hf_*.py`, `extract_intermediates.py`
- `inspect_onnx_model.py`, `save_python_kv.py`
- `verify_decoder_internals.py`, `verify_onnx_model.py`

### Data artifacts (~20)
- `large-samediff-model.shard*-of-6.sdnb` (6 shards)
- `op-timing/*.csv` (6 files)
- `datavecTests/seg0/*.dat` (4 files)
- `tmp-bert-input.fb`
- `triton-optimization-journal.md`
- Other data/config files

### ADRs (2 — only those actually changed in the diff)
- `ADRs/0056 - Libnd4j Native Test Integration.md` — Run libnd4j GTest suites as JUnit 5 DynamicTest via Maven Surefire
- `ADRs/0075 - Libnd4j Native Test Integration.md` — Duplicate of 0056, should be deleted

## Review Focus

- DSP tests — verify they exercise all execution modes
- LLM/VLM tests — correctness criteria (token matching, accuracy)
- Benchmark scripts — ensure they use correct flags
- No test should disable DSP or use workarounds
