# PR Split Plan

## Scope

- Source branch: `ag_new_release_updates_2`
- Snapshot basis: local changes + commits ahead of `origin/ag_new_release_updates_2`
- Excluded from split/commit by default: `sample-compilation.txt`, `session-openclaw.md` (session/log scratch files)

## Existing Open PRs Referenced

- #10229 (`ag_new_release_updates_2` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10229
- #10370 (`pr/adrs-documentation` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10370
- #10371 (`pr/build-system` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10371
- #10375 (`pr/libnd4j-core` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10375
- #10376 (`pr/libnd4j-graph-ops` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10376
- #10377 (`pr/java-ops-factory` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10377
- #10378 (`pr/nd4j-common` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10378
- #10379 (`pr/tests` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10379
- #10380 (`pr/contrib-tools` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10380
- #10381 (`pr/java-core-infra` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10381
- #10382 (`pr/onnx-runtime` -> `master`): https://github.com/deeplearning4j/deeplearning4j/pull/10382

## Push Status

| Branch | PR | File Count | Status |
|---|---:|---:|---|
| `pr/adrs-documentation` | #10370 | 5 | `pending` |
| `pr/build-system` | #10371 | 34 | `pending` |
| `pr/contrib-tools` | #10380 | 3 | `pending` |
| `pr/libnd4j-core` | #10375 | 16 | `pending` |
| `pr/libnd4j-graph-ops` | #10376 | 46 | `pending` |
| `pr/java-core-infra` | #10381 | 36 | `pending` |
| `pr/java-ops-factory` | #10377 | 23 | `pending` |
| `pr/nd4j-common` | #10378 | 2 | `pending` |
| `pr/onnx-runtime` | #10382 | 4 | `pending` |
| `pr/tests` | #10379 | 10 | `pending` |

## Branch File Mapping

### pr/adrs-documentation (PR #10370)

ADR updates, optimization notes, and split-plan tracking.

Files (5):
- `ADRs/0061 - DynamicShapePlan Execution.md`
- `ADRs/0067 - Scaled Dot-Product Attention Optimization.md`
- `ADRs/0071 - Triton Graph Backend.md`
- `PR_SPLIT_PLAN.md`
- `optimization-journal.md`

### pr/build-system (PR #10371)

GitHub workflows, build scripts, CMake/build plumbing, and root/build POM alignment.

Files (34):
- `.github/actions/setup-ccache-linux/action.yml`
- `.github/actions/setup-ccache-macos/action.yml`
- `.github/actions/setup-ccache-windows/action.yml`
- `.github/workflows/build-deploy-android-arm64.yml`
- `.github/workflows/build-deploy-android-x86_64.yml`
- `.github/workflows/build-deploy-cross-platform.yml`
- `.github/workflows/build-deploy-linux-arm64.yml`
- `.github/workflows/build-deploy-linux-cuda-12.6.yml`
- `.github/workflows/build-deploy-linux-cuda-12.9.yml`
- `.github/workflows/build-deploy-linux-x86_64.yml`
- `.github/workflows/build-deploy-mac-arm64.yml`
- `.github/workflows/build-deploy-mac.yml`
- `.github/workflows/build-deploy-windows-cuda-12.6.yml`
- `.github/workflows/build-deploy-windows-cuda-12.9.yml`
- `.github/workflows/build-deploy-windows.yml`
- `.github/workflows/cpu-sanity-check-tests.yaml`
- `.github/workflows/run-gpu-tests-sanity-checks.yml`
- `.github/workflows/test_multiple_arch.yaml`
- `change-cuda-versions.sh`
- `libnd4j/buildnativeoperations.sh`
- `libnd4j/cmake/CudaCleanup.cmake`
- `libnd4j/cmake/CudaConfiguration.cmake`
- `libnd4j/cmake/Dependencies.cmake`
- `libnd4j/cmake/FindNCCL.cmake`
- `libnd4j/cmake/MainBuildFlow.cmake`
- `libnd4j/cmake/Options.cmake`
- `libnd4j/cmake/SmartCcache.cmake`
- `libnd4j/cmake/install_triton.sh`
- `libnd4j/cmake/patch_triton_no_amd.sh`
- `libnd4j/pom.xml`
- `nd4j/nd4j-tensorflow-lite/pom.xml`
- `nd4j/nd4j-tokenizers/pom.xml`
- `nd4j/nd4j-tvm/pom.xml`
- `pom.xml`

### pr/contrib-tools (PR #10380)

Contrib and codegen support module build updates.

Files (3):
- `codegen/blas-lapack-generator/pom.xml`
- `contrib/benchmarking_nd4j/pom.xml`
- `contrib/blas-lapack-generator/pom.xml`

### pr/libnd4j-core (PR #10375)

Core runtime/memory/environment/native infrastructure changes in libnd4j.

Files (16):
- `libnd4j/include/array/DataBuffer.h`
- `libnd4j/include/array/NDArray.hXX`
- `libnd4j/include/array/cpu/DataBuffer.cpp`
- `libnd4j/include/array/cuda/DataBuffer.cu`
- `libnd4j/include/execution/cuda/CudaGraphScheduler.h`
- `libnd4j/include/helpers/MmulHelper.h`
- `libnd4j/include/helpers/cuda/MmulHelper.cu`
- `libnd4j/include/helpers/cuda/PointersManager.cu`
- `libnd4j/include/legacy/NativeOps.h`
- `libnd4j/include/legacy/cpu/NativeOps_dsp.cpp`
- `libnd4j/include/legacy/cuda/NativeOps_dsp.cu`
- `libnd4j/include/legacy/impl/Environment.cpp`
- `libnd4j/include/memory/cuda/CudaMemoryPool.cu`
- `libnd4j/include/mlir/runtime/MLIREngine.cpp`
- `libnd4j/include/mlir/runtime/MLIREngine.h`
- `libnd4j/include/system/Environment.h`

### pr/libnd4j-graph-ops (PR #10376)

Graph backend, Triton/NVRTC/PTX, and graph/ops execution correctness updates.

Files (46):
- `libnd4j/include/graph/FusionPass.h`
- `libnd4j/include/graph/GraphBackend.h`
- `libnd4j/include/graph/NativeDynamicShapePlan.h`
- `libnd4j/include/graph/cpu/AclGraphBackend.cpp`
- `libnd4j/include/graph/cpu/AclGraphBackend.h`
- `libnd4j/include/graph/cpu/OneDnnGraphBackend.cpp`
- `libnd4j/include/graph/cpu/OneDnnGraphBackend.h`
- `libnd4j/include/graph/gpu/GpuKernelLauncher.cu`
- `libnd4j/include/graph/gpu/GpuKernelLauncher.h`
- `libnd4j/include/graph/gpu/NvrtcGraphBackend.cpp`
- `libnd4j/include/graph/gpu/NvrtcGraphBackend.h`
- `libnd4j/include/graph/gpu/NvrtcKernelBuilder.cpp`
- `libnd4j/include/graph/gpu/NvrtcKernelBuilder.h`
- `libnd4j/include/graph/gpu/NvrtcKernelCache.cu`
- `libnd4j/include/graph/gpu/NvrtcKernelCache.h`
- `libnd4j/include/graph/gpu/OpCategoryTable.h`
- `libnd4j/include/graph/gpu/PtxGraphBackend.cpp`
- `libnd4j/include/graph/gpu/PtxGraphBackend.h`
- `libnd4j/include/graph/gpu/TritonGraphBackend.cpp`
- `libnd4j/include/graph/gpu/TritonGraphBackend.h`
- `libnd4j/include/graph/gpu/TritonIRBuilder.cpp`
- `libnd4j/include/graph/gpu/TritonIRBuilder.h`
- `libnd4j/include/graph/gpu/TritonTargetDispatch.cpp`
- `libnd4j/include/graph/gpu/TritonTargetDispatch.h`
- `libnd4j/include/graph/impl/Context.cpp`
- `libnd4j/include/graph/impl/FusionPass.cpp`
- `libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp`
- `libnd4j/include/graph/impl/NativePlanCompiler.cpp`
- `libnd4j/include/ops/declarable/generic/nn/dora_matmul.cpp`
- `libnd4j/include/ops/declarable/generic/nn/dot_product_attention.cpp`
- `libnd4j/include/ops/declarable/generic/nn/fused_elementwise_chain.cpp`
- `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp`
- `libnd4j/include/ops/declarable/generic/nn/loha_matmul.cpp`
- `libnd4j/include/ops/declarable/generic/nn/lokr_matmul.cpp`
- `libnd4j/include/ops/declarable/generic/nn/lora_matmul.cpp`
- `libnd4j/include/ops/declarable/generic/nn/multi_lora_matmul.cpp`
- `libnd4j/include/ops/declarable/generic/nn/selective_scan.cpp`
- `libnd4j/include/ops/declarable/generic/nn/smooth_quant.cpp`
- `libnd4j/include/ops/declarable/headers/llm.h`
- `libnd4j/include/ops/declarable/helpers/cpu/windowed_attention.cpp`
- `libnd4j/include/ops/declarable/helpers/cuda/col2im.cu`
- `libnd4j/include/ops/declarable/helpers/cuda/fusedElementwiseChain.cu`
- `libnd4j/include/ops/declarable/helpers/cuda/windowed_attention.cu`
- `libnd4j/include/ops/declarable/helpers/fusedElementwiseChain.h`
- `libnd4j/include/ops/declarable/impl/DeclarableOp.cpp`
- `libnd4j/include/system/op_boilerplate.h`

### pr/java-core-infra (PR #10381)

SameDiff execution/runtime core infrastructure and CUDA backend bindings.

Files (36):
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/CollectiveCommunicator.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/CollectiveCommunicatorFactory.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DspCompilationMode.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanCompiler.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/DynamicShapePlanExecutor.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/GraphExecutionMode.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/LocalCollectiveCommunicator.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/NcclCommunicator.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/PipelineParallelRunner.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/TensorParallelConfig.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/execution/TensorParallelRunner.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/internal/InferenceSession.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/optimize/GraphOptimizer.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/optimize/optimizations/AttentionFusionOptimizations.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/optimize/optimizations/NormalizationFusionOptimizations.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/optimize/optimizations/QuantizationOptimizations.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/peft/LoraAdapterCache.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/buffer/DataType.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ndarray/LazyINDArray.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/CompiledGraphFunction.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Environment.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/GraphFunction.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/GraphScope.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/Nd4j.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/util/DeviceLocal.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java`
- `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/CudaEnvironment.java`
- `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/bindings/Nd4jCuda.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/DecoderUtils.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/DraftModelSpeculator.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/KVCacheHostOffloader.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/KVCachePrefixTree.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/Speculator.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/StaticKvCacheDecodeLoop.java`
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/TreeAttentionVerifier.java`

### pr/java-ops-factory (PR #10377)

Java op wrappers/factories and codegen op surface updates.

Files (23):
- `codegen/op-codegen/src/main/ops/org/nd4j/codegen/ops/NeuralNetwork.kt`
- `codegen/op-codegen/src/main/ops/org/nd4j/codegen/ops/SDBaseOps.kt`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/ops/SDBaseOps.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/ops/SDNN.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/AwqMatmul.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/BooleanAnd.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/BooleanOr.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/BooleanXor.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/ColumnParallelLinear.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/DecoderMaskedMha.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/Fp8Matmul.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/FusedElementwiseChain.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/FusedGemmSwiglu.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/FusedNormQuantize.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/GpuTopKSample.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/GpuTopPSample.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/MoeGate.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/MultiLoraMatmul.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/RowParallelLinear.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/SelectiveScan.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/impl/transforms/custom/SmoothQuant.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/ops/NDBase.java`
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/factory/ops/NDNN.java`

### pr/nd4j-common (PR #10378)

Shared ND4J configuration constants/properties.

Files (2):
- `nd4j/nd4j-common/src/main/java/org/nd4j/common/config/ND4JEnvironmentVars.java`
- `nd4j/nd4j-common/src/main/java/org/nd4j/common/config/ND4JSystemProperties.java`

### pr/onnx-runtime (PR #10382)

ONNX runtime/import integration and VLM ONNX cache updates.

Files (4):
- `nd4j/nd4j-onnxruntime/pom.xml`
- `nd4j/samediff-import/samediff-import-onnx/pom.xml`
- `nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations/SimplifiedLayerNormalization.kt`
- `nd4j/samediff-vlm/src/main/java/org/eclipse/deeplearning4j/vlm/model/OnnxModelCache.java`

### pr/tests (PR #10379)

Platform test updates for DSP/Triton/VLM behavior.

Files (10):
- `platform-tests/bin/java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/TestDSPExecutionCorrectness.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/TestGraphOptimizerFusions.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/autodiff/samediff/TritonGraphBackendTest.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/linalg/factory/GraphScopeTest.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/linalg/ops/FusedElementwiseChainTest.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/linalg/ops/TRTLLMFeatureParityOpsTest.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/nd4j/linalg/ops/TensorParallelIntegrationTest.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/vlm/TestSmolDoclingOptimizedPipeline.java`
- `platform-tests/src/test/java/org/eclipse/deeplearning4j/vlm/TestVLMModelImportPipeline.java`
