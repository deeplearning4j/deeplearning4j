/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("source-lint")
class NnapiOutputStagingContractTest {

    @Test
    void deferredNnapiCompilationCannotReturnToHostWarmupAfterReplayStarts()
            throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String segments = Files.readString(root.resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp"));
        String runtimePlan = Files.readString(root.resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp"));
        String compiler = Files.readString(root.resolve(
                "libnd4j/include/graph/impl/NativePlanCompiler.cpp"));
        String cpuPlatform = Files.readString(root.resolve(
                "libnd4j/include/graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp"));
        String outputUtils = Files.readString(root.resolve(
                "libnd4j/include/graph/DspSegmentOutputUtils.h"));
        String plan = Files.readString(root.resolve(
                "libnd4j/include/graph/NativeDynamicShapePlan.h"));

        assertTrue(plan.contains(
                        "bool isShapesFrozen() const { return planLifecycle_.isShapesFrozen() || planLifecycle_.isReplaying(); }"),
                "The canonical frozen predicate must include replay steady state");
        assertTrue(segments.contains("GRAPH_BACKEND_PREREQUISITE_BLOCKED")
                        && segments.contains("lowering.prerequisiteBlocked()")
                        && cpuPlatform.contains("lowering.prerequisiteBlocked()")
                        && cpuPlatform.contains("DSP_THROW(COMPILE, \"%s\", message.c_str())"),
                "Missing finalized compiler inputs must be terminal before fallback or sealing");
        assertTrue(segments.contains("attemptedArtifactMatchesCurrentShape")
                        && segments.contains("COMMITTED_GRAPH_BACKEND_FAILURE")
                        && segments.contains("STALE_GRAPH_BACKEND_ARTIFACT_REJECTED")
                        && segments.contains(
                                "return DspExecutionResult(result.status, true, false);"),
                "A current-shape committed NNAPI failure must remain terminal while obsolete shape-drift artifacts return to pre-execution resolution");
        assertTrue(runtimePlan.contains("compiler-required phase cannot seal")
                        && compiler.contains("compiler calibration: preserved")
                        && outputUtils.contains("disableInPlaceConsumersOfSlots("),
                "Legacy warmup preservation may remain for other backends, but compiler sealing must stay fail closed");
    }

    @Test
    void tensorG3UsesCalibratedSignedPointwiseEdgeTpuIslands() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String source = Files.readString(
                root.resolve("libnd4j/include/graph/cpu/NnapiGraphBackend.cpp"));
        String artifact = Files.readString(
                root.resolve("libnd4j/include/graph/cpu/NnapiGraphBackend.h"));

        assertTrue(source.contains(
                        "nnapi-edgetpu-signed-pointwise-v7-dynamic-boundary-safe")
                        && source.contains(
                                "std::string(deviceName) != requiredDeviceName_")
                        && source.contains("isSignedPointwiseSegment(")
                        && source.contains("name == \"add\"")
                        && source.contains("name == \"multiply\"")
                        && source.contains("name == \"sigmoid\"")
                        && source.contains(
                                "slot.wiring.inputSourceIndices[0] == slot.wiring.inputSourceIndices[1]")
                        && source.contains(
                                "selectedDeviceFeatureLevel_ < ANEURALNETWORKS_FEATURE_LEVEL_4"),
                "Tensor G3 admission must be exact-device, signed-INT8, and exclude precision-critical normalization squares");
        assertTrue(source.contains("kCompiledBoundaryCalibrationHeadroom = 2.0")
                        && source.contains("deriveGuardedCalibrationBucket(")
                        && source.contains("observedAbsoluteMaximum <= 0.0f")
                        && source.contains(
                                "signed pointwise calibration requires a positive observed range")
                        && source.contains("std::frexp(guarded, &exponent)")
                        && source.contains("std::ldexp(1.0, exponent)")
                        && !source.contains(
                                "digest.mixValue(operand.observedAbsoluteMaximum)"),
                "Signed pointwise calibration must reserve deterministic bucketed headroom and avoid cache churn within a bucket");
        assertTrue(source.contains("broadcastPointwiseDimensions(")
                        && source.contains(
                                "first->admittedAbsoluteMaximum) +")
                        && source.contains(
                                "first->admittedAbsoluteMaximum) *")
                        && source.contains("output.scale = 1.0f / 256.0f")
                        && source.contains("output.zeroPoint = -128")
                        && source.contains("output.scale > inputScaleProduct")
                        && source.contains("std::nextafter("),
                "ADD/MUL ranges and LOGISTIC's fixed signed quantization contract must be explicit");
        assertTrue(source.contains(
                        "NNAPI_SIGNED_POINTWISE_LOWERING_COMMITTED")
                        && source.contains(
                                "NNAPI_SIGNED_POINTWISE_PLAN_RANGE_REJECTED")
                        && source.contains(
                                "CompiledModel::BoundaryTransform::QUANTIZE_ASYMM_SIGNED")
                        && source.contains(
                                "CompiledModel::BoundaryTransform::DEQUANTIZE_ASYMM_SIGNED")
                        && source.contains(
                                "compiled->backendCacheAbi.c_str()")
                        && source.contains("makeRequestedOutputIdentity(")
                        && source.contains("NNAPI_MODEL_IO_IDENTITY")
                        && source.contains(
                                "existing->requestedOutputIdentity ==")
                        && source.contains(
                                "completeArtifactDigest.mixString(compiled.modelIoIdentity)")
                        && source.contains(
                                "entry.wasCompiled = compiledSourceSlots.count(i) != 0"),
                "Pointwise islands must reuse guarded INT8 staging and cache/audit the emitted device artifact");
        assertTrue(source.contains("nnapi-q4k-conv-v14-finalized-per-op-calibration")
                        && source.contains("sdx.nnapi.q4.calibration.v1")
                        && source.contains("parseFinalizedQ4KCalibration(")
                        && source.contains("calibration sample count must be an integer of at least 32")
                        && source.contains("calibration dataset digest must be lowercase SHA-256")
                        && source.contains("NNAPI_FINALIZED_Q4K_CALIBRATION")
                        && source.contains("kCompiledQ4OutputInteriorQuantMax = 126.0f")
                        && source.contains(
                                "contract.outputScale * kCompiledQ4OutputInteriorQuantMax")
                        && !source.contains("kCompiledQ4UnstableFourActivationBucket")
                        && !source.contains("kCompiledQ4UnstableUnitOutputBucket"),
                "Q4 ranges must come from source-bound per-op calibration while code 127 stays an overflow sentinel");
        assertTrue(source.contains("findPublishedBoundaryCalibration(")
                        && source.contains("NNAPI_INHERITED_BOUNDARY")
                        && source.contains("NNAPI_FINALIZED_Q4K_CALIBRATION")
                        && source.contains(
                                "signed pointwise internal qmatmul input requires an")
                        && source.contains("isRmsNormalizationOutputSource(")
                        && source.contains(
                                "signed pointwise RMS-normalization input requires")
                        && source.contains(
                                "signed pointwise dynamic internal input requires an inherited")
                        && source.contains("!slots[producerSlot].frozenConstantSlot()")
                        && source.contains(
                                "mapping.sourceBufferIdentity != sourceBuffer")
                        && source.contains(
                                "(!signedPointwise && mapping.quantizationZeroPoint != 0)")
                        && source.contains("const bool affineSigned =")
                        && artifact.contains(
                                "DataBuffer* sourceBufferIdentity = nullptr"),
                "Adjacent compiled artifacts must inherit plan-scoped quantization envelopes, reject unbounded host matmul inputs, and preserve affine zero points");
        assertTrue(source.contains("NNAPI_Q4K_OUTPUT_RANGE_REJECTED")
                        && source.contains("quantized_endpoint_saturation")
                        && source.contains("observed_min=%.8g observed_max=%.8g observed_absmax=%.8g")
                        && source.contains("quantizedValues[element] < -126")
                        && source.contains("quantizedValues[element] > 126")
                        && source.contains("return Status::KERNEL_FAILURE;"),
                "Q4 output endpoint saturation must fail closed before dequantized copyback");
        assertTrue(artifact.contains(
                        "float calibrationAbsoluteMaximum = -1.0f")
                        && artifact.contains("std::string backendCacheAbi;")
                        && artifact.contains("std::string requestedOutputIdentity;")
                        && artifact.contains("std::string modelIoIdentity;"),
                "The compiled artifact must own its admitted range, lowering ABI, and exact model I/O identity");
        assertTrue(source.contains(
                        "isQ4KQMatMul(slots[slotIndex]) ||")
                        && source.contains("NNAPI_COMPILATION_AUDIT_REJECTED")
                        && source.contains("reason=uncovered_source_slot"),
                "Q4 feature gating and complete source-slot audit must fail before artifact publication");
    }

    @Test
    void tensorG3TextSessionOwnsOneFixedPrecompiledRollingPlan() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String generation = Files.readString(root.resolve(
                "libnd4j/include/legacy/impl/SdxGenerationSession.cpp"));
        String runtimeHeader = Files.readString(root.resolve(
                "libnd4j/include/dsp/runtime/dsp_runtime_c.h"));
        String metadata = Files.readString(root.resolve(
                "libnd4j/include/legacy/impl/SdxTextGenerationMetadata.cpp"));
        String nativeOps = Files.readString(root.resolve(
                "libnd4j/include/legacy/impl/NativeOps_dsp_shared.cpp"));
        String sdk = Files.readString(root.resolve(
                "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-sdx-model/src/main/java/"
                        + "org/nd4j/dsp/model/SdxPlatformSdk.java"));

        assertTrue(runtimeHeader.contains("int32_t fixed_context_capacity;")
                        && runtimeHeader.contains("sdxGetGenerationContextCapacity"),
                "The stable session ABI must carry and expose one fixed physical capacity");
        int contextFactory = generation.indexOf("createBoundContext(");
        int soleContextCall = generation.indexOf("createBoundContext(", contextFactory + 1);
        assertTrue(contextFactory >= 0 && soleContextCall > contextFactory
                        && generation.indexOf("createBoundContext(", soleContextCall + 1) < 0,
                "A generation session must contain one context factory and exactly one context/plan call");
        assertTrue(!generation.contains("prefillContext")
                        && !generation.contains("clearExecutionState")
                        && generation.contains("initializeFixedPlan(session.get())")
                        && generation.contains("precompileBoundContext(")
                        && generation.contains("kMaxFixedPlanConvergencePasses")
                        && generation.contains("while ((!session->hasExecutionReport ||")
                        && generation.contains("SDX_FIXED_PLAN_CONVERGENCE")
                        && !generation.contains("stablePass < 2")
                        && generation.contains("plan_phase != 2")
                        && generation.contains("resetFixedExecutionState(session, &error)")
                        && generation.contains("droppedPromptTokens")
                        && generation.contains("effectivePrompt =")
                        && generation.contains("generateToContextLimit")
                        && generation.contains("modelVariableShape(")
                        && generation.contains("FIXED_PLAN_KV_SHAPES_DERIVED"),
                "Load, prompt ingestion, rolling-window generation, and reset must reuse one REPLAYING plan");
        assertTrue(generation.contains(
                        "bool resetFixedExecutionState(\n    sdx_generation_session_t* session, std::string* error)")
                        && generation.contains(
                                "if (array == nullptr) array = findNamed(session->decodeOwned, name);")
                        && generation.contains(
                                "fixed execution reset could not resolve input:")
                        && generation.contains(
                                "return fail(session, SDX_STATUS_EXECUTION_FAILED, error);"),
                "Fixed-state reset must validate every KV/recurrent binding and propagate failure");
        int convergenceLoop = generation.indexOf(
                "while ((!session->hasExecutionReport ||");
        int convergenceReset = generation.indexOf(
                "resetFixedExecutionState(session, &error)", convergenceLoop);
        int convergenceExecute = generation.indexOf(
                "status = executeFixedStep(", convergenceLoop);
        assertTrue(convergenceLoop >= 0
                        && convergenceReset > convergenceLoop
                        && convergenceExecute > convergenceReset,
                "Every lifecycle-convergence pass must reset recurrent/KV state before executing the fixed plan");
        assertTrue(metadata.contains("readOptionalKvShapeTemplates(")
                        && metadata.contains(
                                "KV shape arrays must both be absent or match the KV layer count")
                        && nativeOps.contains("getLoadedModelVariableShape(")
                        && nativeOps.contains("variable->shape()"),
                "Legacy bundles may omit duplicated KV templates only when the loaded FlatGraph supplies them");
        assertTrue(sdk.contains("false,\n                256))"),
                "Tensor G3 capacity must be provider-owned rather than app-owned");
    }

    @Test
    void q4KMatmulLoweringIsBackendOwnedQuantizedAndCacheSeparated()
            throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        Path backend = root.resolve("libnd4j/include/graph/cpu/NnapiGraphBackend.cpp");
        Path header = backend.resolveSibling("NnapiGraphBackend.h");
        Path plan = root.resolve("libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp");
        Path segments = root.resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp");
        Path backendContract = root.resolve("libnd4j/include/graph/GraphBackend.h");
        Path resolver = root.resolve("libnd4j/include/graph/GraphBackendResolver.h");
        assertTrue(Files.isRegularFile(backend), "NNAPI backend source was not found");
        assertTrue(Files.isRegularFile(header), "NNAPI backend header was not found");
        assertTrue(Files.isRegularFile(plan), "DSP execution-plan source was not found");

        String source = Files.readString(backend);
        String artifact = Files.readString(header);
        String executionPlan = Files.readString(plan);
        String segmentExecution = Files.readString(segments);
        String backendPolicy = Files.readString(backendContract);
        String resolverPolicy = Files.readString(resolver);

        assertTrue(source.contains("name == \"ggml_qmatmul\"")
                        && source.contains("args.iArgs[0] != 8")
                        && source.contains("Q4_K packed weights must be an immutable external constant")
                        && source.contains("non_standalone_compile"),
                "Only standalone immutable Q4_K ggml_qmatmul contracts may enter NNAPI lowering");
        assertTrue(source.contains("decodeQ4KBlock(")
                        && source.contains("symmetricInt8ScaleFromAbsMax(")
                        && source.contains("quantizeSymmetricInt8("),
                "Q4_K weights must be stream-decoded and symmetrically quantized per output channel");
        assertTrue(source.contains("ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED")
                        && source.contains("ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL")
                        && source.contains("ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(")
                        && source.contains("artifact.perChannelBiasScales")
                        && source.contains("activationScale * channelScale")
                        && source.contains("effective channel scales are implicitly")
                        && !source.contains("reason=bias_per_channel_params")
                        && source.contains("NNAPI_Q4K_EMITTED slot=%d op=CONV_2D emitted_ops=1")
                        && source.contains("shape_adaptation=metadata_only"),
                "The EdgeTPU artifact must be one signed-INT8 per-channel 1x1 convolution");
        assertTrue(artifact.contains("std::vector<int8_t> filter;")
                        && artifact.contains("std::vector<float> perChannelScales;")
                        && artifact.contains("std::vector<float> perChannelBiasScales;")
                        && artifact.contains("std::vector<int32_t> zeroBias;")
                        && artifact.contains("std::vector<QuantizedQ4KConstant> q4kConstants;")
                        && artifact.contains("std::mutex executionMutex;"),
                "Converted constants and execution ownership must remain in the segment artifact");
        assertTrue(source.contains("parseFinalizedQ4KCalibration(")
                        && source.contains("policy.precompileBeforeFirstExecution = true")
                        && source.contains("policy.allowsShapeOnlyWarmup = true")
                        && source.contains("policy.deferCompilationUntilPlanFreeze = false")
                        && source.contains("policy.requiresPrecommitFunctionalWarmup = false")
                        && source.contains("QUANTIZE_ASYMM_SIGNED")
                        && source.contains("DEQUANTIZE_ASYMM_SIGNED")
                        && source.contains("quantizeSymmetricInt8(")
                        && source.contains("std::memcpy(staging.data(), buffer, bufferSize)")
                        && source.contains("NNAPI_Q4K_INPUT_QUANTIZED")
                        && source.contains("NNAPI_Q4K_OUTPUT_DEQUANTIZED")
                        && source.contains("NNAPI_Q4K_PLAN_RANGE_REJECTED")
                        && source.contains("return Status::KERNEL_FAILURE;"),
                "DSP must compile from finalized metadata before the first value-producing NNAPI execution");
        assertTrue(backendPolicy.contains("GraphBackendCompilationReadiness")
                        && backendPolicy.contains("compilationReadiness(")
                        && resolverPolicy.contains("backend->compilationReadiness(")
                        && resolverPolicy.contains("prerequisiteBlockedBackend")
                        && segmentExecution.contains("GRAPH_BACKEND_PREREQUISITE_BLOCKED")
                        && executionPlan.contains("BACKEND_PRECOMPILE_FASTPATH"),
                "Every graph backend must share one fail-closed compilation-readiness lifecycle");
        assertTrue(!source.contains("kTensorG3Q4ActivationScale")
                        && !source.contains("kTensorG3Q4OutputScale")
                        && !source.contains("0.03125f")
                        && !source.contains("0.0625f"),
                "NNAPI lowering must never embed model-independent activation/output scales");
        assertTrue(source.contains("contract.packedWeightSnapshot.resize(")
                        && source.contains("contract.packedWeightSnapshot.data()")
                        && source.contains("#if !defined(__ANDROID_API__) || __ANDROID_API__ < 29"),
                "Weight digest/conversion must share one snapshot and API-28 admission must fail early");
        assertTrue(source.contains("mixNnapiCacheToken(loweringAwareCacheToken,")
                        && source.contains("compiled->loweringCacheIdentity")
                        && source.contains("existing->sourceWeightIdentity == currentSourceWeightIdentity")
                        && source.contains("existing->sourceLoweringIdentity ==")
                        && source.contains("NNAPI_DEVICE_CACHE_IDENTITY")
                        && source.contains("cacheGeneration_ != cacheGenerationAtStart"),
                "Artifact and driver-cache reuse must distinguish weights and lowering ABI");
        assertTrue(executionPlan.contains("backendMergeRangeAdmitted")
                        && executionPlan.contains("backend-range-rejected"),
                "The execution plan must not merge across a backend-rejected quantized boundary");
    }

    @Test
    void tensorG3AcceleratorContractReachesCppAndArtifactVerification()
            throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        Path options = root.resolve("libnd4j/cmake/Options.cmake");
        Path aarVerifier = root.resolve("libnd4j/tools/mobile/verify-android-accelerator-aar.sh");

        assertTrue(Files.isRegularFile(options), "CMake options source was not found");
        assertTrue(Files.isRegularFile(aarVerifier), "Android AAR verifier was not found");

        String cmake = Files.readString(options);
        String aar = Files.readString(aarVerifier);
        assertTrue(cmake.contains("add_definitions(-DSD_NNAPI_ACCELERATOR_ONLY=1)")
                        && cmake.contains("-DSD_NNAPI_REQUIRED_DEVICE_NAME=${SD_NNAPI_REQUIRED_DEVICE_NAME}")
                        && cmake.contains("SD_NNAPI_ACCELERATOR_ONLY requires SD_NNAPI_REQUIRED_DEVICE_NAME")
                        && cmake.contains("SD_NNAPI_ACCELERATOR_ONLY requires Android API 29+"),
                "Tensor G3 deployment settings must be C++ compile definitions, not metadata only");
        for (String symbol : new String[]{
                "ANeuralNetworks_getDeviceCount",
                "ANeuralNetworks_getDevice",
                "ANeuralNetworksDevice_getName",
                "ANeuralNetworksDevice_getType",
                "ANeuralNetworksDevice_getFeatureLevel",
                "ANeuralNetworksModel_getSupportedOperationsForDevices",
                "ANeuralNetworksCompilation_createForDevices"}) {
            assertTrue(aar.contains(symbol),
                    "The AAR verifier must require pinned-device symbol " + symbol);
        }
        assertTrue(aar.contains("forbidden generic NNAPI compilation")
                        && aar.contains("google-edgetpu"),
                "Tensor G3 artifacts must reject generic NNAPI and retain the required device fingerprint");
    }

    @Test
    void androidCpuImporterExcludesUnusedCompilerRuntime() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        Path producer = root.resolve(
                "nd4j/sdx-aot/src/main/android/build-android-cpu-importer-sdk.sh");
        Path acceleratorProducer = root.resolve(
                "libnd4j/tools/mobile/build-android-accelerator.sh");
        Path nativePom = root.resolve(
                "nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native/pom.xml");
        assertTrue(Files.isRegularFile(producer),
                "Android CPU importer producer was not found");
        assertTrue(Files.isRegularFile(acceleratorProducer),
                "Android accelerator producer was not found");
        assertTrue(Files.isRegularFile(nativePom),
                "ND4J native Maven descriptor was not found");

        String source = Files.readString(producer);
        String acceleratorSource = Files.readString(acceleratorProducer);
        String nativePomSource = Files.readString(nativePom);
        assertTrue(source.contains("run_native_platform_stage compile -Dlibnd4j.triton=OFF")
                        && source.contains("-Dlibnd4j.triton=OFF \\")
                        && source.contains("'triton=off'")
                        && source.contains("triton_cpu_included=false"),
                "The conversion-only CPU importer must not compile or advertise Triton");
        assertTrue(source.contains("libsdx_cpu.so|libLLVM.so|libMLIR.so")
                        && source.contains(
                                "CPU importer closure contains unused compiler runtime")
                        && source.contains(
                                "CPU importer deployment contains unused compiler runtime")
                        && source.contains("compiler_runtime_included=false"),
                "The CPU importer must reject LLVM/MLIR compiler payloads before APK packaging");
        assertTrue(!source.contains("run_native_platform_stage compile -Dlibnd4j.triton=ON")
                        && !source.contains("Triton CPU and its LLVM/MLIR runtime closure remain"),
                "The importer producer must not retain the obsolete compiler-enabled contract");
        assertTrue(source.contains("':(exclude,glob)libnd4j/**/*.cu'")
                        && source.contains("':(exclude,glob)libnd4j/**/*.cuh'")
                        && source.contains("unrelated CUDA edits must")
                        && source.contains("nd4j/nd4j-backends/nd4j-api-parent/nd4j-api"),
                "Android CPU cache/source manifests must exclude CUDA translation units while retaining managed DSP lifecycle sources");
        assertTrue(source.contains("dd if=\"$source_library\" of=\"$destination_library\"")
                        && source.contains("iflag=fullblock")
                        && source.contains("managed native byte copy changed content")
                        && source.contains(
                                "managed native payload changed during Android CPU importer publication"),
                "Deployment stripping must operate on verified userspace-copied bytes and revalidate the immutable managed payload");
        assertTrue(nativePomSource.contains("<exclude name=\"*.so.tmp*\"/>")
                        && source.contains(
                                "stale_native_linker_outputs=(\"$NATIVE_BUILD_DIR\"/*.so.tmp*)")
                        && acceleratorSource.contains(
                                "stale_native_linker_outputs=(\"$NATIVE_BUILD_DIR\"/*.so.tmp*)")
                        && source.contains("rm -f -- \"${stale_native_linker_outputs[@]}\"")
                        && acceleratorSource.contains(
                                "rm -f -- \"${stale_native_linker_outputs[@]}\""),
                "Android native packaging must exclude and remove abandoned atomic-linker temporary libraries");
    }

    @Test
    void androidPlanTeardownReleasesAndAccountsForNnapiConstants() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String planHeader = Files.readString(
                root.resolve("libnd4j/include/graph/NativeDynamicShapePlan.h"));
        String cpuLifecycle = Files.readString(root.resolve(
                "libnd4j/include/graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp"));
        String nnapiHeader = Files.readString(
                root.resolve("libnd4j/include/graph/cpu/NnapiGraphBackend.h"));
        String nnapiSource = Files.readString(
                root.resolve("libnd4j/include/graph/cpu/NnapiGraphBackend.cpp"));

        assertTrue(cpuLifecycle.contains("platformReleaseSegmentGpuResources()")
                        && cpuLifecycle.contains("seg.resetGraphBackend();"),
                "CPU/Android session teardown must release direct graph backend artifacts");
        assertTrue(planHeader.contains("compiledGraphBackendArtifactOwnedBytes")
                        && planHeader.contains(
                                "total += segment.compiledGraphBackendArtifactOwnedBytes")
                        && planHeader.contains("compiledGraphBackendArtifactOwnedBytes = 0;"),
                "The plan cache must count and clear opaque backend artifact memory");
        assertTrue(nnapiHeader.contains("size_t ownedBytes() const")
                        && nnapiHeader.contains(
                                "std::vector<QuantizedQ4KConstant>().swap(q4kConstants)")
                        && nnapiSource.contains(
                                "this, shapeKey, compiled, compiled->ownedBytes()"),
                "NNAPI must publish its Q4 constant footprint and free capacity after model teardown");
    }

    @Test
    void cudaPlanTeardownReleasesAndAccountsForCaptureWorkspaces() throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String planHeader = Files.readString(
                root.resolve("libnd4j/include/graph/NativeDynamicShapePlan.h"));
        String planLifecycle = Files.readString(root.resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp"));
        String cudaLifecycle = Files.readString(root.resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan_cuda.cu"));

        assertTrue(planHeader.contains("total += sharedCaptureWorkspaceBytes_")
                        && planHeader.contains("total += cublasWorkspaceSize_")
                        && planHeader.contains("!handle->isWorkspaceExternal()")
                        && planHeader.contains("total += handle->getWorkspaceBytes()"),
                "The plan-cache budget must include shared and handle-owned replay/cuBLAS workspaces");
        int releaseStart = planLifecycle.indexOf(
                "int NativeDynamicShapePlan::releaseGpuIntermediates()");
        int graphsQuiesced = planLifecycle.indexOf(
                "platformReleaseSegmentGpuResources();", releaseStart);
        int deferredDeletesFlushed = planLifecycle.indexOf(
                "flushDeferredSlotDeletes();", releaseStart);
        int untrackedRetired = planLifecycle.indexOf(
                "std::memset(untrackedOutputCache_", releaseStart);
        int captureArenaReleased = planLifecycle.indexOf(
                "platformFreeCaptureWorkspace();", untrackedRetired);
        assertTrue(releaseStart >= 0 && graphsQuiesced > releaseStart
                        && deferredDeletesFlushed > graphsQuiesced
                        && untrackedRetired > deferredDeletesFlushed
                        && captureArenaReleased > untrackedRetired,
                "Graphs must quiesce before buffer retirement, while capture arenas stay registered until all interior DataBuffers retire");
        assertTrue(cudaLifecycle.contains(
                                "pool.unregisterCaptureWorkspace(sharedCaptureWorkspace_)")
                        && cudaLifecycle.contains(
                                "pool.free(sharedCaptureWorkspace_, workspaceDevice, nullptr)")
                        && cudaLifecycle.contains("handle->releaseWorkspace(")
                        && cudaLifecycle.contains("removeDirtyStream(")
                        && cudaLifecycle.contains("ownedStreamDeviceId_")
                        && cudaLifecycle.contains("invalidateCacheForSegments(")
                        && cudaLifecycle.contains("sharedCaptureWorkspace_ = nullptr")
                        && cudaLifecycle.contains("sharedCaptureWorkspaceBytes_ = 0"),
                "Session reset/passivation must release shared and handle-owned capture arenas after safe retirement");
        assertTrue(planLifecycle.contains(
                                "std::memset(untrackedOutputCache_")
                        && !planLifecycle.substring(releaseStart).contains(
                                "untrackedOutputCacheSize_ = 0"),
                "A cold cached plan must retain its untracked-output pointer table for re-warmup");
    }

    @Test
    void directGraphBackendCompilationIsDistinctFromReplayHandles()
            throws Exception {
        Path root = Path.of("").toAbsolutePath().normalize().resolve("..").normalize();
        String lifecycle = Files.readString(
                root.resolve("libnd4j/include/graph/DspSegmentLifecycle.h"));
        String plan = Files.readString(
                root.resolve("libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp"));
        String segments = Files.readString(
                root.resolve("libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp"));
        String cpu = Files.readString(
                root.resolve("libnd4j/include/graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp"));

        assertTrue(lifecycle.contains("markDirectGraphBackendCompiled")
                        && lifecycle.contains("SEALED:DIRECT_COMPILED")
                        && lifecycle.contains("exec.outcome = SegmentExecOutcome::DIRECT_COMPILED"),
                "Direct compiled artifacts need an explicit validated lifecycle transition");
        assertTrue(cpu.contains("GraphBackendArtifactKind::DIRECT_COMPILED")
                        && cpu.contains("SegmentLifecycle::markDirectGraphBackendCompiled")
                        && cpu.contains("executeSegmentWithSpecificBackend("),
                "CPU precompile and frozen replay must publish and execute direct backend artifacts");
        int eagerPrecompile = cpu.indexOf(
                "void NativeDynamicShapePlan::platformPrecompileSegments(");
        int eagerFallbackPolicy = cpu.indexOf(
                "ModeContract::forMode(graphExecutionMode_).allowsFallback", eagerPrecompile);
        int eagerFallbackBackend = cpu.indexOf(
                "seg.def.selectedBackend = SelectedBackend::EMULATED_REPLAY", eagerFallbackPolicy);
        int eagerFallbackHandoff = cpu.indexOf(
                "SegmentLifecycle::prepareFunctionalReplayHandoff(", eagerFallbackBackend);
        assertTrue(eagerPrecompile >= 0
                        && eagerFallbackPolicy > eagerPrecompile
                        && eagerFallbackBackend > eagerFallbackPolicy
                        && eagerFallbackHandoff > eagerFallbackBackend,
                "Eager ARM-hybrid precompile must hand rejected NNAPI/ACL ranges to explicit replay before sealing");
        assertTrue(segments.contains("seg.setResolvedGraphBackend(backend, request);")
                        && segments.contains("SegmentLifecycle::markDirectGraphBackendCompiled")
                        && segments.contains("backendIdentityChanged")
                        && segments.contains("seg.resolvedGraphBackend != backend"),
                "Lazy and same-shape recovery must recompile before publishing direct execution");
        assertTrue(plan.contains("allFrozenDispatchUnitsReady")
                        && plan.contains("SegmentExecOutcome::DIRECT_COMPILED")
                        && plan.contains("segmentHasReadyDirectArtifact"),
                "Direct readiness must be validated separately from graph replay handles");
        assertTrue(plan.contains(
                        "return seg.def.isCapturable && !seg.def.allFrozenConstants &&"),
                "All-frozen constant segments have no runtime work and must not block plan replay promotion");
        assertTrue(Files.readString(
                        root.resolve("libnd4j/include/graph/NativeDynamicShapePlan.h"))
                        .contains("def.shapeKeyState.reset();"),
                "Destroying backend ownership must also invalidate its compilation key");
    }

    @Test
    void everyOutputUsesTheCompiledDescriptorAndOwnedStorageUntilNnapiCompletes()
            throws Exception {
        Path backend = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/graph/cpu/NnapiGraphBackend.cpp")
                .normalize();
        Path backendHeader = backend.resolveSibling("NnapiGraphBackend.h");
        Path segmentExecutor = backend
                .getParent()
                .resolve("../impl/NativeDynamicShapePlan_segments.cpp")
                .normalize();
        Path planLifecycle = segmentExecutor.resolveSibling("NativeDynamicShapePlan.cpp");
        assertTrue(Files.isRegularFile(backend), "NNAPI backend source was not found at " + backend);
        assertTrue(Files.isRegularFile(backendHeader),
                "NNAPI backend header was not found at " + backendHeader);
        assertTrue(Files.isRegularFile(segmentExecutor),
                "DSP segment executor source was not found at " + segmentExecutor);
        assertTrue(Files.isRegularFile(planLifecycle),
                "DSP lifecycle source was not found at " + planLifecycle);

        String source = Files.readString(backend);
        String header = Files.readString(backendHeader);
        String segmentSource = Files.readString(segmentExecutor);
        String lifecycleSource = Files.readString(planLifecycle);
        int paddingOptIn = source.indexOf(
                "ANeuralNetworksExecution_enableInputAndOutputPadding(execution, true);");
        int staging = source.indexOf("struct StagedOutputBuffer {");
        int descriptorValidation = source.indexOf(
                "matchesCompiledDescriptor(arr, mapping.sourceDataType,",
                staging);
        int alignmentQuery = source.indexOf(
                "ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(",
                descriptorValidation);
        int paddingQuery = source.indexOf(
                "ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput(",
                alignmentQuery);
        int alignedAddress = source.indexOf(
                "const uintptr_t alignedDataAddress =",
                paddingQuery);
        int binding = source.indexOf(
                "void* buffer = staging.data();",
                alignedAddress);
        int wait = source.indexOf(
                "result = ANeuralNetworksEvent_wait(event);",
                binding);
        int guardValidation = source.indexOf(
                "const bool prefixGuardIntact = std::all_of(",
                wait);
        int typedView = source.indexOf(
                "NDArray boundOutput(staging.data(), 'c', stagingShape,",
                guardValidation);
        int copyBack = source.indexOf(
                "arr->assign(&boundOutput);",
                typedView);
        int preCopybackValidation = source.indexOf(
                "stage=before_copyback arr=%p db=%p",
                wait);
        int backendExecute = segmentSource.indexOf(
                "auto status = backend->executeSegment(");
        int liveSlotSnapshot = lifecycleSource.indexOf(
                "const size_t liveSlotCapacity = totalOutputSlots_ > 0");
        int sharedAliasProtection = lifecycleSource.indexOf(
                "shouldRetainDeferredSlotArray(arr, exactWrapperLive,",
                liveSlotSnapshot);
        int ownerRequeue = lifecycleSource.indexOf(
                "deferredSlotDeletes_.push_back(arr);",
                sharedAliasProtection);
        int teardownClassification = lifecycleSource.indexOf(
                "for (NDArray* arr : ownedArrays) {");
        int teardownViewDelete = lifecycleSource.indexOf(
                "for (NDArray* arr : viewArrays) delete arr;",
                teardownClassification);
        int teardownOwnerDelete = lifecycleSource.indexOf(
                "for (NDArray* arr : owningArrays) delete arr;",
                teardownViewDelete);
        int teardownClear = lifecycleSource.indexOf(
                "planOwnedArrays_.clear();",
                teardownOwnerDelete);
        int cacheConfiguration = source.indexOf(
                "ANeuralNetworksCompilation_setCaching(");
        int compilationFinish = source.indexOf(
                "ANeuralNetworksCompilation_finish(compilation);",
                cacheConfiguration);
        int gatherLowering = source.indexOf("if (nnapiOp == ANEURALNETWORKS_GATHER) {");
        int gatherIndices = source.indexOf(
                "const uint32_t indicesOperand = inputOperands[1];", gatherLowering);
        int gatherAxis = source.indexOf(
                "inputOperands.push_back(addScalarOperand(model, axis, nextOperand));",
                gatherIndices);
        int gatherReinsertIndices = source.indexOf(
                "inputOperands.push_back(indicesOperand);", gatherAxis);
        int embeddingContract = source.indexOf(
                "getEmbeddingLookupContract(");
        int embeddingInt64 = source.indexOf(
                "lookups->dataType() != DataType::INT64", embeddingContract);
        int batchedLookupFlatten = source.indexOf(
                "failed to flatten embedding lookups", embeddingInt64);
        int embeddingLookup = source.indexOf(
                "lowered=EMBEDDING_LOOKUP", batchedLookupFlatten);
        int batchedShapeRestore = source.indexOf(
                "failed to restore batched embedding", embeddingLookup);
        int emittedOperationCount = source.indexOf(
                "operationSourceSlots.size()", batchedShapeRestore);
        int checkedIndexNarrowing = source.indexOf(
                "NNAPI_GATHER_INDEX_RANGE", emittedOperationCount);
        int reductionLowering = source.indexOf(
                "if (nnapiOp == ANEURALNETWORKS_MEAN ||");
        int reductionKeepDims = source.indexOf(
                "inputOperands.push_back(addBoolOperand(model, keepDims, nextOperand));",
                reductionLowering);

        assertTrue(header.contains("DataType sourceDataType;")
                        && header.contains("DataType bindingDataType;")
                        && header.contains("std::vector<LongType> dimensions;"),
                "Compiled NNAPI mappings must retain their source and binding descriptors");
        assertTrue(descriptorValidation >= 0,
                "Live DSP output metadata must match the compiled operand descriptor");
        assertTrue(paddingOptIn >= 0 && paddingOptIn < staging,
                "NNAPI padding support must be enabled before any operand is bound");
        assertTrue(descriptorValidation > staging,
                "Every staged output must validate the live array against its compiled descriptor");
        assertTrue(alignmentQuery > staging,
                "Output staging must query the selected compilation's preferred alignment");
        assertTrue(paddingQuery > alignmentQuery,
                "Output staging must query the selected compilation's preferred padding");
        assertTrue(alignedAddress > paddingQuery,
                "The driver-visible output address must satisfy the preferred alignment");
        assertTrue(binding > alignedAddress, "NNAPI must bind aligned owned staging storage");
        assertTrue(wait > binding, "Aligned staging storage must survive until the NNAPI event completes");
        assertTrue(guardValidation > wait,
                "The output guard must be validated after NNAPI completes");
        assertTrue(preCopybackValidation > wait && preCopybackValidation < guardValidation,
                "NNAPI must reject a closed target before validating or copying staged bytes back");
        assertTrue(typedView > guardValidation,
                "NDArray metadata must not exist while NNAPI can write the output");
        assertTrue(copyBack > typedView,
                "Validated staging must copy back through a post-execution typed view");
        assertTrue(!source.contains("NDArray* boundOutput = arr;"),
                "NNAPI must not write directly into mutable DynamicShapePlan arrays");
        assertTrue(
                source.contains("NNAPI_OUTPUT_STAGING seg[%d-%d] output=%u source_slot=%d"),
                "Detailed DSP diagnostics must identify the output and source slot");
        assertTrue(
                source.contains("static_cast<size_t>(outputLength) * elementSize"),
                "NNAPI output capacity must be derived from the compiled binding descriptor");
        assertTrue(
                source.contains("const size_t boundBytes = (rawBytes + paddingMask) & ~paddingMask;")
                        && source.contains("execution, idx, nullptr, buffer, boundBytes"),
                "NNAPI must receive the padded output capacity, not the raw tensor byte count");
        assertTrue(
                source.contains("NNAPI_OUTPUT_GUARD_CORRUPTION seg[%d-%d] output=%u"),
                "A write outside the aligned padded output must fail with segment diagnostics");
        assertTrue(backendExecute >= 0,
                "The compiled graph backend must remain the segment execution path");
        assertTrue(!segmentSource.contains("graph-backend-closed-output-replacement"),
                "Frozen DSP slots must never be replaced to paper over a closed shared buffer");
        assertTrue(liveSlotSnapshot >= 0
                        && sharedAliasProtection > liveSlotSnapshot
                        && ownerRequeue > sharedAliasProtection,
                "Deferred deletion must use one live-slot snapshot and retain only wrappers that can close a live DataBuffer");
        assertTrue(teardownClassification >= 0
                        && teardownViewDelete > teardownClassification
                        && teardownOwnerDelete > teardownViewDelete
                        && teardownClear > teardownOwnerDelete,
                "Plan teardown must classify live wrappers once, then delete views before owners");
        assertTrue(
                !lifecycleSource.substring(teardownViewDelete, teardownClear).contains("isView()"),
                "Plan teardown must never query a wrapper again after deletion has started");
        assertTrue(
                header.contains("bool compileSegment(const GraphBackendRequest& request,"),
                "NNAPI must consume the request-aware compile path instead of dropping device cache metadata");
        assertTrue(
                cacheConfiguration >= 0 && compilationFinish > cacheConfiguration,
                "NNAPI driver caching must be configured before compilation is finished");
        assertTrue(
                source.contains("deviceCompilationCacheModelKey, startSlot, endSlot, shapeKey"),
                "The NNAPI cache token must distinguish the model, segment, and concrete shape");
        assertTrue(
                source.contains("NNAPI_DEVICE_CACHE_REJECTED")
                        && !source.substring(cacheConfiguration, compilationFinish)
                                .contains("return false;"),
                "A driver cache rejection must remain a non-fatal optimization miss");
        assertTrue(gatherLowering >= 0
                        && gatherIndices > gatherLowering
                        && gatherAxis > gatherIndices
                        && gatherReinsertIndices > gatherAxis,
                "NNAPI GATHER operands must be emitted as input, axis, indices");
        assertTrue(embeddingContract >= 0
                        && embeddingInt64 > embeddingContract
                        && batchedLookupFlatten > embeddingInt64
                        && embeddingLookup > batchedLookupFlatten
                        && batchedShapeRestore > embeddingLookup,
                "Batched INT64 embedding gathers must lower through flatten, "
                        + "EMBEDDING_LOOKUP, and output-shape restoration");
        assertTrue(emittedOperationCount > batchedShapeRestore,
                "NNAPI device classification must size its support vector from "
                        + "emitted NNAPI operations, not DSP slots");
        assertTrue(checkedIndexNarrowing > emittedOperationCount,
                "INT64 token ids must be range-checked before NNAPI INT32 binding");
        assertTrue(reductionLowering >= 0 && reductionKeepDims > reductionLowering,
                "NNAPI reduction keep_dims must be emitted as a BOOL scalar");
        assertTrue(!source.contains("ANeuralNetworksModel_relaxComputationFloat32toFloat16"),
                "Transformer logits must not be globally relaxed to FP16 range and precision");
    }

    @Test
    void armHybridKeepsQwenShapeOpsInsideCompiledBackends()
            throws Exception {
        Path backend = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/graph/cpu/AclGraphBackend.cpp")
                .normalize();
        Path header = backend.resolveSibling("AclGraphBackend.h");
        Path architecture = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/architecture/LLaMAArchitecture.java")
                .normalize();
        assertTrue(Files.isRegularFile(backend),
                "ACL graph backend source was not found at " + backend);
        assertTrue(Files.isRegularFile(header),
                "ACL graph backend header was not found at " + header);
        assertTrue(Files.isRegularFile(architecture),
                "LLaMA architecture source was not found at " + architecture);

        String source = Files.readString(backend);
        String headerSource = Files.readString(header);
        String architectureSource = Files.readString(architecture);
        int singletonAdmission = source.indexOf(
                "bool AclGraphBackend::canResolveSlot(");
        int gatherValidation = source.indexOf("arm_compute::NEGather::validate(");
        int checkedNarrowing = source.indexOf("ACL_GATHER_INDEX_RANGE");
        int perExecutionStaging = source.indexOf("gather->stageIndices(indices)");
        int gatherRun = source.indexOf("entry.function->run();", perExecutionStaging);
        int fullAudit = source.indexOf("functionsBuilt == expectedFunctions && completeAudit");
        int segmentOwnership = source.indexOf(
                "seg.setCompiledGraphBackendArtifact(this, shapeKey, compiled);");
        int compileInvalidationLock = source.indexOf(
                "std::lock_guard<std::mutex> registryLock(cacheMtx_);");
        int denseBindingGuard = source.indexOf("!isDenseCOrder(array)");
        int stagedOutputTracking = source.indexOf("stagedProducedSlots.insert(slotIdx)");
        int stagedOutputCopyback = source.indexOf(
                "stagedProducedSlots.count(slotIdx) != 0", stagedOutputTracking);
        int int64SubtractLowering = source.indexOf("ACL_INT64_SUBTRACT_LOWERING");
        int int64SubtractValidation = source.indexOf(
                "subtract->validateBindings(left, right, output)");
        int int64SubtractRun = source.indexOf(
                "entry.function->run();", int64SubtractValidation);
        int int64Mapping = source.indexOf(
                "case DataType::INT64: return arm_compute::DataType::S64;");
        int scalarPhysicalShape = source.indexOf("shape.set(0, 1);");
        int subtractSingletonAdmission = source.indexOf(
                "isSubtractName(slots[slotIndex].ident.opName)");
        int subtractRhsContract = source.indexOf("return rightValue == 1");
        int subtractMinOverflow = source.indexOf(
                "std::numeric_limits<LongType>::min() + rightValue");
        int subtractMaxOverflow = source.indexOf(
                "std::numeric_limits<LongType>::max() + rightValue");
        int scalarStaging = source.indexOf(
                "std::memcpy(tensor->buffer(), array->buffer(), array->sizeOfT());");
        int scalarCopyback = source.indexOf(
                "std::memcpy(arr->buffer(), tensor->buffer(), arr->sizeOfT());");

        assertTrue(singletonAdmission >= 0,
                "ARM hybrid capability partitioning must see singleton ACL gathers");
        assertTrue(gatherValidation > singletonAdmission,
                "ACL gather must validate the concrete table/index/output descriptors");
        assertTrue(checkedNarrowing >= 0,
                "INT64 token IDs must be range checked before ACL S32 staging");
        assertTrue(perExecutionStaging > gatherValidation
                        && perExecutionStaging > checkedNarrowing
                        && gatherRun > perExecutionStaging,
                "Changing decode token IDs must be staged before every ACL gather execution");
        assertTrue(fullAudit > gatherValidation,
                "ACL compilation must cover every admitted DSP slot");
        assertTrue(segmentOwnership > fullAudit,
                "Mutable ACL tensors must belong to the plan segment, not a process-wide range cache");
        assertTrue(compileInvalidationLock > fullAudit
                        && compileInvalidationLock < segmentOwnership,
                "ACL artifact construction and publication must be atomic with invalidation");
        assertTrue(denseBindingGuard > segmentOwnership
                        && stagedOutputTracking > denseBindingGuard
                        && stagedOutputCopyback > stagedOutputTracking,
                "ACL must reject unsupported layouts and copy every staged output back");
        assertTrue(int64SubtractLowering > gatherValidation
                        && int64SubtractValidation > denseBindingGuard
                        && int64SubtractRun > int64SubtractValidation
                        && stagedOutputCopyback > int64SubtractRun,
                "The scalar INT64 shape subtraction must stay in the compiled ARM backend");
        assertTrue(int64Mapping >= 0 && scalarPhysicalShape > int64Mapping,
                "ND4J INT64 scalars must map to one physical ACL S64 element");
        assertTrue(subtractSingletonAdmission >= 0 && subtractRhsContract >= 0
                        && subtractMinOverflow > subtractRhsContract
                        && subtractMaxOverflow > subtractMinOverflow,
                "Singleton subtraction must enforce RHS one and both overflow boundaries");
        assertTrue(scalarStaging > denseBindingGuard && scalarCopyback > scalarStaging,
                "Staged scalar inputs and outputs require explicit one-element copies");
        assertTrue(architectureSource.contains("actualSequenceLength.sub(one)"),
                "Qwen final-position selection must remain covered by the compiled scalar subtraction");
        assertTrue(headerSource.contains("std::weak_ptr<AclFunctionGroup>")
                        && headerSource.contains("std::mutex executionMtx;"),
                "ACL artifacts must be plan-owned and serialized per compiled group");
        assertTrue(!source.contains("canFuseActivation(")
                        && !source.contains("fused with previous op"),
                "ACL must not skip a following activation without configuring real fused execution");
    }

    @Test
    void fusedKernelTailsCannotBeSkippedAcrossBackendSegmentBoundaries()
            throws Exception {
        Path lifecycle = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/graph/impl/NativeDynamicShapePlan.cpp")
                .normalize();
        assertTrue(Files.isRegularFile(lifecycle),
                "DSP lifecycle source was not found at " + lifecycle);

        String source = Files.readString(lifecycle);
        int helper = source.indexOf(
                "static int disableFusedChainsAcrossSegmentBoundaries(");
        int segmentMap = source.indexOf(
                "std::vector<int> slotToSegment", helper);
        int boundaryCheck = source.indexOf(
                "slotToSegment[chainSlot] != headSegment", segmentMap);
        int clearTail = source.indexOf(
                "slots[chainSlot].fusedChain.isFusedChainTail = false;", boundaryCheck);
        int clearHead = source.indexOf(
                "head.fusedChain.isFusedChainHead = false;", clearTail);
        int clearLength = source.indexOf(
                "head.fusedChain.fusedChainLength = 0;", clearHead);
        int initialApply = source.indexOf(
                "int applied = FusionPass::applyFusions(", clearLength);
        int initialGuard = source.indexOf("\"initial-fusion\"", initialApply);
        int freezeApply = source.indexOf(
                "int applied = FusionPass::applyFusions(", initialGuard);
        int freezeResegment = source.indexOf("resegmentForFreeze();", freezeApply);
        int freezeGuard = source.indexOf("\"freeze-fusion\"", freezeResegment);
        int segmentBuild = source.indexOf(
                "void NativeDynamicShapePlan::buildSegments()", freezeGuard);
        int segmentGuard = source.indexOf("\"segment-build\"", segmentBuild);

        assertTrue(helper >= 0 && segmentMap > helper && boundaryCheck > segmentMap,
                "Fused chains must be checked against the final slot-to-segment map");
        assertTrue(clearTail > boundaryCheck && clearHead > clearTail
                        && clearLength > clearHead,
                "A cross-segment chain must restore tail execution before clearing its head");
        assertTrue(initialApply > clearLength && initialGuard > initialApply,
                "Serialized plans must reject cross-segment fusion before their first warmup");
        assertTrue(freezeApply > initialGuard && freezeResegment > freezeApply
                        && freezeGuard > freezeResegment,
                "Freeze-time fusion must be validated against the final rebuilt segment map");
        assertTrue(segmentBuild > freezeGuard && segmentGuard > segmentBuild,
                "Every rebuilt segment map must revalidate fused-kernel tail skips");
        assertTrue(source.contains("FUSION_SEGMENT_BOUNDARY"),
                "Cross-segment fusion rejection needs durable DSP diagnostics");
    }

    @Test
    void nativeOpSanityLogFingerprintsLogicalOutputsAcrossExecutionPaths()
            throws Exception {
        Path segmentExecutor = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/graph/impl/NativeDynamicShapePlan_segments.cpp")
                .normalize();
        Path platformPom = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("pom.xml");
        Path nnapiBackend = segmentExecutor
                .getParent()
                .resolve("../cpu/NnapiGraphBackend.cpp")
                .normalize();
        assertTrue(Files.isRegularFile(segmentExecutor),
                "DSP segment executor source was not found at " + segmentExecutor);
        assertTrue(Files.isRegularFile(platformPom),
                "platform-tests pom was not found at " + platformPom);
        assertTrue(Files.isRegularFile(nnapiBackend),
                "NNAPI backend source was not found at " + nnapiBackend);

        String source = Files.readString(segmentExecutor);
        String pom = Files.readString(platformPom);
        String nnapiSource = Files.readString(nnapiBackend);
        int scanCap = source.indexOf("kOpSanityMaxScannedValues =");
        int helper = source.indexOf("struct OpSanitySummary", scanCap);
        int hash = source.indexOf("summary->valueHash *= 0x100000001b3ULL;", helper);
        int logicalLayout = source.indexOf(
                "shape::strideDescendingCAscendingF(array->shapeInfo())", hash);
        int boundedScan = source.indexOf(
                "std::min(length, kOpSanityMaxScannedValues)", logicalLayout);
        int finiteChecks = source.indexOf("summary->nanCount++", boundedScan);
        int invalidBuffer = source.indexOf("state=invalid-buffer", finiteChecks);
        int deadState = source.indexOf("state=dead", invalidBuffer);
        int trigger = source.indexOf("diagnosticsNativeDump()", finiteChecks);
        int sync = source.indexOf("output->forceSyncToHost();", trigger);
        int event = source.indexOf("\"OP_SANITY backend=%s", sync);
        int boundaryHelper = source.indexOf("recordSegmentBoundaryOpSanity(", event);
        int backendExecution = source.indexOf("backend->executeSegment(", boundaryHelper);
        int backendRecord = source.indexOf("recordSegmentBoundaryOpSanity(", backendExecution);
        int slotExecution = source.indexOf("executeSlot(stepIdx", backendRecord);
        int slotRecord = source.indexOf("\"SLOT_BY_SLOT\"", slotExecution);

        assertTrue(scanCap >= 0 && helper > scanCap && hash > helper
                        && logicalLayout > hash && boundedScan > logicalLayout
                        && finiteChecks > boundedScan,
                "Operation sanity must sample logical values deterministically and calculate content statistics");
        assertTrue(trigger > finiteChecks,
                "Operation sanity logging must remain separately gated from full DSP diagnostics");
        assertTrue(sync > trigger && event > sync,
                "Triggered sanity logging must synchronize before emitting value evidence");
        assertTrue(event > finiteChecks,
                "Each record must include a value hash plus NaN/Inf/finite statistics");
        assertTrue(source.indexOf("scanned=%lld coverage=%s", event) >= event,
                "Large output records must disclose bounded sampling coverage");
        assertTrue(source.indexOf("arr=%p db=%p primary=%p special=%p offset=%lld", event) >= event,
                "Operation sanity records must expose wrapper and storage identities for alias diagnosis");
        assertTrue(source.indexOf("std::is_integral<T>::value", helper) < finiteChecks,
                "Integer hashes must preserve exact typed bits instead of rounding through double");
        assertTrue(invalidBuffer > finiteChecks && deadState > invalidBuffer,
                "Invalid and dead outputs must be reported without dereferencing stale buffers");
        assertTrue(source.contains("call->captureActive"),
                "Host functional replay must report values when no hardware capture is active");
        assertTrue(!source.substring(helper, event).contains("ews()"),
                "Operation sanity logging must not use deprecated element-wise stride checks");
        assertTrue(boundaryHelper > event && backendExecution > boundaryHelper
                        && backendRecord > backendExecution,
                "Compiled graph backends must log only their materialized segment-boundary outputs");
        assertTrue(slotExecution > backendRecord && slotRecord > slotExecution,
                "Explicit execution must log every completed slot using the same record format");
        assertTrue(pom.contains("<ND4J_DSP_NATIVE_DUMP_OUTPUTS>"
                        + "${nd4j.dsp.native.dumpOutputs}</ND4J_DSP_NATIVE_DUMP_OUTPUTS>")
                        && pom.contains("<ND4J_DSP_DIAG_EXEC_LIMIT>"
                        + "${nd4j.dsp.diagExecLimit}</ND4J_DSP_DIAG_EXEC_LIMIT>"),
                "Surefire must pass the opt-in trigger and execution bound into its forked JVM");
        assertTrue(nnapiSource.contains("diagDetailLimit()")
                        && nnapiSource.contains("emittedAliasCount < aliasDetailLimit")
                        && nnapiSource.contains("NNAPI_OUTPUT_TARGET_ALIAS_SUMMARY")
                        && nnapiSource.contains("aliases_omitted=%d"),
                "NNAPI identity tracing must bound alias detail and summarize omitted entries");
    }

    @Test
    void sdxImporterAcceptsTheOpSanityDiagnosticWireMode() throws Exception {
        Path importer = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../nd4j/sdx-aot/src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxGgufModelPreparer.java")
                .normalize();
        assertTrue(Files.isRegularFile(importer), "SDX importer source was not found at " + importer);

        String source = Files.readString(importer);
        int clears = source.indexOf(
                "System.clearProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS)");
        int mode = source.indexOf("case \"op_sanity\":", clears);
        int verify = source.indexOf(
                "System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS, \"VERIFY\")",
                mode);
        int full = source.indexOf(
                "System.setProperty(ND4JSystemProperties.DSP_DIAGNOSTICS_LEVEL, \"full\")",
                verify);
        int nativeDump = source.indexOf(
                "System.setProperty(ND4JSystemProperties.DSP_NATIVE_DUMP_OUTPUTS, \"true\")",
                full);

        assertTrue(clears >= 0 && mode > clears && verify > mode && full > verify
                        && nativeDump > full,
                "The importer must accept op_sanity as a complete VERIFY/full/native-dump mode");
    }

    @Test
    void tensorG3RawQ4PreparationProducesCalibrationAfterCanonicalization()
            throws Exception {
        Path importer = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../nd4j/sdx-aot/src/main/java/org/eclipse/deeplearning4j/sdx/aot/SdxGgufModelPreparer.java")
                .normalize();
        String source = Files.readString(importer);

        int canonicalIdentity = source.indexOf(
                "canonicalIdentity = SdxSourceIdentity.identify(canonical)");
        int calibration = source.indexOf("prepareTensorG3Q4Calibration(", canonicalIdentity);
        assertTrue(canonicalIdentity >= 0
                        && calibration > canonicalIdentity
                        && source.contains("SdxTensorG3Q4Calibration.calibrate(")
                        && source.contains("writeTensorG3Q4Profile(")
                        && source.contains("cachedTargetMatchesCanonicalProfile(")
                        && source.contains("compileOptionsBuilder.quantizationConfig(quantizationConfig)")
                        && source.contains("targetSoc, quantizationConfig != null")
                        && !source.contains("options.path(\"quantizationConfigPath\").asText(null)")
                        && !source.contains("requiresFinalizedQ4Calibration()"),
                "Raw Tensor G3 Q4 import must generate source-bound calibration internally after canonical SDZ identity exists");
    }

    @Test
    void generationPipelineRetiresFrozenBorrowerBeforeClosingRetainedInputs()
            throws Exception {
        Path pipeline = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/GenerationPipeline.java")
                .normalize();
        String source = Files.readString(pipeline);
        int close = source.indexOf(
                "Every retained generation state below owns arrays");
        int reset = source.indexOf("decoder.resetSession()", close);
        int clear = source.indexOf("decoder.clearDynamicShapePlanCache()", reset);
        int retainedState = source.indexOf("InGraphKvState cachedOneShot", clear);
        int stateClose = source.indexOf("cachedOneShot.close()", retainedState);

        assertTrue(close >= 0 && reset > close && clear > reset
                        && retainedState > clear && stateClose > retainedState,
                "Pipeline close must retire the decoder session/native plan before releasing retained fixed-buffer inputs");
    }

    @Test
    void androidAotObjectCacheIncludesExactManagedClassBytes() throws Exception {
        Path identity = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../nd4j/sdx-aot/src/main/android/native-image-object-identity.sh")
                .normalize();
        Path builder = identity.resolveSibling("build-android-aot-sdk.sh");
        assertTrue(Files.isRegularFile(identity), "AOT object identity helper was not found at " + identity);
        assertTrue(Files.isRegularFile(builder), "AOT builder was not found at " + builder);

        String identitySource = Files.readString(identity);
        String builderSource = Files.readString(builder);
        assertTrue(identitySource.contains("SDX_NATIVE_IMAGE_OBJECT_STAGE_FORMAT=7")
                        && identitySource.contains("target-scoped-managed-content-v5")
                        && identitySource.contains("classes_sha256=$CLASSES_SHA256")
                        && identitySource.contains("model_classes_sha256=$MODEL_CLASSES_SHA256")
                        && identitySource.contains("fresh_class_builds_sha256=$FRESH_CLASS_BUILDS_SHA256")
                        && identitySource.contains("classpath_manifest_sha256=$CLASSPATH_MANIFEST_SHA256"),
                "Native Image machine-code reuse must be keyed by exact compiled class bytes");
        assertTrue(builderSource.contains("validate_compatible_native_image_object_stage()")
                        && builderSource.contains("source_manifest_sha256=*) continue")
                        && !builderSource.contains("classes_sha256=*) continue"),
                "Compatible object reuse may ignore final-link source paths, but never managed class bytes");
        assertTrue(builderSource.contains("nd4j/nd4j-ggml")
                        && builderSource.contains("[nd4j-ggml]=\"$FRESH_CLASSES_ROOT/nd4j-ggml\"")
                        && builderSource.contains("[nd4j-ggml]=\"org/nd4j/ggml/GGMLModelImport.class\"")
                        && builderSource.contains("CURRENT_CLASSPATH_IDS=(nd4j-native-runtime tokenizers-native-preset tokenizers-native nd4j-ggml")
                        && builderSource.contains("  nd4j-ggml\n"),
                "Android AOT cache identity must compile and hash exact nd4j-ggml classes instead of reusing a SNAPSHOT jar");
    }

    @Test
    void recurrentStateCrossesThePrefillDecodeBoundaryThroughHostOwnedStorage()
            throws Exception {
        Path session = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/legacy/impl/SdxGenerationSession.cpp")
                .normalize();
        assertTrue(Files.isRegularFile(session), "SDX generation source was not found at " + session);

        String source = Files.readString(session);
        int copyStart = source.indexOf("bool copyRecurrentArrayInto(");
        int copyEnd = source.indexOf("\n}\n\n}  // namespace", copyStart);
        assertTrue(copyStart >= 0 && copyEnd > copyStart,
                "The recurrent-state transfer helper must remain explicit");
        String transfer = source.substring(copyStart, copyEnd);

        assertTrue(
                transfer.contains("source->forceSyncToHost();"),
                "Borrowed accelerator output must be synchronized before leaving the prefill context");
        assertTrue(
                transfer.contains("destination->forceSyncToHost();"),
                "The decode state must be host-authoritative before the explicit copy");
        assertTrue(
                transfer.contains("void* sourceBuffer = source->buffer();"),
                "The transfer must validate that the prefill output exposes host storage");
        assertTrue(
                transfer.contains("void* destinationBuffer = destination->buffer();"),
                "The decode state must expose independently owned host storage");
        assertTrue(
                transfer.contains("std::memcpy(destinationBuffer, sourceBuffer, bytes);"),
                "Recurrent state must cross the context boundary through an explicit host copy");
        assertTrue(
                transfer.contains("destination->tickWriteHost();"),
                "The copied decode state must be marked host-authoritative");
        assertTrue(
                !transfer.contains("destination->assign(source);"),
                "Generic assignment must not retain device-only ownership across the public ABI boundary");
    }
}
