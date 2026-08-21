/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class NnapiOutputStagingContractTest {

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
        int descriptorValidation = source.indexOf(
                "matchesCompiledDescriptor(arr, mapping.sourceDataType,");
        int paddingOptIn = source.indexOf(
                "ANeuralNetworksExecution_enableInputAndOutputPadding(execution, true);");
        int staging = source.indexOf(
                "struct StagedOutputBuffer {",
                descriptorValidation);
        int alignmentQuery = source.indexOf(
                "ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput(",
                staging);
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
        assertTrue(staging > descriptorValidation,
                "Every output must receive independent aligned staging storage");
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
        assertTrue(Files.isRegularFile(segmentExecutor),
                "DSP segment executor source was not found at " + segmentExecutor);
        assertTrue(Files.isRegularFile(platformPom),
                "platform-tests pom was not found at " + platformPom);

        String source = Files.readString(segmentExecutor);
        String pom = Files.readString(platformPom);
        int scanCap = source.indexOf("kOpSanityMaxScannedValues = 4096");
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
