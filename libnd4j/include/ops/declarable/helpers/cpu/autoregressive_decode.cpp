/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <ops/declarable/helpers/autoregressive_decode.h>
#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <execution/LaunchContext.h>
#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspPhaseUtils.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/logger.h>
#include <system/env_functions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

static std::string nestedPlanFailureDetail() {
    auto* launchContext = LaunchContext::defaultContext();
    auto* errorReference = launchContext != nullptr ? launchContext->errorReference() : nullptr;
    const char* detail = errorReference != nullptr ? errorReference->errorMessage() : nullptr;
    return detail != nullptr && detail[0] != '\0'
               ? std::string(detail)
               : std::string("nested plan returned without native failure detail");
}

// ─── CPU helpers (equivalent to CUDA kernels) ────────────────────────────────

/**
 * CPU: look up a single row from the embedding table.
 * Copies embeddingTable[tokenId, :] into outputEmbed [1, 1, hidden].
 */
template <typename T>
static void embedLookupCpu(const void* vEmbTable, void* vOutput,
                           LongType tokenId, LongType hidden, LongType tableRowStride) {
    auto embTable = reinterpret_cast<const T*>(vEmbTable);
    auto output = reinterpret_cast<T*>(vOutput);
    LongType baseOffset = tokenId * tableRowStride;
    for (LongType i = 0; i < hidden; i++) {
        output[i] = embTable[baseOffset + i];
    }
}

/**
 * CPU: update attention mask for the next decode step.
 */
template <typename T>
static void updateAttentionMaskCpu(void* vMask, LongType position, LongType maxKvLen) {
    auto mask = reinterpret_cast<T*>(vMask);
    if (position < maxKvLen) {
        mask[position] = static_cast<T>(1);
    }
}

/**
 * CPU: update causal mask for the next decode step.
 */
template <typename T>
static void updateCausalMaskCpu(void* vMask, LongType position, LongType maskLen) {
    auto mask = reinterpret_cast<T*>(vMask);
    if (position >= 0 && position < maskLen) {
        mask[position] = static_cast<T>(0);
    }
}

/**
 * Re-mask a speculative suffix after MTP rejection. The predictor may have
 * written farther than the accepted prefix; those KV slots stay allocated but
 * must be invisible until they are overwritten by a later accepted path.
 */
template <typename T>
static void maskCausalRangeCpu(void* vMask, LongType begin, LongType end, LongType maskLen) {
    auto mask = reinterpret_cast<T*>(vMask);
    const float maskFill = (sizeof(T) == 2) ? -65504.0f : -1e9f;
    begin = std::max<LongType>(0, begin);
    end = std::min<LongType>(end, maskLen);
    for (LongType position = begin; position < end; position++) {
        mask[position] = static_cast<T>(maskFill);
    }
}

/**
 * CPU: refill the GGUF W-wide causal mask for one decode step.
 *
 * The [1,1,W,maxKvLen] additive bias frozen into the plan encodes a linear
 * speculative chain: query slot w sits at absolute position currentPos + w and
 * may attend every column c <= currentPos + w (committed past, lower window
 * slots, self). The freeze-time mask from DecoderInputBuilder encodes that band
 * at the freeze position only, and updateCausalMaskCpu's single flat-index
 * write only ever advances row 0 — draft rows would stay stuck at the freeze
 * geometry. Refill all W rows in-place each step. Inactive rows get the same
 * causal band so their softmax rows stay finite (outputs ignored).
 */
template <typename T>
static void refillWindowCausalMaskCpu(void* vMask, LongType wMax, LongType maxKvLen,
                                      LongType currentPos) {
    // Match DecoderInputBuilder.buildInGraphWindowMask fill values: -65504 for
    // 2-byte float types (half/bfloat16), -1e9 for float/double.
    const float maskFill = (sizeof(T) == 2) ? -65504.0f : -1e9f;
    auto mask = reinterpret_cast<T*>(vMask);
    for (LongType w = 0; w < wMax; w++) {
        T* row = mask + w * maxKvLen;
        const LongType boundary = currentPos + w;
        for (LongType c = 0; c < maxKvLen; c++) {
            row[c] = (c <= boundary) ? static_cast<T>(0.0f) : static_cast<T>(maskFill);
        }
    }
}

/**
 * CPU: build initial attention mask from prefill length.
 */
template <typename T>
static void buildInitialMaskCpu(void* vMask, LongType prefillSeqLen, LongType maxKvLen) {
    auto mask = reinterpret_cast<T*>(vMask);
    for (LongType i = 0; i < prefillSeqLen && i < maxKvLen; i++) {
        mask[i] = static_cast<T>(1);
    }
}

/**
 * CPU: argmax over a float/half row [vocabSize].
 * Writes the index to output[0] as INT64.
 */
template <typename T>
static void argmaxCpu(const void* vLogits, void* vOutput, LongType vocabSize) {
    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<LongType*>(vOutput);
    if (vocabSize <= 0) {
        output[0] = 0;
        return;
    }
    T maxVal = logits[0];
    LongType maxIdx = 0;
    for (LongType i = 1; i < vocabSize; i++) {
        if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
        }
    }
    output[0] = maxIdx;
}

// Recurrent state is fed back through host-owned external inputs. NDArray::assign
// is not sufficient here: on accelerator plans it may retain device-authoritative
// storage and leave the host pointer null on the next tensor-view submission.
static bool copyRecurrentFeedback(NDArray* source, NDArray* destination) {
    if (source == nullptr || destination == nullptr ||
        source->lengthOf() != destination->lengthOf() ||
        source->dataType() != destination->dataType()) {
        return false;
    }

    source->forceSyncToHost();
    destination->forceSyncToHost();
    const size_t elementSize = DataTypeUtils::sizeOfElement(source->dataType());
    const LongType length = source->lengthOf();
    if (length < 0 || (elementSize != 0 &&
                       static_cast<uint64_t>(length) >
                           std::numeric_limits<size_t>::max() / elementSize)) {
        return false;
    }
    const size_t bytes = static_cast<size_t>(length) * elementSize;
    auto* sourceData = source->dataBuffer();
    auto* destinationData = destination->dataBuffer();
    if (bytes > 0 && (sourceData == nullptr || destinationData == nullptr)) {
        return false;
    }
    if (bytes > 0 && destination->buffer() == nullptr) {
        destinationData->allocatePrimary();
    }
    void* sourceBuffer = source->buffer();
    void* destinationBuffer = destination->buffer();
    if (bytes > 0 && (sourceBuffer == nullptr || destinationBuffer == nullptr)) {
        return false;
    }
    if (bytes > 0 &&
        (sourceData->getLenInBytes() < bytes ||
         destinationData->getLenInBytes() < bytes)) {
        return false;
    }
    if (bytes > 0) {
        std::memcpy(destinationBuffer, sourceBuffer, bytes);
        destination->tickWriteHost();
    }
    return true;
}

// Shared typed implementation for predictor and verification rows. Keep ties at
// the first vocabulary index, matching the existing greedy comparison contract.
template <typename T>
static LongType speculativeArgmaxCpu(const void* buffer, LongType vocabSize) {
    const auto* logits = reinterpret_cast<const T*>(buffer);
    LongType bestIdx = 0;
    T best = logits[0];
    for (LongType v = 1; v < vocabSize; v++) {
        if (logits[v] > best) {
            best = logits[v];
            bestIdx = v;
        }
    }
    return bestIdx;
}

// Sample one value from the head of a logits row into `out` for the rerun NaN
// guard. Template-typed by BUILD_SINGLE_SELECTOR so the probe covers every
// FLOAT dtype rather than assuming FLOAT32.
template <typename T>
static void sampleFirstRowValueCpu(const void* valuePtr, void* out) {
    *static_cast<float*>(out) = static_cast<float>(*reinterpret_cast<const T*>(valuePtr));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main CPU Implementation — equivalent logic to autoregressiveDecode (CUDA impl)
// ═══════════════════════════════════════════════════════════════════════════════

void autoregressiveDecode(
    NDArray* prefillEmbeddings,
    NDArray* embeddingTable,
    NDArray* inputIds,
    NDArray* attentionMask,
    NDArray* positionIds,
    NDArray** staticKvBuffers,
    int numKvPairs,
    NDArray* generatedTokenIds,
    NDArray* tokenCount,
    NDArray* timingInfo,
    int maxNewTokens,
    int prefillSeqLen,
    const std::vector<int>& stopTokenIds,
    const std::vector<std::vector<int>>& stopTokenSequences,
    const std::vector<int>& stopTokenHistory,
    double temperature,
    int topK,
    double topP,
    double repPenalty,
    LaunchContext* context,
    AutoregressiveDecodeConfig* config) {
    StopSequenceMatcher stopMatcher(stopTokenIds, stopTokenSequences);
    bool historyMatchedStop = stopMatcher.prime(stopTokenHistory);
    RepetitionLoopMatcher repetitionMatcher(
        config != nullptr ? config->nativeRepetitionLoopMaxPeriod : 0,
        config != nullptr ? config->nativeRepetitionLoopMaxRepeats : 0);
    bool historyMatchedRepetition = repetitionMatcher.prime(stopTokenHistory);

    // Initialize outputs
    LongType zero = 0;
    float zeroF = 0.0f;
    generatedTokenIds->assign(zero);
    tokenCount->assign(zero);
    timingInfo->assign(zeroF);
    if (config != nullptr) config->nativeFinishReason = 0;

    // Validate that we have a plan to execute — hard error, not silent return.
    REQUIRE_TRUE(config != nullptr && config->planHandle != nullptr, 0,
                 "autoregressive_decode: no plan handle provided. "
                 "The Java side MUST pass a compiled NativeDynamicShapePlan via config->planHandle. "
                 "config=%p planHandle=%p",
                 config, config ? config->planHandle : nullptr);

    if (historyMatchedStop && stopTerminationAllowed(config, 0)) return;
    if (historyMatchedRepetition) {
        config->nativeFinishReason = 1;
        if (timingInfo->lengthOf() > 6) timingInfo->p(6, -1.0f);
        return;
    }

    auto plan = config->planHandle;
    AutoregressiveP0Counters p0;
    bool p0RepairActive = false;
    bool p0MaintenanceActive = false;

    // ── Timing ──
    std::vector<double> stepTimesMs;
    std::vector<int> stepTokenCounts;
    stepTimesMs.reserve(maxNewTokens);
    stepTokenCounts.reserve(maxNewTokens);
    auto loopStart = std::chrono::high_resolution_clock::now();

    // ── Internal state ──
    LongType currentPosition = static_cast<LongType>(prefillSeqLen);
    auto hidden = embeddingTable->sizeAt(1);
    auto vocabSize = embeddingTable->sizeAt(0);
    auto embTableRowStride = embeddingTable->strideAt(0);

    // ── Build internal attention mask if not provided ──
    NDArray* internalMask = nullptr;
    LongType maxKvLen = 0;
    if (attentionMask != nullptr) {
        maxKvLen = attentionMask->sizeAt(-1);
    } else {
        maxKvLen = prefillSeqLen + maxNewTokens;
        std::vector<LongType> maskShape = {1, 1, 1, maxKvLen};
        internalMask = NDArrayFactory::create('c', maskShape, DataType::FLOAT32, context);
        internalMask->assign(zeroF);
        BUILD_SINGLE_SELECTOR(internalMask->dataType(), buildInitialMaskCpu,
                              (internalMask->buffer(), prefillSeqLen, maxKvLen),
                              SD_COMMON_TYPES);
        attentionMask = internalMask;
    }

    // ── Build internal position_ids if not provided ──
    NDArray* internalPosIds = nullptr;
    if (positionIds == nullptr) {
        std::vector<LongType> posShape = {1, 1};
        internalPosIds = NDArrayFactory::create('c', posShape, DataType::INT64, context);
        internalPosIds->p(0, static_cast<LongType>(prefillSeqLen));
        positionIds = internalPosIds;
    }

    // ── Working buffers ──
    // Reuse prefillEmbeddings for embed lookup (same as CUDA path).
    NDArray* decodeEmbedding = prefillEmbeddings;

    // Token sample output: single INT64 scalar
    std::vector<LongType> sampleShape = {1};
    NDArray* sampledToken = NDArrayFactory::create('c', sampleShape, DataType::INT64, context);

    int tokensGenerated = 0;

    // ── Get plan's external inputs from the persistent OpaqueContext ──
    auto* extCtx = reinterpret_cast<graph::Context*>(config->extInputContext);
    int numExtInputs = config->numPlanExternalInputs;

    std::vector<NDArray*> extInputsVec(numExtInputs);
    if (extCtx != nullptr) {
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = extCtx->array(i);
        }
    } else if (config->planExternalInputs != nullptr) {
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = config->planExternalInputs[i];
        }
    }
    NDArray** extInputs = extInputsVec.data();

    // ── Extract causal mask from ext inputs (if present) ──
    NDArray* causalMask = nullptr;
    LongType causalMaskLen = 0;
    if (config->causalMaskExtIdx >= 0 && config->causalMaskExtIdx < numExtInputs) {
        causalMask = extInputsVec[config->causalMaskExtIdx];
        if (causalMask != nullptr) {
            causalMaskLen = causalMask->sizeAt(-1);
        }
    }

    // GGUF in-graph models have no separate 0/1 attention mask: the pipeline passes
    // the additive causal mask as this op's attentionMask input. Writing 0/1-mask
    // semantics (mask[pos]=1) into that additive bias plants a +1 self-attention
    // bonus at row 0 every step — greedy (always row 0) and speculative rows >= 1
    // then compute different hidden states for the SAME token, breaking lossless
    // speculative equivalence. When the two masks share a buffer, the causal-mask
    // maintenance owns every update and the 0/1 update must not run.
    const bool attnMaskAliasesCausal = attentionMask != nullptr && causalMask != nullptr
        && attentionMask->dataBuffer() == causalMask->dataBuffer();

    // ── Extract attn_mask_reformat from ext inputs (if present) ──
    NDArray* attnMaskReformat = nullptr;
    LongType attnMaskReformatLen = 0;
    if (config->attnMaskReformatExtIdx >= 0 && config->attnMaskReformatExtIdx < numExtInputs) {
        attnMaskReformat = extInputsVec[config->attnMaskReformatExtIdx];
        if (attnMaskReformat != nullptr) {
            attnMaskReformatLen = attnMaskReformat->sizeAt(-1);
        }
    }

    // Plan outputs: allocate array for plan to fill
    int numPlanOutputs = plan->getNumRequestedOutputs();
    std::vector<NDArray*> planOutputsVec(numPlanOutputs, nullptr);
    NDArray** planOutputs = planOutputsVec.data();
    const bool useScalarTarget = config->scalarPlanHandle != nullptr;
    std::vector<NDArray*> scalarInputs(config->scalarNumPlanExternalInputs, nullptr);
    std::vector<NDArray*> scalarOutputs(config->scalarNumPlanOutputs, nullptr);
    if (useScalarTarget) {
        auto* scalarContext = reinterpret_cast<graph::Context*>(config->scalarExtInputContext);
        for (int i = 0; i < config->scalarNumPlanExternalInputs; ++i) scalarInputs[i] = scalarContext->array(i);
    }
    // DEEP PRE-VERIFICATION RECURRENT SNAPSHOTS (CPU mirror of the CUDA
    // scalar-binding aliasing fix): prepareScalarTarget's recurrent copy is
    // skipped for every scalar input whose DataBuffer is identical to the target
    // window ext input, so with a shared-buffer binding there was NO private
    // snapshot at all - the verify pass mutated the live window state in place
    // and the rerun double-advanced through the rejected draft rows. These
    // dedicated owned scratch arrays are captured immediately BEFORE each
    // verification execution (never read from at capture time, so buffer
    // identity cannot disable them) and are the single restore source for BOTH
    // rerun geometries. Slot layout: [0, numGdnStatePairs) are GDN pairs,
    // [numGdnStatePairs, +numConvStatePairs) are conv pairs, each paired with
    // its TARGET-domain ext input index. Allocated lazily once per decode call;
    // freed with the other internal allocations in the cleanup section.
    std::vector<NDArray*> stateSnapshotArrays_cpu;
    std::vector<int> stateSnapshotExtIdx_cpu;
    // Capture every recurrent state ext input (GDN + conv pairs) into the
    // dedicated owned snapshot arrays. Called immediately before the plan
    // execution that may mutate the live ext inputs (the verification pass), so
    // the snapshot is genuinely PRE-verification regardless of whether the
    // scalar plan's "private" arrays share buffers with the window ext inputs.
    auto capturePreVerificationState_cpu = [&]() {
        for (int s = 0; s < config->numGdnStatePairs; s++) {
            int extIdx = config->gdnStateExtIndices != nullptr
                ? config->gdnStateExtIndices[s] : -1;
            NDArray* src = (extIdx >= 0 && extIdx < numExtInputs) ? extInputs[extIdx] : nullptr;
            if (src == nullptr) continue;
            if (static_cast<int>(stateSnapshotArrays_cpu.size()) <= s) {
                stateSnapshotArrays_cpu.resize(s + 1, nullptr);
                stateSnapshotExtIdx_cpu.resize(s + 1, -1);
            }
            if (stateSnapshotArrays_cpu[s] == nullptr
                    || stateSnapshotArrays_cpu[s]->dataType() != src->dataType()
                    || stateSnapshotArrays_cpu[s]->lengthOf() != src->lengthOf()) {
                // Own allocation - never aliases the live ext input, so the
                // snapshot survives any in-place mutation the plan applies to
                // ext inputs.
                delete stateSnapshotArrays_cpu[s];
                std::vector<LongType> snapShape;
                snapShape.reserve(src->rankOf());
                for (int d = 0; d < src->rankOf(); d++) snapShape.push_back(src->sizeAt(d));
                stateSnapshotArrays_cpu[s] = NDArrayFactory::create(
                    'c', snapShape, src->dataType(), context);
                stateSnapshotExtIdx_cpu[s] = extIdx;
            }
            NDArray* snap = stateSnapshotArrays_cpu[s];
            NDArray::preparePrimaryUse({snap}, {src});
            std::memcpy(snap->buffer(), src->buffer(),
                        src->lengthOf() * src->sizeOfT());
            p0.snapshotBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            snap->tickWriteHost();
            NDArray::registerPrimaryUse({snap}, {src});
        }
        for (int s = 0; s < config->numConvStatePairs; s++) {
            int extIdx = config->convStateExtIndices != nullptr
                ? config->convStateExtIndices[s] : -1;
            int slot = config->numGdnStatePairs + s;
            NDArray* src = (extIdx >= 0 && extIdx < numExtInputs) ? extInputs[extIdx] : nullptr;
            if (src == nullptr) continue;
            if (static_cast<int>(stateSnapshotArrays_cpu.size()) <= slot) {
                stateSnapshotArrays_cpu.resize(slot + 1, nullptr);
                stateSnapshotExtIdx_cpu.resize(slot + 1, -1);
            }
            if (stateSnapshotArrays_cpu[slot] == nullptr
                    || stateSnapshotArrays_cpu[slot]->dataType() != src->dataType()
                    || stateSnapshotArrays_cpu[slot]->lengthOf() != src->lengthOf()) {
                delete stateSnapshotArrays_cpu[slot];
                std::vector<LongType> snapShape;
                snapShape.reserve(src->rankOf());
                for (int d = 0; d < src->rankOf(); d++) snapShape.push_back(src->sizeAt(d));
                stateSnapshotArrays_cpu[slot] = NDArrayFactory::create(
                    'c', snapShape, src->dataType(), context);
                stateSnapshotExtIdx_cpu[slot] = extIdx;
            }
            NDArray* snap = stateSnapshotArrays_cpu[slot];
            NDArray::preparePrimaryUse({snap}, {src});
            std::memcpy(snap->buffer(), src->buffer(),
                        src->lengthOf() * src->sizeOfT());
            p0.snapshotBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            snap->tickWriteHost();
            NDArray::registerPrimaryUse({snap}, {src});
        }
    };
    // Restore the deep pre-verification snapshots into the LIVE window ext
    // inputs the window plan reads (both rerun geometries read this storage:
    // the window plan directly, and the scalar plan indirectly -
    // prepareScalarTarget, re-run by the caller after this restore, re-stages
    // its width-1 arrays FROM the live ext inputs).
    auto restorePreVerificationState_cpu = [&]() {
        for (size_t s = 0; s < stateSnapshotArrays_cpu.size(); ++s) {
            NDArray* snap = stateSnapshotArrays_cpu[s];
            int ti = stateSnapshotExtIdx_cpu[s];
            NDArray* windowArr = (ti >= 0 && ti < numExtInputs) ? extInputs[ti] : nullptr;
            if (snap == nullptr || windowArr == nullptr
                    || snap->dataType() != windowArr->dataType()
                    || snap->lengthOf() != windowArr->lengthOf()) continue;
            NDArray::preparePrimaryUse({windowArr}, {snap});
            std::memcpy(windowArr->buffer(), snap->buffer(),
                        snap->lengthOf() * snap->sizeOfT());
            p0.restoreBytes += static_cast<std::uint64_t>(snap->lengthOf() * snap->sizeOfT());
            windowArr->tickWriteHost();
            NDArray::registerPrimaryUse({windowArr}, {snap});
        }
    };
    // K=1 MTP scalar-rerun parity (CUDA mirror): prepareScalarTarget also serves as
    // the PRE-VERIFICATION SNAPSHOT RESTORE for accepted-prefix state reruns. Calling
    // it before a rerun re-establishes the private pre-verification recurrent
    // snapshots (recurrent entries copied FROM the live window ext inputs, which hold
    // them until the verify pass mutates them in place) and geometry to the rerun's
    // asl=1 - for BOTH rerun geometries: the scalar plan's private width-1 arrays,
    // and the live window ext inputs the W plan reads (a bindingless window rerun has
    // only those).
    auto prepareScalarTarget = [&]() {
        for (int i = 0; i < config->scalarNumPlanExternalInputs; ++i) {
            NDArray* dst = scalarInputs[i];
            NDArray* src = extInputs[config->scalarInputToTarget[i]];
            if (dst->dataBuffer() == src->dataBuffer()) continue;
            bool geometry = i == config->scalarInputIdsExtIdx || i == config->scalarCausalMaskExtIdx
                || i == config->scalarPositionOffsetExtIdx || i == config->scalarCachePositionExtIdx
                || i == config->scalarActualSequenceLengthExtIdx;
            // Recurrent decision in the TARGET index domain: scalar input i maps to
            // target index ti; recurrent iff ti equals a target-domain GDN/conv
            // state index. The gdn/conv arrays are NEVER re-mapped through the
            // scalar-indexed vector (they are already target indices).
            const int ti = config->scalarInputToTarget[i];
            bool recurrent = false;
            for (int s = 0; s < config->numGdnStatePairs && !recurrent; ++s) {
                recurrent = config->gdnStateExtIndices != nullptr
                    && ti == config->gdnStateExtIndices[s];
            }
            for (int s = 0; s < config->numConvStatePairs && !recurrent; ++s) {
                recurrent = config->convStateExtIndices != nullptr
                    && ti == config->convStateExtIndices[s];
            }
            // Mirror the CUDA contract: refresh geometry + recurrent snapshots only;
            // weights and derived inputs keep their captured values.
            if (!geometry && !recurrent) continue;
            REQUIRE_TRUE(dst->lengthOf() <= src->lengthOf(), 0,
                         "autoregressive_decode: scalar source is smaller than captured input");
            NDArray::preparePrimaryUse({dst}, {src});
            std::memcpy(dst->buffer(), src->buffer(), dst->lengthOf() * dst->sizeOfT());
            NDArray::registerPrimaryUse({dst}, {src});
        }
        scalarInputs[config->scalarActualSequenceLengthExtIdx]->p(0, static_cast<LongType>(1));
    };
    // NOTE (rerun geometry ordering): at the accepted-prefix rerun site the
    // deep restore runs first (restorePreVerificationState_cpu), THEN the
    // rerun's asl write, THEN prepareScalarTarget() re-stages the scalar
    // arrays. prepareScalarTarget copies geometry from the live ext inputs,
    // so the asl write must precede it for a scalar rerun to observe asl=1.
    auto executeScalarTarget = [&]() {
        DSP_DIAG(KV_CACHE, "SCALAR_TARGET_SELECTED plan=%p idsWidth=1 maskRows=1 position=%lld inputs=%d outputs=%d",
                 config->scalarPlanHandle, static_cast<long long>(currentPosition),
                 config->scalarNumPlanExternalInputs, config->scalarNumPlanOutputs);
        Status status = config->scalarPlanHandle->execute(
            scalarInputs.data(), config->scalarNumPlanExternalInputs,
            scalarOutputs.data(), config->scalarNumPlanOutputs, nullptr);
        if (status == Status::OK) {
            for (int i = 0; i < numPlanOutputs; ++i) {
                planOutputs[i] = scalarOutputs[config->targetOutputToScalar[i]];
                REQUIRE_TRUE(planOutputs[i] != nullptr, 0,
                             "autoregressive_decode: scalar target returned a null requested output");
            }
            auto* logits = scalarOutputs[config->scalarLogitsOutputIdx];
            REQUIRE_TRUE(logits->rankOf() >= 2 && logits->rankOf() <= 3
                             && logits->sizeAt(0) == 1
                             && (logits->rankOf() == 2 || logits->sizeAt(1) == 1), 0,
                         "autoregressive_decode: scalar target returned non-scalar logits geometry");
        }
        return status;
    };

    REQUIRE_TRUE(extCtx != nullptr || config->planExternalInputs != nullptr, 0,
                 "autoregressive_decode: no external input source. "
                 "Either extInputContext (OpaqueContext*) or planExternalInputs (NDArray**) "
                 "must be non-null. Both are null — cannot wire plan inputs.");

    // Placeholder inputs: host-written by Java each step → force H2D on DSP stream
    if (config->embeddingsExtIdx >= 0) plan->markExternalInputPlaceholder(config->embeddingsExtIdx);
    if (config->maskExtIdx >= 0) plan->markExternalInputPlaceholder(config->maskExtIdx);
    if (config->posIdsExtIdx >= 0) plan->markExternalInputPlaceholder(config->posIdsExtIdx);
    if (config->inputIdsExtIdx >= 0) plan->markExternalInputPlaceholder(config->inputIdsExtIdx);
    if (config->causalMaskExtIdx >= 0) plan->markExternalInputPlaceholder(config->causalMaskExtIdx);
    if (config->attnMaskReformatExtIdx >= 0) plan->markExternalInputPlaceholder(config->attnMaskReformatExtIdx);
    if (config->positionOffsetExtIdx >= 0) plan->markExternalInputPlaceholder(config->positionOffsetExtIdx);
    if (config->cachePositionExtIdx >= 0) plan->markExternalInputPlaceholder(config->cachePositionExtIdx);
    if (config->actualSequenceLengthExtIdx >= 0) {
        plan->markExternalInputPlaceholder(config->actualSequenceLengthExtIdx);
    }
    // GDN/conv state: device-written via D2D copy on DSP stream — NOT placeholder
    if (config->numGdnStatePairs > 0 && config->gdnStateExtIndices != nullptr) {
        for (int s = 0; s < config->numGdnStatePairs; s++) {
            int extIdx = config->gdnStateExtIndices[s];
            if (extIdx >= 0) plan->markExternalInputVariable(extIdx);
        }
    }
    if (config->numConvStatePairs > 0 && config->convStateExtIndices != nullptr) {
        for (int s = 0; s < config->numConvStatePairs; s++) {
            int extIdx = config->convStateExtIndices[s];
            if (extIdx >= 0) plan->markExternalInputVariable(extIdx);
        }
    }
    // KV cache: device-written by attention kernels — NOT placeholder
    if (config->kvInputExtIndices != nullptr) {
        for (int kv = 0; kv < 2 * numKvPairs; kv++) {
            int kvIdx = config->kvInputExtIndices[kv];
            if (kvIdx >= 0) plan->markExternalInputVariable(kvIdx);
        }
    }

    // ── ADR 0106 Phase 1: window substrate ──────────────────────────────────
    // When activeWindow > 1, we replace the single-token mask/posIds with
    // the pre-allocated window tensors. These are updated in-place each step
    // so device addresses remain stable (pointer-stability contract, ADR 0105).
    // The MASK_FILL value for float masks that mark positions as inaccessible.
    constexpr float WINDOW_MASK_FILL = -3.4028235e+38f;

    const bool useWindowSubstrate = (config->windowMax > 1
                                     && config->windowGridMask != nullptr
                                     && (config->windowPositionGrid != nullptr || config->planOwnsKvScatter));

    // ── ADR 0106 Phase 2 speculative decode state (CPU) ───────────────────
    const int specK_cpu = config->speculativeK;
    const bool useNgram_cpu = (specK_cpu > 0
                               && config->speculatorType == 1
                               && useWindowSubstrate
                               && config->windowMax >= specK_cpu + 1);
    const bool useMtp_cpu = (specK_cpu > 0
                             && config->speculatorType == 2
                             && useWindowSubstrate
                             && config->windowMax >= specK_cpu + 1
                             && config->mtpPlanHandle != nullptr
                             && config->mtpExtInputContext != nullptr);
    const bool useSpeculative_cpu = useNgram_cpu || useMtp_cpu;

    // Host-side n-gram tables learned only from verified output tokens.
    std::unordered_map<LongType, LongType> ngramTable_cpu;
    std::unordered_map<LongType, std::unordered_map<LongType, LongType>> trigramTable_cpu;
    if (useNgram_cpu) {
        ngramTable_cpu.reserve(256);
        trigramTable_cpu.reserve(256);
    }
    LongType specPreviousToken_cpu = -1;
    LongType specCurrentToken_cpu = -1;

    // Qwen3.5's bundled predictor is an independent scalar DSP plan. Its
    // external-input addresses are stable for the whole decode call.
    // MTP METADATA vs MTP DRAFTING (review round 4, finding D/5 - CUDA
    // mirror): the predictor RESOURCES (plan, context, ext-input wiring) are
    // gated on metadata presence, NOT on useMtp_cpu (which additionally
    // requires specK>0). With the Java-side K=0 wiring fixed, the session
    // now attaches MTP resources at effective K=0, and the CPU epilogue's
    // maintenance/publication paths must keep the predictor state advancing
    // across scalar-only stretches ("MTP resources present, drafting
    // disabled, predictor maintained").
    const bool mtpMetadataReady_cpu = config->mtpPlanHandle != nullptr
                                      && config->mtpExtInputContext != nullptr;
    graph::NativeDynamicShapePlan* mtpPlan_cpu = (useMtp_cpu || mtpMetadataReady_cpu)
        ? config->mtpPlanHandle : nullptr;
    graph::Context* mtpContext_cpu = (useMtp_cpu || mtpMetadataReady_cpu)
        ? reinterpret_cast<graph::Context*>(config->mtpExtInputContext) : nullptr;
    std::vector<NDArray*> mtpExtInputsVec_cpu;
    std::vector<NDArray*> mtpPlanOutputsVec_cpu;
    NDArray** mtpExtInputs_cpu = nullptr;
    NDArray** mtpPlanOutputs_cpu = nullptr;
    int mtpNumExtInputs_cpu = 0;
    int mtpNumOutputs_cpu = 0;
    LongType mtpMaskLen_cpu = 0;

    if (mtpPlan_cpu != nullptr && mtpContext_cpu != nullptr) {
        REQUIRE_TRUE(mtpContext_cpu != nullptr, 0,
                     "autoregressive_decode: MTP CPU context is null");
        mtpNumExtInputs_cpu = config->mtpNumPlanExternalInputs;
        mtpNumOutputs_cpu = mtpPlan_cpu->getNumRequestedOutputs();
        REQUIRE_TRUE(mtpNumExtInputs_cpu > 0 && mtpNumOutputs_cpu > 0, 0,
                     "autoregressive_decode: invalid MTP CPU plan dimensions inputs=%d outputs=%d",
                     mtpNumExtInputs_cpu, mtpNumOutputs_cpu);
        auto validMtpExtIdx_cpu = [&](int idx) {
            return idx >= 0 && idx < mtpNumExtInputs_cpu;
        };
        REQUIRE_TRUE(validMtpExtIdx_cpu(config->mtpInputIdsExtIdx)
                         && validMtpExtIdx_cpu(config->mtpTargetHiddenExtIdx)
                         && validMtpExtIdx_cpu(config->mtpCausalMaskExtIdx)
                         && validMtpExtIdx_cpu(config->mtpPositionOffsetExtIdx)
                         && validMtpExtIdx_cpu(config->mtpCachePositionExtIdx)
                         && validMtpExtIdx_cpu(config->mtpKvInputExtIndices[0])
                         && validMtpExtIdx_cpu(config->mtpKvInputExtIndices[1]),
                     0, "autoregressive_decode: MTP CPU external-input index is out of range");

        mtpExtInputsVec_cpu.resize(mtpNumExtInputs_cpu);
        for (int i = 0; i < mtpNumExtInputs_cpu; i++) {
            mtpExtInputsVec_cpu[i] = mtpContext_cpu->array(i);
        }
        mtpExtInputsVec_cpu[config->mtpInputIdsExtIdx] = config->mtpInputIds;
        mtpExtInputsVec_cpu[config->mtpTargetHiddenExtIdx] = config->mtpTargetHidden;
        mtpExtInputsVec_cpu[config->mtpCausalMaskExtIdx] = config->mtpCausalMask;
        mtpExtInputsVec_cpu[config->mtpPositionOffsetExtIdx] = config->mtpPositionOffset;
        mtpExtInputsVec_cpu[config->mtpCachePositionExtIdx] = config->mtpCachePosition;
        mtpExtInputsVec_cpu[config->mtpKvInputExtIndices[0]] = config->mtpKvBuffers[0];
        mtpExtInputsVec_cpu[config->mtpKvInputExtIndices[1]] = config->mtpKvBuffers[1];
        mtpExtInputs_cpu = mtpExtInputsVec_cpu.data();

        mtpPlanOutputsVec_cpu.resize(mtpNumOutputs_cpu, nullptr);
        mtpPlanOutputs_cpu = mtpPlanOutputsVec_cpu.data();
        mtpMaskLen_cpu = config->mtpCausalMask->sizeAt(-1);

        mtpPlan_cpu->markExternalInputPlaceholder(config->mtpInputIdsExtIdx);
        mtpPlan_cpu->markExternalInputPlaceholder(config->mtpTargetHiddenExtIdx);
        mtpPlan_cpu->markExternalInputPlaceholder(config->mtpCausalMaskExtIdx);
        mtpPlan_cpu->markExternalInputPlaceholder(config->mtpPositionOffsetExtIdx);
        mtpPlan_cpu->markExternalInputPlaceholder(config->mtpCachePositionExtIdx);
        mtpPlan_cpu->markExternalInputVariable(config->mtpKvInputExtIndices[0]);
        mtpPlan_cpu->markExternalInputVariable(config->mtpKvInputExtIndices[1]);
    }

    // CPU-side argmax helper: returns argmax over T* logits of length vocabSize.
    // Used in the speculative path to evaluate multiple rows of logits.
    auto cpuArgmax = [&](const void* logitsRowPtr, LongType vocabSize,
                          DataType dtype) -> LongType {
        REQUIRE_TRUE(logitsRowPtr != nullptr && vocabSize > 0, 0,
                     "autoregressive_decode: speculative logits row must be non-empty");
        BUILD_SINGLE_SELECTOR(dtype, return speculativeArgmaxCpu,
                              (logitsRowPtr, vocabSize), SD_FLOAT_TYPES);
    };

    // Helper: byte stride for one logits row given the dtype.
    auto logitsByteStride = [&](DataType dtype, LongType vocabSize) -> LongType {
        return vocabSize * DataTypeUtils::sizeOfElement(dtype);
    };

    auto executeMtpCpu = [&](LongType tokenId, LongType position) -> LongType {
        // PREDICTOR ROW MAPPING (packet 2, CPU mirror of executeMtpCuda): the
        // argument is a TARGET input-token position P; the predictor consumes
        // the pair (x_(P+1), h_P) at predictor row r = P - 1 (rope = r,
        // slot = r). Callers keep target coordinates; this boundary converts
        // exactly once. Bounds: P >= 1 and the converted row must fit the
        // predictor mask and both KV buffers BEFORE any predictor-cache
        // indexing - a caller bug surfaces here as a loud failure, not a clamp.
        REQUIRE_TRUE(position >= 1, 0,
                     "autoregressive_decode: MTP CPU target token position "
                     "must be >= 1, got %lld",
                     (long long)position);
        const LongType predictorRow = position - 1;
        // CACHE LAYOUT CONTRACT (review round 4, finding E): the MTP predictor
        // KV cache is BSHD [batch, maxSeqLen, heads, dim] (kv_scatter.h:148,
        // kvInPlaceWriteBSHD reads cacheMaxSeqLen = sizeAt(1)) - the SEQUENCE
        // dimension is dim 1, unambiguously. No max() heuristic: a transposed
        // cache where heads > seq would otherwise pass the old check.
        REQUIRE_TRUE(config->mtpKvBuffers[0] != nullptr && config->mtpKvBuffers[1] != nullptr,
                     0, "autoregressive_decode: MTP CPU predictor KV buffers are unavailable");
        REQUIRE_TRUE(config->mtpKvBuffers[0]->rankOf() == 4 && config->mtpKvBuffers[1]->rankOf() == 4,
                     0, "autoregressive_decode: MTP CPU predictor KV buffers must be rank 4 "
                        "[batch, maxSeqLen, heads, dim]");
        const LongType kvRows0_cpu = config->mtpKvBuffers[0]->sizeAt(1);
        const LongType kvRows1_cpu = config->mtpKvBuffers[1]->sizeAt(1);
        REQUIRE_TRUE(predictorRow < mtpMaskLen_cpu
                && predictorRow < kvRows0_cpu
                && predictorRow < kvRows1_cpu,
            0,
            "autoregressive_decode: MTP CPU predictor row %lld (target position %lld) "
            "is outside cache/mask capacity (seq capacity %lld/%lld, mask %lld)",
            (long long)predictorRow, (long long)position,
            (long long)kvRows0_cpu, (long long)kvRows1_cpu, (long long)mtpMaskLen_cpu);
        // DRAFTING gate: the maintenance callers below may run executeMtpCpu
        // with useMtp_cpu false (resource-present K=0); a null plan is still
        // a hard error.
        REQUIRE_TRUE(mtpPlan_cpu != nullptr && mtpContext_cpu != nullptr, 0,
                     "autoregressive_decode: attempted MTP CPU execution while MTP is disabled");
        config->mtpInputIds->p(0, tokenId);
        config->mtpPositionOffset->p(0, predictorRow);
        config->mtpCachePosition->p(0, predictorRow);
        BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskCpu,
                              (config->mtpCausalMask->buffer(), predictorRow, mtpMaskLen_cpu),
                              SD_FLOAT_TYPES);

        if (p0RepairActive) {
            p0.predictorRepairForwards++;
            p0.predictorRepairLmHeadForwards++;
        } else if (p0MaintenanceActive) {
            p0.predictorMaintenanceForwards++;
        } else {
            p0.predictorProposalForwards++;
        }
        Status mtpStatus = mtpPlan_cpu->execute(
            mtpExtInputs_cpu, mtpNumExtInputs_cpu,
            mtpPlanOutputs_cpu, mtpNumOutputs_cpu,
            nullptr);
        std::string mtpFailureDetail;
        if (mtpStatus != Status::OK) mtpFailureDetail = nestedPlanFailureDetail();
        REQUIRE_TRUE(mtpStatus == Status::OK, 0,
                     "%s [autoregressive_decode nested MTP CPU plan position=%lld, status=%s (%d)]",
                     mtpFailureDetail.c_str(), (long long)position,
                     graph::dsp::dspStatusName(mtpStatus), static_cast<int>(mtpStatus));
        REQUIRE_TRUE(config->mtpLogitsOutputIdx >= 0
                         && config->mtpLogitsOutputIdx < mtpNumOutputs_cpu
                         && mtpPlanOutputs_cpu[config->mtpLogitsOutputIdx] != nullptr,
                     0, "autoregressive_decode: MTP CPU logits output is unavailable");
        REQUIRE_TRUE(config->mtpHiddenOutputIdx >= 0
                         && config->mtpHiddenOutputIdx < mtpNumOutputs_cpu
                         && mtpPlanOutputs_cpu[config->mtpHiddenOutputIdx] != nullptr,
                     0, "autoregressive_decode: MTP CPU hidden output is unavailable");

        NDArray* mtpLogits = mtpPlanOutputs_cpu[config->mtpLogitsOutputIdx];
        LongType mtpVocab = mtpLogits->sizeAt(mtpLogits->rankOf() - 1);
        LongType draft = cpuArgmax(mtpLogits->buffer(), mtpVocab, mtpLogits->dataType());
        config->mtpTargetHidden->assign(mtpPlanOutputs_cpu[config->mtpHiddenOutputIdx]);
        return draft;
    };

    auto setMtpTargetCarryCpu = [&](NDArray* targetHiddenRows, int row) {
        REQUIRE_TRUE(targetHiddenRows != nullptr && targetHiddenRows->rankOf() == 3, 0,
                     "autoregressive_decode: target hidden output must be rank 3 for MTP carry");
        REQUIRE_TRUE(row >= 0 && row < targetHiddenRows->sizeAt(1), 0,
                     "autoregressive_decode: MTP carry row %d outside target hidden sequence %lld",
                     row, (long long)targetHiddenRows->sizeAt(1));
        std::vector<LongType> hiddenSlice{
            0, 1, static_cast<LongType>(row), static_cast<LongType>(row + 1),
            0, targetHiddenRows->sizeAt(2)};
        NDArray* hiddenRow = (*targetHiddenRows)(hiddenSlice, true);
        config->mtpTargetHidden->assign(hiddenRow);
        delete hiddenRow;
    };

    LongType totalSpeculativeProposed = 0;
    LongType totalSpeculativeAccepted = 0;
    LongType speculativeStepCount = 0;

    // Adaptive MTP chain-depth cap (mirrors the CUDA helper): recursive drafts
    // feed the predictor its own output hidden — out-of-distribution for heads
    // trained only on trunk hidden. Positions whose evaluations never accept
    // cost a full predictor execution per step for nothing; cap past them.
    int mtpChainCap_cpu = specK_cpu;
    int mtpPosEvaluated_cpu[33] = {};
    int mtpPosAccepted_cpu[33] = {};
    constexpr int MTP_CHAIN_CAP_MIN_EVALS_CPU = 12;

    for (int step = 0; step < maxNewTokens; step++) {
        // Cancellation is observed only at a committed step boundary. This
        // keeps KV/recurrent state coherent for a later continuation.
        if (config->cancelCallback != nullptr &&
                config->cancelCallback(config->callbackUserData)) {
            break;
        }
        // Multi-token speculative steps advance tokensGenerated faster than the
        // step counter — without this check the next step writes past the
        // generatedTokenIds buffer (maxNewTokens-sized) and over-reports count.
        if (tokensGenerated >= maxNewTokens) break;
        const int tokensBeforeStep = tokensGenerated;
        auto stepStart = std::chrono::high_resolution_clock::now();

        // ── Step 1: Update plan external inputs for this decode step ──
        if (config->embeddingsExtIdx >= 0 && config->embeddingsExtIdx < numExtInputs) {
            extInputs[config->embeddingsExtIdx] = decodeEmbedding;
        }

        // ── ADR 0106 Phase 2: build proposals for this step (CPU) ───────────
        int proposedCount_cpu = 0;
        int order3Hits_cpu = 0;
        int order2Hits_cpu = 0;
        LongType draftIds_cpu[33] = {};

        int maxPropose_cpu = (specK_cpu < 32) ? specK_cpu : 32;
        int remainingOutput_cpu = maxNewTokens - tokensGenerated;
        int outputDraftCapacity_cpu = remainingOutput_cpu - 1;
        if (outputDraftCapacity_cpu < maxPropose_cpu) {
            maxPropose_cpu = outputDraftCapacity_cpu;
        }
        LongType remainingKv_cpu = maxKvLen - currentPosition;
        LongType kvDraftCapacity_cpu = remainingKv_cpu - 1;
        if (kvDraftCapacity_cpu < static_cast<LongType>(maxPropose_cpu)) {
            maxPropose_cpu = kvDraftCapacity_cpu > 0
                ? static_cast<int>(kvDraftCapacity_cpu) : 0;
        }
        if (maxPropose_cpu < 0) maxPropose_cpu = 0;
        if (useMtp_cpu && maxPropose_cpu > mtpChainCap_cpu) maxPropose_cpu = mtpChainCap_cpu;

        if (useNgram_cpu && specCurrentToken_cpu >= 0) {
            LongType previous = specPreviousToken_cpu;
            LongType current = specCurrentToken_cpu;
            for (int p = 0; p < maxPropose_cpu; p++) {
                LongType next = -1;
                bool found = false;
                if (previous >= 0) {
                    auto outer = trigramTable_cpu.find(previous);
                    if (outer != trigramTable_cpu.end()) {
                        auto inner = outer->second.find(current);
                        if (inner != outer->second.end()) {
                            next = inner->second;
                            found = true;
                            order3Hits_cpu++;
                        }
                    }
                }
                if (!found) {
                    auto backoff = ngramTable_cpu.find(current);
                    if (backoff != ngramTable_cpu.end()) {
                        next = backoff->second;
                        found = true;
                        order2Hits_cpu++;
                    }
                }
                if (!found) break;
                draftIds_cpu[p] = next;
                proposedCount_cpu++;
                previous = current;
                current = next;
            }
            DSP_DIAG(KV_CACHE,
                     "NGRAM_PROPOSE step=%d previous=%lld current=%lld proposed=%d order3=%d order2=%d",
                     step, (long long)specPreviousToken_cpu, (long long)specCurrentToken_cpu,
                     proposedCount_cpu, order3Hits_cpu, order2Hits_cpu);
        } else if (useMtp_cpu) {
            LongType mtpToken = inputIds->e<LongType>(0);
            if (maxPropose_cpu == 0) {
                // Keep the predictor cache aligned even when only one target token
                // fits in the remaining output/KV envelope.
                p0MaintenanceActive = true;
                (void)executeMtpCpu(mtpToken, currentPosition);
                p0MaintenanceActive = false;
            } else {
                for (int p = 0; p < maxPropose_cpu; p++) {
                    LongType draft = executeMtpCpu(mtpToken, currentPosition + p);
                    draftIds_cpu[p] = draft;
                    proposedCount_cpu++;
                    mtpToken = draft;
                }
            }
            DSP_DIAG(KV_CACHE,
                     "MTP_PROPOSE step=%d basePos=%lld proposed=%d draft=[%lld,%lld,%lld,%lld]",
                     step, (long long)currentPosition, proposedCount_cpu,
                     (long long)draftIds_cpu[0], (long long)draftIds_cpu[1],
                     (long long)draftIds_cpu[2], (long long)draftIds_cpu[3]);
        }

        if (proposedCount_cpu > 0) {
            config->activeWindow = 1 + proposedCount_cpu;
        }

        if (useSpeculative_cpu && proposedCount_cpu > 0 && config->planOwnsKvScatter
                && inputIds->lengthOf() >= proposedCount_cpu + 1) {
            for (int p = 0; p < proposedCount_cpu; p++) {
                inputIds->p(0LL, static_cast<LongType>(p + 1), draftIds_cpu[p]);
            }
        }

        // ── ADR 0106 Phase 1: window mask and position grid update ──────────
        // When W>1, fill the fixed [1,1,W_max,past+W_max] mask and [1,W_max] posGrid
        // in-place, then wire them into the plan's ext inputs in place of the 1-wide
        // tensors. When W=1 the existing path runs unmodified.
        if (useWindowSubstrate) {
            NDArray* wMask = config->windowGridMask;
            NDArray* wPos  = config->windowPositionGrid;
            LongType wMax  = config->windowMax;
            LongType aW    = config->activeWindow;
            // wMask shape: [1,1,wMax,past+wMax] — treat as flat rows of length (past+wMax)
            LongType rowLen = wMask->sizeAt(3);  // past_len + wMax

            // Fill entire mask with MASK_FILL (all masked).
            // assign() requires a non-const reference, so copy into a local variable.
            float maskFillVal = WINDOW_MASK_FILL;
            wMask->assign(maskFillVal);

            // wMask is [1, 1, wMax, rowLen] — use 4D indexing p(batch, head, w, k, value).
            float zeroVal = 0.0f;
            for (LongType w = 0; w < wMax; w++) {
                // Keep every fixed-width row causal, including inactive padding rows.
                // This avoids all-masked softmax NaNs; actual_sequence_length controls
                // which rows are allowed to update recurrent state.
                LongType causalEnd = std::min(rowLen, currentPosition + w + 1);
                for (LongType k = 0; k < causalEnd; k++) {
                    wMask->p(0LL, 0LL, w, k, zeroVal);
                }
            }

            // Fill window position grid when the target graph has an explicit position_ids input.
            if (wPos != nullptr) {
                for (LongType w = 0; w < wMax; w++) {
                    LongType pos = (w < aW) ? (currentPosition + w) : currentPosition;
                    wPos->p(0LL, w, pos);
                }
            }

            // Wire window tensors into ext inputs (replacing 1-wide mask/posIds)
            if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
                extInputs[config->maskExtIdx] = wMask;
            }
            if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
                extInputs[config->posIdsExtIdx] = wPos;
            }
        } else {
            if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
                extInputs[config->maskExtIdx] = attentionMask;
            }
            if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
                extInputs[config->posIdsExtIdx] = positionIds;
            }
        }

        if (config->inputIdsExtIdx >= 0 && config->inputIdsExtIdx < numExtInputs) {
            extInputs[config->inputIdsExtIdx] = inputIds;
        }

        if (causalMask != nullptr && config->causalMaskExtIdx >= 0 && config->causalMaskExtIdx < numExtInputs) {
            extInputs[config->causalMaskExtIdx] = causalMask;
        }

        if (config->kvInputExtIndices != nullptr && staticKvBuffers != nullptr) {
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvIdx = config->kvInputExtIndices[kv];
                if (kvIdx >= 0 && kvIdx < numExtInputs) {
                    extInputs[kvIdx] = staticKvBuffers[kv];
                }
            }
        }


        if (config->actualSequenceLengthExtIdx >= 0
                && config->actualSequenceLengthExtIdx < numExtInputs) {
            NDArray* actualSeqLen = extInputs[config->actualSequenceLengthExtIdx];
            if (actualSeqLen != nullptr) {
                actualSeqLen->p(0, static_cast<LongType>(config->activeWindow));
            }
        }

        // ── Step 1b: Pre-unmask the CURRENT position in causal mask ──
        // GGUF only (planOwnsKvScatter == true): the attention op writes KV at
        // cache_position = currentPosition in-place, then attends to the full buffer
        // including that position. Pre-unmasking is required for correct self-attention.
        //
        // ONNX/external-scatter path (planOwnsKvScatter == false): KV scatter happens
        // AFTER execution. Position currentPosition is empty during plan execution —
        // pre-unmasking it here exposes an empty-KV slot that the Java reference does
        // not expose, causing logit divergence at the step where currentPosition first
        // exceeds all positions already unmasked by the Java warmup.
        if (config->planOwnsKvScatter) {
            if (causalMask != nullptr && currentPosition >= 0 && currentPosition < causalMaskLen) {
                if (causalMask->rankOf() == 4 && causalMask->sizeAt(2) > 1) {
                    // W-wide window mask: the per-row causal band moves with
                    // currentPosition every step — a single-column unmask only
                    // ever advances row 0 (flat index < maxKvLen). Refill all rows.
                    BUILD_SINGLE_SELECTOR(causalMask->dataType(), refillWindowCausalMaskCpu,
                                          (causalMask->buffer(), causalMask->sizeAt(2),
                                           causalMask->sizeAt(3), currentPosition),
                                          SD_FLOAT_TYPES);
                } else {
                    BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskCpu,
                                          (causalMask->buffer(), currentPosition, causalMaskLen),
                                          SD_FLOAT_TYPES);
                }
            }
            if (attnMaskReformat != nullptr && currentPosition >= 0 && currentPosition < attnMaskReformatLen) {
                BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskCpu,
                                      (attnMaskReformat->buffer(), currentPosition, attnMaskReformatLen),
                                      SD_FLOAT_TYPES);
            }
            if (!attnMaskAliasesCausal && currentPosition >= 0 && currentPosition < maxKvLen) {
                BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskCpu,
                                      (attentionMask->buffer(), currentPosition, maxKvLen),
                                      SD_COMMON_TYPES);
            }
        }

        // ── Step 2: Execute plan ──
        // On CPU, use execute() instead of executeSteadyState() (which is CUDA-only).

        // ADR 0107 V2: inject scale buffers into the thread-local registry so that
        // dot_product_attention_v2 can look them up by INT8 KV cache pointer identity.
        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr
            && config->kvInputExtIndices != nullptr) {
            static thread_local std::vector<NDArray*> tl_kvQuantPtrs;
            int N = numKvPairs;
            tl_kvQuantPtrs.resize(N);
            for (int ki = 0; ki < N; ki++) {
                int extIdx = config->kvInputExtIndices[ki];
                tl_kvQuantPtrs[ki] = (extIdx >= 0 && extIdx < numExtInputs)
                    ? extInputs[extIdx] : nullptr;
            }
            setKvScaleRegistry(tl_kvQuantPtrs.data(), config->kvScaleBuffers, N);
        }

        if (useSpeculative_cpu && proposedCount_cpu > 0) {
            // DEEP pre-verification snapshot (CPU mirror of the CUDA scalar-binding
            // aliasing fix): copy the recurrent state ext inputs into DEDICATED owned
            // arrays so a rerun can advance consumed rows from the pre-step state
            // instead of the post-verification state this step's verify pass leaves
            // in place. Gated on proposedCount_cpu > 0: with no proposals the state
            // commit happens inline (no rerun fires), so no snapshot is consumed and
            // the ext inputs already hold the authoritative committed state.
            // Runs for BOTH bindings: with a shared-buffer scalar binding,
            // prepareScalarTarget copied nothing and the rerun would execute from
            // post-verify state (mtp-fix-gate2 CUDA NaN guard, step=1).
            // Called BEFORE prepareScalarTarget and BEFORE the verification plan
            // execution - the snapshot is genuinely pre-verify.
            capturePreVerificationState_cpu();
        }
        if (useScalarTarget) prepareScalarTarget();
        if (proposedCount_cpu > 0) p0.targetVerificationForwards++;
        Status planStatus = useScalarTarget && proposedCount_cpu == 0 ? executeScalarTarget() : plan->execute(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            nullptr);

        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
            clearKvScaleRegistry();
        }

        std::string planFailureDetail;
        if (planStatus != Status::OK) planFailureDetail = nestedPlanFailureDetail();
        REQUIRE_TRUE(planStatus == Status::OK, 0,
                     "%s [autoregressive_decode nested plan step=%d, status=%s (%d), "
                     "frozen=%d, numExt=%d, numOutputs=%d]",
                     planFailureDetail.c_str(), step,
                     graph::dsp::dspStatusName(planStatus), static_cast<int>(planStatus),
                     plan->isShapesFrozen() ? 1 : 0,
                     numExtInputs, numPlanOutputs);

        REQUIRE_TRUE(config->logitsOutputIdx >= 0 && config->logitsOutputIdx < numPlanOutputs, 0,
                     "autoregressive_decode: logitsOutputIdx=%d out of range [0,%d) at step %d. "
                     "The plan has fewer outputs than expected or logitsOutputIdx was not set.",
                     config->logitsOutputIdx, numPlanOutputs, step);
        REQUIRE_TRUE(planOutputs[config->logitsOutputIdx] != nullptr, 0,
                     "autoregressive_decode: logits output NDArray* is null at step %d (idx=%d). "
                     "Plan returned OK but did not populate the logits output slot.",
                     step, config->logitsOutputIdx);

#if defined(SD_VULKAN)
        // Vulkan owns the forward and KV buffers, while the canonical scalar
        // sampling policy is intentionally host-orchestrated for the first
        // mobile runtime. Synchronize exactly at the logits sampling boundary;
        // the per-step input/mask writes below are synchronized back to the
        // device before the next replay. A Vulkan sampler can replace this one
        // boundary later without changing the session or JavaCPP APIs.
        planOutputs[config->logitsOutputIdx]->syncToHost();
#endif

        // Validate logits output buffer is not stale/closed.
        {
            NDArray* logitsArr = planOutputs[config->logitsOutputIdx];
            auto* logitsDb = logitsArr->dataBuffer();
            REQUIRE_TRUE(logitsDb != nullptr, 0,
                         "autoregressive_decode: logits DataBuffer is null at step %d. "
                         "Output array exists but has no backing buffer — likely a stale slot.",
                         step);
            REQUIRE_TRUE(!logitsDb->isClosed(), 0,
                         "autoregressive_decode: logits DataBuffer is CLOSED at step %d. "
                         "The plan reused a freed buffer — stale slot reuse bug.",
                         step);
            REQUIRE_TRUE(logitsArr->buffer() != nullptr, 0,
                         "autoregressive_decode: logits host buffer is null at step %d. "
                         "Buffer exists but has no host allocation.",
                         step);
        }

        // ── ADR 0106 Phase 2: accepted-prefix recurrent-state commit ────────────
        // The verification forward ran with actual_sequence_length = 1 + proposedCount,
        // which advances GDN/conv recurrent state through ALL proposed rows. When any
        // draft is rejected, that state includes rejected tokens — the next step would
        // decode from polluted state (first token divergence lands at the step after
        // the first partial/zero acceptance). Compute acceptance HERE, from the FIRST
        // pass's logits, and on partial acceptance or terminal truncation re-execute
        // with actual_sequence_length = emitted count. The feedback copies below
        // advance through the consumed input prefix only. Emission
        // uses the FIRST pass's argmaxes (captured into specRowArgmax_cpu); the
        // downstream speculative block consumes these instead of recomputing from the
        // re-run's logits (whose rows beyond the accepted prefix are not meaningful).
        int specAccepted_cpu = -1;   // -1 = not a proposing step
        int specConsumed_cpu = 0;
        bool specShouldStop_cpu = false;
        LongType specRowArgmax_cpu[33] = {};
        // IMMUTABLE VERIFICATION WINNERS (review round 5, finding 1): captured
        // once per proposing step from the FIRST verification pass, never
        // overwritten by any rerun readout. The rerun disagreement comparison
        // and any recovery path must compare against THIS array, not against
        // specRowArgmax_cpu[0] (which the scalar-binding rerun block replaces
        // with the rerun winner - comparing against it made the disagreement
        // gate self-compare and skip shortening/matcher repair on real A->B
        // flips). Declared at step scope: the finalize block below reads it
        // outside the capture scope.
        LongType verifyRowArgmax_cpu[33] = {};
        // Exact pre-provisional-accept matcher checkpoint: the truncated multi-row
        // commit path (RERUN_SHORTEN_REEXEC, review round 3 finding 3) restores it
        // before re-accepting the single authoritative token, so the suffix never
        // retains rows from the invalidated verification sequence.
        StopSequenceMatcher::Snapshot matcherPreLoopSnapshot_cpu;
        if (useSpeculative_cpu && proposedCount_cpu > 0
                && planOutputs[config->logitsOutputIdx] != nullptr
                && planOutputs[config->logitsOutputIdx]->rankOf() == 3) {
            NDArray* firstPassLogits = planOutputs[config->logitsOutputIdx];
            LongType fpVocab = firstPassLogits->sizeAt(2);
            LongType fpStride = logitsByteStride(firstPassLogits->dataType(), fpVocab);
            const char* fpBase = reinterpret_cast<const char*>(firstPassLogits->buffer());
            int fpRows = 1 + proposedCount_cpu;
            for (int row = 0; row < fpRows && row < 33; row++) {
                specRowArgmax_cpu[row] = cpuArgmax(fpBase + row * fpStride, fpVocab,
                                                   firstPassLogits->dataType());
            }
            // VERIFIER VALIDITY GATE, CPU mirror (review round 6, finding 3):
            // every ACTIVE verification row whose result feeds an acceptance
            // decision or an emitted token must be NaN-free. The CUDA gate runs
            // in the multi-row argmax kernel; here the full-row scan happens on
            // the host through the dtype-selective sampler. Without this gate a
            // fully accepted batch (rerun/recovery guard never runs) could emit
            // a correction/bonus token decided from an all-NaN row (cpuArgmax
            // keeps index 0). Rows beyond the active prefix are not validated.
            {
                bool anyInvalid = false;
                int firstInvalidRow = -1;
                for (int row = 0; row < fpRows && !anyInvalid; row++) {
                    for (LongType v = 0; v < fpVocab && !anyInvalid; v++) {
                        float sampled = 0.0f;
                        BUILD_SINGLE_SELECTOR(firstPassLogits->dataType(), sampleFirstRowValueCpu,
                                              (fpBase + row * fpStride + v * firstPassLogits->sizeOfT(),
                                               &sampled),
                                              SD_FLOAT_TYPES);
                        if (std::isnan(sampled)) {
                            anyInvalid = true;
                            firstInvalidRow = row;
                        }
                    }
                }
                REQUIRE_TRUE(!anyInvalid, 0,
                             "autoregressive_decode: SPEC VERIFY VALIDITY GUARD step=%d "
                             "rows=%d proposed=%d - verification logits row %d contains "
                             "NaN; refusing to accept or emit from invalid results "
                             "(cause requires a dedicated trace)",
                             step, fpRows, proposedCount_cpu, firstInvalidRow);
            }
            specAccepted_cpu = 0;
            while (specAccepted_cpu < proposedCount_cpu &&
                   specRowArgmax_cpu[specAccepted_cpu] == draftIds_cpu[specAccepted_cpu]) {
                specAccepted_cpu++;
            }
            // Pre-rerun row-0 verification argmax, for the NaN-guard diagnostic
            // (the rerun refresh below overwrites specRowArgmax_cpu[0]).
            const LongType specRowArgmaxOriginal_cpu = specRowArgmax_cpu[0];
            // Snapshot the immutable verification winners for this step (see
            // the declaration above for the round-5 rationale).
            for (int i = 0; i < 33; i++) verifyRowArgmax_cpu[i] = specRowArgmax_cpu[i];

            // Adaptive chain-cap accounting (see declaration above the step loop).
            // Count UNCONDITIONALLY: row p's argmax is the target's continuation
            // of the draft prefix, so draft[p] == argmax[p] measures chain quality
            // at position p even when an earlier draft missed (the lossless accept
            // rule stays sequential — this only feeds the cap statistic).
            if (useMtp_cpu) {
                for (int p = 0; p < proposedCount_cpu && p < 33; p++) {
                    mtpPosEvaluated_cpu[p]++;
                    if (specRowArgmax_cpu[p] == draftIds_cpu[p]) mtpPosAccepted_cpu[p]++;
                }
                for (int p = 1; p < mtpChainCap_cpu && p < 33; p++) {
                    if (mtpPosEvaluated_cpu[p] >= MTP_CHAIN_CAP_MIN_EVALS_CPU
                            && mtpPosAccepted_cpu[p] == 0) {
                        DSP_DIAG(KV_CACHE,
                                 "MTP_CHAIN_CAP: capping chain depth %d -> %d "
                                 "(pos%d evaluated=%d accepted=0; recursive drafts unproductive)",
                                 mtpChainCap_cpu, p, p, mtpPosEvaluated_cpu[p]);
                        mtpChainCap_cpu = p;
                        break;
                    }
                }
            }

            // One emitted output consumes one input row: base + prior accepted
            // drafts. An accepted terminal token has NOT itself been consumed.
            // Determine this boundary before recurrent feedback or predictor
            // carry. T1 (audit F3, CPU mirror): the rerun below may REPLACE the
            // final emission (rerunRefreshedToken), so the matcher suffix is
            // fed the consumed rows PROVISIONALLY here (to preserve mid-batch
            // stop semantics and suffix state) and ROLLED BACK before the
            // authoritative accept below when the rerun rewrote row 0. The
            // earlier approach (skipping the in-loop accept entirely) broke
            // terminal truncation and mid-batch stops (red b8e04d8e).
            // COMMIT POLICY (allowMultiRowCommit): false (shipped default) caps
            // the consume at one row - bit-exact greedy parity through the
            // validated scalar width-1 path. true (experimental) consumes the
            // full accepted prefix. CUDA mirror: identical cap expression.
            const int commitCap_cpu = config->allowMultiRowCommit ? specAccepted_cpu + 1 : 1;
            matcherPreLoopSnapshot_cpu = stopMatcher.snapshot();
            while (specConsumed_cpu < commitCap_cpu
                    && tokensGenerated + specConsumed_cpu < maxNewTokens) {
                LongType token = specRowArgmax_cpu[specConsumed_cpu];
                specConsumed_cpu++;
                bool matchedStop = stopMatcher.accept(token);
                specShouldStop_cpu = matchedStop
                    && stopTerminationAllowed(config, tokensGenerated + specConsumed_cpu);
                if (specShouldStop_cpu) break;
            }

            if (specConsumed_cpu < 1 + proposedCount_cpu
                    && config->actualSequenceLengthExtIdx >= 0
                    && config->actualSequenceLengthExtIdx < numExtInputs
                    && extInputs[config->actualSequenceLengthExtIdx] != nullptr) {
                NDArray* aslArr = extInputs[config->actualSequenceLengthExtIdx];
                aslArr->p(0, static_cast<LongType>(specConsumed_cpu));
                DSP_DIAG(KV_CACHE,
                         "SPEC_STATE_RERUN step=%d proposed=%d accepted=%d — re-executing "
                         "with actual_sequence_length=%d for accepted-prefix state commit",
                         step, proposedCount_cpu, specAccepted_cpu, specConsumed_cpu);
                if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr
                    && config->kvInputExtIndices != nullptr) {
                    static thread_local std::vector<NDArray*> tl_kvQuantPtrsRerun;
                    tl_kvQuantPtrsRerun.resize(numKvPairs);
                    for (int ki = 0; ki < numKvPairs; ki++) {
                        int extIdx = config->kvInputExtIndices[ki];
                        tl_kvQuantPtrsRerun[ki] = (extIdx >= 0 && extIdx < numExtInputs)
                            ? extInputs[extIdx] : nullptr;
                    }
                    setKvScaleRegistry(tl_kvQuantPtrsRerun.data(), config->kvScaleBuffers, numKvPairs);
                }
                // PLAN SELECTION (review round 2, CUDA mirror): the width-one
                // scalar plan can only serve a single-row commit. A multi-row
                // accepted-prefix commit MUST route through the window plan,
                // whose outputs describe the full consumed prefix - otherwise
                // hidden-row indexing below reads a width-one plan as if it
                // were multi-row.
                const bool multiRowRerun_cpu = specConsumed_cpu > 1;
                const bool scalarRerun_cpu = useScalarTarget && !multiRowRerun_cpu;
                if (scalarRerun_cpu) config->activeWindow = 1;
                else if (multiRowRerun_cpu) config->activeWindow = specConsumed_cpu;
                // PRE-VERIFICATION SNAPSHOT RESTORE (CUDA mirror, K=1 scalar-rerun
                // state-poisoning fix + shared-buffer aliasing fix): before
                // executing the rerun, restore the DEEP pre-verification recurrent
                // snapshots (owned arrays captured before this step's verify pass)
                // into the live window ext inputs the verify pass mutated in place
                // through ALL proposed rows, including the rejected suffix. The
                // restore covers BOTH rerun geometries: the window plan reads the
                // live ext inputs directly, and the scalar plan reads them through
                // the prepareScalarTarget() re-stage below (its recurrent copy runs
                // whenever the scalar arrays are NOT buffer-identical; when they ARE
                // identical, the scalar arrays ARE the just-restored live storage).
                // The old scalar-sourced restore with its buffer-identity skip was a
                // no-op exactly for shared-buffer bindings - the mtp-fix-gate2 NaN
                // guard failure mode.
                // Scalar-geometry re-stage (CUDA mirror): after the deep restore,
                // refresh the private width-1 arrays FROM the restored live ext
                // inputs - recurrent to the pre-verification state, geometry to
                // the rerun's asl=1 - so executeScalarTarget below cannot replay
                // stale post-verify state left by an earlier different-width run.
                restorePreVerificationState_cpu();
                if (useScalarTarget) {
                    prepareScalarTarget();
                }
                p0.acceptedPrefixReruns++;
                Status rerunStatus = scalarRerun_cpu ? executeScalarTarget() : plan->execute(
                    extInputs, numExtInputs,
                    planOutputs, numPlanOutputs,
                    nullptr);
                if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
                    clearKvScaleRegistry();
                }
                std::string rerunFailureDetail;
                if (rerunStatus != Status::OK) rerunFailureDetail = nestedPlanFailureDetail();
                REQUIRE_TRUE(rerunStatus == Status::OK, 0,
                             "%s [autoregressive_decode accepted-prefix re-execution step=%d, "
                             "status=%s (%d), accepted=%d of %d]",
                             rerunFailureDetail.c_str(), step,
                             graph::dsp::dspStatusName(rerunStatus),
                             static_cast<int>(rerunStatus), specAccepted_cpu,
                             proposedCount_cpu);
                if (useScalarTarget) {
                    // RERUN WINNER INTO SEPARATE STORAGE (review round 5,
                    // finding 1): the rerun's row-0 readout replaces the
                    // specRowArgmax_cpu[0] slot (downstream publication reads
                    // that array), but the IMMUTABLE verification winners
                    // (verifyRowArgmax_cpu) captured above are what every
                    // disagreement comparison uses.
                    NDArray* scalarLogits = planOutputs[config->logitsOutputIdx];
                    NDArray::preparePrimaryUse({}, {scalarLogits});
                    specRowArgmax_cpu[0] = cpuArgmax(scalarLogits->buffer(),
                        scalarLogits->sizeAt(scalarLogits->rankOf() - 1), scalarLogits->dataType());
                    NDArray::registerPrimaryUse({}, {scalarLogits});
                }

                // FAIL-LOUD NaN GUARD (K=1 state-poisoning regression, CUDA
                // mirror): FULL-ROW probe over the rerun's logits (finding 5:
                // was the first 8 entries only) plus a probe of the FIRST GDN
                // state pair's output row from the rerun pass itself (probing
                // the ext input here would read the still-uncommitted pre-verify
                // state instead of what the rerun just produced). NaN here means
                // the rerun executed from mutated (post-verification) recurrent
                // state; committing it would poison every later step. Fail
                // loudly naming geometry and step - never continue with
                // poisoned state.
                bool rerunLogitsNan_cpu = false;
                {
                    NDArray* rerunLogitsArr = planOutputs[config->logitsOutputIdx];
                    const LongType rerunVocabLocal =
                        rerunLogitsArr->sizeAt(rerunLogitsArr->rankOf() - 1);
                    const LongType probeVocab = rerunVocabLocal;
                    if (probeVocab > 0) {
                        NDArray::preparePrimaryUse({}, {rerunLogitsArr});
                        const char* base = reinterpret_cast<const char*>(rerunLogitsArr->buffer());
                        for (LongType v = 0; v < probeVocab; v++) {
                            float sampled = 0.0f;
                            BUILD_SINGLE_SELECTOR(rerunLogitsArr->dataType(),
                                                  sampleFirstRowValueCpu,
                                                  (base + v * rerunLogitsArr->sizeOfT(), &sampled),
                                                  SD_FLOAT_TYPES);
                            if (std::isnan(sampled)) {
                                rerunLogitsNan_cpu = true;
                                break;
                            }
                        }
                        NDArray::registerPrimaryUse({}, {rerunLogitsArr});
                    }
                }
                bool rerunStateNan_cpu = false;
                if (config->numGdnStatePairs > 0
                        && config->gdnStateOutputIndices != nullptr) {
                    int gdnOut0 = config->gdnStateOutputIndices[0];
                    NDArray* gdnOut = (gdnOut0 >= 0 && gdnOut0 < numPlanOutputs)
                        ? planOutputs[gdnOut0] : nullptr;
                    // DTYPE-SAFE STATE SAMPLE (review round 5, finding 3, CPU
                    // mirror): convert through the state's OWN dtype selector -
                    // no FP32-only gating, no raw reinterpretation.
                    if (gdnOut != nullptr && gdnOut->lengthOf() >= 4) {
                        NDArray::preparePrimaryUse({}, {gdnOut});
                        const char* stateBase = reinterpret_cast<const char*>(gdnOut->buffer());
                        for (LongType i = 0; i < 4; i++) {
                            float sampled = 0.0f;
                            BUILD_SINGLE_SELECTOR(gdnOut->dataType(), sampleFirstRowValueCpu,
                                                  (stateBase + i * gdnOut->sizeOfT(), &sampled),
                                                  SD_FLOAT_TYPES);
                            if (std::isnan(sampled)) {
                                rerunStateNan_cpu = true;
                                break;
                            }
                        }
                        NDArray::registerPrimaryUse({}, {gdnOut});
                    }
                }
                REQUIRE_TRUE(!(rerunLogitsNan_cpu || rerunStateNan_cpu), 0,
                             "autoregressive_decode: SPEC RERUN NaN GUARD step=%d "
                             "geometry=%s rerunArgmax=%lld verifyRow0=%lld "
                             "rerunLogitsNaN=%d rerunGdnStateNaN=%d - the "
                             "accepted-prefix rerun executed from mutated recurrent "
                             "state; refusing to commit poisoned state",
                             step, scalarRerun_cpu ? "scalar-width-1" : "window",
                             (long long)specRowArgmax_cpu[0],
                             (long long)specRowArgmaxOriginal_cpu,
                             rerunLogitsNan_cpu ? 1 : 0, rerunStateNan_cpu ? 1 : 0);
            }
        }

        // Commit the target model's accepted hidden state into the reusable MTP
        // carry. Predictor KV writes beyond a rejected prefix remain allocated,
        // but their mask entries are restored before the next draft chain.
        // REVIEW ROUND 3 ORDERING FIX (finding 3, CPU): the predictor prefix
        // repair and retained-pair publication below used to run BEFORE the
        // emission finalize, publishing carryRow/pending token from the
        // PROVISIONAL verification sequence even when the finalize truncated
        // the commit to one row and re-executed. The block now runs AFTER the
        // finalize (see MTP PUBLICATION below); only the hoisted geometry
        // variables are declared here.
        int carryRow_cpu = proposedCount_cpu > 0 ? specConsumed_cpu - 1 : 0;
        LongType nextMtpPosition_cpu = currentPosition + carryRow_cpu + 1;
        // Packet P2: the proposal-write horizon is derived directly from the
        // ORIGINAL proposedCount_cpu inside the publication block (never from
        // this mutable variable - the shortened-prefix branch resets it while
        // the proposal rows still exist). Kept only for the SHORTEN_REEXEC
        // branch's own geometry reset below.
        LongType mtpWrittenThrough_cpu = proposedCount_cpu > 0
            ? currentPosition + proposedCount_cpu - 1 : currentPosition;

        // ── Step 2b: GDN/conv recurrent state feedback ──
        // Copy state outputs back to ext inputs for the next decode step.
        // This is critical for hybrid architectures (e.g. Qwen with GDN layers).
        // Without this, GDN layers see frozen state from warmup and degenerate.
        //
        // State mappings are strict. A missing or incompatible recurrent pair is
        // a graph/runtime error; silently skipping it changes model semantics.
        if (config->numGdnStatePairs > 0) {
            REQUIRE_TRUE(config->gdnStateExtIndices != nullptr &&
                         config->gdnStateOutputIndices != nullptr, 0,
                         "autoregressive_decode: GDN state mappings are missing for %d pairs at step %d",
                         config->numGdnStatePairs, step);
            for (int s = 0; s < config->numGdnStatePairs; s++) {
                int outIdx = config->gdnStateOutputIndices[s];
                int extIdx = config->gdnStateExtIndices[s];
                REQUIRE_TRUE(outIdx >= 0 && outIdx < numPlanOutputs &&
                             extIdx >= 0 && extIdx < numExtInputs,
                             0, "autoregressive_decode: invalid GDN state mapping at step %d pair %d",
                             step, s);
                NDArray* src = planOutputs[outIdx];
                NDArray* dst = extInputs[extIdx];
                REQUIRE_TRUE(src != nullptr && dst != nullptr, 0,
                             "autoregressive_decode: null GDN state mapping at step %d pair %d",
                             step, s);
                REQUIRE_TRUE(copyRecurrentFeedback(src, dst), 0,
                             "autoregressive_decode: GDN state feedback copy failed at step %d pair %d",
                             step, s);
                p0.stateCommitBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            }
        }
        if (config->numConvStatePairs > 0) {
            REQUIRE_TRUE(config->convStateExtIndices != nullptr &&
                         config->convStateOutputIndices != nullptr, 0,
                         "autoregressive_decode: conv state mappings are missing for %d pairs at step %d",
                         config->numConvStatePairs, step);
            for (int s = 0; s < config->numConvStatePairs; s++) {
                int outIdx = config->convStateOutputIndices[s];
                int extIdx = config->convStateExtIndices[s];
                if (s == 0 && step < 4) {
                    bool valid = outIdx >= 0 && outIdx < numPlanOutputs && planOutputs[outIdx] != nullptr
                        && extIdx >= 0 && extIdx < numExtInputs && extInputs[extIdx] != nullptr;
                    DSP_DIAG(KV_CACHE,
                        "CONV_FB_PROBE step=%d pair=0 outIdx=%d extIdx=%d valid=%d src[0..2]=%.6f,%.6f,%.6f dstPre[0..2]=%.6f,%.6f,%.6f",
                        step, outIdx, extIdx, (int)valid,
                        valid && planOutputs[outIdx]->lengthOf() > 0 ? planOutputs[outIdx]->e<float>(0) : -999.0f,
                        valid && planOutputs[outIdx]->lengthOf() > 1 ? planOutputs[outIdx]->e<float>(1) : -999.0f,
                        valid && planOutputs[outIdx]->lengthOf() > 2 ? planOutputs[outIdx]->e<float>(2) : -999.0f,
                        valid && extInputs[extIdx]->lengthOf() > 0 ? extInputs[extIdx]->e<float>(0) : -999.0f,
                        valid && extInputs[extIdx]->lengthOf() > 1 ? extInputs[extIdx]->e<float>(1) : -999.0f,
                        valid && extInputs[extIdx]->lengthOf() > 2 ? extInputs[extIdx]->e<float>(2) : -999.0f);
                }
                REQUIRE_TRUE(outIdx >= 0 && outIdx < numPlanOutputs &&
                             extIdx >= 0 && extIdx < numExtInputs,
                             0, "autoregressive_decode: invalid conv state mapping at step %d pair %d",
                             step, s);
                NDArray* src = planOutputs[outIdx];
                NDArray* dst = extInputs[extIdx];
                REQUIRE_TRUE(src != nullptr && dst != nullptr, 0,
                             "autoregressive_decode: null conv state mapping at step %d pair %d",
                             step, s);
                REQUIRE_TRUE(copyRecurrentFeedback(src, dst), 0,
                             "autoregressive_decode: conv state feedback copy failed at step %d pair %d",
                             step, s);
                p0.stateCommitBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            }
        }

        // ── Step 3: Token sampling ──
        NDArray* logitsOutput = planOutputs[config->logitsOutputIdx];

        // Validate logits rank before accessing shape dimensions.
        auto logitsRank = logitsOutput->rankOf();
        REQUIRE_TRUE(logitsRank >= 2 && logitsRank <= 3, 0,
                     "autoregressive_decode: logitsOutput rank is %lld (expected 2 or 3) at step %d. "
                     "lengthOf=%lld, logitsOutputIdx=%d, numPlanOutputs=%d. "
                     "The plan output at this index is not logits — check logitsOutputIdx mapping.",
                     (long long)logitsRank, step,
                     (long long)logitsOutput->lengthOf(),
                     config->logitsOutputIdx, numPlanOutputs);

        LongType logitsSeqLen;
        LongType logitsVocab;
        if (logitsRank == 3) {
            logitsSeqLen = logitsOutput->sizeAt(1);
            logitsVocab = logitsOutput->sizeAt(2);
        } else {
            logitsSeqLen = 1;
            logitsVocab = logitsOutput->sizeAt(1);
        }

        REQUIRE_TRUE(logitsVocab > 0, 0,
                     "autoregressive_decode: logits vocab dimension is 0 at step %d. "
                     "Cannot perform token selection on empty vocabulary.",
                     step);

        // ── ADR 0106 Phase 2 speculative path OR Phase 1 scalar path (CPU) ──
        if (useSpeculative_cpu && proposedCount_cpu > 0 && (logitsRank == 3 || useScalarTarget)) {
            // ── Speculative: consume the FIRST-pass argmaxes + acceptance ────────
            // Both were computed in the accepted-prefix state-commit block right
            // after plan execution. The logits buffer may now hold the accepted-
            // prefix re-run (whose rows beyond the accepted prefix are not
            // meaningful for emission), so recomputing here would be wrong.
            // Accept rule recap: input row i contains draftIds[i-1] for i > 0, so
            // target logits row i predicts the token after that input — row 0
            // validates draft 0, row 1 validates draft 1, etc. On first mismatch at
            // j, emit accepted drafts [0,j) then argmax[j] as the correction token;
            // if all match, argmax[proposedCount] is the bonus token. For the
            // accepted prefix rowArgmax[i] == draftIds_cpu[i], so the store loop
            // below emits rowArgmax[0..n-1] directly.
            LongType rowArgmax[33];
            for (int i = 0; i < 33; i++) rowArgmax[i] = verifyRowArgmax_cpu[i];
            int acceptedDrafts = specAccepted_cpu >= 0 ? specAccepted_cpu : 0;
            int n = specConsumed_cpu;
            // Rerun-refreshed emission (CUDA mirror): when the accepted-prefix
            // rerun produced the authoritative state, its row-0 logits are the
            // greedy readout; emission must match that state or the next step's
            // inputs diverge from what was emitted. The consumed row fed to the
            // matcher above was the PROVISIONAL verify row-0 argmax; roll the
            // suffix back to its pre-step state and re-accept the authoritative
            // token so the suffix/shouldStop describe what was actually emitted.
            LongType rerunRefreshedToken_cpu = -1;
            // Rerun fired whenever the consumed prefix is shorter than the
            // window (specConsumed < 1 + proposedCount): the rerun pass owns
            // the committed state. Its row 0 is the authoritative readout even
            // when the graph still exports W logits rows. The old n==1 width
            // guard left multi-row flips unrefreshed (review round 3, finding 3).
            if (logitsOutput != nullptr
                    && planOutputs[config->logitsOutputIdx] != nullptr
                    && specConsumed_cpu < 1 + proposedCount_cpu
                    && useMtp_cpu) {
                NDArray* rerunLogits = planOutputs[config->logitsOutputIdx];
                LongType rerunVocab = useScalarTarget
                    ? rerunLogits->sizeAt(rerunLogits->rankOf() - 1) : rerunLogits->sizeAt(2);
                if (rerunVocab > 0) {
                    LongType refreshed = cpuArgmax(rerunLogits->buffer(), rerunVocab,
                                                   rerunLogits->dataType());
                    // DISAGREEMENT GATE (CUDA mirror of rerunRefreshedToken !=
                    // argmaxDst[0]): the authoritative refresh rewrites the
                    // emission only when the rerun's row-0 readout DISAGREES
                    // with the VERIFY row 0 - compared against the IMMUTABLE
                    // verification winner (review round 5, finding 1), never
                    // against specRowArgmax_cpu[0] which the scalar-binding
                    // rerun block already replaced with the rerun winner.
                    const LongType supersededRow0_cpu = verifyRowArgmax_cpu[0];
                    const bool rerunDisagrees = refreshed != supersededRow0_cpu;
                    if (rerunDisagrees) {
                        DSP_DIAG(KV_CACHE,
                                 "RERUN_EMISSION_REFRESH step=%d verify=%lld rerun=%lld "
                                 "- emitting the asl=1 authoritative argmax",
                                 step, (long long)supersededRow0_cpu, (long long)refreshed);
                        rowArgmax[0] = refreshed;
                        // ONE COMMITTED SEQUENCE (review round 5, finding 1B):
                        // specRowArgmax_cpu[0] is the predictor-publication token
                        // source; leaving it at the superseded verify winner while
                        // rowArgmax[0] carries the rerun winner made a one-row
                        // flip emit B and publish A to the predictor. The
                        // multi-row SHORTEN_REEXEC branch below also writes this
                        // slot (with its own width-1 readout).
                        specRowArgmax_cpu[0] = refreshed;
                        rerunRefreshedToken_cpu = refreshed;
                    }
                    if (rerunDisagrees && specConsumed_cpu > 1) {
                        // SHORTENED-PREFIX STATE RECOVERY (review round 3, finding
                        // 3, CPU mirror of CUDA RERUN_SHORTEN_REEXEC): the multi-row
                        // rerun's planOutputs hold recurrent state AFTER
                        // specConsumed_cpu consumed inputs, but the finalized
                        // emission is now ONE token. Committing that state would
                        // pair a 1-token history with m-token recurrence. Re-derive
                        // BOTH from one width-1 execution: restore the pre-verify
                        // snapshots (owned arrays, safe to restore again), set
                        // asl=1, re-stage the scalar geometry, re-execute, and take
                        // the emission readout AND the committed state from THIS
                        // pass. The state feedback + predictor publication below
                        // then read width-1-consistent outputs.
                        DSP_DIAG(KV_CACHE,
                                 "RERUN_TRUNCATE_COMMIT step=%d supersededRow0=%lld "
                                 "rerunRow0=%lld committed=%d -> n=1 (RERUN_SHORTEN_REEXEC "
                                 "follows)",
                                 step, (long long)supersededRow0_cpu, (long long)refreshed,
                                 (long long)specConsumed_cpu);
                        specConsumed_cpu = 1;
                        n = 1;
                        // Width-1 geometry + pre-verify state restore (mirrors the
                        // rerun site's ordering: restore FIRST, then asl, then the
                        // scalar re-stage).
                        restorePreVerificationState_cpu();
                        if (config->actualSequenceLengthExtIdx >= 0
                                && config->actualSequenceLengthExtIdx < numExtInputs
                                && extInputs[config->actualSequenceLengthExtIdx] != nullptr) {
                            extInputs[config->actualSequenceLengthExtIdx]->p(
                                0, static_cast<LongType>(1));
                        }
                        if (useScalarTarget) {
                            prepareScalarTarget();
                        }
                        config->activeWindow = 1;
                        p0.shortenedRecoveryForwards++;
                        Status shortenStatus = useScalarTarget
                            ? executeScalarTarget()
                            : plan->execute(extInputs, numExtInputs,
                                            planOutputs, numPlanOutputs, nullptr);
                        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
                            clearKvScaleRegistry();
                        }
                        std::string shortenFailureDetail;
                        if (shortenStatus != Status::OK)
                            shortenFailureDetail = nestedPlanFailureDetail();
                        REQUIRE_TRUE(shortenStatus == Status::OK, 0,
                                     "%s [autoregressive_decode shortened-prefix re-execution "
                                     "step=%d, status=%s (%d)]",
                                     shortenFailureDetail.c_str(), step,
                                     graph::dsp::dspStatusName(shortenStatus),
                                     static_cast<int>(shortenStatus));
                        // Authoritative row-0 readout from the width-1 pass.
                        NDArray* shortenLogits = planOutputs[config->logitsOutputIdx];
                        REQUIRE_TRUE(shortenLogits != nullptr, 0,
                                     "autoregressive_decode: shortened rerun logits output is "
                                     "null at step %d", step);
                        LongType shortenVocab = shortenLogits->sizeAt(shortenLogits->rankOf() - 1);
                        LongType shortenedToken = cpuArgmax(shortenLogits->buffer(), shortenVocab,
                                                            shortenLogits->dataType());
                        // NaN probe on the width-1 pass (same poisoning class as the
                        // guarded pass-1 rerun; stays loud). FINDING 5: the probe
                        // covers the FULL row, not the first 8 entries, and every
                        // float dtype via sampleFirstRowValueCpu's selector.
                        {
                            NDArray::preparePrimaryUse({}, {shortenLogits});
                            const LongType probeVocabS = shortenVocab;
                            const char* baseS = reinterpret_cast<const char*>(shortenLogits->buffer());
                            for (LongType v = 0; v < probeVocabS; v++) {
                                float sampled = 0.0f;
                                BUILD_SINGLE_SELECTOR(shortenLogits->dataType(),
                                                      sampleFirstRowValueCpu,
                                                      (baseS + v * shortenLogits->sizeOfT(), &sampled),
                                                      SD_FLOAT_TYPES);
                                REQUIRE_TRUE(!std::isnan(sampled), 0,
                                             "autoregressive_decode: SHORTEN REEXEC NaN step=%d - "
                                             "width-1 re-execution produced NaN logits; refusing to "
                                             "commit", step);
                            }
                            NDArray::registerPrimaryUse({}, {shortenLogits});
                        }
                        // Emission AND state now come from this pass. The
                        // FINALIZED EMISSION SEQUENCE (review round 5, finding
                        // 1): both arrays that downstream consumers read are
                        // rewritten here so emission, matcher, predictor
                        // publication, and metrics all share ONE committed
                        // sequence (the no-scalar-binding one-row flip
                        // previously published the OLD pending token from
                        // specRowArgmax_cpu while emitting the new one).
                        rerunRefreshedToken_cpu = shortenedToken;
                        rowArgmax[0] = shortenedToken;
                        specRowArgmax_cpu[0] = shortenedToken;
                        // SHORTENED EMISSION REPAIR: the truncation invalidates the
                        // stale verification suffix rows [1..); zero them so the
                        // store loop below emits exactly the width-1 readout.
                        for (int i = 1; i < 33; i++) rowArgmax[i] = 0;
                        // Restore the exact pre-provisional-accept matcher state,
                        // then re-accept ONLY the authoritative token - the suffix
                        // never retains rows from the invalidated sequence.
                        stopMatcher.restore(matcherPreLoopSnapshot_cpu);
                        bool matchedStopShort = stopMatcher.accept(rowArgmax[0]);
                        specShouldStop_cpu = matchedStopShort
                            && stopTerminationAllowed(config, tokensGenerated + specConsumed_cpu);
                        // carryRow for the target-hidden publication below: the
                        // width-1 pass's hidden output row 0 is the hidden AFTER
                        // consuming the single committed row.
                        carryRow_cpu = 0;
                        nextMtpPosition_cpu = currentPosition + 1;
                        mtpWrittenThrough_cpu = currentPosition;
                        specRowArgmax_cpu[0] = shortenedToken;

                        // RECURRENT FEEDBACK AFTER RECOVERY (review round 4,
                        // finding C/4): the ordinary GDN/conv feedback above ran
                        // from the MULTI-ROW rerun outputs, but the shortened
                        // width-1 re-execution just produced REPLACEMENT outputs
                        // in the same planOutputs slots. Re-commit the feedback
                        // from those finalized outputs so the retained state
                        // advances through the FINALIZED consumed-input prefix
                        // (width 1), not the superseded multi-row prefix. This
                        // matters whenever a graph's recurrent outputs do not
                        // alias their input storage; an in-place graph conceals
                        // it. The publication is deliberately the LAST write: no
                        // token storage or predictor work below re-runs the
                        // target.
                        if (config->numGdnStatePairs > 0
                                && config->gdnStateExtIndices != nullptr
                                && config->gdnStateOutputIndices != nullptr) {
                            for (int s = 0; s < config->numGdnStatePairs; s++) {
                                int outIdx = config->gdnStateOutputIndices[s];
                                int extIdx = config->gdnStateExtIndices[s];
                                REQUIRE_TRUE(outIdx >= 0 && outIdx < numPlanOutputs
                                                 && extIdx >= 0 && extIdx < numExtInputs,
                                             0, "autoregressive_decode: invalid GDN state mapping "
                                                "after shortened rerun at step %d pair %d", step, s);
                                NDArray* src = planOutputs[outIdx];
                                NDArray* dst = extInputs[extIdx];
                                REQUIRE_TRUE(src != nullptr && dst != nullptr, 0,
                                             "autoregressive_decode: null GDN state mapping "
                                             "after shortened rerun at step %d pair %d", step, s);
                                REQUIRE_TRUE(copyRecurrentFeedback(src, dst), 0,
                                             "autoregressive_decode: GDN state feedback copy failed "
                                             "after shortened rerun at step %d pair %d", step, s);
                                p0.stateCommitBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
                            }
                        }
                        if (config->numConvStatePairs > 0
                                && config->convStateExtIndices != nullptr
                                && config->convStateOutputIndices != nullptr) {
                            for (int s = 0; s < config->numConvStatePairs; s++) {
                                int outIdx = config->convStateOutputIndices[s];
                                int extIdx = config->convStateExtIndices[s];
                                REQUIRE_TRUE(outIdx >= 0 && outIdx < numPlanOutputs
                                                 && extIdx >= 0 && extIdx < numExtInputs,
                                             0, "autoregressive_decode: invalid conv state mapping "
                                                "after shortened rerun at step %d pair %d", step, s);
                                NDArray* src = planOutputs[outIdx];
                                NDArray* dst = extInputs[extIdx];
                                REQUIRE_TRUE(src != nullptr && dst != nullptr, 0,
                                             "autoregressive_decode: null conv state mapping "
                                             "after shortened rerun at step %d pair %d", step, s);
                                REQUIRE_TRUE(copyRecurrentFeedback(src, dst), 0,
                                             "autoregressive_decode: conv state feedback copy failed "
                                             "after shortened rerun at step %d pair %d", step, s);
                                p0.stateCommitBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
                            }
                        }
                    }
                }
            }
            // T1 (audit F3): authoritative stop state. Roll back the provisional
            // accept for the rewritten row, then feed the matcher the FINAL
            // emitted token exactly once. Mid-batch accepts from rows below the
            // rewritten one are real emissions and stay in the suffix.
            // (The truncated multi-row path above already restored the exact
            // pre-loop matcher state and re-accepted row 0.)
            if (rerunRefreshedToken_cpu >= 0) {
                stopMatcher.rollback(1);
                bool matchedStop = stopMatcher.accept(rowArgmax[0]);
                specShouldStop_cpu = matchedStop
                    && stopTerminationAllowed(config, tokensGenerated + n);
            }

            // MTP PUBLICATION (moved after the finalize, review round 3): the
            // predictor prefix repair and retained-pair publication now read the
            // FINALIZED emission sequence and the POST-truncation geometry, so a
            // truncated multi-row commit publishes the width-1-conditioned pair
            // instead of the provisional verification sequence.
            if (useMtp_cpu) {
                REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                                 && config->targetHiddenOutputIdx < numPlanOutputs
                                 && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                             0, "autoregressive_decode: target hidden output is unavailable for MTP");

                // Predictor-side accepted-prefix repair (CUDA mirror): rewrite every
                // committed position's predictor KV row as fused(committed token,
                // target hidden at q-1). Chained proposal rows carry self-propagated
                // hidden; a fully accepted K=1 step leaves the bonus row unwritten by
                // the prefix. The target's rerun above repairs only the target plan.
                // specRowArgmax_cpu row j is the accepted draft for j < specAccepted
                // and the correction/bonus for the final committed row. On a
                // truncated commit (specConsumed_cpu forced to 1) this loop runs
                // zero times - the stale draft-conditioned rows are masked below.
                p0RepairActive = true;
                for (int j = 0; j + 1 < specConsumed_cpu; j++) {
                    LongType repairPosition = currentPosition + 1 + j;
                    setMtpTargetCarryCpu(
                        planOutputs[config->targetHiddenOutputIdx], j);
                    (void)executeMtpCpu(specRowArgmax_cpu[j], repairPosition);
                    DSP_DIAG(KV_CACHE,
                             "MTP_PREFIX_REPAIR step=%d position=%lld committedRow=%d "
                             "carryRow=%d — rewriting predictor KV row with target hidden",
                             step, (long long)repairPosition, j, carryRow_cpu);
                }
                p0RepairActive = false;

                // Remask the entire future predictor tail. Adaptive K may shrink,
                // so rows left unmasked by a wider prior proposal must not remain
                // visible in a later predictor call.
                const LongType retainedPredictorEnd_cpu =
                    currentPosition + static_cast<LongType>(specConsumed_cpu) - 1;
                if (retainedPredictorEnd_cpu < mtpMaskLen_cpu) {
                    BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), maskCausalRangeCpu,
                                          (config->mtpCausalMask->buffer(), retainedPredictorEnd_cpu,
                                           mtpMaskLen_cpu, mtpMaskLen_cpu),
                                          SD_FLOAT_TYPES);
                }

                setMtpTargetCarryCpu(planOutputs[config->targetHiddenOutputIdx], carryRow_cpu);
                // Pending-input publication (packet P2): nextMtpPosition_cpu is
                // the next pending TARGET token position; its predictor row is
                // nextMtpPosition_cpu - 1 (rope = slot = row). The token itself
                // stays specRowArgmax_cpu[carryRow_cpu] (the finalized emission).
                const LongType nextPredictorRow_cpu = nextMtpPosition_cpu - 1;
                config->mtpPositionOffset->p(0, nextPredictorRow_cpu);
                config->mtpCachePosition->p(0, nextPredictorRow_cpu);
                if (proposedCount_cpu > 0) {
                    config->mtpInputIds->p(0, specRowArgmax_cpu[carryRow_cpu]);
                }
            }
            totalSpeculativeProposed += proposedCount_cpu;
            // Accepted drafts ACTUALLY EMITTED (review round 2): count each
            // emitted token that still equals its draft. A scalar refresh that
            // flipped row 0 away from its draft means that draft was not emitted.
            int acceptedEmitted_cpu = 0;
            for (int i = 0; i < acceptedDrafts && i < n; i++) {
                if (rowArgmax[i] == draftIds_cpu[i]) acceptedEmitted_cpu++;
            }
            totalSpeculativeAccepted += acceptedEmitted_cpu;
            speculativeStepCount++;
            REQUIRE_TRUE(specConsumed_cpu > 0, 0,
                         "autoregressive_decode: speculative consume committed zero rows "
                         "with proposedCount=%d at step %d", proposedCount_cpu, step);

            // Gated diagnostic event: mirrors the CUDA helper's SPEC_STEP event.
            DSP_DIAG(KV_CACHE,
                     "SPEC_STEP step=%d basePos=%lld proposed=%d accepted=%d "
                     "draft=[%lld,%lld,%lld,%lld] argmaxRaw=[%lld,%lld,%lld,%lld,%lld]",
                     step, (long long)currentPosition, proposedCount_cpu, acceptedDrafts,
                     (long long)draftIds_cpu[0], (long long)draftIds_cpu[1],
                     (long long)draftIds_cpu[2], (long long)draftIds_cpu[3],
                     (long long)rowArgmax[0], (long long)rowArgmax[1],
                     (long long)rowArgmax[2], (long long)rowArgmax[3],
                     (long long)rowArgmax[4]);

            // ── Store accepted tokens ──────────────────────────────────────────
            bool shouldStop = specShouldStop_cpu;
            int storedCount = 0;
            for (int i = 0; i < n && tokensGenerated < maxNewTokens; i++) {
                LongType tok = rowArgmax[i];
                generatedTokenIds->p(tokensGenerated, tok);
                tokensGenerated++;
                storedCount++;
                if (config->tokenCallback != nullptr) {
                    config->tokenCallback(tok, config->callbackUserData);
                }
            }

            // ── Advance currentPosition by storedCount ──────────────────────────
            LongType basePosition = currentPosition;
            for (int i = 0; i < storedCount; i++) {
                LongType kvJust = currentPosition;
                currentPosition++;

                // Unmask attention mask for kvJust (skipped when it aliases the
                // additive causal mask — see attnMaskAliasesCausal).
                if (!attnMaskAliasesCausal && kvJust >= 0 && kvJust < maxKvLen) {
                    BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskCpu,
                                          (attentionMask->buffer(), kvJust, maxKvLen),
                                          SD_COMMON_TYPES);
                }
                // Unmask causal mask
                {
                    LongType cmPos = config->planOwnsKvScatter ? kvJust : currentPosition;
                    if (causalMask != nullptr && cmPos >= 0 && cmPos < causalMaskLen) {
                        BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskCpu,
                                              (causalMask->buffer(), cmPos, causalMaskLen),
                                              SD_FLOAT_TYPES);
                    }
                }
            }
            (void)basePosition;
            // Publish the committed KV prefix on terminal window rows, rather
            // than leaving the speculative verification suffix visible.
            if ((shouldStop || tokensGenerated == maxNewTokens) && useWindowSubstrate) {
                NDArray* mask = config->windowGridMask;
                LongType rowLen = mask->sizeAt(-1);
                BUILD_SINGLE_SELECTOR(mask->dataType(), maskCausalRangeCpu,
                                      (mask->buffer(), currentPosition, rowLen, rowLen), SD_FLOAT_TYPES);
                size_t rowBytes = static_cast<size_t>(rowLen) * mask->sizeOfT();
                for (LongType row = 1; row < mask->lengthOf() / rowLen; row++) {
                    std::memcpy(static_cast<char*>(mask->buffer()) + row * rowBytes,
                                mask->buffer(), rowBytes);
                }
            }

            // ── Update n-gram tables from the verified emission sequence ─────────
            if (useNgram_cpu) {
                LongType previous = specPreviousToken_cpu;
                LongType current = specCurrentToken_cpu;
                for (int i = 0; i < storedCount; i++) {
                    LongType tok = rowArgmax[i];
                    if (current >= 0) {
                        ngramTable_cpu[current] = tok;
                        if (previous >= 0) {
                            trigramTable_cpu[previous][current] = tok;
                        }
                    }
                    previous = current;
                    current = tok;
                }
                specPreviousToken_cpu = previous;
                specCurrentToken_cpu = current;
            }

            LongType nextTokenId = rowArgmax[storedCount - 1];

            // Restore activeWindow to 1 for next step (will be set fresh by proposal)
            config->activeWindow = 1;

            auto stepEnd = std::chrono::high_resolution_clock::now();
            double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
            stepTimesMs.push_back(stepMs);
            stepTokenCounts.push_back(tokensGenerated - tokensBeforeStep);

            // Publish the pending final token and positions even on termination.
            // ── Step 6: Embedding lookup for next step ─────────────────────────
            if (config->embeddingsExtIdx >= 0) {
                REQUIRE_TRUE(nextTokenId >= 0 && nextTokenId < vocabSize, 0,
                             "autoregressive_decode speculative CPU: nextTokenId=%lld out of range at step %d.",
                             (long long)nextTokenId, step);
                BUILD_SINGLE_SELECTOR(embeddingTable->dataType(), embedLookupCpu,
                                      (embeddingTable->buffer(), decodeEmbedding->buffer(),
                                       nextTokenId, hidden, embTableRowStride),
                                      SD_COMMON_TYPES);
            }
            positionIds->p(0, currentPosition);
            inputIds->p(0, nextTokenId);

            if (config->positionOffsetExtIdx >= 0 && config->positionOffsetExtIdx < numExtInputs) {
                NDArray* posOffset = extInputs[config->positionOffsetExtIdx];
                if (posOffset != nullptr) posOffset->p(0, currentPosition);
            }
            if (config->cachePositionExtIdx >= 0 && config->cachePositionExtIdx < numExtInputs) {
                NDArray* cachePos = extInputs[config->cachePositionExtIdx];
                if (cachePos != nullptr) cachePos->p(0, currentPosition);
            }

            if (shouldStop) break;

            continue;  // skip Phase 1 scalar path for this step
        }

        // ── Phase 1 scalar path (W=1 or no proposals) ───────────────────────
        // Restore activeWindow to 1 if speculative path set it but produced no proposals.
        if (useSpeculative_cpu && proposedCount_cpu == 0) {
            config->activeWindow = 1;
        }

        // ADR 0106 Phase 1: when W>1 the logits output has shape [1, W_max, vocab].
        // For the greedy/sample policy we sample from position 0 (first window slot).
        // Phase 2 (speculative) will inspect all W position-logits and apply policy.
        // When W=1 the slice is identical to the full output (no overhead, no copy).
        NDArray* logitsForSample = logitsOutput;
        NDArray* logitsSlice = nullptr;  // owned by this scope if allocated
        if (useWindowSubstrate && logitsRank == 3 && logitsSeqLen > 1) {
            // Slice position 0: logitsOutput[0, 0, :] → keeps batch+vocab dims.
            // operator()(idx) flat format: {dim0Start,dim0End, dim1Start,dim1End, dim2Start,dim2End}
            std::vector<LongType> sliceIdx{0, 1, 0, 1, 0, logitsVocab};
            logitsSlice = (*logitsOutput)(sliceIdx, true);
            logitsForSample = logitsSlice;
        }

        TokenSampleConfig stepSampleConfig = config != nullptr ? config->sampleConfig : TokenSampleConfig();
        LongType baseSeed = stepSampleConfig.seed;
        int generatedOffset = stepSampleConfig.generatedTokenOffset;
        stepSampleConfig.temperature = temperature;
        stepSampleConfig.topK = topK;
        stepSampleConfig.topP = topP;
        stepSampleConfig.repPenalty = repPenalty;
        // Force batchMax/windowMax to 1 for the scalar selection step
        // (policy drives the W-wide selection; the substrate just runs the forward).
        // Also reset SPECULATIVE strategy to GREEDY: in the scalar fallback path
        // TOKEN_SAMPLE_SPECULATIVE(3) is not handled by tokenSamplePolicy.
        stepSampleConfig.batchMax = 1;
        stepSampleConfig.windowMax = 1;
        stepSampleConfig.activeBatch = 1;
        stepSampleConfig.activeWindow = 1;
        if (stepSampleConfig.strategy == TOKEN_SAMPLE_SPECULATIVE) {
            stepSampleConfig.strategy = TOKEN_SAMPLE_GREEDY;
        }
        stepSampleConfig.seed = baseSeed > 0 ? baseSeed + static_cast<LongType>(step) : 0;
        stepSampleConfig.generatedTokenOffset = generatedOffset + step;
        stepSampleConfig.stopTokenIds = stopTokenIds.empty() ? nullptr : stopTokenIds.data();
        stepSampleConfig.stopTokenCount = static_cast<int>(stopTokenIds.size());

        TokenSampleResult sampleResult;
        if (step > 0) {
            NDArray* tokensSoFar = (*generatedTokenIds)({0, step}, true);
            tokenSamplePolicy(logitsForSample, sampledToken, tokensSoFar,
                              stepSampleConfig, &sampleResult, context);
            delete tokensSoFar;
        } else {
            tokenSamplePolicy(logitsForSample, sampledToken, inputIds,
                              stepSampleConfig, &sampleResult, context);
        }
        if (logitsSlice != nullptr) {
            delete logitsSlice;
            logitsSlice = nullptr;
        }

        LongType nextTokenId = sampledToken->e<LongType>(0);

        // Gated diagnostic event: per-step scalar-path record (host reads only —
        // this is the CPU helper). Mirrors the intent of the CUDA SPEC_STEP event
        // for the non-proposing path: shows position bookkeeping and the selected
        // token so W>1-vs-W=1 divergences can be localized to a step.
        if (DSP_DIAG_ENABLED(KV_CACHE)) {
            LongType posOffV = -1, cachePosV = -1;
            if (config->positionOffsetExtIdx >= 0 && config->positionOffsetExtIdx < numExtInputs
                    && extInputs[config->positionOffsetExtIdx] != nullptr) {
                posOffV = extInputs[config->positionOffsetExtIdx]->e<LongType>(0);
            }
            if (config->cachePositionExtIdx >= 0 && config->cachePositionExtIdx < numExtInputs
                    && extInputs[config->cachePositionExtIdx] != nullptr) {
                cachePosV = extInputs[config->cachePositionExtIdx]->e<LongType>(0);
            }
            float l0 = 0, l1 = 0;
            if (logitsOutput != nullptr && logitsOutput->lengthOf() >= 2) {
                l0 = logitsOutput->e<float>(0);
                l1 = logitsOutput->e<float>(1);
            }
            DSP_DIAG(KV_CACHE,
                     "SCALAR_STEP step=%d pos=%lld posOff=%lld cachePos=%lld tok=%lld "
                     "repPen=%.3f l0=%.4f l1=%.4f",
                     step, (long long)currentPosition, (long long)posOffV, (long long)cachePosV,
                     (long long)nextTokenId, (double)repPenalty, l0, l1);
        }

        // ADR 0106 Phase 2: learn the verified scalar transition.
        if (useNgram_cpu) {
            if (specCurrentToken_cpu >= 0) {
                ngramTable_cpu[specCurrentToken_cpu] = nextTokenId;
                if (specPreviousToken_cpu >= 0) {
                    trigramTable_cpu[specPreviousToken_cpu][specCurrentToken_cpu] = nextTokenId;
                }
            }
            specPreviousToken_cpu = specCurrentToken_cpu;
            specCurrentToken_cpu = nextTokenId;
        }
        // ── MTP SCALAR-PATH MAINTENANCE + AUTHORITATIVE PUBLICATION (review
        // rounds 4+5, findings B/2 + D/5, CPU mirror of the CUDA epilogue) ─
        // This block runs whenever the SCALAR path executes and MTP metadata
        // exists - i.e. whenever the proposing-path publication block did NOT
        // run. Two distinct cases:
        //   K=0 (useMtp_cpu false): no predictor forward ran for the just-
        //      consumed token, so MAINTENANCE must consume the SAVED pre-step
        //      pair - the pending token already in mtpInputIds at predictor
        //      row currentPosition - 1. Do NOT overwrite the token with
        //      nextTokenId: that is the NEWLY EMITTED token and would write
        //      the next token into the previous token's KV row (encoded-oracle
        //      signature: key 702 where 701 is required).
        //   K>0 with ZERO proposals (useMtp_cpu true, proposedCount_cpu == 0):
        //      the proposal stage already ran a predictor forward for the
        //      current row (executeMtpCpu with maxPropose_cpu == 0), which
        //      installed its RECURSIVE self-hidden as the carry - no second
        //      maintenance forward is needed, only the authoritative replace.
        //   In BOTH cases the PUBLICATION step is mandatory: refresh the
        //   carry from the target's hidden row 0 and publish the newly
        //   emitted token as the next pending input at row currentPosition
        //   (the split-call boundary defect: the old code updated only the
        //   token/scalars on this path, leaving the recursive self-hidden
        //   installed as the carry).
        // When drafting is ON and proposals WERE produced, the proposing
        // path's publication block already handled carry + pending input for
        // the committed prefix; nothing to do here.
        if (mtpMetadataReady_cpu && (!useMtp_cpu || proposedCount_cpu == 0)) {
            REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                             && config->targetHiddenOutputIdx < numPlanOutputs
                             && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                         0, "autoregressive_decode: target hidden output is unavailable for MTP");
            // 1. Maintenance consumes the saved pair (token untouched) - K=0
            //    only. With K>0 zero-proposal the predictor already executed
            //    the current row; only its carry needs replacing.
            if (!useMtp_cpu && currentPosition >= 1) {
                config->mtpPositionOffset->p(0, currentPosition - 1);
                config->mtpCachePosition->p(0, currentPosition - 1);
                BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskCpu,
                                      (config->mtpCausalMask->buffer(), currentPosition - 1,
                                       mtpMaskLen_cpu),
                                      SD_FLOAT_TYPES);
                p0MaintenanceActive = true;
                (void)executeMtpCpu(config->mtpInputIds->e<LongType>(0), currentPosition);
                p0MaintenanceActive = false;
            }
            // 2. Authoritative publication: target-conditioned carry + newly
            //    emitted pending token at row currentPosition.
            setMtpTargetCarryCpu(planOutputs[config->targetHiddenOutputIdx], 0);
            config->mtpInputIds->p(0, nextTokenId);
            config->mtpPositionOffset->p(0, currentPosition);
            config->mtpCachePosition->p(0, currentPosition);
        }

        // Store in output and notify the reusable session layer. The callback
        // never owns decoder buffers and cannot interrupt a partially committed
        // step; cancellation is observed at the next loop boundary.
        generatedTokenIds->p(tokensGenerated, nextTokenId);
        tokensGenerated++;
        if (config->tokenCallback != nullptr) {
            config->tokenCallback(nextTokenId, config->callbackUserData);
        }

        if (step < 10 && env_isVerbose()) {
          sd_debug("CPU_DECODE_STEP[%d/%d]: nextTokenId=%lld currentPosition=%lld stopTokenCount=%d\n",
                    step, maxNewTokens, (long long)nextTokenId, (long long)currentPosition,
                    (int)stopTokenIds.size());
        }

        // ── Step 4: Check stop condition ──
        bool matchedStop = stopMatcher.accept(nextTokenId);
        bool shouldStop = matchedStop && stopTerminationAllowed(config, tokensGenerated);
        bool matchedRepetition = repetitionMatcher.accept(nextTokenId);

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
        stepTimesMs.push_back(stepMs);
        stepTokenCounts.push_back(tokensGenerated - tokensBeforeStep);

        if (shouldStop) break;
        if (matchedRepetition) {
            config->nativeFinishReason = 1;
            break;
        }

        // ── Step 5: KV scatter — copy present KV into static buffers ──
        if (!config->planOwnsKvScatter &&
            config->kvOutputIndices != nullptr && staticKvBuffers != nullptr && numKvPairs > 0) {
            std::vector<KvScatterEntry> entries(2 * numKvPairs);
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvOutIdx = config->kvOutputIndices[kv];
                REQUIRE_TRUE(kvOutIdx >= 0 && kvOutIdx < numPlanOutputs, 0,
                             "autoregressive_decode: KV output index %d out of range [0,%d) "
                             "at step %d kv=%d",
                             kvOutIdx, numPlanOutputs, step, kv);
                NDArray* presentKv = planOutputs[kvOutIdx];
                NDArray* staticBuf = staticKvBuffers[kv];
                REQUIRE_TRUE(presentKv != nullptr, 0,
                             "autoregressive_decode: KV output[%d] (planOutput[%d]) is null "
                             "at step %d — plan did not produce this output.",
                             kv, kvOutIdx, step);
                REQUIRE_TRUE(staticBuf != nullptr, 0,
                             "autoregressive_decode: static KV buffer[%d] is null at step %d.",
                             kv, step);
                REQUIRE_TRUE(presentKv->buffer() != nullptr, 0,
                             "autoregressive_decode: KV output[%d] has null host buffer "
                             "at step %d — stale or uninitialized output.",
                             kv, step);
                REQUIRE_TRUE(staticBuf->buffer() != nullptr, 0,
                             "autoregressive_decode: static KV[%d] has null host buffer "
                             "at step %d — buffer was freed or never allocated.",
                             kv, step);

                entries[kv].srcPtr = presentKv->buffer();
                entries[kv].dstPtr = staticBuf->buffer();
                entries[kv].heads = presentKv->sizeAt(1);
                entries[kv].srcSeqLen = presentKv->sizeAt(2);
                entries[kv].dstSeqLen = staticBuf->sizeAt(2);
                entries[kv].dim = presentKv->sizeAt(3);
                entries[kv].lastPos = presentKv->sizeAt(2) - 1;
                entries[kv].cachePos = currentPosition;
            }

            REQUIRE_TRUE(staticKvBuffers[0] != nullptr, 0,
                         "autoregressive_decode: staticKvBuffers[0] is null at step %d — "
                         "cannot determine KV data type for scatter.",
                         step);
            kvScatterBatched(entries.data(), 2 * numKvPairs,
                             staticKvBuffers[0]->dataType(), context);

        }

        // ── Step 6: Embedding lookup for next token ──
        // Only perform embedding lookup if we have an embeddings ext input to update.
        // In single-model mode (embeddingsExtIdx == -1), the model handles its own
        // embedding lookup internally, so we skip this step.
        if (config->embeddingsExtIdx >= 0) {
            REQUIRE_TRUE(nextTokenId >= 0 && nextTokenId < vocabSize, 0,
                         "autoregressive_decode: nextTokenId=%lld out of range [0,%lld) at step %d. "
                         "Argmax/sampling returned an invalid token ID.",
                         (long long)nextTokenId, (long long)vocabSize, step);
            REQUIRE_TRUE(embeddingTable->buffer() != nullptr, 0,
                         "autoregressive_decode: embeddingTable host buffer is null at step %d.",
                         step);
            REQUIRE_TRUE(decodeEmbedding->buffer() != nullptr, 0,
                         "autoregressive_decode: decodeEmbedding host buffer is null at step %d.",
                         step);
            BUILD_SINGLE_SELECTOR(embeddingTable->dataType(), embedLookupCpu,
                                  (embeddingTable->buffer(),
                                   decodeEmbedding->buffer(),
                                   nextTokenId, hidden, embTableRowStride),
                                  SD_COMMON_TYPES);
        }

        // ── Advance position BEFORE updating mask/posIds for the next step ──
        currentPosition++;

        // ── Step 7: Update input buffers for next step ──
        LongType kvJustWritten = currentPosition - 1;
        // 0/1 unmask skipped when the attention mask aliases the additive causal
        // mask (see attnMaskAliasesCausal).
        if (!attnMaskAliasesCausal && kvJustWritten >= 0 && kvJustWritten < maxKvLen) {
            BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskCpu,
                                  (attentionMask->buffer(), kvJustWritten, maxKvLen),
                                  SD_COMMON_TYPES);
        }

        // Causal mask: for ONNX/external-scatter (planOwnsKvScatter == false), unmask
        // currentPosition (the NEXT write slot), matching Java's advance-one-ahead pattern.
        // For GGUF (planOwnsKvScatter == true), unmask kvJustWritten.
        {
            LongType causalMaskUnmaskPos = config->planOwnsKvScatter ? kvJustWritten : currentPosition;
            if (causalMask != nullptr && causalMaskUnmaskPos >= 0 && causalMaskUnmaskPos < causalMaskLen) {
                BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskCpu,
                                      (causalMask->buffer(), causalMaskUnmaskPos, causalMaskLen),
                                      SD_FLOAT_TYPES);
            }

            LongType attnReformatUnmaskPos = kvJustWritten;
            if (attnMaskReformat != nullptr && attnReformatUnmaskPos >= 0 && attnReformatUnmaskPos < attnMaskReformatLen) {
                BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskCpu,
                                      (attnMaskReformat->buffer(), attnReformatUnmaskPos, attnMaskReformatLen),
                                      SD_FLOAT_TYPES);
            }
        }

        // Update position_ids
        positionIds->p(0, currentPosition);

        // Update input_ids
        inputIds->p(0, nextTokenId);

        // ── Update in-graph KV cache scalars (GGUF pattern) ──
        // position_offset and cache_position are scalar ext inputs that the
        // attention op reads for RoPE position and KV write position.
        if (config->positionOffsetExtIdx >= 0 && config->positionOffsetExtIdx < numExtInputs) {
            NDArray* posOffset = extInputs[config->positionOffsetExtIdx];
            if (posOffset != nullptr) {
                posOffset->p(0, currentPosition);
                posOffset->syncToDevice();
            }
        }
        if (config->cachePositionExtIdx >= 0 && config->cachePositionExtIdx < numExtInputs) {
            NDArray* cachePos = extInputs[config->cachePositionExtIdx];
            if (cachePos != nullptr) {
                cachePos->p(0, currentPosition);
                cachePos->syncToDevice();
            }
        }

        // ── Sync mutable inputs to device ──────────────────────────────────
        // Steps 6-7 wrote to the CPU host buffers. On CUDA, the plan reads from
        // GPU device buffers. Without graph capture (e.g. TRITON_NO_GC), there is
        // no captured H2D memcpy node, so we must sync explicitly. On CPU this is
        // a no-op.
        decodeEmbedding->syncToDevice();
        if (useWindowSubstrate) {
            // Sync the window tensors — on CUDA they must be device-authoritative
            // before the next plan execution. On CPU this is a no-op.
            config->windowGridMask->syncToDevice();
            // In-graph KV plans use scalar position inputs, not a position grid.
            if (config->windowPositionGrid != nullptr) {
                config->windowPositionGrid->syncToDevice();
            }
        } else {
            attentionMask->syncToDevice();
            positionIds->syncToDevice();
        }
        inputIds->syncToDevice();
        if (causalMask != nullptr) {
            causalMask->syncToDevice();
        }
        if (attnMaskReformat != nullptr) {
            attnMaskReformat->syncToDevice();
        }
        for (int kv = 0; kv < 2 * numKvPairs; kv++) {
            if (staticKvBuffers != nullptr && staticKvBuffers[kv] != nullptr) {
                staticKvBuffers[kv]->syncToDevice();
            }
        }
    }

    // ── Write token count ──
    tokenCount->p(0, static_cast<LongType>(tokensGenerated));

    // ── Compute timing stats ──
    auto loopEnd = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(loopEnd - loopStart).count();

    timingInfo->p(7, static_cast<float>(totalSpeculativeProposed));
    timingInfo->p(8, static_cast<float>(totalSpeculativeAccepted));
    timingInfo->p(9, static_cast<float>(speculativeStepCount));
    if (!stepTimesMs.empty()) {
        double avgMs = totalMs / stepTimesMs.size();
        // Throughput counts finalized emitted tokens; latency remains per step.
        double tokPerSec = totalMs > 0.0 ? (tokensGenerated * 1000.0 / totalMs) : 0.0;

        std::vector<double> sorted = stepTimesMs;
        std::sort(sorted.begin(), sorted.end());
        double p50 = sorted[sorted.size() / 2];
        double p99 = sorted[std::min<size_t>(sorted.size() - 1,
                                              static_cast<size_t>(sorted.size() * 0.99))];

        timingInfo->p(0, static_cast<float>(totalMs));
        timingInfo->p(1, static_cast<float>(avgMs));
        timingInfo->p(2, static_cast<float>(tokPerSec));
        timingInfo->p(3, static_cast<float>(p50));
        timingInfo->p(4, static_cast<float>(p99));

        // Late-steady throughput (steps 60+): excludes warmup bimodal oscillation.
        constexpr int LATE_STEADY_START = 60;
        if (static_cast<int>(stepTimesMs.size()) > LATE_STEADY_START) {
            double lateSteadyTotalMs = 0.0;
            int lateSteadyCount = 0;
            LongType lateSteadyTokens = 0;
            for (int i = LATE_STEADY_START; i < static_cast<int>(stepTimesMs.size()); i++) {
                lateSteadyTotalMs += stepTimesMs[i];
                lateSteadyTokens += stepTokenCounts[i];
                lateSteadyCount++;
            }
            double lateSteadyAvgMs = lateSteadyTotalMs / lateSteadyCount;
            double lateSteadyTokPerSec = lateSteadyTotalMs > 0.0
                ? lateSteadyTokens * 1000.0 / lateSteadyTotalMs : 0.0;
            timingInfo->p(5, static_cast<float>(lateSteadyTokPerSec));
            timingInfo->p(6, static_cast<float>(lateSteadyAvgMs));
        } else {
            timingInfo->p(5, static_cast<float>(tokPerSec));
            timingInfo->p(6, static_cast<float>(avgMs));
        }
    }
    p0.finalizedTokens = tokensGenerated;
    p0.proposals = totalSpeculativeProposed;
    p0.acceptedDrafts = totalSpeculativeAccepted;
    p0.speculativeSteps = static_cast<int>(speculativeStepCount);
    if (config->nativeFinishReason == 1 && timingInfo->lengthOf() > 6) {
        timingInfo->p(6, -1.0f);
    }

    // ── Cleanup internal allocations ──
    // Free the deep pre-verification recurrent snapshots (owned scratch NDArrays
    // captured before each verification execution; never alias the live ext
    // inputs, see the stateSnapshotArrays_cpu declaration above).
    for (size_t s = 0; s < stateSnapshotArrays_cpu.size(); ++s) {
        if (stateSnapshotArrays_cpu[s] != nullptr) {
            delete stateSnapshotArrays_cpu[s];
            stateSnapshotArrays_cpu[s] = nullptr;
        }
    }
    stateSnapshotArrays_cpu.clear();
    stateSnapshotExtIdx_cpu.clear();
    delete sampledToken;
    if (internalMask != nullptr) {
        delete internalMask;
    }
    if (internalPosIds != nullptr) {
        delete internalPosIds;
    }
    DSP_DIAG(KV_CACHE,
             "MTP_P0_CPU finalized=%lld proposals=%lld accepted=%lld steps=%d "
             "targetVerify=%d reruns=%d shortened=%d predictorProposal=%d "
             "predictorRepair=%d predictorMaintenance=%d repairLmHead=%d "
             "snapshotBytes=%llu restoreBytes=%llu stateCommitBytes=%llu",
             (long long)p0.finalizedTokens, (long long)p0.proposals,
             (long long)p0.acceptedDrafts, p0.speculativeSteps,
             p0.targetVerificationForwards, p0.acceptedPrefixReruns,
             p0.shortenedRecoveryForwards, p0.predictorProposalForwards,
             p0.predictorRepairForwards, p0.predictorMaintenanceForwards,
             p0.predictorRepairLmHeadForwards,
             (unsigned long long)p0.snapshotBytes,
             (unsigned long long)p0.restoreBytes,
             (unsigned long long)p0.stateCommitBytes);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
