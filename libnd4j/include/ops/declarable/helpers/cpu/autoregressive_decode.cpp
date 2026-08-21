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
#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/logger.h>
#include <system/env_functions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

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
    double temperature,
    int topK,
    double topP,
    double repPenalty,
    LaunchContext* context,
    AutoregressiveDecodeConfig* config) {

    // Initialize outputs
    LongType zero = 0;
    float zeroF = 0.0f;
    generatedTokenIds->assign(zero);
    tokenCount->assign(zero);
    timingInfo->assign(zeroF);

    // Validate that we have a plan to execute — hard error, not silent return.
    REQUIRE_TRUE(config != nullptr && config->planHandle != nullptr, 0,
                 "autoregressive_decode: no plan handle provided. "
                 "The Java side MUST pass a compiled NativeDynamicShapePlan via config->planHandle. "
                 "config=%p planHandle=%p",
                 config, config ? config->planHandle : nullptr);

    auto plan = config->planHandle;

    // ── Timing ──
    std::vector<double> stepTimesMs;
    stepTimesMs.reserve(maxNewTokens);
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
    graph::NativeDynamicShapePlan* mtpPlan_cpu = useMtp_cpu ? config->mtpPlanHandle : nullptr;
    graph::Context* mtpContext_cpu = useMtp_cpu
        ? reinterpret_cast<graph::Context*>(config->mtpExtInputContext) : nullptr;
    std::vector<NDArray*> mtpExtInputsVec_cpu;
    std::vector<NDArray*> mtpPlanOutputsVec_cpu;
    NDArray** mtpExtInputs_cpu = nullptr;
    NDArray** mtpPlanOutputs_cpu = nullptr;
    int mtpNumExtInputs_cpu = 0;
    int mtpNumOutputs_cpu = 0;
    LongType mtpMaskLen_cpu = 0;

    if (useMtp_cpu) {
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
        LongType bestIdx = 0;
        if (dtype == DataType::FLOAT32) {
            auto* lp = reinterpret_cast<const float*>(logitsRowPtr);
            float bestVal = lp[0];
            for (LongType v = 1; v < vocabSize; v++) {
                if (lp[v] > bestVal) { bestVal = lp[v]; bestIdx = v; }
            }
        } else if (dtype == DataType::HALF) {
            // FP16: use float for comparison
            auto* lp = reinterpret_cast<const float16*>(logitsRowPtr);
            float bestVal = (float)lp[0];
            for (LongType v = 1; v < vocabSize; v++) {
                float val = (float)lp[v];
                if (val > bestVal) { bestVal = val; bestIdx = v; }
            }
        } else if (dtype == DataType::DOUBLE) {
            auto* lp = reinterpret_cast<const double*>(logitsRowPtr);
            double bestVal = lp[0];
            for (LongType v = 1; v < vocabSize; v++) {
                if (lp[v] > bestVal) { bestVal = lp[v]; bestIdx = v; }
            }
        } else {
            // Fallback: single row — let sampledToken handle it via tokenSamplePolicy
            bestIdx = 0;
        }
        return bestIdx;
    };

    // Helper: byte stride for one logits row given the dtype.
    auto logitsByteStride = [&](DataType dtype, LongType vocabSize) -> LongType {
        switch (dtype) {
            case DataType::FLOAT32: return vocabSize * sizeof(float);
            case DataType::HALF: return vocabSize * 2;
            case DataType::DOUBLE:  return vocabSize * sizeof(double);
            default: return vocabSize * sizeof(float);
        }
    };

    auto executeMtpCpu = [&](LongType tokenId, LongType position) -> LongType {
        REQUIRE_TRUE(useMtp_cpu, 0,
                     "autoregressive_decode: attempted MTP CPU execution while MTP is disabled");
        config->mtpInputIds->p(0, tokenId);
        config->mtpPositionOffset->p(0, position);
        config->mtpCachePosition->p(0, position);
        BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskCpu,
                              (config->mtpCausalMask->buffer(), position, mtpMaskLen_cpu),
                              SD_FLOAT_TYPES);

        Status mtpStatus = mtpPlan_cpu->execute(
            mtpExtInputs_cpu, mtpNumExtInputs_cpu,
            mtpPlanOutputs_cpu, mtpNumOutputs_cpu,
            nullptr);
        REQUIRE_TRUE(mtpStatus == Status::OK, 0,
                     "autoregressive_decode: MTP CPU plan failed at position %lld with status %d",
                     (long long)position, static_cast<int>(mtpStatus));
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
                (void)executeMtpCpu(mtpToken, currentPosition);
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

        Status planStatus = plan->execute(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            nullptr);

        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
            clearKvScaleRegistry();
        }

        REQUIRE_TRUE(planStatus == Status::OK, 0,
                     "autoregressive_decode: plan execution FAILED at step %d with status %d. "
                     "Plan state: frozen=%d numExt=%d numOutputs=%d. "
                     "This is NOT recoverable — fix the plan execution failure.",
                     step, static_cast<int>(planStatus),
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
        // pass's logits, and on partial acceptance re-execute the plan with
        // actual_sequence_length = 1 + acceptedDrafts so the state outputs consumed by
        // the feedback copies below advance through the accepted prefix only. Emission
        // uses the FIRST pass's argmaxes (captured into specRowArgmax_cpu); the
        // downstream speculative block consumes these instead of recomputing from the
        // re-run's logits (whose rows beyond the accepted prefix are not meaningful).
        int specAccepted_cpu = -1;   // -1 = not a proposing step
        LongType specRowArgmax_cpu[33] = {};
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
            specAccepted_cpu = 0;
            while (specAccepted_cpu < proposedCount_cpu &&
                   specRowArgmax_cpu[specAccepted_cpu] == draftIds_cpu[specAccepted_cpu]) {
                specAccepted_cpu++;
            }

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

            if (specAccepted_cpu < proposedCount_cpu
                    && config->actualSequenceLengthExtIdx >= 0
                    && config->actualSequenceLengthExtIdx < numExtInputs
                    && extInputs[config->actualSequenceLengthExtIdx] != nullptr) {
                NDArray* aslArr = extInputs[config->actualSequenceLengthExtIdx];
                aslArr->p(0, static_cast<LongType>(1 + specAccepted_cpu));
                DSP_DIAG(KV_CACHE,
                         "SPEC_STATE_RERUN step=%d proposed=%d accepted=%d — re-executing "
                         "with actual_sequence_length=%d for accepted-prefix state commit",
                         step, proposedCount_cpu, specAccepted_cpu, 1 + specAccepted_cpu);
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
                Status rerunStatus = plan->execute(
                    extInputs, numExtInputs,
                    planOutputs, numPlanOutputs,
                    nullptr);
                if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
                    clearKvScaleRegistry();
                }
                REQUIRE_TRUE(rerunStatus == Status::OK, 0,
                             "autoregressive_decode: accepted-prefix state re-execution FAILED "
                             "at step %d with status %d (accepted=%d of %d).",
                             step, static_cast<int>(rerunStatus), specAccepted_cpu,
                             proposedCount_cpu);
            }
        }

        // Commit the target model's accepted hidden state into the reusable MTP
        // carry. Predictor KV writes beyond a rejected prefix remain allocated,
        // but their mask entries are restored before the next draft chain.
        if (useMtp_cpu) {
            REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                             && config->targetHiddenOutputIdx < numPlanOutputs
                             && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                         0, "autoregressive_decode: target hidden output is unavailable for MTP");

            int carryRow_cpu = proposedCount_cpu > 0
                ? std::max(0, specAccepted_cpu) : 0;
            LongType nextMtpPosition_cpu = currentPosition + carryRow_cpu + 1;
            LongType mtpProcessedThrough_cpu = currentPosition;

            if (proposedCount_cpu > 0) {
                mtpProcessedThrough_cpu = currentPosition + proposedCount_cpu - 1;
                if (carryRow_cpu == proposedCount_cpu) {
                    // K predictor calls produce K drafts but consume only the base
                    // token plus drafts [0,K-2]. Consume the final accepted draft
                    // so predictor KV is aligned with the target's bonus token.
                    (void)executeMtpCpu(draftIds_cpu[proposedCount_cpu - 1],
                                        currentPosition + proposedCount_cpu);
                    mtpProcessedThrough_cpu = currentPosition + proposedCount_cpu;
                }
            }

            if (nextMtpPosition_cpu <= mtpProcessedThrough_cpu) {
                BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), maskCausalRangeCpu,
                                      (config->mtpCausalMask->buffer(), nextMtpPosition_cpu,
                                       mtpProcessedThrough_cpu + 1, mtpMaskLen_cpu),
                                      SD_FLOAT_TYPES);
            }

            setMtpTargetCarryCpu(planOutputs[config->targetHiddenOutputIdx], carryRow_cpu);
            config->mtpPositionOffset->p(0, nextMtpPosition_cpu);
            config->mtpCachePosition->p(0, nextMtpPosition_cpu);
            if (proposedCount_cpu > 0) {
                config->mtpInputIds->p(0, specRowArgmax_cpu[carryRow_cpu]);
            }
        }

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
                        valid ? planOutputs[outIdx]->e<float>(0) : -999.0f,
                        valid ? planOutputs[outIdx]->e<float>(1) : -999.0f,
                        valid ? planOutputs[outIdx]->e<float>(2) : -999.0f,
                        valid ? extInputs[extIdx]->e<float>(0) : -999.0f,
                        valid ? extInputs[extIdx]->e<float>(1) : -999.0f,
                        valid ? extInputs[extIdx]->e<float>(2) : -999.0f);
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
        if (useSpeculative_cpu && proposedCount_cpu > 0 && logitsRank == 3) {
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
            for (int i = 0; i < 33; i++) rowArgmax[i] = specRowArgmax_cpu[i];
            int acceptedDrafts = specAccepted_cpu >= 0 ? specAccepted_cpu : 0;
            int n = acceptedDrafts + 1;
            totalSpeculativeProposed += proposedCount_cpu;
            totalSpeculativeAccepted += acceptedDrafts;
            speculativeStepCount++;

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
            bool shouldStop = false;
            int storedCount = 0;
            for (int i = 0; i < n && tokensGenerated < maxNewTokens; i++) {
                LongType tok = rowArgmax[i];
                generatedTokenIds->p(tokensGenerated, tok);
                tokensGenerated++;
                storedCount++;
                if (config->tokenCallback != nullptr) {
                    config->tokenCallback(tok, config->callbackUserData);
                }
                for (int s : stopTokenIds) {
                    if (tok == static_cast<LongType>(s)) { shouldStop = true; break; }
                }
                if (shouldStop) break;
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

            if (shouldStop) break;

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
        if (useMtp_cpu) {
            // The target hidden row was committed immediately after target
            // execution. Pair it with the just-sampled, still-unwritten token.
            config->mtpInputIds->p(0, nextTokenId);
            config->mtpPositionOffset->p(0, currentPosition + 1);
            config->mtpCachePosition->p(0, currentPosition + 1);
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
        bool shouldStop = false;
        for (int s : stopTokenIds) {
            if (nextTokenId == static_cast<LongType>(s)) {
                shouldStop = true;
                if (env_isVerbose()) {
                  sd_debug("CPU_DECODE_STEP[%d]: STOP matched token %lld == stopId %d\n",
                            step, (long long)nextTokenId, s);
                }
                break;
            }
        }

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepMs = std::chrono::duration<double, std::milli>(stepEnd - stepStart).count();
        stepTimesMs.push_back(stepMs);

        if (shouldStop) break;

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
            config->windowPositionGrid->syncToDevice();
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
        double tokPerSec = stepTimesMs.size() > 0 ? (stepTimesMs.size() * 1000.0 / totalMs) : 0.0;

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
            for (int i = LATE_STEADY_START; i < static_cast<int>(stepTimesMs.size()); i++) {
                lateSteadyTotalMs += stepTimesMs[i];
                lateSteadyCount++;
            }
            double lateSteadyAvgMs = lateSteadyTotalMs / lateSteadyCount;
            double lateSteadyTokPerSec = lateSteadyCount * 1000.0 / lateSteadyTotalMs;
            timingInfo->p(5, static_cast<float>(lateSteadyTokPerSec));
            timingInfo->p(6, static_cast<float>(lateSteadyAvgMs));
        } else {
            timingInfo->p(5, static_cast<float>(tokPerSec));
            timingInfo->p(6, static_cast<float>(avgMs));
        }
    }

    // ── Cleanup internal allocations ──
    delete sampledToken;
    if (internalMask != nullptr) {
        delete internalMask;
    }
    if (internalPosIds != nullptr) {
        delete internalPosIds;
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
