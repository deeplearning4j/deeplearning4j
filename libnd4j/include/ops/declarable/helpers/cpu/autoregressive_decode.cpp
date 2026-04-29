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
#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/logger.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
    if (position < maskLen) {
        mask[position] = static_cast<T>(0);
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

// ═══════════════════════════════════════════════════════════════════════════════
// Main CPU Implementation — equivalent logic to autoregressiveDecodeCuda
// ═══════════════════════════════════════════════════════════════════════════════

void autoregressiveDecodeCpu(
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

    for (int step = 0; step < maxNewTokens; step++) {
        auto stepStart = std::chrono::high_resolution_clock::now();

        // ── Step 1: Update plan external inputs for this decode step ──
        if (config->embeddingsExtIdx >= 0 && config->embeddingsExtIdx < numExtInputs) {
            extInputs[config->embeddingsExtIdx] = decodeEmbedding;
        }

        if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
            extInputs[config->maskExtIdx] = attentionMask;
        }

        if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
            extInputs[config->posIdsExtIdx] = positionIds;
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


        // ── Step 2: Execute plan ──
        // On CPU, use execute() instead of executeSteadyState() (which is CUDA-only).
        Status planStatus = plan->execute(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            nullptr);

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

        // ── Step 2b: GDN/conv recurrent state feedback ──
        // Copy state outputs back to ext inputs for the next decode step.
        // This is critical for hybrid architectures (e.g. Qwen with GDN layers).
        // Without this, GDN layers see frozen state from warmup and degenerate.
        //
        // Safety: validate element count, dtype, and DataBuffer byte capacity before
        // each assign() to prevent buffer overruns when plan output shape diverges
        // from the external input shape.
        if (config->numGdnStatePairs > 0 && config->gdnStateExtIndices != nullptr
            && config->gdnStateOutputIndices != nullptr) {
            for (int s = 0; s < config->numGdnStatePairs; s++) {
                int outIdx = config->gdnStateOutputIndices[s];
                int extIdx = config->gdnStateExtIndices[s];
                if (outIdx >= 0 && outIdx < numPlanOutputs && planOutputs[outIdx] != nullptr
                    && extIdx >= 0 && extIdx < numExtInputs && extInputs[extIdx] != nullptr) {
                    NDArray* src = planOutputs[outIdx];
                    NDArray* dst = extInputs[extIdx];
                    // Validate element count
                    if (src->lengthOf() != dst->lengthOf()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: GDN state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] length=%lld != extInput[%d] length=%lld — shape mismatch",
                            step, s, outIdx, (long long)src->lengthOf(),
                            extIdx, (long long)dst->lengthOf());
                        continue;
                    }
                    // Validate dtype
                    if (src->dataType() != dst->dataType()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: GDN state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] dtype=%d != extInput[%d] dtype=%d — type mismatch",
                            step, s, outIdx, (int)src->dataType(),
                            extIdx, (int)dst->dataType());
                        continue;
                    }
                    // Validate DataBuffer byte capacity (guards against overrun)
                    auto* srcDb = src->dataBuffer();
                    auto* dstDb = dst->dataBuffer();
                    if (srcDb != nullptr && dstDb != nullptr &&
                        srcDb->getLenInBytes() > dstDb->getLenInBytes()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: GDN state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] byte size=%zu exceeds extInput[%d] DataBuffer capacity=%zu — overrun prevented",
                            step, s, outIdx, srcDb->getLenInBytes(),
                            extIdx, dstDb->getLenInBytes());
                        continue;
                    }
                    dst->assign(src);
                }
            }
        }
        if (config->numConvStatePairs > 0 && config->convStateExtIndices != nullptr
            && config->convStateOutputIndices != nullptr) {
            for (int s = 0; s < config->numConvStatePairs; s++) {
                int outIdx = config->convStateOutputIndices[s];
                int extIdx = config->convStateExtIndices[s];
                if (outIdx >= 0 && outIdx < numPlanOutputs && planOutputs[outIdx] != nullptr
                    && extIdx >= 0 && extIdx < numExtInputs && extInputs[extIdx] != nullptr) {
                    NDArray* src = planOutputs[outIdx];
                    NDArray* dst = extInputs[extIdx];
                    // Validate element count
                    if (src->lengthOf() != dst->lengthOf()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: conv state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] length=%lld != extInput[%d] length=%lld — shape mismatch",
                            step, s, outIdx, (long long)src->lengthOf(),
                            extIdx, (long long)dst->lengthOf());
                        continue;
                    }
                    // Validate dtype
                    if (src->dataType() != dst->dataType()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: conv state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] dtype=%d != extInput[%d] dtype=%d — type mismatch",
                            step, s, outIdx, (int)src->dataType(),
                            extIdx, (int)dst->dataType());
                        continue;
                    }
                    // Validate DataBuffer byte capacity (guards against overrun)
                    auto* srcDb = src->dataBuffer();
                    auto* dstDb = dst->dataBuffer();
                    if (srcDb != nullptr && dstDb != nullptr &&
                        srcDb->getLenInBytes() > dstDb->getLenInBytes()) {
                        DSP_DIAG(FALLBACK,
                            "autoregressive_decode: conv state feedback SKIPPED at step %d pair %d: "
                            "plan output[%d] byte size=%zu exceeds extInput[%d] DataBuffer capacity=%zu — overrun prevented",
                            step, s, outIdx, srcDb->getLenInBytes(),
                            extIdx, dstDb->getLenInBytes());
                        continue;
                    }
                    dst->assign(src);
                }
            }
        }

        // ── Step 3: Token sampling ──
        NDArray* logitsOutput = planOutputs[config->logitsOutputIdx];
        LongType logitsSeqLen = logitsOutput->sizeAt(1);
        LongType logitsVocab = logitsOutput->sizeAt(2);

        if (temperature <= 0.0 || (topK <= 1 && topP <= 0.0)) {
            // Greedy: argmax over last-position logits
            REQUIRE_TRUE(logitsVocab > 0, 0,
                         "autoregressive_decode: logits vocab dimension is 0 at step %d. "
                         "Cannot perform argmax on empty vocabulary.",
                         step);
            LongType lastPosOffset = (logitsSeqLen - 1) * logitsVocab;
            const void* logitsPtr = static_cast<const char*>(logitsOutput->buffer())
                                    + lastPosOffset * logitsOutput->sizeOfT();

            BUILD_SINGLE_SELECTOR(logitsOutput->dataType(), argmaxCpu,
                                  (logitsPtr, sampledToken->buffer(), logitsVocab),
                                  SD_COMMON_TYPES);
        } else {
            // Sampling: use tokenSampleCpu
            tokenSampleCpu(logitsOutput, sampledToken, temperature, topK, topP,
                           static_cast<LongType>(step), context);
        }

        LongType nextTokenId = sampledToken->e<LongType>(0);

        // Store in output
        generatedTokenIds->p(tokensGenerated, nextTokenId);
        tokensGenerated++;

        // ── Step 4: Check stop condition ──
        bool shouldStop = false;
        for (int s : stopTokenIds) {
            if (nextTokenId == static_cast<LongType>(s)) {
                shouldStop = true;
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

            // On CPU, tickWriteDevice is a no-op, but we call it for consistency
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                staticKvBuffers[kv]->tickWriteDevice();
            }
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
        if (kvJustWritten >= 0 && kvJustWritten < maxKvLen) {
            BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskCpu,
                                  (attentionMask->buffer(), kvJustWritten, maxKvLen),
                                  SD_COMMON_TYPES);
        }

        if (causalMask != nullptr && currentPosition < causalMaskLen) {
            BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskCpu,
                                  (causalMask->buffer(), currentPosition, causalMaskLen),
                                  SD_FLOAT_TYPES);
        }

        if (attnMaskReformat != nullptr && kvJustWritten >= 0 && kvJustWritten < attnMaskReformatLen) {
            BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskCpu,
                                  (attnMaskReformat->buffer(), kvJustWritten, attnMaskReformatLen),
                                  SD_FLOAT_TYPES);
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
        attentionMask->syncToDevice();
        positionIds->syncToDevice();
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
