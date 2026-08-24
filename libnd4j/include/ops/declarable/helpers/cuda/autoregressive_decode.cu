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

#include <system/op_boilerplate.h>
#include <system/env_functions.h>
#include <helpers/logger.h>
#include <ops/declarable/helpers/autoregressive_decode.h>
#include <ops/declarable/helpers/token_sample.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <graph/Context.h>
#include <graph/DspDiagnostics.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

// ─── CUDA Kernels ────────────────────────────────────────────────────────────

/**
 * CUDA kernel: look up a single row from the embedding table.
 *
 * Given embeddingTable [vocabSize, hidden] and a token ID, copies
 * embeddingTable[tokenId, :] into outputEmbed [1, 1, hidden].
 *
 * One block, blockDim.x threads — each thread copies hidden/blockDim.x elements.
 */
template <typename T>
static SD_KERNEL void embedLookupKernel(const void* vEmbTable,
                                         void* vOutput,
                                         LongType tokenId,
                                         LongType hidden,
                                         LongType tableRowStride) {
    auto embTable = reinterpret_cast<const T*>(vEmbTable);
    auto output = reinterpret_cast<T*>(vOutput);

    LongType baseOffset = tokenId * tableRowStride;
    for (LongType i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[i] = embTable[baseOffset + i];
    }
}

/**
 * Launcher for embedLookupKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void embedLookupLauncher(const cudaStream_t* stream, const void* embTable,
                                 void* output, LongType tokenId,
                                 LongType hidden, LongType tableRowStride) {
    embedLookupKernel<T><<<1, 256, 0, *stream>>>(embTable, output, tokenId, hidden, tableRowStride);
}

/**
 * CUDA kernel: update attention mask for the next decode step.
 *
 * Sets mask[position] = 1.0 (unmask the new position).
 * The mask is [1, 1, 1, maxKvLen] for single-token decode.
 */
template <typename T>
static SD_KERNEL void updateAttentionMaskKernel(void* vMask,
                                                  LongType position,
                                                  LongType maxKvLen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto mask = reinterpret_cast<T*>(vMask);
        if (position < maxKvLen) {
            mask[position] = static_cast<T>(1);
        }
    }
}

/**
 * Launcher for updateAttentionMaskKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void updateAttentionMaskLauncher(const cudaStream_t* stream,
                                         void* vMask,
                                         LongType position,
                                         LongType maxKvLen) {
    updateAttentionMaskKernel<T><<<1, 1, 0, *stream>>>(vMask, position, maxKvLen);
}

/**
 * CUDA kernel: update position_ids for the next decode step.
 *
 * Sets positionIds[0] = newPosition.
 */
static SD_KERNEL void updatePositionIdsKernel(void* vPositionIds,
                                                LongType newPosition) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto posIds = reinterpret_cast<LongType*>(vPositionIds);
        posIds[0] = newPosition;
    }
}

/**
 * CUDA kernel: update input_ids for the next decode step.
 *
 * Sets inputIds[0] = newTokenId.
 */
static SD_KERNEL void updateInputIdsKernel(void* vInputIds,
                                             LongType newTokenId) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto inputIds = reinterpret_cast<LongType*>(vInputIds);
        inputIds[0] = newTokenId;
    }
}

/**
 * CUDA kernel: update causal mask for the next decode step.
 *
 * Sets causalMask[position] = 0.0f (unmask the new position).
 * The causal mask is [1, 1, 1, maskLen] FLOAT for single-token decode,
 * filled with MASK_FILL (-3.4028235e+38f) for masked positions and 0.0f
 * for unmasked positions. Each decode step unmasks one more position.
 */
template <typename T>
static SD_KERNEL void updateCausalMaskKernel(void* vMask,
                                               LongType position,
                                               LongType maskLen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto mask = reinterpret_cast<T*>(vMask);
        if (position < maskLen) {
            mask[position] = static_cast<T>(0);
        }
    }
}

/**
 * Launcher for updateCausalMaskKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void updateCausalMaskLauncher(const cudaStream_t* stream,
                                      void* vMask,
                                      LongType position,
                                      LongType maskLen) {
    updateCausalMaskKernel<T><<<1, 1, 0, *stream>>>(vMask, position, maskLen);
}

/**
 * CUDA kernel: remask a rejected speculative suffix in a scalar predictor cache.
 *
 * Predictor KV writes are harmless past the accepted prefix only while the
 * corresponding additive-mask entries remain inaccessible. Later predictor
 * steps overwrite those slots before making them visible again.
 */
template <typename T>
static SD_KERNEL void maskCausalRangeKernel(void* vMask,
                                            LongType begin,
                                            LongType end,
                                            LongType maskLen,
                                            float maskFill) {
    auto mask = reinterpret_cast<T*>(vMask);
    begin = begin < 0 ? 0 : begin;
    end = end > maskLen ? maskLen : end;
    for (LongType position = begin + blockIdx.x * blockDim.x + threadIdx.x;
         position < end;
         position += static_cast<LongType>(gridDim.x) * blockDim.x) {
        mask[position] = static_cast<T>(maskFill);
    }
}

template <typename T>
static void maskCausalRangeLauncher(const cudaStream_t* stream,
                                    void* vMask,
                                    LongType begin,
                                    LongType end,
                                    LongType maskLen) {
    begin = std::max<LongType>(0, begin);
    end = std::min<LongType>(end, maskLen);
    if (begin >= end) return;
    constexpr int threads = 256;
    int blocks = static_cast<int>((end - begin + threads - 1) / threads);
    float maskFill = (sizeof(T) == 2) ? -65504.0f : -1e9f;
    maskCausalRangeKernel<T><<<blocks, threads, 0, *stream>>>(
        vMask, begin, end, maskLen, maskFill);
}

/**
 * CUDA kernel: refill the GGUF W-wide causal mask for one decode step.
 *
 * The [1,1,W,maxKvLen] additive bias frozen into the plan encodes a linear
 * speculative chain: query slot w sits at absolute position currentPos + w and
 * may attend every column c <= currentPos + w (committed past, lower window
 * slots, self). The freeze-time mask from DecoderInputBuilder encodes that band
 * at the freeze position only, and updateCausalMaskKernel's single flat-index
 * write only ever advances row 0 — draft rows would stay stuck at the freeze
 * geometry. Refill all W rows in-place each step. Inactive rows get the same
 * causal band so their softmax rows stay finite (outputs ignored).
 */
template <typename T>
static SD_KERNEL void refillWindowCausalMaskKernel(void* vMask,
                                                    LongType wMax,
                                                    LongType maxKvLen,
                                                    LongType currentPos,
                                                    float maskFill) {
    LongType totalElems = wMax * maxKvLen;
    auto mask = reinterpret_cast<T*>(vMask);
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalElems;
         idx += static_cast<LongType>(gridDim.x) * blockDim.x) {
        LongType w = idx / maxKvLen;
        LongType c = idx % maxKvLen;
        mask[idx] = (c <= currentPos + w) ? static_cast<T>(0.0f) : static_cast<T>(maskFill);
    }
}

template <typename T>
static void refillWindowCausalMaskLauncher(const cudaStream_t* stream,
                                           void* vMask,
                                           LongType wMax,
                                           LongType maxKvLen,
                                           LongType currentPos) {
    // Match DecoderInputBuilder.buildInGraphWindowMask fill values: -65504 for
    // 2-byte float types (half/bfloat16 — exp() underflows to 0 either way),
    // -1e9 for float/double.
    float maskFill = (sizeof(T) == 2) ? -65504.0f : -1e9f;
    LongType totalElems = wMax * maxKvLen;
    int threads = 256;
    int blocks = static_cast<int>((totalElems + threads - 1) / threads);
    refillWindowCausalMaskKernel<T><<<blocks, threads, 0, *stream>>>(
        vMask, wMax, maxKvLen, currentPos, maskFill);
}

/**
 * CUDA kernel: build initial attention mask from prefill length.
 *
 * Sets mask[0..prefillSeqLen-1] = 1, rest stays 0.
 * Mask is pre-zeroed by the caller.
 */
template <typename T>
static SD_KERNEL void buildInitialMaskKernel(void* vMask, LongType prefillSeqLen, LongType maxKvLen) {
    LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < prefillSeqLen && idx < maxKvLen) {
        reinterpret_cast<T*>(vMask)[idx] = static_cast<T>(1);
    }
}

/**
 * Launcher for buildInitialMaskKernel.
 */
template <typename T>
static void buildInitialMaskLauncher(const cudaStream_t* stream, void* vMask,
                                      LongType prefillSeqLen, LongType maxKvLen) {
    int threads = 256;
    int blocks = (prefillSeqLen + threads - 1) / threads;
    buildInitialMaskKernel<T><<<blocks, threads, 0, *stream>>>(vMask, prefillSeqLen, maxKvLen);
}

// ─── ADR 0106 Phase 1: Window substrate CUDA kernels ─────────────────────────

/**
 * CUDA kernel: fill the fixed [1,1,W_max,past+W_max] window attention mask for one step.
 *
 * Each thread handles one element of the mask. The mask layout (flattened [W_max*(past+W_max)]):
 *   row w, col k: maskData[w*(past+W_max) + k]
 *     = 0.0f  if k < currentPos           (attend to past KV)
 *     = 0.0f  if k == currentPos + w      (attend to self, causal)
 *     = MASK_FILL otherwise               (masked)
 *   rows w >= activeWindow: entirely MASK_FILL
 *
 * Grid: 1D over all elements. One block sufficient for W_max <= 32, rowLen <= 4096.
 */
static SD_KERNEL void fillWindowMaskKernel(void* vMask,
                                            LongType wMax,
                                            LongType rowLen,
                                            LongType currentPos,
                                            LongType activeWindow,
                                            float maskFill) {
    LongType totalElems = wMax * rowLen;
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalElems;
         idx += gridDim.x * blockDim.x) {
        LongType w = idx / rowLen;
        LongType k = idx % rowLen;
        float val;
        if (k <= currentPos + w) {
            // Keep inactive fixed-width rows causal too: all-masked softmax rows can
            // contaminate fused W-wide kernels. activeWindow gates recurrent commits.
            val = 0.0f;
        } else {
            val = maskFill;   // mask future positions
        }
        reinterpret_cast<float*>(vMask)[idx] = val;
    }
}

/**
 * CUDA kernel: fill the fixed [1, W_max] window position grid for one step.
 *
 * grid[w] = currentPos + w  for w < activeWindow
 * grid[w] = currentPos      for w >= activeWindow (irrelevant; masked)
 */
static SD_KERNEL void fillWindowPositionGridKernel(void* vPos,
                                                    LongType wMax,
                                                    LongType currentPos,
                                                    LongType activeWindow) {
    for (LongType w = blockIdx.x * blockDim.x + threadIdx.x; w < wMax; w += gridDim.x * blockDim.x) {
        LongType pos = (w < activeWindow) ? (currentPos + w) : currentPos;
        reinterpret_cast<LongType*>(vPos)[w] = pos;
    }
}

// ─── Argmax helper (greedy decode) ───────────────────────────────────────────

/**
 * CUDA kernel: find argmax over a float/half row [vocabSize].
 * Writes the index to output[0] as INT64.
 *
 * Block-level reduction using shared memory.
 */
template <typename T>
static SD_KERNEL void argmaxKernel(const void* vLogits, void* vOutput, LongType vocabSize) {
    extern __shared__ char smem[];
    auto sMaxVal = reinterpret_cast<T*>(smem);
    auto sMaxIdx = reinterpret_cast<LongType*>(smem + blockDim.x * sizeof(T));

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<LongType*>(vOutput);

    T localMax = static_cast<T>(-1e30);
    LongType localIdx = 0;

    for (LongType i = threadIdx.x; i < vocabSize; i += blockDim.x) {
        T val = logits[i];
        if (val > localMax) {
            localMax = val;
            localIdx = i;
        }
    }

    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sMaxVal[threadIdx.x + stride] > sMaxVal[threadIdx.x]) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[0] = sMaxIdx[0];
    }
}

/**
 * Launcher for argmaxKernel — called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void argmaxLauncher(const cudaStream_t* stream, const void* logitsPtr,
                            void* outputPtr, LongType vocabSize) {
    int threads = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxKernel<T><<<1, threads, smemSize, *stream>>>(logitsPtr, outputPtr, vocabSize);
}

// ─── ADR 0106 Phase 2: n-gram speculative decoding kernels ───────────────────

/**
 * CUDA kernel: look up W token rows from the embedding table for speculative
 * multi-token prefill.
 *
 * Writes embeddingTable[tokenIds[w], :] into outputEmbed[w * hidden .. (w+1)*hidden - 1]
 * for w = 0..numTokens-1.
 *
 * Grid: numTokens blocks; up to 256 threads per block.
 * Each block handles one token's hidden vector.
 */
template <typename T>
static SD_KERNEL void embedLookupMultiTokenKernel(const void* vEmbTable,
                                                   void* vOutput,
                                                   const LongType* tokenIds,
                                                   LongType numTokens,
                                                   LongType hidden,
                                                   LongType tableRowStride) {
    LongType w = blockIdx.x;
    if (w >= numTokens) return;
    auto embTable = reinterpret_cast<const T*>(vEmbTable);
    auto output   = reinterpret_cast<T*>(vOutput);
    LongType tokId     = tokenIds[w];
    LongType baseOffset = tokId * tableRowStride;
    LongType outOffset  = w * hidden;
    for (LongType i = threadIdx.x; i < hidden; i += blockDim.x) {
        output[outOffset + i] = embTable[baseOffset + i];
    }
}

/**
 * Launcher for embedLookupMultiTokenKernel.
 * numTokens blocks, 256 threads each.
 */
template <typename T>
static void embedLookupMultiTokenLauncher(const cudaStream_t* stream,
                                          const void* embTable,
                                          void* output,
                                          const LongType* tokenIds,
                                          LongType numTokens,
                                          LongType hidden,
                                          LongType tableRowStride) {
    if (numTokens <= 0) return;
    embedLookupMultiTokenKernel<T><<<static_cast<int>(numTokens), 256, 0, *stream>>>(
        embTable, output, tokenIds, numTokens, hidden, tableRowStride);
}

/**
 * CUDA kernel: find argmax independently for each of numRows rows of a
 * contiguous [numRows, vocabSize] logits buffer.
 *
 * Writes output[row] = argmax of logits[row, :].
 * One block per row; shared memory holds per-thread (maxVal, maxIdx) pairs.
 */
template <typename T>
static SD_KERNEL void argmaxMultiRowKernel(const void* vLogits, void* vOutput,
                                            LongType numRows, LongType vocabSize) {
    extern __shared__ char smem[];
    auto sMaxVal = reinterpret_cast<T*>(smem);
    auto sMaxIdx = reinterpret_cast<LongType*>(smem + blockDim.x * sizeof(T));

    LongType row = blockIdx.x;
    if (row >= numRows) return;

    auto logits = reinterpret_cast<const T*>(vLogits) + row * vocabSize;
    auto output = reinterpret_cast<LongType*>(vOutput);

    T       localMax = static_cast<T>(-1e30);
    LongType localIdx = 0;
    for (LongType i = threadIdx.x; i < vocabSize; i += blockDim.x) {
        T val = logits[i];
        if (val > localMax) { localMax = val; localIdx = i; }
    }
    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            if (sMaxVal[threadIdx.x + stride] > sMaxVal[threadIdx.x]) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = sMaxIdx[0];
}

/**
 * Launcher for argmaxMultiRowKernel.
 * numRows blocks, 256 threads per block with smem for (maxVal, maxIdx) pairs.
 */
template <typename T>
static void argmaxMultiRowLauncher(const cudaStream_t* stream, const void* logitsPtr,
                                   void* outputPtr, LongType numRows, LongType vocabSize) {
    if (numRows <= 0) return;
    int threads  = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxMultiRowKernel<T><<<static_cast<int>(numRows), threads, smemSize, *stream>>>(
        logitsPtr, outputPtr, numRows, vocabSize);
}

// ─── Main Implementation ─────────────────────────────────────────────────────

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

    auto stream = context->getCudaStream();

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

    DSP_DIAG(KV_CACHE,
             "AUTOREGRESSIVE_DECODE_CUDA entered plan=%p maxNewTokens=%d prefillSeqLen=%d "
             "embExtIdx=%d maskExtIdx=%d posExtIdx=%d idsExtIdx=%d causalExtIdx=%d "
             "attnReformatExtIdx=%d numKvPairs=%d logitsOutIdx=%d",
             plan, maxNewTokens, prefillSeqLen,
             config->embeddingsExtIdx, config->maskExtIdx, config->posIdsExtIdx,
             config->inputIdsExtIdx, config->causalMaskExtIdx,
             config->attnMaskReformatExtIdx, numKvPairs, config->logitsOutputIdx);

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
    // Shape: [1, 1, 1, maxKvLen] — single-token decode mask
    NDArray* internalMask = nullptr;
    LongType maxKvLen = 0;
    if (attentionMask != nullptr) {
        maxKvLen = attentionMask->sizeAt(-1);
    } else {
        // Allocate: maxKvLen = prefillSeqLen + maxNewTokens
        maxKvLen = prefillSeqLen + maxNewTokens;
        std::vector<LongType> maskShape = {1, 1, 1, maxKvLen};
        internalMask = NDArrayFactory::create('c', maskShape, DataType::FLOAT32, context);
        internalMask->assign(zeroF);
        // Fill prefill positions
        NDArray::prepareSpecialUse({internalMask}, {});
        BUILD_SINGLE_SELECTOR(internalMask->dataType(), buildInitialMaskLauncher,
                              (stream, internalMask->specialBuffer(), prefillSeqLen, maxKvLen),
                              SD_COMMON_TYPES);
        NDArray::registerSpecialUse({internalMask}, {});
        attentionMask = internalMask;
    }

    // ── Build internal position_ids if not provided ──
    // Shape: [1, 1] — single-token decode
    NDArray* internalPosIds = nullptr;
    if (positionIds == nullptr) {
        std::vector<LongType> posShape = {1, 1};
        internalPosIds = NDArrayFactory::create('c', posShape, DataType::INT64, context);
        internalPosIds->p(0, static_cast<LongType>(prefillSeqLen));
        positionIds = internalPosIds;
    }

    // ── Working buffers ──
    // Reuse prefillEmbeddings (Java's decodeEmbeddings [1,1,hidden]) for embed lookup.
    // CRITICAL: Do NOT allocate a new NDArray — the CUDA graph was captured with
    // prefillEmbeddings' device address as the embeddings ext input. Using a new
    // allocation would change the address, causing externalAddrsMatch() to fail,
    // which forces fallback to phaseReplay (broken ext input sync → degenerate output).
    NDArray* decodeEmbedding = prefillEmbeddings;

    // Token sample output: single INT64 scalar
    std::vector<LongType> sampleShape = {1};
    NDArray* sampledToken = NDArrayFactory::create('c', sampleShape, DataType::INT64, context);

    // Logits slice: [vocabSize] — last-position logits from plan output
    // We'll point into the plan's output buffer directly when possible

    int tokensGenerated = 0;

    // ── Get plan's external inputs from the persistent OpaqueContext ──
    // The Java DynamicShapePlanExecutor caches an OpaqueContext with all ext inputs
    // registered via setGraphContextInputArray(). That context persists across calls.
    // We read NDArray* pointers from it via ctx->array(i).
    auto* extCtx = reinterpret_cast<graph::Context*>(config->extInputContext);
    int numExtInputs = config->numPlanExternalInputs;

    // Build ext inputs array from the context's registered inputs
    std::vector<NDArray*> extInputsVec(numExtInputs);
    if (extCtx != nullptr) {
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = extCtx->array(i);
        }
    } else if (config->planExternalInputs != nullptr) {
        // Fallback: use directly passed array (legacy path)
        for (int i = 0; i < numExtInputs; i++) {
            extInputsVec[i] = config->planExternalInputs[i];
        }
    }
    NDArray** extInputs = extInputsVec.data();

    // ── Extract causal mask from ext inputs (if present) ──
    // The causal mask is a plan external input at config->causalMaskExtIdx.
    // It's [1, 1, 1, maskLen] FLOAT, filled with MASK_FILL for masked positions.
    // We need to update it per step: unmask position currentPosition with 0.0f.
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
    // speculative equivalence (divergence compounds through the attention stack).
    // When the two masks share a buffer, the causal-mask maintenance owns every
    // update and the 0/1 update must not run.
    const bool attnMaskAliasesCausal = attentionMask != nullptr && causalMask != nullptr
        && attentionMask->dataBuffer() == causalMask->dataBuffer();

    // ── Extract attn_mask_reformat from ext inputs (if present) ──
    // The attn_mask_reformat override bypasses the model's internal subgraph
    // which produces incorrect masks for padded static-KV decode. We delta-update
    // it each step just like the causal mask.
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

    bool stepTimingEnabled = plan->isExecutionTimingEnabled();

    // ADR 0106 Phase 1: window substrate flag.
    // When activeWindow > 1 and the pre-allocated window tensors are present,
    // we use the fixed [1,1,W_max,past+W_max] mask + [1,W_max] position grid
    // instead of the 1-wide tensors. Addresses are stable — kernels update in-place.
    // When activeWindow == 1, this is false and the existing path runs unchanged.
    //
    // ADR 0106 Phase 2 extension: when speculativeK > 0 AND the two-model path is
    // active (embeddingsExtIdx >= 0) AND window tensors aren't externally provided,
    // we allocate them here internally. The GGUF path (embeddingsExtIdx == -1) cannot
    // use the window substrate without re-freezing with [1,W] input_ids shapes, so it
    // falls back to scalar decode (n-gram table still builds for future use).
    constexpr float WINDOW_MASK_FILL = -3.4028235e+38f;
    NDArray* internalWindowGridMask = nullptr;
    NDArray* internalWindowPositionGrid = nullptr;
    const int specK_pre = config->speculativeK;
    const int wMaxForAlloc = config->windowMax;

    if (specK_pre > 0 && config->speculatorType == 1
            && config->embeddingsExtIdx >= 0   // two-model (ONNX) path only
            && wMaxForAlloc >= specK_pre + 1
            && config->windowGridMask == nullptr) {
        // Allocate internal window tensors for the ONNX speculative path.
        // [1, 1, W_max, past+W_max]: causal window mask per proposal slot.
        // [1, W_max]: position grid for W-wide position_ids.
        // These are updated in-place each step by fillWindowMaskKernel / fillWindowPositionGridKernel.
        LongType kLen = static_cast<LongType>(maxKvLen);  // past+W_max ≈ maxKvLen
        std::vector<LongType> wMaskShape = {1, 1, (LongType)wMaxForAlloc, kLen};
        std::vector<LongType> wPosShape  = {1, (LongType)wMaxForAlloc};
        internalWindowGridMask = NDArrayFactory::create('c', wMaskShape, DataType::FLOAT32, context);
        internalWindowPositionGrid = NDArrayFactory::create('c', wPosShape, DataType::INT64, context);
        float initialWindowMask = WINDOW_MASK_FILL;
        LongType initialWindowPosition = 0;
        internalWindowGridMask->assign(initialWindowMask);
        internalWindowPositionGrid->assign(initialWindowPosition);
        NDArray::prepareSpecialUse({internalWindowGridMask, internalWindowPositionGrid}, {});
        NDArray::registerSpecialUse({internalWindowGridMask, internalWindowPositionGrid}, {});
        config->windowGridMask = internalWindowGridMask;
        config->windowPositionGrid = internalWindowPositionGrid;
    }

    const bool useWindowSubstrate = (config->windowMax > 1
                                     && config->windowGridMask != nullptr
                                     && (config->windowPositionGrid != nullptr
                                         || config->planOwnsKvScatter));

    if (useWindowSubstrate) {
        // Mark window tensors as VARIABLE (device-written in-place each step).
        // Same reasoning as embeddings/mask/posIds above.
        if (config->maskExtIdx >= 0) plan->markExternalInputVariable(config->maskExtIdx);
        if (config->posIdsExtIdx >= 0) plan->markExternalInputVariable(config->posIdsExtIdx);
    }

    // Tier 1c: Pinned memory for D2H token readback — enables true async DMA
    // instead of driver-managed staging through a bounce buffer.
    LongType* pinnedTokenId = nullptr;
    cudaError_t pinErr = cudaMallocHost(&pinnedTokenId, sizeof(LongType));
    if (pinErr != cudaSuccess) {
        // Fallback: use stack variable (unpinned, staging copy)
        pinnedTokenId = nullptr;
    }
    LongType stackTokenId = 0;  // fallback if pinned alloc fails

    // Capture-safe requested-output discriminator. Four raw 64-bit samples per
    // output are copied asynchronously immediately after plan execution and read
    // only after the token path's existing synchronization. No extra sync, tick,
    // host value read, or execution-mode change is introduced.
    constexpr int PLAN_OUTPUT_FP_SAMPLES = 4;
    uint64_t* pinnedPlanOutputSamples = nullptr;
    std::vector<size_t> planOutputBytes(numPlanOutputs, 0);
    std::vector<void*> planOutputDevicePtrs(numPlanOutputs, nullptr);
    LongType planOutputFingerprintInvocation = -1;
    if (DSP_DIAG_ENABLED(KV_CACHE) && numPlanOutputs > 0) {
        static thread_local LongType tlPlanOutputFingerprintInvocation = 0;
        planOutputFingerprintInvocation = ++tlPlanOutputFingerprintInvocation;
        cudaError_t fpPinErr = cudaMallocHost(
            &pinnedPlanOutputSamples,
            static_cast<size_t>(numPlanOutputs) * PLAN_OUTPUT_FP_SAMPLES * sizeof(uint64_t));
        if (fpPinErr != cudaSuccess) pinnedPlanOutputSamples = nullptr;
    }

    // Accepted-prefix state discriminator. Samples are queued after the authoritative
    // state commit and consumed only at an existing token synchronization (or the final
    // synchronization). A per-step pinned ring prevents a later replay from overwriting
    // an earlier asynchronous D2H sample before it is reported.
    constexpr int COMMITTED_STATE_FP_SAMPLES = 4;
    std::vector<int> committedStateExtIndices;
    std::vector<int> committedStateKinds;
    std::vector<int> committedStatePairIndices;
    if (config->gdnStateExtIndices != nullptr) {
        for (int pair = 0; pair < config->numGdnStatePairs; pair++) {
            int extIdx = config->gdnStateExtIndices[pair];
            if (extIdx >= 0 && extIdx < numExtInputs) {
                committedStateExtIndices.push_back(extIdx);
                committedStateKinds.push_back(0);
                committedStatePairIndices.push_back(pair);
            }
        }
    }
    if (config->convStateExtIndices != nullptr) {
        for (int pair = 0; pair < config->numConvStatePairs; pair++) {
            int extIdx = config->convStateExtIndices[pair];
            if (extIdx >= 0 && extIdx < numExtInputs) {
                committedStateExtIndices.push_back(extIdx);
                committedStateKinds.push_back(1);
                committedStatePairIndices.push_back(pair);
            }
        }
    }
    const int committedStateCount = static_cast<int>(committedStateExtIndices.size());
    const size_t committedStateRecordStride =
        static_cast<size_t>(committedStateCount) * COMMITTED_STATE_FP_SAMPLES;
    uint64_t* pinnedCommittedStateSamples = nullptr;
    std::vector<size_t> committedStateBytes(committedStateCount, 0);
    std::vector<void*> committedStateDevicePtrs(committedStateCount, nullptr);
    std::vector<char> committedStateQueued(std::max(0, maxNewTokens), 0);
    std::vector<char> committedStateEmitted(std::max(0, maxNewTokens), 0);
    std::vector<char> committedStateSpeculative(std::max(0, maxNewTokens), 0);
    std::vector<LongType> committedStateNextPosition(std::max(0, maxNewTokens), -1);
    if (DSP_DIAG_ENABLED(KV_CACHE) && committedStateCount > 0 && maxNewTokens > 0) {
        cudaError_t statePinErr = cudaMallocHost(
            &pinnedCommittedStateSamples,
            static_cast<size_t>(maxNewTokens) * committedStateRecordStride * sizeof(uint64_t));
        if (statePinErr != cudaSuccess) pinnedCommittedStateSamples = nullptr;
    }

    // Pre-execution mirror of the committed-state discriminator: samples the same
    // ext state arrays at the same offsets immediately BEFORE each step's main
    // verification pass. Comparing PRE_EXEC_STATE_FP at step N+1 against
    // COMMITTED_STATE_FP at step N detects any mutation of the committed arrays
    // between the commit and the next target execution (e.g. by predictor-plan
    // executions scheduled in between).
    uint64_t* pinnedPreExecStateSamples = nullptr;
    std::vector<char> preExecStateQueued(std::max(0, maxNewTokens), 0);
    std::vector<char> preExecStateEmitted(std::max(0, maxNewTokens), 0);
    std::vector<LongType> preExecStatePosition(std::max(0, maxNewTokens), -1);
    if (pinnedCommittedStateSamples != nullptr) {
        cudaError_t preStatePinErr = cudaMallocHost(
            &pinnedPreExecStateSamples,
            static_cast<size_t>(maxNewTokens) * committedStateRecordStride * sizeof(uint64_t));
        if (preStatePinErr != cudaSuccess) pinnedPreExecStateSamples = nullptr;
    }

    auto queueCommittedStateSamples = [&](int recordStep, LongType nextPosition, bool speculative) {
        if (pinnedCommittedStateSamples == nullptr
                || recordStep < 0 || recordStep >= maxNewTokens) {
            return;
        }
        uint64_t* record = pinnedCommittedStateSamples
            + static_cast<size_t>(recordStep) * committedStateRecordStride;
        std::fill(record, record + committedStateRecordStride, 0ULL);
        for (int stateIdx = 0; stateIdx < committedStateCount; stateIdx++) {
            NDArray* state = extInputs[committedStateExtIndices[stateIdx]];
            auto* db = state != nullptr ? state->dataBuffer() : nullptr;
            if (db == nullptr || !db->isValid() || db->isClosed()
                    || state->specialBuffer() == nullptr) {
                continue;
            }
            const size_t bytes =
                static_cast<size_t>(state->lengthOf()) * state->sizeOfT();
            if (bytes == 0) continue;
            committedStateBytes[stateIdx] = bytes;
            committedStateDevicePtrs[stateIdx] = state->specialBuffer();
            const size_t sampleWidth = std::min(sizeof(uint64_t), bytes);
            const size_t maxOffset = bytes - sampleWidth;
            for (int sample = 0; sample < COMMITTED_STATE_FP_SAMPLES; sample++) {
                const size_t offset =
                    maxOffset * static_cast<size_t>(sample)
                    / static_cast<size_t>(COMMITTED_STATE_FP_SAMPLES - 1);
                cudaMemcpyAsync(
                    record + static_cast<size_t>(stateIdx) * COMMITTED_STATE_FP_SAMPLES + sample,
                    static_cast<const char*>(state->specialBuffer()) + offset,
                    sampleWidth, cudaMemcpyDeviceToHost, *stream);
            }
        }
        committedStateQueued[recordStep] = 1;
        committedStateSpeculative[recordStep] = speculative ? 1 : 0;
        committedStateNextPosition[recordStep] = nextPosition;
    };

    auto emitCommittedStateSamples = [&](int maxReadyStep) {
        if (pinnedCommittedStateSamples == nullptr) return;
        maxReadyStep = std::min(maxReadyStep, maxNewTokens - 1);
        for (int recordStep = 0; recordStep <= maxReadyStep; recordStep++) {
            if (!committedStateQueued[recordStep] || committedStateEmitted[recordStep]) continue;
            const uint64_t* record = pinnedCommittedStateSamples
                + static_cast<size_t>(recordStep) * committedStateRecordStride;
            uint64_t aggregate = 1469598103934665603ULL;
            for (int stateIdx = 0; stateIdx < committedStateCount; stateIdx++) {
                const uint64_t* samples =
                    record + static_cast<size_t>(stateIdx) * COMMITTED_STATE_FP_SAMPLES;
                uint64_t hash = 1469598103934665603ULL;
                for (int sample = 0; sample < COMMITTED_STATE_FP_SAMPLES; sample++) {
                    hash ^= samples[sample];
                    hash *= 1099511628211ULL;
                }
                hash ^= static_cast<uint64_t>(committedStateBytes[stateIdx]);
                hash *= 1099511628211ULL;
                aggregate ^= hash;
                aggregate *= 1099511628211ULL;
                DSP_DIAG(
                    KV_CACHE,
                    "COMMITTED_STATE_FP step=%d nextPos=%lld path=%s kind=%s pair=%d "
                    "ext=%d bytes=%zu device=%p hash=%016llx "
                    "samples=[%016llx,%016llx,%016llx,%016llx]",
                    recordStep,
                    static_cast<long long>(committedStateNextPosition[recordStep]),
                    committedStateSpeculative[recordStep] ? "spec" : "scalar",
                    committedStateKinds[stateIdx] == 0 ? "gdn" : "conv",
                    committedStatePairIndices[stateIdx],
                    committedStateExtIndices[stateIdx],
                    committedStateBytes[stateIdx],
                    committedStateDevicePtrs[stateIdx],
                    static_cast<unsigned long long>(hash),
                    static_cast<unsigned long long>(samples[0]),
                    static_cast<unsigned long long>(samples[1]),
                    static_cast<unsigned long long>(samples[2]),
                    static_cast<unsigned long long>(samples[3]));
            }
            DSP_DIAG(
                KV_CACHE,
                "COMMITTED_STATE_FP_AGG step=%d nextPos=%lld path=%s states=%d hash=%016llx",
                recordStep,
                static_cast<long long>(committedStateNextPosition[recordStep]),
                committedStateSpeculative[recordStep] ? "spec" : "scalar",
                committedStateCount,
                static_cast<unsigned long long>(aggregate));
            committedStateEmitted[recordStep] = 1;
        }
    };

    auto queuePreExecStateSamples = [&](int recordStep, LongType position) {
        if (pinnedPreExecStateSamples == nullptr
                || recordStep < 0 || recordStep >= maxNewTokens) {
            return;
        }
        uint64_t* record = pinnedPreExecStateSamples
            + static_cast<size_t>(recordStep) * committedStateRecordStride;
        std::fill(record, record + committedStateRecordStride, 0ULL);
        for (int stateIdx = 0; stateIdx < committedStateCount; stateIdx++) {
            NDArray* state = extInputs[committedStateExtIndices[stateIdx]];
            auto* db = state != nullptr ? state->dataBuffer() : nullptr;
            if (db == nullptr || !db->isValid() || db->isClosed()
                    || state->specialBuffer() == nullptr) {
                continue;
            }
            const size_t bytes =
                static_cast<size_t>(state->lengthOf()) * state->sizeOfT();
            if (bytes == 0) continue;
            const size_t sampleWidth = std::min(sizeof(uint64_t), bytes);
            const size_t maxOffset = bytes - sampleWidth;
            for (int sample = 0; sample < COMMITTED_STATE_FP_SAMPLES; sample++) {
                const size_t offset =
                    maxOffset * static_cast<size_t>(sample)
                    / static_cast<size_t>(COMMITTED_STATE_FP_SAMPLES - 1);
                cudaMemcpyAsync(
                    record + static_cast<size_t>(stateIdx) * COMMITTED_STATE_FP_SAMPLES + sample,
                    static_cast<const char*>(state->specialBuffer()) + offset,
                    sampleWidth, cudaMemcpyDeviceToHost, *stream);
            }
        }
        preExecStateQueued[recordStep] = 1;
        preExecStatePosition[recordStep] = position;
    };

    auto emitPreExecStateSamples = [&](int maxReadyStep) {
        if (pinnedPreExecStateSamples == nullptr) return;
        maxReadyStep = std::min(maxReadyStep, maxNewTokens - 1);
        for (int recordStep = 0; recordStep <= maxReadyStep; recordStep++) {
            if (!preExecStateQueued[recordStep] || preExecStateEmitted[recordStep]) continue;
            const uint64_t* record = pinnedPreExecStateSamples
                + static_cast<size_t>(recordStep) * committedStateRecordStride;
            for (int stateIdx = 0; stateIdx < committedStateCount; stateIdx++) {
                const uint64_t* samples =
                    record + static_cast<size_t>(stateIdx) * COMMITTED_STATE_FP_SAMPLES;
                DSP_DIAG(
                    KV_CACHE,
                    "PRE_EXEC_STATE_FP step=%d pos=%lld kind=%s pair=%d ext=%d "
                    "samples=[%016llx,%016llx,%016llx,%016llx]",
                    recordStep,
                    static_cast<long long>(preExecStatePosition[recordStep]),
                    committedStateKinds[stateIdx] == 0 ? "gdn" : "conv",
                    committedStatePairIndices[stateIdx],
                    committedStateExtIndices[stateIdx],
                    static_cast<unsigned long long>(samples[0]),
                    static_cast<unsigned long long>(samples[1]),
                    static_cast<unsigned long long>(samples[2]),
                    static_cast<unsigned long long>(samples[3]));
            }
            preExecStateEmitted[recordStep] = 1;
        }
    };

    // Step-input discriminator: dump mask columns and fixed KV-cache rows as the
    // step's main pass saw them. Called from a stream-synchronized point on both
    // the speculative and scalar paths so the two are directly comparable —
    // distinguishes mask asymmetry from KV-row content divergence between W-wide
    // and W=1 writes of the same logical positions.
    auto dumpStepInputSlices = [&](const char* path, int stepIdx, LongType basePos) {
        if (!DSP_DIAG_ENABLED(KV_CACHE)) return;
        auto dumpMaskSlice = [&](NDArray* mask, const char* name, LongType rowOffset) {
            constexpr LongType DUMP_FROM = 14, DUMP_N = 21;
            if (mask == nullptr || mask->specialBuffer() == nullptr
                    || mask->dataType() != DataType::FLOAT32
                    || rowOffset + DUMP_FROM + DUMP_N > mask->lengthOf()) {
                return;
            }
            float vals[DUMP_N] = {};
            cudaMemcpyAsync(vals,
                            static_cast<const char*>(mask->specialBuffer())
                                + (rowOffset + DUMP_FROM) * sizeof(float),
                            DUMP_N * sizeof(float), cudaMemcpyDeviceToHost, *stream);
            cudaStreamSynchronize(*stream);
            char buf[512];
            int off = 0;
            for (LongType i = 0; i < DUMP_N && off < (int)sizeof(buf) - 16; i++) {
                off += snprintf(buf + off, sizeof(buf) - off, "%s%.3g",
                                i ? "," : "", vals[i]);
            }
            DSP_DIAG(KV_CACHE, "MASK_SLICE path=%s step=%d base=%lld %s[%lld..%lld]=[%s]",
                     path, stepIdx, basePos, name, DUMP_FROM, DUMP_FROM + DUMP_N - 1, buf);
        };
        dumpMaskSlice(attentionMask, "attn01", 0);
        if (causalMask != nullptr && causalMask->rankOf() == 4) {
            LongType wRows = causalMask->sizeAt(2);
            LongType mCols = causalMask->sizeAt(3);
            dumpMaskSlice(causalMask, "causal_r0", 0);
            dumpMaskSlice(causalMask, "causal_rLast", (wRows - 1) * mCols);
        } else if (causalMask != nullptr) {
            dumpMaskSlice(causalMask, "causal_flat", 0);
        }
        // Fixed KV rows 18..21 of layer-0 key and value caches (first 2 values each):
        // same logical positions every call, so W-wide vs W=1 writes are comparable.
        auto dumpKvRows = [&](int extIdx, const char* name) {
            if (extIdx < 0 || extIdx >= numExtInputs) return;
            NDArray* cache = extInputs[extIdx];
            if (cache == nullptr || cache->specialBuffer() == nullptr
                    || cache->rankOf() != 4
                    || cache->dataType() != DataType::FLOAT32) {
                return;
            }
            LongType kvLen = cache->sizeAt(1);
            LongType rowStride = cache->sizeAt(2) * cache->sizeAt(3);
            constexpr LongType ROW_FROM = 18, ROW_TO = 21, VALS = 2;
            if (ROW_TO >= kvLen) return;
            float vals[(ROW_TO - ROW_FROM + 1) * VALS] = {};
            for (LongType r = ROW_FROM; r <= ROW_TO; r++) {
                cudaMemcpyAsync(vals + (r - ROW_FROM) * VALS,
                                static_cast<const char*>(cache->specialBuffer())
                                    + r * rowStride * sizeof(float),
                                VALS * sizeof(float), cudaMemcpyDeviceToHost, *stream);
            }
            cudaStreamSynchronize(*stream);
            DSP_DIAG(KV_CACHE,
                     "KV_ROW_SLICE path=%s step=%d base=%lld %s rows18..21=[%.6g,%.6g|%.6g,%.6g|%.6g,%.6g|%.6g,%.6g]",
                     path, stepIdx, basePos, name,
                     vals[0], vals[1], vals[2], vals[3],
                     vals[4], vals[5], vals[6], vals[7]);
        };
        if (config->kvInputExtIndices != nullptr && numKvPairs > 0) {
            dumpKvRows(config->kvInputExtIndices[0], "key0");
            dumpKvRows(config->kvInputExtIndices[numKvPairs], "val0");
            // Per-layer depth bisection: first 2 values of row 19 of every layer's
            // key cache. The first layer whose row-19 write diverges between the
            // speculative and scalar paths is where corruption enters the stack.
            char depthBuf[1024];
            int depthOff = 0;
            for (int ki = 0; ki < numKvPairs && depthOff < (int)sizeof(depthBuf) - 48; ki++) {
                int extIdx = config->kvInputExtIndices[ki];
                float v[2] = {};
                bool ok = false;
                if (extIdx >= 0 && extIdx < numExtInputs) {
                    NDArray* cache = extInputs[extIdx];
                    if (cache != nullptr && cache->specialBuffer() != nullptr
                            && cache->rankOf() == 4 && cache->sizeAt(1) > 19
                            && cache->dataType() == DataType::FLOAT32) {
                        LongType rowStride = cache->sizeAt(2) * cache->sizeAt(3);
                        cudaMemcpyAsync(v,
                                        static_cast<const char*>(cache->specialBuffer())
                                            + 19 * rowStride * sizeof(float),
                                        2 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
                        ok = true;
                    }
                }
                if (ok) cudaStreamSynchronize(*stream);
                depthOff += snprintf(depthBuf + depthOff, sizeof(depthBuf) - depthOff,
                                     "%sL%d:%.5g,%.5g", ki ? " " : "", ki,
                                     ok ? v[0] : 0.0f, ok ? v[1] : 0.0f);
            }
            DSP_DIAG(KV_CACHE, "KV_DEPTH_ROW19 path=%s step=%d base=%lld [%s]",
                     path, stepIdx, basePos, depthBuf);
        }
    };

    // ── ADR 0106 Phase 2: speculative decode state ─────────────────────────
    // N-gram and bundled Qwen3.5 MTP share the W-wide target verifier. MTP has
    // its own scalar plan, context, KV cache, and device-written mutable inputs.
    const int specK = config->speculativeK;
    const bool targetWindowReady = (specK > 0
                                    && useWindowSubstrate
                                    && config->windowMax >= specK + 1);
    const bool useNgram = targetWindowReady && config->speculatorType == 1;
    const bool useMtp = (targetWindowReady
                         && config->speculatorType == 2
                         && config->mtpPlanHandle != nullptr
                         && config->mtpExtInputContext != nullptr);
    const bool useSpeculative = useNgram || useMtp;

    // Host-side n-gram tables learned only from verified output tokens.
    // Order-3 preserves one token of context; order-2 remains the backoff when
    // that context has not been observed yet. Both live only for this call.
    std::unordered_map<LongType, LongType> ngramTable;
    std::unordered_map<LongType, std::unordered_map<LongType, LongType>> trigramTable;
    if (useNgram) {
        ngramTable.reserve(256);
        trigramTable.reserve(256);
    }

    // Last two verified tokens; -1 means the context is not available yet.
    LongType specPreviousToken = -1;
    LongType specCurrentToken = -1;

    // Pinned buffer for target multi-row argmax results (up to specK+1 rows).
    LongType* pinnedArgmax = nullptr;
    LongType stackArgmax[33] = {};  // specK <= 32 is enforced below
    if (useSpeculative) {
        if (specK > 32) {
            DSP_DIAG(KV_CACHE, "SPEC_K_CAP requested=%d effective=32", specK);
        }
        cudaError_t argmaxPinErr = cudaMallocHost(&pinnedArgmax, (specK + 1) * sizeof(LongType));
        if (argmaxPinErr != cudaSuccess) pinnedArgmax = nullptr;
    }

    // Stable device buffers for target argmax rows and scalar MTP drafts.
    NDArray* specArgmaxDevice = nullptr;
    NDArray* mtpDraftDevice = nullptr;
    if (useSpeculative) {
        std::vector<LongType> argmaxShape = {static_cast<LongType>(specK + 1)};
        specArgmaxDevice = NDArrayFactory::create('c', argmaxShape, DataType::INT64, context);
        if (useMtp) {
            mtpDraftDevice = NDArrayFactory::create('c', argmaxShape, DataType::INT64, context);
        }
    }

    // MTP drafts are copied to host only alongside the target acceptance readback.
    LongType* pinnedDraftIds = nullptr;
    LongType stackDraftIds[33] = {};
    if (useMtp) {
        cudaError_t draftPinErr = cudaMallocHost(&pinnedDraftIds, (specK + 1) * sizeof(LongType));
        if (draftPinErr != cudaSuccess) pinnedDraftIds = nullptr;
    }

    // ── Qwen3.5 bundled MTP predictor plan ────────────────────────────────
    graph::NativeDynamicShapePlan* mtpPlan = useMtp ? config->mtpPlanHandle : nullptr;
    auto* mtpContext = useMtp
        ? reinterpret_cast<graph::Context*>(config->mtpExtInputContext) : nullptr;
    std::vector<NDArray*> mtpExtInputsVec;
    std::vector<NDArray*> mtpPlanOutputsVec;
    NDArray** mtpExtInputs = nullptr;
    NDArray** mtpPlanOutputs = nullptr;
    int mtpNumExtInputs = 0;
    int mtpNumOutputs = 0;
    LongType mtpMaskLen = 0;

    if (useMtp) {
        mtpNumExtInputs = config->mtpNumPlanExternalInputs;
        mtpNumOutputs = mtpPlan->getNumRequestedOutputs();
        auto validMtpExtIdx = [&](int idx) {
            return idx >= 0 && idx < mtpNumExtInputs;
        };
        REQUIRE_TRUE(mtpContext != nullptr && mtpNumExtInputs > 0 && mtpNumOutputs > 0, 0,
                     "autoregressive_decode: invalid CUDA MTP plan/context inputs=%d outputs=%d",
                     mtpNumExtInputs, mtpNumOutputs);
        REQUIRE_TRUE(validMtpExtIdx(config->mtpInputIdsExtIdx)
                         && validMtpExtIdx(config->mtpTargetHiddenExtIdx)
                         && validMtpExtIdx(config->mtpCausalMaskExtIdx)
                         && validMtpExtIdx(config->mtpPositionOffsetExtIdx)
                         && validMtpExtIdx(config->mtpCachePositionExtIdx)
                         && validMtpExtIdx(config->mtpKvInputExtIndices[0])
                         && validMtpExtIdx(config->mtpKvInputExtIndices[1]),
                     0, "autoregressive_decode: CUDA MTP external-input index is out of range");
        REQUIRE_TRUE(config->mtpInputIds != nullptr
                         && config->mtpTargetHidden != nullptr
                         && config->mtpCausalMask != nullptr
                         && config->mtpPositionOffset != nullptr
                         && config->mtpCachePosition != nullptr
                         && config->mtpKvBuffers[0] != nullptr
                         && config->mtpKvBuffers[1] != nullptr,
                     0, "autoregressive_decode: CUDA MTP retained input is null");

        mtpExtInputsVec.resize(mtpNumExtInputs);
        for (int i = 0; i < mtpNumExtInputs; i++) {
            mtpExtInputsVec[i] = mtpContext->array(i);
        }
        mtpExtInputsVec[config->mtpInputIdsExtIdx] = config->mtpInputIds;
        mtpExtInputsVec[config->mtpTargetHiddenExtIdx] = config->mtpTargetHidden;
        mtpExtInputsVec[config->mtpCausalMaskExtIdx] = config->mtpCausalMask;
        mtpExtInputsVec[config->mtpPositionOffsetExtIdx] = config->mtpPositionOffset;
        mtpExtInputsVec[config->mtpCachePositionExtIdx] = config->mtpCachePosition;
        mtpExtInputsVec[config->mtpKvInputExtIndices[0]] = config->mtpKvBuffers[0];
        mtpExtInputsVec[config->mtpKvInputExtIndices[1]] = config->mtpKvBuffers[1];
        mtpExtInputs = mtpExtInputsVec.data();

        mtpPlanOutputsVec.resize(mtpNumOutputs, nullptr);
        mtpPlanOutputs = mtpPlanOutputsVec.data();
        mtpMaskLen = config->mtpCausalMask->sizeAt(-1);

        // Every scalar/carry/mask input is written on this CUDA stream. VARIABLE
        // gives the predictor plan stable staging and D2D refresh semantics without
        // forcing stale host data back over those device-authoritative values.
        mtpPlan->markExternalInputVariable(config->mtpInputIdsExtIdx);
        mtpPlan->markExternalInputVariable(config->mtpTargetHiddenExtIdx);
        mtpPlan->markExternalInputVariable(config->mtpCausalMaskExtIdx);
        mtpPlan->markExternalInputVariable(config->mtpPositionOffsetExtIdx);
        mtpPlan->markExternalInputVariable(config->mtpCachePositionExtIdx);
        for (int kv = 0; kv < 2; kv++) {
            int kvIdx = config->mtpKvInputExtIndices[kv];
            mtpPlan->markExternalInputVariable(kvIdx);
            mtpPlan->registerDeviceManagedExternalInput(config->mtpKvBuffers[kv]);
        }
    }

    // KV_CACHE-gated chain probe: per chain exec, sample the carry-in hidden,
    // input token, and hidden-out (async D2H on the exec stream, drained by the
    // acceptance path's existing sync — no new sync points). Diagnoses whether
    // the draft chain's hidden/token carry is visible to the predictor plan.
    float mtpChainCarryIn[33][4] = {};
    float mtpChainHidOut[33][4] = {};
    LongType mtpChainTok[33] = {};
    int mtpChainSampled = 0;

    // Adaptive MTP chain-depth cap. Recursive drafting feeds the predictor its
    // OWN output hidden — out-of-distribution for heads trained only on trunk
    // hidden (measured: Qwen3.5-0.8B bundled head hits 41% at position 0 and
    // 0/51 at position 1 even when position 0's token was correct). Positions
    // that never accept still cost one full predictor execution per step and
    // widen the verification window, so once a position has enough evaluations
    // with zero accepts, stop proposing past it. Counters persist across the
    // whole generation (this function IS the decode loop).
    int mtpChainCap = specK;
    int mtpPosEvaluated[33] = {};
    int mtpPosAccepted[33] = {};
    constexpr int MTP_CHAIN_CAP_MIN_EVALS = 12;

    auto executeMtpCuda = [&](LongType position, int draftSlot, bool writeTargetRow) {
        REQUIRE_TRUE(useMtp && mtpDraftDevice != nullptr, 0,
                     "autoregressive_decode: attempted CUDA MTP execution while disabled");
        REQUIRE_TRUE(draftSlot >= 0 && draftSlot <= specK, 0,
                     "autoregressive_decode: CUDA MTP draft slot %d outside [0,%d]",
                     draftSlot, specK);

        const bool chainProbe = DSP_DIAG_ENABLED(KV_CACHE) && draftSlot < 33
            && config->mtpTargetHidden->dataType() == DataType::FLOAT32
            && config->mtpTargetHidden->lengthOf() >= 4;
        if (chainProbe) {
            cudaMemcpyAsync(mtpChainCarryIn[draftSlot],
                            config->mtpTargetHidden->specialBuffer(),
                            4 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
            cudaMemcpyAsync(&mtpChainTok[draftSlot],
                            config->mtpInputIds->specialBuffer(),
                            sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
            if (draftSlot + 1 > mtpChainSampled) mtpChainSampled = draftSlot + 1;
        }

        NDArray::prepareSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition, config->mtpCausalMask}, {});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpPositionOffset->specialBuffer(), position);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpCachePosition->specialBuffer(), position);
        BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskLauncher,
                              (stream, config->mtpCausalMask->specialBuffer(),
                               position, mtpMaskLen),
                              SD_FLOAT_TYPES);
        NDArray::registerSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition, config->mtpCausalMask}, {});

        Status mtpStatus = mtpPlan->executeSteadyState(
            mtpExtInputs, mtpNumExtInputs,
            mtpPlanOutputs, mtpNumOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
        REQUIRE_TRUE(mtpStatus == Status::OK, 0,
                     "autoregressive_decode: CUDA MTP plan failed at position %lld with status %d",
                     (long long)position, static_cast<int>(mtpStatus));
        REQUIRE_TRUE(config->mtpLogitsOutputIdx >= 0
                         && config->mtpLogitsOutputIdx < mtpNumOutputs
                         && mtpPlanOutputs[config->mtpLogitsOutputIdx] != nullptr,
                     0, "autoregressive_decode: CUDA MTP logits output is unavailable");
        REQUIRE_TRUE(config->mtpHiddenOutputIdx >= 0
                         && config->mtpHiddenOutputIdx < mtpNumOutputs
                         && mtpPlanOutputs[config->mtpHiddenOutputIdx] != nullptr,
                     0, "autoregressive_decode: CUDA MTP hidden output is unavailable");

        NDArray* mtpLogits = mtpPlanOutputs[config->mtpLogitsOutputIdx];
        NDArray* mtpHidden = mtpPlanOutputs[config->mtpHiddenOutputIdx];
        REQUIRE_TRUE(mtpLogits->rankOf() >= 2 && mtpLogits->rankOf() <= 3, 0,
                     "autoregressive_decode: CUDA MTP logits rank %lld is invalid",
                     (long long)mtpLogits->rankOf());
        LongType mtpVocab = mtpLogits->sizeAt(mtpLogits->rankOf() - 1);
        void* draftPtr = static_cast<char*>(mtpDraftDevice->specialBuffer())
                         + static_cast<size_t>(draftSlot) * sizeof(LongType);
        NDArray::prepareSpecialUse({mtpDraftDevice}, {mtpLogits});
        BUILD_SINGLE_SELECTOR(mtpLogits->dataType(), argmaxLauncher,
                              (stream, mtpLogits->specialBuffer(), draftPtr, mtpVocab),
                              SD_FLOAT_TYPES);
        NDArray::registerSpecialUse({mtpDraftDevice}, {mtpLogits});

        REQUIRE_TRUE(mtpHidden->lengthOf() == config->mtpTargetHidden->lengthOf()
                         && mtpHidden->dataType() == config->mtpTargetHidden->dataType(),
                     0, "autoregressive_decode: CUDA MTP hidden carry shape/type mismatch");
        size_t hiddenBytes = static_cast<size_t>(mtpHidden->lengthOf()) * mtpHidden->sizeOfT();
        NDArray::prepareSpecialUse({config->mtpTargetHidden}, {mtpHidden});
        cudaMemcpyAsync(config->mtpTargetHidden->specialBuffer(), mtpHidden->specialBuffer(),
                        hiddenBytes, cudaMemcpyDeviceToDevice, *stream);
        NDArray::registerSpecialUse({config->mtpTargetHidden}, {mtpHidden});

        if (chainProbe && mtpHidden->dataType() == DataType::FLOAT32) {
            cudaMemcpyAsync(mtpChainHidOut[draftSlot], mtpHidden->specialBuffer(),
                            4 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
        }

        NDArray::prepareSpecialUse({config->mtpInputIds}, {mtpDraftDevice});
        cudaMemcpyAsync(config->mtpInputIds->specialBuffer(), draftPtr,
                        sizeof(LongType), cudaMemcpyDeviceToDevice, *stream);
        NDArray::registerSpecialUse({config->mtpInputIds}, {mtpDraftDevice});

        if (writeTargetRow) {
            REQUIRE_TRUE(config->planOwnsKvScatter
                             && inputIds->dataType() == DataType::INT64
                             && inputIds->lengthOf() > draftSlot + 1,
                         0, "autoregressive_decode: target input cannot receive CUDA MTP draft %d",
                         draftSlot);
            void* targetTokenPtr = static_cast<LongType*>(inputIds->specialBuffer())
                                   + draftSlot + 1;
            NDArray::prepareSpecialUse({inputIds}, {mtpDraftDevice});
            cudaMemcpyAsync(targetTokenPtr, draftPtr, sizeof(LongType),
                            cudaMemcpyDeviceToDevice, *stream);
            NDArray::registerSpecialUse({inputIds}, {mtpDraftDevice});
        }
    };

    auto setMtpTargetCarryCuda = [&](NDArray* targetHiddenRows, int row) {
        REQUIRE_TRUE(useMtp && targetHiddenRows != nullptr
                         && targetHiddenRows->rankOf() == 3,
                     0, "autoregressive_decode: target hidden output must be rank 3 for CUDA MTP");
        REQUIRE_TRUE(row >= 0 && row < targetHiddenRows->sizeAt(1)
                         && targetHiddenRows->strideAt(2) == 1
                         && config->mtpTargetHidden->lengthOf() == targetHiddenRows->sizeAt(2)
                         && config->mtpTargetHidden->dataType() == targetHiddenRows->dataType(),
                     0, "autoregressive_decode: CUDA MTP target carry row/shape/type mismatch");
        size_t rowBytes = static_cast<size_t>(targetHiddenRows->sizeAt(2))
                          * targetHiddenRows->sizeOfT();
        const void* source = static_cast<const char*>(targetHiddenRows->specialBuffer())
                             + static_cast<size_t>(row)
                                   * targetHiddenRows->strideAt(1)
                                   * targetHiddenRows->sizeOfT();
        NDArray::prepareSpecialUse({config->mtpTargetHidden}, {targetHiddenRows});
        cudaMemcpyAsync(config->mtpTargetHidden->specialBuffer(), source,
                        rowBytes, cudaMemcpyDeviceToDevice, *stream);
        NDArray::registerSpecialUse({config->mtpTargetHidden}, {targetHiddenRows});
    };

    auto setMtpNextInputCuda = [&](NDArray* tokenSource,
                                   LongType tokenIndex,
                                   LongType nextPosition) {
        REQUIRE_TRUE(useMtp && tokenSource != nullptr
                         && tokenSource->dataType() == DataType::INT64
                         && tokenIndex >= 0 && tokenIndex < tokenSource->lengthOf(),
                     0, "autoregressive_decode: invalid CUDA MTP next-token source");
        const void* tokenPtr = static_cast<const LongType*>(tokenSource->specialBuffer())
                               + tokenIndex;
        NDArray::prepareSpecialUse(
            {config->mtpInputIds, config->mtpPositionOffset, config->mtpCachePosition},
            {tokenSource});
        cudaMemcpyAsync(config->mtpInputIds->specialBuffer(), tokenPtr,
                        sizeof(LongType), cudaMemcpyDeviceToDevice, *stream);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpPositionOffset->specialBuffer(), nextPosition);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpCachePosition->specialBuffer(), nextPosition);
        NDArray::registerSpecialUse(
            {config->mtpInputIds, config->mtpPositionOffset, config->mtpCachePosition},
            {tokenSource});
    };

    // ── Mark decode-loop-modified ext inputs as VARIABLE ────────────────────
    DSP_DIAG(KV_CACHE,
             "AUTOREGRESSIVE_DECODE_CUDA markExternalInputVariable plan=%p numExternalInputs=%d",
             plan, numExtInputs);
    // The native decode loop writes fresh data to these ext inputs every step
    // (embed lookup, mask update, position update, input_ids update). The plan's
    // default classification marks them as non-variable (SOURCE_VARIABLE = model
    // weight), which means:
    //   1. No staging buffers allocated for them
    //   2. ensureAndSyncStagingBuffers skips D2D refresh
    //   3. Merged CUDA graphs that captured gap ops reading from the Java-side
    //      warmup addresses will read stale data if the OpaqueContext provides
    //      different NDArray pointers.
    //
    // markExternalInputVariable fixes this by:
    //   - Allocating plan-owned staging buffers for these inputs
    //   - D2D-refreshing them each step in ensureAndSyncStagingBuffers
    //   - Invalidating arg tables so they point to the stable staging addresses
    //
    // This MUST happen before the decode loop so the first execution allocates
    // staging buffers and subsequent executions refresh them.
    // These ext inputs are DEVICE-written IN-PLACE by THIS op's own kernels every
    // step (embedLookupKernel, updateAttentionMaskKernel, updatePositionIdsKernel,
    // updateInputIdsKernel, updateCausalMaskLauncher) and committed device-authoritative
    // via registerSpecialUse({...}). They are NOT host-fed — Java never writes them
    // per step in the native decode loop. They must therefore be VARIABLE (protected +
    // address-stable + staging D2D-refreshed each step), exactly like the GDN/conv/KV
    // inputs below — NOT PLACEHOLDER.
    //
    // PLACEHOLDER means "host-written → force H2D" (externalInputIsPlaceholder_ ==
    // force-H2D, NDArray.h). On replay, performPreReplaySync would H2D-copy the STALE
    // host buffer over the fresh device value the kernel just wrote, the captured graph
    // would then recompute the PREVIOUS step's forward pass, and the decode sticks on a
    // single token (java/native match steps 0-4 then native repeats the step-4 token).
    // Placeholder also leaves them unprotected (isProtectedExternalInput == !placeholder)
    // so the captured graph can bake a stale Java-warmup address.
    if (config->embeddingsExtIdx >= 0) plan->markExternalInputVariable(config->embeddingsExtIdx);
    if (config->maskExtIdx >= 0) plan->markExternalInputVariable(config->maskExtIdx);
    if (config->posIdsExtIdx >= 0) plan->markExternalInputVariable(config->posIdsExtIdx);
    if (config->inputIdsExtIdx >= 0) plan->markExternalInputVariable(config->inputIdsExtIdx);
    if (config->causalMaskExtIdx >= 0) plan->markExternalInputVariable(config->causalMaskExtIdx);
    if (config->attnMaskReformatExtIdx >= 0) plan->markExternalInputVariable(config->attnMaskReformatExtIdx);
    if (config->positionOffsetExtIdx >= 0) plan->markExternalInputVariable(config->positionOffsetExtIdx);
    if (config->cachePositionExtIdx >= 0) plan->markExternalInputVariable(config->cachePositionExtIdx);
    if (config->actualSequenceLengthExtIdx >= 0) {
        plan->markExternalInputVariable(config->actualSequenceLengthExtIdx);
    }
    // GDN/conv state: device-written via D2D copy on DSP stream each step.
    // Mark as variable (participates in dependency tracking) but NOT placeholder
    // (must NOT H2D — device buffer is authoritative, host buffer is stale).
    if (config->numGdnStatePairs > 0 && config->gdnStateExtIndices != nullptr) {
        for (int s = 0; s < config->numGdnStatePairs; s++) {
            int extIdx = config->gdnStateExtIndices[s];
            if (extIdx >= 0) {
                plan->markExternalInputVariable(extIdx);
                if (extIdx < numExtInputs) plan->registerDeviceManagedExternalInput(extInputs[extIdx]);
            }
        }
    }
    if (config->numConvStatePairs > 0 && config->convStateExtIndices != nullptr) {
        for (int s = 0; s < config->numConvStatePairs; s++) {
            int extIdx = config->convStateExtIndices[s];
            if (extIdx >= 0) {
                plan->markExternalInputVariable(extIdx);
                if (extIdx < numExtInputs) plan->registerDeviceManagedExternalInput(extInputs[extIdx]);
            }
        }
    }
    // KV cache: device-written by attention kernels in-place each step.
    // Keep the caller-owned device buffer authoritative. Generic staging is
    // input-only: redirecting an in-place cache through it would strand the
    // mutation in the plan-owned copy and reuse would observe a stale cache.
    if (config->kvInputExtIndices != nullptr) {
        for (int kv = 0; kv < 2 * numKvPairs; kv++) {
            int kvIdx = config->kvInputExtIndices[kv];
            if (kvIdx >= 0) {
                plan->markExternalInputVariable(kvIdx);
                if (kvIdx < numExtInputs) {
                    plan->registerDeviceManagedExternalInput(extInputs[kvIdx]);
                }
            }
        }
    }

    LongType totalSpeculativeProposed = 0;
    LongType totalSpeculativeAccepted = 0;
    LongType speculativeStepCount = 0;

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
        // decodeEmbedding IS prefillEmbeddings (same NDArray, same device address).
        // The embed lookup kernel writes into it in-place each step, keeping the
        // device address stable for CUDA graph replay (externalAddrsMatch).
        if (config->embeddingsExtIdx >= 0 && config->embeddingsExtIdx < numExtInputs) {
            extInputs[config->embeddingsExtIdx] = decodeEmbedding;
        }

        // ── ADR 0106 Phase 2: build proposals for this step ─────────────────
        int proposedCount = 0;
        int order3Hits = 0;
        int order2Hits = 0;
        LongType draftIds[33] = {};

        int maxPropose = (specK < 32) ? specK : 32;
        int remainingOutput = maxNewTokens - tokensGenerated;
        int outputDraftCapacity = remainingOutput - 1;
        if (outputDraftCapacity < maxPropose) maxPropose = outputDraftCapacity;
        LongType remainingKv = maxKvLen - currentPosition;
        LongType kvDraftCapacity = remainingKv - 1;
        if (kvDraftCapacity < static_cast<LongType>(maxPropose)) {
            maxPropose = kvDraftCapacity > 0 ? static_cast<int>(kvDraftCapacity) : 0;
        }
        if (maxPropose < 0) maxPropose = 0;
        if (useMtp && maxPropose > mtpChainCap) maxPropose = mtpChainCap;

        if (useNgram && specCurrentToken >= 0) {
            LongType previous = specPreviousToken;
            LongType current = specCurrentToken;
            for (int p = 0; p < maxPropose; p++) {
                LongType next = -1;
                bool found = false;
                if (previous >= 0) {
                    auto outer = trigramTable.find(previous);
                    if (outer != trigramTable.end()) {
                        auto inner = outer->second.find(current);
                        if (inner != outer->second.end()) {
                            next = inner->second;
                            found = true;
                            order3Hits++;
                        }
                    }
                }
                if (!found) {
                    auto backoff = ngramTable.find(current);
                    if (backoff != ngramTable.end()) {
                        next = backoff->second;
                        found = true;
                        order2Hits++;
                    }
                }
                if (!found) break;
                draftIds[p] = next;
                proposedCount++;
                previous = current;
                current = next;
            }
            DSP_DIAG(KV_CACHE,
                     "NGRAM_PROPOSE step=%d previous=%lld current=%lld proposed=%d order3=%d order2=%d",
                     step, (long long)specPreviousToken, (long long)specCurrentToken,
                     proposedCount, order3Hits, order2Hits);
        } else if (useMtp) {
            if (maxPropose == 0) {
                // Consume the base token so predictor KV stays aligned even when
                // the output/KV envelope has room only for the target token.
                executeMtpCuda(currentPosition, 0, false);
            } else {
                for (int p = 0; p < maxPropose; p++) {
                    // Each call consumes the current predictor input, emits one
                    // device-resident draft, chains predictor hidden/token state,
                    // and writes that draft directly into target input row p+1.
                    executeMtpCuda(currentPosition + p, p, true);
                    proposedCount++;
                }
            }
            DSP_DIAG(KV_CACHE,
                     "MTP_PROPOSE_QUEUED step=%d basePos=%lld proposed=%d",
                     step, (long long)currentPosition, proposedCount);
        }

        if (proposedCount > 0) {
            config->activeWindow = 1 + proposedCount;
        }

        // N-gram drafts originate on the host. MTP drafts already occupy the
        // target's W-wide input rows through stream-ordered D2D copies above.
        if (useNgram && proposedCount > 0 && config->planOwnsKvScatter
                && inputIds->lengthOf() >= proposedCount + 1) {
            NDArray::prepareSpecialUse({inputIds}, {});
            cudaMemcpyAsync(static_cast<LongType*>(inputIds->specialBuffer()) + 1,
                            draftIds, proposedCount * sizeof(LongType),
                            cudaMemcpyHostToDevice, *stream);
            NDArray::registerSpecialUse({inputIds}, {});
        }

        // ADR 0106 Phase 1: window substrate mask + position grid.
        // When W>1, fill the fixed window tensors in-place via GPU kernels and wire
        // them into the ext inputs in place of the 1-wide attention mask and position IDs.
        // Device addresses stay stable (pointer-stability, ADR 0105).
        if (useWindowSubstrate) {
            NDArray* wMask = config->windowGridMask;
            NDArray* wPos  = config->windowPositionGrid;
            LongType wMax  = static_cast<LongType>(config->windowMax);
            LongType aW    = static_cast<LongType>(config->activeWindow);
            LongType rowLen = wMask->sizeAt(3);  // past_len + wMax

            // Fill window mask on GPU: one thread per element
            if (wPos != nullptr) NDArray::prepareSpecialUse({wMask, wPos}, {});
            else NDArray::prepareSpecialUse({wMask}, {});
            LongType totalElems = wMax * rowLen;
            int threads = 256;
            int blocks = static_cast<int>((totalElems + threads - 1) / threads);
            fillWindowMaskKernel<<<blocks, threads, 0, *stream>>>(
                wMask->specialBuffer(), wMax, rowLen, currentPosition, aW, WINDOW_MASK_FILL);

            if (wPos != nullptr) {
                fillWindowPositionGridKernel<<<1, static_cast<int>(wMax), 0, *stream>>>(
                    wPos->specialBuffer(), wMax, currentPosition, aW);
                NDArray::registerSpecialUse({wMask, wPos}, {});
            } else {
                NDArray::registerSpecialUse({wMask}, {});
            }

            if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
                extInputs[config->maskExtIdx] = wMask;
            }
            if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
                extInputs[config->posIdsExtIdx] = wPos;
            }
        } else {
            // W=1 path: existing 1-wide tensors (bit-identical to pre-ADR behaviour)
            if (config->maskExtIdx >= 0 && config->maskExtIdx < numExtInputs) {
                extInputs[config->maskExtIdx] = attentionMask;
            }
            if (config->posIdsExtIdx >= 0 && config->posIdsExtIdx < numExtInputs) {
                extInputs[config->posIdsExtIdx] = positionIds;
            }
        }

        // Input IDs
        if (config->inputIdsExtIdx >= 0 && config->inputIdsExtIdx < numExtInputs) {
            extInputs[config->inputIdsExtIdx] = inputIds;
        }

        // Causal mask: wire into ext inputs (same pointer every step, updated in-place)
        if (causalMask != nullptr && config->causalMaskExtIdx >= 0 && config->causalMaskExtIdx < numExtInputs) {
            extInputs[config->causalMaskExtIdx] = causalMask;
        }

        // KV cache inputs: point to static buffers
        if (config->kvInputExtIndices != nullptr && staticKvBuffers != nullptr) {
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvIdx = config->kvInputExtIndices[kv];
                if (kvIdx >= 0 && kvIdx < numExtInputs) {
                    extInputs[kvIdx] = staticKvBuffers[kv];
                }
            }
        }


        // Recurrent GDN/causal-conv kernels must process exactly the live target
        // verification width. The Java warmup initializes this scalar to 1; speculative
        // proposal construction above may expand activeWindow for this replay.
        if (config->actualSequenceLengthExtIdx >= 0
                && config->actualSequenceLengthExtIdx < numExtInputs) {
            NDArray* actualSeqLen = extInputs[config->actualSequenceLengthExtIdx];
            if (actualSeqLen != nullptr) {
                NDArray::prepareSpecialUse({actualSeqLen}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                    actualSeqLen->specialBuffer(), static_cast<LongType>(config->activeWindow));
                NDArray::registerSpecialUse({actualSeqLen}, {});
            }
        }

        // ── Step 1b: Pre-unmask the CURRENT position in causal mask ──
        // GGUF only (planOwnsKvScatter == true): the dotProductAttentionV2 op writes
        // KV at cache_position = currentPosition in-place, then attends to the full
        // buffer including that position. Pre-unmasking currentPosition is required so
        // the token can attend to its own newly-written KV entry.
        //
        // ONNX/external-scatter path (planOwnsKvScatter == false): KV scatter happens
        // AFTER execution via kvScatterBatched. Position currentPosition in the static
        // KV buffer is EMPTY during plan execution — attending to it reads zeros, giving
        // wrong logits. The post-execution mask update unmasks kvJustWritten (the PREVIOUS
        // position that was just written) for the NEXT step; the current query position
        // is always exposed via mask[totalSeqLen-1] (padded layout set by Java warmup).
        // Step 1b pre-unmask of currentPosition — GATED on planOwnsKvScatter (verified correct
        // by experiment: removing the gate gives step-7 native=87 vs java=2008). For the
        // external-scatter path (planOwnsKvScatter==false, ONNX/SmolDocling) the current token's
        // K/V is NOT in the cache at currentPosition during plan execution (scatter is post-exec)
        // — it is provided at the PADDED query slot (mask[totalSeqLen-1]). Pre-unmasking
        // currentPosition would attend an EMPTY cache slot → wrong logits. GGUF in-graph scatter
        // DOES have the current K/V at currentPosition, so it pre-unmasks.
        if (config->planOwnsKvScatter) {
            if (causalMask != nullptr && currentPosition >= 0 && currentPosition < causalMaskLen) {
                if (causalMask->rankOf() == 4 && causalMask->sizeAt(2) > 1) {
                    // W-wide window mask: the per-row causal band moves with
                    // currentPosition every step — a single-column unmask only
                    // ever advances row 0 (flat index < maxKvLen). Refill all rows.
                    NDArray::prepareSpecialUse({causalMask}, {});
                    BUILD_SINGLE_SELECTOR(causalMask->dataType(), refillWindowCausalMaskLauncher,
                                          (stream, causalMask->specialBuffer(),
                                           causalMask->sizeAt(2), causalMask->sizeAt(3), currentPosition),
                                          SD_FLOAT_TYPES);
                    NDArray::registerSpecialUse({causalMask}, {});
                } else {
                    NDArray::prepareSpecialUse({causalMask}, {});
                    BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskLauncher,
                                          (stream, causalMask->specialBuffer(), currentPosition, causalMaskLen),
                                          SD_FLOAT_TYPES);
                    NDArray::registerSpecialUse({causalMask}, {});
                }
            }
            if (attnMaskReformat != nullptr && currentPosition >= 0 && currentPosition < attnMaskReformatLen) {
                NDArray::prepareSpecialUse({attnMaskReformat}, {});
                BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskLauncher,
                                      (stream, attnMaskReformat->specialBuffer(), currentPosition, attnMaskReformatLen),
                                      SD_FLOAT_TYPES);
                NDArray::registerSpecialUse({attnMaskReformat}, {});
            }
            // Also unmask the attention mask (0/1 mask) for GGUF in-graph KV.
            // Skipped when it aliases the additive causal mask (see attnMaskAliasesCausal).
            if (!attnMaskAliasesCausal && currentPosition >= 0 && currentPosition < maxKvLen) {
                NDArray::prepareSpecialUse({attentionMask}, {});
                BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskLauncher,
                                      (stream, attentionMask->specialBuffer(), currentPosition, maxKvLen),
                                      SD_COMMON_TYPES);
                NDArray::registerSpecialUse({attentionMask}, {});
            }
        }

        auto tWireEnd = stepTimingEnabled ? std::chrono::high_resolution_clock::now() : stepStart;

        // ── Step 2: Execute plan ──
        // Use executeSteadyState() for the hot decode path. For step >= 4 in
        // REPLAYING phase, this eliminates ~200ms/step of CPU overhead (slot
        // scans, lifecycle checks, shape validation). For earlier steps or
        // pre-REPLAYING phase, it automatically falls back to full execute().
        //
        // ALL decode-loop-written ext inputs (embeddings, attn/causal/reformat masks,
        // position_ids, input_ids, position_offset, GDN/conv state, KV cache) are marked
        // VARIABLE — device-authoritative, never placeholder. This op's kernels write
        // them in-place and registerSpecialUse leaves the device buffer authoritative,
        // so performPreReplaySync respects actuality (isPrimaryActual) and skips H2D — a
        // forced H2D (placeholder behavior) would clobber the fresh device value with
        // stale host data. Staging D2D refreshes each into the captured graph every step.

        // ADR 0107 V2: inject scale buffers into the thread-local registry so that
        // dot_product_attention_v2 can look them up by INT8 KV cache pointer identity.
        // The registry is set per-step (before executeSteadyState) and cleared after.
        // extInputs[kvInputExtIndices[0..N-1]] are the INT8 key cache NDArrays (at original
        // variable name indices). Scale arrays are parallel (indexed [0..N-1]=key, [N..2N-1]=val).
        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr
            && config->kvInputExtIndices != nullptr && config->numGdnStatePairs >= 0) {
            // Gather the N INT8 key cache NDArray pointers from extInputs
            static thread_local std::vector<NDArray*> tl_kvQuantPtrs;
            int N = numKvPairs;
            tl_kvQuantPtrs.resize(N);
            for (int ki = 0; ki < N; ki++) {
                int extIdx = config->kvInputExtIndices[ki];  // first N = key caches
                tl_kvQuantPtrs[ki] = (extIdx >= 0 && extIdx < numExtInputs)
                    ? extInputs[extIdx] : nullptr;
            }
            setKvScaleRegistry(tl_kvQuantPtrs.data(), config->kvScaleBuffers, N);
        }

        queuePreExecStateSamples(step, currentPosition);
        Status planStatus = plan->executeSteadyState(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));

        // Clear the scale registry immediately after plan execution (no stale refs).
        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
            clearKvScaleRegistry();
        }

        // Validate plan output every step — these are O(1) pointer/flag checks,
        // negligible cost compared to the plan execution itself.
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
            REQUIRE_TRUE(logitsArr->specialBuffer() != nullptr, 0,
                         "autoregressive_decode: logits specialBuffer (device ptr) is null at step %d. "
                         "Buffer exists but has no device allocation — missing syncToDevice or stale buffer.",
                         step);
        }

        if (pinnedPlanOutputSamples != nullptr && step < 2) {
            const size_t sampleCount =
                static_cast<size_t>(numPlanOutputs) * PLAN_OUTPUT_FP_SAMPLES;
            std::fill(pinnedPlanOutputSamples, pinnedPlanOutputSamples + sampleCount, 0ULL);
            std::fill(planOutputBytes.begin(), planOutputBytes.end(), 0);
            std::fill(planOutputDevicePtrs.begin(), planOutputDevicePtrs.end(), nullptr);

            for (int outputIdx = 0; outputIdx < numPlanOutputs; outputIdx++) {
                NDArray* output = planOutputs[outputIdx];
                auto* db = output != nullptr ? output->dataBuffer() : nullptr;
                if (db == nullptr || !db->isValid() || db->isClosed() || db->special() == nullptr) {
                    continue;
                }

                const size_t bytes = db->getLenInBytes();
                if (bytes == 0) continue;
                planOutputBytes[outputIdx] = bytes;
                planOutputDevicePtrs[outputIdx] = db->special();

                const size_t sampleWidth = std::min(sizeof(uint64_t), bytes);
                const size_t maxOffset = bytes - sampleWidth;
                for (int sample = 0; sample < PLAN_OUTPUT_FP_SAMPLES; sample++) {
                    const size_t offset =
                        maxOffset * static_cast<size_t>(sample) /
                        static_cast<size_t>(PLAN_OUTPUT_FP_SAMPLES - 1);
                    cudaMemcpyAsync(
                        &pinnedPlanOutputSamples[
                            static_cast<size_t>(outputIdx) * PLAN_OUTPUT_FP_SAMPLES + sample],
                        static_cast<const char*>(db->special()) + offset,
                        sampleWidth, cudaMemcpyDeviceToHost, *stream);
                }
            }
        }

        auto emitPlanOutputFingerprints = [&]() {
            if (pinnedPlanOutputSamples == nullptr || step >= 2) return;
            for (int outputIdx = 0; outputIdx < numPlanOutputs; outputIdx++) {
                const uint64_t* samples =
                    pinnedPlanOutputSamples +
                    static_cast<size_t>(outputIdx) * PLAN_OUTPUT_FP_SAMPLES;
                uint64_t hash = 1469598103934665603ULL;
                for (int sample = 0; sample < PLAN_OUTPUT_FP_SAMPLES; sample++) {
                    hash ^= samples[sample];
                    hash *= 1099511628211ULL;
                }
                hash ^= static_cast<uint64_t>(planOutputBytes[outputIdx]);
                hash *= 1099511628211ULL;
                NDArray* output = planOutputs[outputIdx];
                DSP_DIAG(
                    KV_CACHE,
                    "PLAN_OUTPUT_FP invocation=%lld step=%d idx=%d role=%s dtype=%d "
                    "length=%lld bytes=%zu device=%p hash=%016llx "
                    "samples=[%016llx,%016llx,%016llx,%016llx]",
                    static_cast<long long>(planOutputFingerprintInvocation), step, outputIdx,
                    outputIdx == config->logitsOutputIdx ? "logits" : "state",
                    output != nullptr ? static_cast<int>(output->dataType()) : -1,
                    output != nullptr ? static_cast<long long>(output->lengthOf()) : 0LL,
                    planOutputBytes[outputIdx], planOutputDevicePtrs[outputIdx],
                    static_cast<unsigned long long>(hash),
                    static_cast<unsigned long long>(samples[0]),
                    static_cast<unsigned long long>(samples[1]),
                    static_cast<unsigned long long>(samples[2]),
                    static_cast<unsigned long long>(samples[3]));
            }
        };

        // NOTE: Do NOT call plan->setShapesFrozen(true) here.
        // The plan auto-seals during its first executeSteadyState() call
        // (which falls back to execute() for the warmup steps), setting
        // shapesFrozen=true and triggering Triton compilation. Calling
        // setShapesFrozen manually after execution violates the plan lifecycle
        // (executeCount > 0) and would skip the warmup/capture phase.
        // Auto-seal handles the transition correctly.

        auto tPlanEnd = stepTimingEnabled ? std::chrono::high_resolution_clock::now() : stepStart;

        // ── Step 2b: GDN/conv recurrent state feedback ──
        // Copy state outputs back to ext inputs for the next decode step.
        // Critical for hybrid architectures (e.g. Qwen with GDN layers).
        // Without this, GDN layers see frozen state from warmup and degenerate.
        //
        // CRITICAL: Use explicit cudaMemcpyAsync on the DECODE LOOP's stream,
        // NOT assign(). assign() uses the array's LaunchContext stream which may
        // differ from the plan execution stream (ctx->dspStream vs LC default).
        // This caused a stream ordering race: assign's memcpy ran on the LC
        // default stream while the next plan->execute() read ext inputs on the
        // DSP stream, with no event synchronization between them.
        //
        // Both plan outputs and ext inputs are always C-contiguous [B,H,D_k,D_v]
        // with same type/length (guaranteed by gated_delta_rule op shape function),
        // so raw memcpy is safe and avoids the stream mismatch entirely.
        auto commitRecurrentState = [&]() {
            if (config->numGdnStatePairs > 0 && config->gdnStateExtIndices != nullptr
                && config->gdnStateOutputIndices != nullptr) {
                for (int s = 0; s < config->numGdnStatePairs; s++) {
                    int outIdx = config->gdnStateOutputIndices[s];
                    int extIdx = config->gdnStateExtIndices[s];
                    if (outIdx >= 0 && outIdx < numPlanOutputs && planOutputs[outIdx] != nullptr
                        && extIdx >= 0 && extIdx < numExtInputs && extInputs[extIdx] != nullptr) {
                        NDArray* src = planOutputs[outIdx];
                        NDArray* dst = extInputs[extIdx];
                        if (src->lengthOf() == dst->lengthOf() && src->dataType() == dst->dataType()) {
                            size_t bytes = src->lengthOf() * src->sizeOfT();
                            NDArray::prepareSpecialUse({dst}, {src});
                            cudaMemcpyAsync(dst->specialBuffer(), src->specialBuffer(),
                                            bytes, cudaMemcpyDeviceToDevice, *stream);
                            NDArray::registerSpecialUse({dst}, {src});
                        }
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
                        if (src->lengthOf() == dst->lengthOf() && src->dataType() == dst->dataType()) {
                            size_t bytes = src->lengthOf() * src->sizeOfT();
                            NDArray::prepareSpecialUse({dst}, {src});
                            cudaMemcpyAsync(dst->specialBuffer(), src->specialBuffer(),
                                            bytes, cudaMemcpyDeviceToDevice, *stream);
                            NDArray::registerSpecialUse({dst}, {src});
                        }
                    }
                }
            }
        };
        // ADR 0106 Phase 2 (accepted-prefix state commit): on proposing steps the
        // verification forward advanced recurrent state through ALL proposed rows
        // (actual_sequence_length = 1 + proposedCount). Committing that state before
        // acceptance is known would poison the next step whenever a draft is
        // rejected. Defer the commit to the speculative accept block, which re-runs
        // the plan with the accepted prefix on partial acceptance before committing.
        const bool deferStateCommit = (useSpeculative && proposedCount > 0);
        if (!deferStateCommit) {
            commitRecurrentState();
            queueCommittedStateSamples(step, currentPosition + 1, false);
        }

        // ── Step 3: Token sampling ──
        // Get logits from plan output at config->logitsOutputIdx
        NDArray* logitsOutput = planOutputs[config->logitsOutputIdx];

        // Validate logits rank before accessing shape dimensions.
        // Expected: rank 2 [batch, vocabSize] or rank 3 [batch, seqLen, vocabSize].
        // A rank-0 (scalar) output means the plan returned a wrong/stale output slot.
        auto logitsRank = logitsOutput->rankOf();
        REQUIRE_TRUE(logitsRank >= 2 && logitsRank <= 3, 0,
                     "autoregressive_decode: logitsOutput rank is %lld (expected 2 or 3) at step %d. "
                     "lengthOf=%lld, logitsOutputIdx=%d, numPlanOutputs=%d. "
                     "The plan output at this index is not logits — check logitsOutputIdx mapping.",
                     (long long)logitsRank, step,
                     (long long)logitsOutput->lengthOf(),
                     config->logitsOutputIdx, numPlanOutputs);

        // logitsOutput shape: [batch, seqLen, vocabSize] (rank 3) or [batch, vocabSize] (rank 2)
        // For rank 3: decode steps have seqLen=1 → [1, 1, vocabSize], prefill → [1, N, vocabSize]
        // For rank 2: always [batch, vocabSize] — treat as seqLen=1
        LongType logitsSeqLen;
        LongType logitsVocab;
        if (logitsRank == 3) {
            logitsSeqLen = logitsOutput->sizeAt(1);
            logitsVocab = logitsOutput->sizeAt(2);
        } else {
            // rank 2: [batch, vocabSize]
            logitsSeqLen = 1;
            logitsVocab = logitsOutput->sizeAt(1);
        }

        // Get pointer to last-position logits (already on device)
        NDArray::prepareSpecialUse({sampledToken}, {logitsOutput});

        REQUIRE_TRUE(logitsVocab > 0, 0,
                     "autoregressive_decode: logits vocab dimension is 0 at step %d. "
                     "Cannot perform token selection on empty vocabulary.",
                     step);

        // ── ADR 0106 Phase 2 speculative path OR Phase 1 scalar path ────────────
        //
        // SPECULATIVE (useSpeculative && proposedCount > 0):
        //   Run argmaxMultiRowLauncher over all (1+proposedCount) rows of logits,
        //   then D2H-sync to get all argmax values on the host. Apply the lossless
        //   accept rule: accept argmax[0] always; accept argmax[i] for i=1..p iff
        //   argmax[i-1] == draftIds[i-1] (i.e. the target agreed with our proposal
        //   at position i-1). Emit all accepted tokens as a batch.
        //
        // SCALAR (everything else): same W=1 path as Phase 1, completely unchanged.

        if (useSpeculative && proposedCount > 0 && logitsRank == 3) {
            // ── Speculative multi-row argmax ──────────────────────────────────────
            // logitsOutput shape: [1, W_max, vocab]. Rows 0..proposedCount are the
            // active positions filled by this step's forward (activeWindow=1+proposedCount).
            int numRows = 1 + proposedCount;
            // The contiguous device ptr for rows 0..numRows-1 is logitsOutput->specialBuffer()
            // (batch=1, so offset 0 IS row 0). Rows are stride-vocabVocab apart (contiguous).
            NDArray::prepareSpecialUse({specArgmaxDevice}, {logitsOutput});
            BUILD_SINGLE_SELECTOR(logitsOutput->dataType(), argmaxMultiRowLauncher,
                                  (stream, logitsOutput->specialBuffer(),
                                   specArgmaxDevice->specialBuffer(),
                                   static_cast<LongType>(numRows),
                                   logitsVocab),
                                  SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({specArgmaxDevice}, {logitsOutput});

            // D2H: target rows and MTP drafts share the acceptance path's
            // existing synchronization. No predictor-side host boundary is added.
            LongType* argmaxDst = pinnedArgmax ? pinnedArgmax : stackArgmax;
            cudaMemcpyAsync(argmaxDst, specArgmaxDevice->specialBuffer(),
                            numRows * sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
            LongType* mtpDraftDst = pinnedDraftIds ? pinnedDraftIds : stackDraftIds;
            if (useMtp) {
                cudaMemcpyAsync(mtpDraftDst, mtpDraftDevice->specialBuffer(),
                                proposedCount * sizeof(LongType),
                                cudaMemcpyDeviceToHost, *stream);
            }

            // Gated diagnostic D2H (rides the existing sync below — no new sync
            // points): sample the first 4 logits of rows 0 and 1 so kernel-path
            // numeric drift between the W-wide verification forward and the W=1
            // sequential forward is observable at value level, not just argmax.
            float specLogitsSample[8] = {};
            if (DSP_DIAG_ENABLED(KV_CACHE) && logitsVocab >= 4 && numRows >= 2
                    && logitsOutput->dataType() == DataType::FLOAT32) {
                const char* lgBase = static_cast<const char*>(logitsOutput->specialBuffer());
                size_t lgRow = static_cast<size_t>(logitsVocab) * logitsOutput->sizeOfT();
                cudaMemcpyAsync(specLogitsSample, lgBase, 4 * logitsOutput->sizeOfT(),
                                cudaMemcpyDeviceToHost, *stream);
                cudaMemcpyAsync(specLogitsSample + 4, lgBase + lgRow,
                                4 * logitsOutput->sizeOfT(),
                                cudaMemcpyDeviceToHost, *stream);
            }

            // Keep every target input at this verification step until acceptance is
            // known. A partial acceptance re-executes the target with a shorter
            // actual_sequence_length; advancing masks/positions before that re-run
            // makes it observe next-step inputs and corrupts the committed state.
            // The CPU helper already follows this ordering.
            LongType basePosition = currentPosition;

            // ── D2H sync: wait for the existing async argmax/draft copies ──
            cudaStreamSynchronize(*stream);
            emitCommittedStateSamples(step - 1);
            emitPreExecStateSamples(step);
            dumpStepInputSlices("spec", step, basePosition);
            emitPlanOutputFingerprints();
            if (useMtp) {
                std::copy(mtpDraftDst, mtpDraftDst + proposedCount, draftIds);
            }
            if (useMtp && DSP_DIAG_ENABLED(KV_CACHE)) {
                // Chain-probe drain: samples were queued on the exec stream during
                // each executeMtpCuda; the synchronize above completed them.
                for (int cp = 0; cp < mtpChainSampled; cp++) {
                    DSP_DIAG(KV_CACHE,
                             "MTP_CHAIN_PROBE step=%d slot=%d tok=%lld "
                             "carryIn=[%.6g,%.6g,%.6g,%.6g] hidOut=[%.6g,%.6g,%.6g,%.6g]",
                             step, cp, (long long)mtpChainTok[cp],
                             mtpChainCarryIn[cp][0], mtpChainCarryIn[cp][1],
                             mtpChainCarryIn[cp][2], mtpChainCarryIn[cp][3],
                             mtpChainHidOut[cp][0], mtpChainHidOut[cp][1],
                             mtpChainHidOut[cp][2], mtpChainHidOut[cp][3]);
                }
                mtpChainSampled = 0;
            }

            // ── Apply lossless accept rule ─────────────────────────────────────────
            // Input row i contains draftIds[i - 1] for i > 0, so target logits row i
            // predicts the token after that input. Therefore row 0 validates draft 0,
            // row 1 validates draft 1, etc. On the first mismatch at j, emit accepted
            // drafts [0,j) followed by target argmax[j] as the correction token. If
            // every draft matches, argmax[proposedCount] is the bonus token.
            // Snapshot raw target argmaxes before the emission rewrite below —
            // consumed by the gated KV_CACHE diagnostic event (host data already
            // synced by the D2H above; no additional synchronization).
            LongType argmaxRaw[8] = {};
            if (DSP_DIAG_ENABLED(KV_CACHE)) {
                for (int i = 0; i < 8 && i <= proposedCount; i++) argmaxRaw[i] = argmaxDst[i];
            }

            int acceptedDrafts = 0;
            while (acceptedDrafts < proposedCount &&
                   argmaxDst[acceptedDrafts] == draftIds[acceptedDrafts]) {
                acceptedDrafts++;
            }

            // Adaptive chain-cap accounting. Count UNCONDITIONALLY: row p's argmax
            // is the target's continuation of the draft prefix, so draft[p] ==
            // argmax[p] measures the head's chain quality at position p even when
            // an earlier draft already missed (the lossless accept rule stays
            // sequential — this only feeds the cap statistic). Unconditional
            // counting reaches MIN_EVALS in MIN_EVALS steps instead of waiting
            // for earlier positions to hit.
            if (useMtp) {
                for (int p = 0; p < proposedCount && p < 33; p++) {
                    mtpPosEvaluated[p]++;
                    if (argmaxDst[p] == draftIds[p]) mtpPosAccepted[p]++;
                }
                for (int p = 1; p < mtpChainCap && p < 33; p++) {
                    if (mtpPosEvaluated[p] >= MTP_CHAIN_CAP_MIN_EVALS && mtpPosAccepted[p] == 0) {
                        DSP_DIAG(KV_CACHE,
                                 "MTP_CHAIN_CAP: capping chain depth %d -> %d "
                                 "(pos%d evaluated=%d accepted=0; recursive drafts unproductive)",
                                 mtpChainCap, p, p, mtpPosEvaluated[p]);
                        mtpChainCap = p;
                        break;
                    }
                }
            }

            // ── ADR 0106 Phase 2: accepted-prefix recurrent-state commit ─────────
            // The forward above ran with actual_sequence_length = 1 + proposedCount,
            // advancing GDN/conv state through ALL proposed rows. On partial/zero
            // acceptance, re-execute with actual_sequence_length = 1 + acceptedDrafts
            // so the state outputs advance through the accepted prefix only; the
            // emission below uses THIS pass's argmaxes (already on the host). The
            // re-run sees the same step inputs (input_ids/mask/positions unchanged;
            // in-graph KV writes are idempotent for the same step). Next step's
            // pre-exec update rewrites actual_sequence_length, so no restore needed.
            if (acceptedDrafts < proposedCount
                    && config->actualSequenceLengthExtIdx >= 0
                    && config->actualSequenceLengthExtIdx < numExtInputs
                    && extInputs[config->actualSequenceLengthExtIdx] != nullptr) {
                NDArray* aslArr = extInputs[config->actualSequenceLengthExtIdx];
                NDArray::prepareSpecialUse({aslArr}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                    aslArr->specialBuffer(), static_cast<LongType>(1 + acceptedDrafts));
                NDArray::registerSpecialUse({aslArr}, {});
                DSP_DIAG(KV_CACHE,
                         "SPEC_STATE_RERUN step=%d proposed=%d accepted=%d — re-executing "
                         "with actual_sequence_length=%d for accepted-prefix state commit",
                         step, proposedCount, acceptedDrafts, 1 + acceptedDrafts);
                if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr
                    && config->kvInputExtIndices != nullptr && config->numGdnStatePairs >= 0) {
                    static thread_local std::vector<NDArray*> tl_kvQuantPtrsRerun;
                    tl_kvQuantPtrsRerun.resize(numKvPairs);
                    for (int ki = 0; ki < numKvPairs; ki++) {
                        int extIdx = config->kvInputExtIndices[ki];
                        tl_kvQuantPtrsRerun[ki] = (extIdx >= 0 && extIdx < numExtInputs)
                            ? extInputs[extIdx] : nullptr;
                    }
                    setKvScaleRegistry(tl_kvQuantPtrsRerun.data(), config->kvScaleBuffers, numKvPairs);
                }
                Status rerunStatus = plan->executeSteadyState(
                    extInputs, numExtInputs,
                    planOutputs, numPlanOutputs,
                    reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
                if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
                    clearKvScaleRegistry();
                }
                REQUIRE_TRUE(rerunStatus == Status::OK, 0,
                             "autoregressive_decode: accepted-prefix state re-execution FAILED "
                             "at step %d with status %d (accepted=%d of %d).",
                             step, static_cast<int>(rerunStatus), acceptedDrafts, proposedCount);
            }
            // Commit recurrent state from the (possibly re-run) accepted-prefix pass.
            commitRecurrentState();
            queueCommittedStateSamples(
                step, basePosition + static_cast<LongType>(acceptedDrafts) + 1, true);

            if (useMtp) {
                REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                                 && config->targetHiddenOutputIdx < numPlanOutputs
                                 && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                             0, "autoregressive_decode: target hidden output is unavailable for CUDA MTP");

                LongType mtpProcessedThrough = basePosition + proposedCount - 1;
                if (acceptedDrafts == proposedCount) {
                    // K predictor calls produce K drafts but consume only the base
                    // plus drafts [0,K-2]. Consume the final accepted draft so the
                    // predictor cache aligns with the target's bonus token.
                    executeMtpCuda(basePosition + proposedCount, proposedCount, false);
                    mtpProcessedThrough = basePosition + proposedCount;
                }

                LongType nextMtpPosition = basePosition + acceptedDrafts + 1;
                if (nextMtpPosition <= mtpProcessedThrough) {
                    NDArray::prepareSpecialUse({config->mtpCausalMask}, {});
                    BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(),
                                          maskCausalRangeLauncher,
                                          (stream, config->mtpCausalMask->specialBuffer(),
                                           nextMtpPosition, mtpProcessedThrough + 1,
                                           mtpMaskLen),
                                          SD_FLOAT_TYPES);
                    NDArray::registerSpecialUse({config->mtpCausalMask}, {});
                }

                setMtpTargetCarryCuda(
                    planOutputs[config->targetHiddenOutputIdx], acceptedDrafts);
                setMtpNextInputCuda(
                    specArgmaxDevice, acceptedDrafts, nextMtpPosition);
            }

            LongType correctionOrBonus = argmaxDst[acceptedDrafts];
            for (int i = 0; i < acceptedDrafts; i++) {
                argmaxDst[i] = draftIds[i];
            }
            argmaxDst[acceptedDrafts] = correctionOrBonus;
            int n = acceptedDrafts + 1;

            totalSpeculativeProposed += proposedCount;
            totalSpeculativeAccepted += acceptedDrafts;
            speculativeStepCount++;

            // Upload the reconstructed lossless emission sequence so the existing D2D
            // storage path remains stream-ordered and avoids host scalar writes.
            NDArray::prepareSpecialUse({specArgmaxDevice}, {});
            cudaMemcpyAsync(specArgmaxDevice->specialBuffer(), argmaxDst,
                            n * sizeof(LongType), cudaMemcpyHostToDevice, *stream);
            NDArray::registerSpecialUse({specArgmaxDevice}, {});

            // ── Store accepted tokens to generatedTokenIds ────────────────────────
            bool shouldStop = false;
            int storedCount = 0;
            NDArray::prepareSpecialUse({generatedTokenIds}, {specArgmaxDevice});
            for (int i = 0; i < n && tokensGenerated < maxNewTokens; i++) {
                LongType tok = argmaxDst[i];
                void* dstPtr = static_cast<char*>(generatedTokenIds->specialBuffer())
                               + tokensGenerated * sizeof(LongType);
                cudaMemcpyAsync(dstPtr,
                                static_cast<char*>(specArgmaxDevice->specialBuffer()) + i * sizeof(LongType),
                                sizeof(LongType), cudaMemcpyDeviceToDevice, *stream);
                tokensGenerated++;
                storedCount++;
                if (config->tokenCallback != nullptr) {
                    config->tokenCallback(tok, config->callbackUserData);
                }
                bool matchedStop = stopMatcher.accept(tok);
                shouldStop = matchedStop && stopTerminationAllowed(config, tokensGenerated);
                if (shouldStop) break;
            }
            NDArray::registerSpecialUse({generatedTokenIds}, {specArgmaxDevice});

            // Gated diagnostic event: the first speculative steps carry the whole
            // correctness story (which drafts were proposed, what the target's
            // per-row argmaxes were, where acceptance stopped). All values are
            // host-side data already produced by the existing D2H sync.
            DSP_DIAG(KV_CACHE,
                     "SPEC_STEP step=%d basePos=%lld proposed=%d accepted=%d stored=%d "
                     "draft=[%lld,%lld,%lld,%lld] argmaxRaw=[%lld,%lld,%lld,%lld,%lld] "
                     "r0=[%.6f,%.6f,%.6f,%.6f] r1=[%.6f,%.6f,%.6f,%.6f]",
                     step, (long long)basePosition, proposedCount, acceptedDrafts,
                     storedCount,
                     (long long)draftIds[0], (long long)draftIds[1],
                     (long long)draftIds[2], (long long)draftIds[3],
                     (long long)argmaxRaw[0], (long long)argmaxRaw[1],
                     (long long)argmaxRaw[2], (long long)argmaxRaw[3],
                     (long long)argmaxRaw[4],
                     specLogitsSample[0], specLogitsSample[1],
                     specLogitsSample[2], specLogitsSample[3],
                     specLogitsSample[4], specLogitsSample[5],
                     specLogitsSample[6], specLogitsSample[7]);

            // Commit the base KV output only after the accepted-prefix re-run has
            // selected the authoritative plan outputs (ONNX path; GGUF scatters in graph).
            if (!config->planOwnsKvScatter &&
                config->kvOutputIndices != nullptr && staticKvBuffers != nullptr && numKvPairs > 0) {
                std::vector<KvScatterEntry> entries(2 * numKvPairs);
                std::vector<NDArray*> scatterWrites;
                std::vector<NDArray*> scatterReads;
                scatterWrites.reserve(2 * numKvPairs);
                scatterReads.reserve(2 * numKvPairs);
                for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                    int kvOutIdx = config->kvOutputIndices[kv];
                    NDArray* presentKv = planOutputs[kvOutIdx];
                    NDArray* staticBuf = staticKvBuffers[kv];
                    REQUIRE_TRUE(presentKv != nullptr && staticBuf != nullptr, 0,
                                 "autoregressive_decode speculative: null KV at step %d kv=%d", step, kv);
                    entries[kv].srcPtr  = presentKv->specialBuffer();
                    entries[kv].dstPtr  = staticBuf->specialBuffer();
                    entries[kv].heads   = presentKv->sizeAt(1);
                    entries[kv].srcSeqLen = presentKv->sizeAt(2);
                    entries[kv].dstSeqLen = staticBuf->sizeAt(2);
                    entries[kv].dim     = presentKv->sizeAt(3);
                    entries[kv].lastPos = presentKv->sizeAt(2) - 1;
                    entries[kv].cachePos = basePosition;
                    scatterWrites.push_back(staticBuf);
                    scatterReads.push_back(presentKv);
                }
                NDArray::prepareSpecialUse(scatterWrites, scatterReads);
                kvScatterBatched(entries.data(), 2 * numKvPairs,
                                 staticKvBuffers[0]->dataType(), context);
                NDArray::registerSpecialUse(scatterWrites, scatterReads);
            }

            // Advance and expose exactly the tokens that were actually stored. This
            // happens after rerun/state commit so rejected draft rows never become
            // next-step inputs, even transiently.
            for (int i = 0; i < storedCount; i++) {
                LongType kvPos = currentPosition;
                currentPosition++;
                {
                    LongType cmPos = config->planOwnsKvScatter ? kvPos : currentPosition;
                    if (causalMask != nullptr && cmPos >= 0 && cmPos < causalMaskLen) {
                        NDArray::prepareSpecialUse({causalMask}, {});
                        BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskLauncher,
                                              (stream, causalMask->specialBuffer(), cmPos, causalMaskLen),
                                              SD_FLOAT_TYPES);
                        NDArray::registerSpecialUse({causalMask}, {});
                    }
                }
                if (!attnMaskAliasesCausal && kvPos >= 0 && kvPos < maxKvLen) {
                    NDArray::prepareSpecialUse({attentionMask}, {});
                    BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskLauncher,
                                          (stream, attentionMask->specialBuffer(), kvPos, maxKvLen),
                                          SD_COMMON_TYPES);
                    NDArray::registerSpecialUse({attentionMask}, {});
                }
                if (attnMaskReformat != nullptr && kvPos >= 0 && kvPos < attnMaskReformatLen) {
                    NDArray::prepareSpecialUse({attnMaskReformat}, {});
                    BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskLauncher,
                                          (stream, attnMaskReformat->specialBuffer(), kvPos, attnMaskReformatLen),
                                          SD_FLOAT_TYPES);
                    NDArray::registerSpecialUse({attnMaskReformat}, {});
                }
            }
            // Update position IDs to the next committed position.
            if (storedCount > 0) {
                NDArray::prepareSpecialUse({positionIds}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(positionIds->specialBuffer(), currentPosition);
                NDArray::registerSpecialUse({positionIds}, {});
            }

            // ── Update n-gram tables from the verified emission sequence ─────────
            // Rejected drafts are never learned. MTP has its own persistent state.
            if (useNgram) {
                LongType previous = specPreviousToken;
                LongType current = specCurrentToken;
                for (int i = 0; i < storedCount; i++) {
                    LongType tok = argmaxDst[i];
                    if (current >= 0) {
                        ngramTable[current] = tok;
                        if (previous >= 0) {
                            trigramTable[previous][current] = tok;
                        }
                        DSP_DIAG(KV_CACHE,
                                 "NGRAM_PUT spec step=%d previous=%lld current=%lld next=%lld "
                                 "order3=%d i=%d stored=%d",
                                 step, (long long)previous, (long long)current, (long long)tok,
                                 previous >= 0 ? 1 : 0, i, storedCount);
                    }
                    previous = current;
                    current = tok;
                }
                specPreviousToken = previous;
                specCurrentToken = current;
            }

            LongType nextTokenId = argmaxDst[storedCount - 1];

            // Restore activeWindow to base value (proposals will be set fresh next step)
            config->activeWindow = (specK > 0) ? 1 : config->activeWindow;

            // Timing
            auto tStopCheck = std::chrono::high_resolution_clock::now();
            double stepMs = std::chrono::duration<double, std::milli>(tStopCheck - stepStart).count();
            stepTimesMs.push_back(stepMs);

            if (shouldStop) break;

            // ── Embedding lookup and input updates for next step ─────────────────
            if (config->embeddingsExtIdx >= 0) {
                REQUIRE_TRUE(nextTokenId >= 0 && nextTokenId < vocabSize, 0,
                             "autoregressive_decode speculative: nextTokenId=%lld out of range at step %d.",
                             (long long)nextTokenId, step);
                NDArray::prepareSpecialUse({decodeEmbedding}, {embeddingTable});
                BUILD_SINGLE_SELECTOR(embeddingTable->dataType(), embedLookupLauncher,
                                      (stream, embeddingTable->specialBuffer(),
                                       decodeEmbedding->specialBuffer(),
                                       nextTokenId, hidden, embTableRowStride),
                                      SD_COMMON_TYPES);
                NDArray::registerSpecialUse({decodeEmbedding}, {embeddingTable});
            }
            NDArray::prepareSpecialUse({inputIds}, {});
            updateInputIdsKernel<<<1, 1, 0, *stream>>>(inputIds->specialBuffer(), nextTokenId);
            NDArray::registerSpecialUse({inputIds}, {});

            // Update GGUF in-graph KV scalars
            if (config->positionOffsetExtIdx >= 0 && config->positionOffsetExtIdx < numExtInputs) {
                NDArray* posOffset = extInputs[config->positionOffsetExtIdx];
                if (posOffset != nullptr) {
                    NDArray::prepareSpecialUse({posOffset}, {});
                    updatePositionIdsKernel<<<1, 1, 0, *stream>>>(posOffset->specialBuffer(), currentPosition);
                    NDArray::registerSpecialUse({posOffset}, {});
                }
            }
            if (config->cachePositionExtIdx >= 0 && config->cachePositionExtIdx < numExtInputs) {
                NDArray* cachePosArr = extInputs[config->cachePositionExtIdx];
                if (cachePosArr != nullptr) {
                    NDArray::prepareSpecialUse({cachePosArr}, {});
                    updatePositionIdsKernel<<<1, 1, 0, *stream>>>(cachePosArr->specialBuffer(), currentPosition);
                    NDArray::registerSpecialUse({cachePosArr}, {});
                }
            }

            // Step timing breakdown (speculative path)
            if (stepTimingEnabled) {
                auto tLoopEnd = std::chrono::high_resolution_clock::now();
                auto planUs = std::chrono::duration_cast<std::chrono::microseconds>(tPlanEnd - stepStart).count();
                auto totalStepUs = std::chrono::duration_cast<std::chrono::microseconds>(tLoopEnd - stepStart).count();
                DSP_DIAG(KV_CACHE,
                         "DECODE_STEP_TIMING step=%d path=SPECULATIVE total=%lldus plan=%lldus "
                         "proposed=%d accepted=%d",
                         step, totalStepUs, planUs, proposedCount, storedCount);
            }

            // Balance the prepareSpecialUse({sampledToken}, {logitsOutput}) called above.
            // In the speculative path we don't use sampledToken — registerSpecialUse to
            // keep the CUDA-graph-capture bookkeeping symmetric.
            NDArray::registerSpecialUse({sampledToken}, {logitsOutput});

            // NOTE: skip the rest of the loop body — we handled everything above.
            continue;
        }

        // ── Phase 1 scalar path (W=1 or no proposals this step) ─────────────────
        // Restore activeWindow to 1 in case the speculative path set it but proposedCount==0.
        if (useSpeculative && proposedCount == 0) {
            config->activeWindow = 1;
        }

        NDArray* logitsForSample = logitsOutput;
        NDArray* logitsSliceCuda = nullptr;
        if (useWindowSubstrate && logitsRank == 3 && logitsSeqLen > 1) {
            // operator()(idx) flat format: {dim0Start,dim0End, dim1Start,dim1End, dim2Start,dim2End}
            std::vector<LongType> sliceIdx{0, 1, 0, 1, 0, logitsVocab};
            logitsSliceCuda = (*logitsOutput)(sliceIdx, true);
            logitsForSample = logitsSliceCuda;
        }

        TokenSampleConfig stepSampleConfig = config != nullptr ? config->sampleConfig : TokenSampleConfig();
        LongType baseSeed = stepSampleConfig.seed;
        int generatedOffset = stepSampleConfig.generatedTokenOffset;
        stepSampleConfig.temperature = temperature;
        stepSampleConfig.topK = topK;
        stepSampleConfig.topP = topP;
        stepSampleConfig.repPenalty = repPenalty;
        // Force scalar B=1/W=1 for the selection step — the substrate runs W-wide
        // but policy selection is still scalar (Phase 2 will extend this).
        // Also reset SPECULATIVE strategy to GREEDY: in the scalar fallback path
        // (proposedCount==0 or no window substrate) we always select greedily.
        // TOKEN_SAMPLE_SPECULATIVE(3) is not handled by tokenSamplePolicy — it
        // would throw "only scalar GREEDY/SAMPLE" if left as-is.
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
            std::vector<LongType> range = {0, static_cast<LongType>(step)};
            NDArray* tokensSoFar = (*generatedTokenIds)(range, true);
            tokenSamplePolicy(logitsForSample, sampledToken, tokensSoFar,
                              stepSampleConfig, &sampleResult, context);
            delete tokensSoFar;
        } else {
            tokenSamplePolicy(logitsForSample, sampledToken, inputIds,
                              stepSampleConfig, &sampleResult, context);
        }

        if (logitsSliceCuda != nullptr) {
            delete logitsSliceCuda;
            logitsSliceCuda = nullptr;
        }

        NDArray::registerSpecialUse({sampledToken}, {logitsOutput});

        // ── Tier 1a: Store token via D2D copy (avoids p() hidden H2D + stream 0 sync) ──
        // generatedTokenIds->p() does host write → syncToDevice() → cudaMemcpyAsync
        // on stream 0 + cudaStreamSynchronize(stream_0) — a hidden pipeline drain.
        // Direct D2D from sampledToken to generatedTokenIds stays on the decode stream.
        {
            void* dstPtr = static_cast<char*>(generatedTokenIds->specialBuffer())
                           + tokensGenerated * sizeof(LongType);
            NDArray::prepareSpecialUse({generatedTokenIds}, {sampledToken});
            cudaMemcpyAsync(dstPtr, sampledToken->specialBuffer(),
                            sizeof(LongType), cudaMemcpyDeviceToDevice, *stream);
            NDArray::registerSpecialUse({generatedTokenIds}, {sampledToken});
        }
        tokensGenerated++;

        // ── Tier 1b: Pre-sync GPU work ──
        // Everything below until the cudaStreamSynchronize only depends on
        // currentPosition (CPU counter) and plan output pointers (already on
        // device). None of it needs the token ID from D2H. Launching these
        // kernels BEFORE the sync overlaps their GPU execution with the
        // async D2H copy and hides their latency behind the sync wait.
        //
        // Advance position BEFORE updates for the NEXT decode step.
        currentPosition++;
        LongType kvJustWritten = currentPosition - 1;

        if (useMtp) {
            REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                             && config->targetHiddenOutputIdx < numPlanOutputs
                             && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                         0, "autoregressive_decode: scalar target hidden output is unavailable for CUDA MTP");
            setMtpTargetCarryCuda(planOutputs[config->targetHiddenOutputIdx], 0);
            setMtpNextInputCuda(sampledToken, 0, currentPosition);
        }

        // ── KV scatter — copy present KV into static buffers ──
        // Moved BEFORE sync: scatter only needs currentPosition (CPU counter)
        // and plan output device pointers. Both are available without the token
        // ID. Since scatter and the next plan execution are on the same stream,
        // CUDA ordering guarantees scatter completes before the next read.
        // Skip manual scatter when the plan's native KV scatter is active
        // (planOwnsKvScatter) — executeKvScatterPostExec handles it with its
        // own device-side position counter via executeSteadyState.
        if (!config->planOwnsKvScatter &&
            config->kvOutputIndices != nullptr && staticKvBuffers != nullptr && numKvPairs > 0) {
            // Build batched KV scatter entries and ownership lists.
            std::vector<KvScatterEntry> entries(2 * numKvPairs);
            std::vector<NDArray*> scatterWrites;
            std::vector<NDArray*> scatterReads;
            scatterWrites.reserve(2 * numKvPairs);
            scatterReads.reserve(2 * numKvPairs);
            for (int kv = 0; kv < 2 * numKvPairs; kv++) {
                int kvOutIdx = config->kvOutputIndices[kv];
                NDArray* presentKv = planOutputs[kvOutIdx];
                NDArray* staticBuf = staticKvBuffers[kv];

                REQUIRE_TRUE(kvOutIdx >= 0 && kvOutIdx < numPlanOutputs, 0,
                             "autoregressive_decode: KV output index %d out of range [0,%d) "
                             "at step %d kv=%d",
                             kvOutIdx, numPlanOutputs, step, kv);
                REQUIRE_TRUE(presentKv != nullptr, 0,
                             "autoregressive_decode: KV output[%d] (planOutput[%d]) is null "
                             "at step %d — plan did not produce this output.",
                             kv, kvOutIdx, step);
                REQUIRE_TRUE(staticBuf != nullptr, 0,
                             "autoregressive_decode: static KV buffer[%d] is null at step %d.",
                             kv, step);
                REQUIRE_TRUE(presentKv->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: KV output[%d] has null device buffer "
                             "at step %d — stale or uninitialized output.",
                             kv, step);
                REQUIRE_TRUE(staticBuf->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: static KV[%d] has null device buffer "
                             "at step %d — buffer was freed or never allocated.",
                             kv, step);

                entries[kv].srcPtr = presentKv->specialBuffer();
                entries[kv].dstPtr = staticBuf->specialBuffer();
                entries[kv].heads = presentKv->sizeAt(1);
                entries[kv].srcSeqLen = presentKv->sizeAt(2);
                entries[kv].dstSeqLen = staticBuf->sizeAt(2);
                entries[kv].dim = presentKv->sizeAt(3);
                entries[kv].lastPos = presentKv->sizeAt(2) - 1;
                entries[kv].cachePos = kvJustWritten;  // currentPosition - 1
                scatterWrites.push_back(staticBuf);
                scatterReads.push_back(presentKv);
            }

            REQUIRE_TRUE(staticKvBuffers[0] != nullptr, 0,
                         "autoregressive_decode: staticKvBuffers[0] is null at step %d — "
                         "cannot determine KV data type for scatter.",
                         step);
            NDArray::prepareSpecialUse(scatterWrites, scatterReads);
            kvScatterBatched(entries.data(), 2 * numKvPairs,
                             staticKvBuffers[0]->dataType(), context);
            NDArray::registerSpecialUse(scatterWrites, scatterReads);
        }

        // Update attention mask: unmask the KV position that was JUST written.
        // Skipped when it aliases the additive causal mask (see attnMaskAliasesCausal).
        if (!attnMaskAliasesCausal && kvJustWritten >= 0 && kvJustWritten < maxKvLen) {
            NDArray::prepareSpecialUse({attentionMask}, {});
            BUILD_SINGLE_SELECTOR(attentionMask->dataType(), updateAttentionMaskLauncher,
                                  (stream, attentionMask->specialBuffer(), kvJustWritten, maxKvLen),
                                  SD_COMMON_TYPES);
            NDArray::registerSpecialUse({attentionMask}, {});
        }

        // Update causal mask: for ONNX/external-scatter path (planOwnsKvScatter == false),
        // unmask currentPosition (the NEXT write position), matching Java's advance-one-ahead
        // pattern in runJavaDecodeLoop (causalMask[cachePos] where cachePos is already incremented).
        // For GGUF (planOwnsKvScatter == true), unmask kvJustWritten (the just-written position)
        // because the in-graph attention already unmasked currentPosition via the pre-unmask above.
        {
            LongType causalMaskUnmaskPos = config->planOwnsKvScatter ? kvJustWritten : currentPosition;
            if (causalMask != nullptr && causalMaskUnmaskPos >= 0 && causalMaskUnmaskPos < causalMaskLen) {
                NDArray::prepareSpecialUse({causalMask}, {});
                BUILD_SINGLE_SELECTOR(causalMask->dataType(), updateCausalMaskLauncher,
                                      (stream, causalMask->specialBuffer(), causalMaskUnmaskPos, causalMaskLen),
                                      SD_FLOAT_TYPES);
                NDArray::registerSpecialUse({causalMask}, {});
            }
        }

        // Update attn_mask_reformat: explicit padded bias mirrors attention_mask.
        // The current query is represented by the final appended slot in the graph,
        // so after external scatter the newly written static KV position becomes
        // visible on the next step.
        {
            LongType attnReformatUnmaskPos = kvJustWritten;
            if (attnMaskReformat != nullptr && attnReformatUnmaskPos >= 0 && attnReformatUnmaskPos < attnMaskReformatLen) {
                NDArray::prepareSpecialUse({attnMaskReformat}, {});
                BUILD_SINGLE_SELECTOR(attnMaskReformat->dataType(), updateCausalMaskLauncher,
                                      (stream, attnMaskReformat->specialBuffer(), attnReformatUnmaskPos, attnMaskReformatLen),
                                      SD_FLOAT_TYPES);
                NDArray::registerSpecialUse({attnMaskReformat}, {});
            }
        }

        // Update position_ids: set to next step's position
        NDArray::prepareSpecialUse({positionIds}, {});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            positionIds->specialBuffer(),
            currentPosition);
        NDArray::registerSpecialUse({positionIds}, {});

        // ── D2H token readback via pinned memory ──
        // Read sampled token ID back to host (single int64).
        // Issue D2H copy on the SAME stream as the argmax kernel — FIFO ordering
        // guarantees the copy starts after argmax completes.
        // Using pinned memory enables true async DMA (no driver bounce buffer).
        //
        // All GPU work that doesn't need the token ID (KV scatter, mask updates,
        // position updates) is launched ABOVE this point. The sync below waits for
        // everything on the stream, including those overlapped kernels.
        LongType* tokenDst = pinnedTokenId ? pinnedTokenId : &stackTokenId;
        *tokenDst = 0;
        auto tSyncStart = stepTimingEnabled ? std::chrono::high_resolution_clock::now() : stepStart;
        cudaMemcpyAsync(tokenDst, sampledToken->specialBuffer(),
                        sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
        // Gated diagnostic D2H (rides the sync below — no new sync points):
        // sample the first 4 logits of the live row for cross-pipeline value
        // comparison against the W-wide verification rows.
        float scalarLogitsSample[4] = {};
        if (DSP_DIAG_ENABLED(KV_CACHE) && logitsOutput->lengthOf() >= 4
                && logitsOutput->dataType() == DataType::FLOAT32) {
            cudaMemcpyAsync(scalarLogitsSample, logitsOutput->specialBuffer(),
                            4 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
        }
        cudaStreamSynchronize(*stream);
        emitCommittedStateSamples(step);
        // NOTE: mask slices here reflect the already-advanced next-step state (the
        // advance kernels launch before this sync); the KV rows are the payload —
        // they hold exactly what this step committed.
        dumpStepInputSlices("scalar", step, currentPosition - 1);
        emitPlanOutputFingerprints();
        auto tSyncEnd = stepTimingEnabled ? std::chrono::high_resolution_clock::now() : stepStart;
        LongType nextTokenId = *tokenDst;
        if (config->tokenCallback != nullptr) {
            config->tokenCallback(nextTokenId, config->callbackUserData);
        }

        // Gated diagnostic event: per-step scalar-path record (host-side counters
        // and the already-synced token only — no additional device reads or syncs).
        // Mirrors the CPU helper's SCALAR_STEP event for step-level divergence
        // localization.
        DSP_DIAG(KV_CACHE, "SCALAR_STEP step=%d pos=%lld tok=%lld proposed=%d "
                 "r0=[%.6f,%.6f,%.6f,%.6f]",
                 step, (long long)(currentPosition - 1), (long long)nextTokenId,
                 proposedCount,
                 scalarLogitsSample[0], scalarLogitsSample[1],
                 scalarLogitsSample[2], scalarLogitsSample[3]);

        // ADR 0106 Phase 2: learn the verified scalar transition.
        if (useNgram) {
            if (specCurrentToken >= 0) {
                ngramTable[specCurrentToken] = nextTokenId;
                if (specPreviousToken >= 0) {
                    trigramTable[specPreviousToken][specCurrentToken] = nextTokenId;
                }
                DSP_DIAG(KV_CACHE,
                         "NGRAM_PUT scalar step=%d previous=%lld current=%lld next=%lld order3=%d",
                         step, (long long)specPreviousToken, (long long)specCurrentToken,
                         (long long)nextTokenId, specPreviousToken >= 0 ? 1 : 0);
            }
            specPreviousToken = specCurrentToken;
            specCurrentToken = nextTokenId;
        }

        // ── Check stop condition ──
        bool matchedStop = stopMatcher.accept(nextTokenId);
        bool shouldStop = matchedStop && stopTerminationAllowed(config, tokensGenerated);
        bool matchedRepetition = repetitionMatcher.accept(nextTokenId);

        auto tStopCheck = std::chrono::high_resolution_clock::now();

        // Compute step time using the stop check timestamp
        // Always measure real wall-clock step time — needed for lateSteady metric even
        // when detailed sub-step timing (stepTimingEnabled) is off.
        double stepMs = std::chrono::duration<double, std::milli>(tStopCheck - stepStart).count();
        stepTimesMs.push_back(stepMs);

        if (shouldStop) break;
        if (matchedRepetition) {
            config->nativeFinishReason = 1;
            break;
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
            NDArray::prepareSpecialUse({decodeEmbedding}, {embeddingTable});
            BUILD_SINGLE_SELECTOR(embeddingTable->dataType(), embedLookupLauncher,
                                  (stream, embeddingTable->specialBuffer(),
                                   decodeEmbedding->specialBuffer(),
                                   nextTokenId, hidden, embTableRowStride),
                                  SD_COMMON_TYPES);
            NDArray::registerSpecialUse({decodeEmbedding}, {embeddingTable});
        }

        // Update input_ids: set to next token (needs nextTokenId from D2H)
        NDArray::prepareSpecialUse({inputIds}, {});
        updateInputIdsKernel<<<1, 1, 0, *stream>>>(
            inputIds->specialBuffer(),
            nextTokenId);
        NDArray::registerSpecialUse({inputIds}, {});

        // ── Update in-graph KV cache scalars (GGUF pattern) ──
        // position_offset and cache_position are scalar ext inputs that the
        // attention op reads for RoPE position and KV write position.
        if (config->positionOffsetExtIdx >= 0 && config->positionOffsetExtIdx < numExtInputs) {
            NDArray* posOffset = extInputs[config->positionOffsetExtIdx];
            if (posOffset != nullptr) {
                NDArray::prepareSpecialUse({posOffset}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                    posOffset->specialBuffer(),
                    currentPosition);
                NDArray::registerSpecialUse({posOffset}, {});
            }
        }
        if (config->cachePositionExtIdx >= 0 && config->cachePositionExtIdx < numExtInputs) {
            NDArray* cachePosArr = extInputs[config->cachePositionExtIdx];
            if (cachePosArr != nullptr) {
                NDArray::prepareSpecialUse({cachePosArr}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                    cachePosArr->specialBuffer(),
                    currentPosition);
                NDArray::registerSpecialUse({cachePosArr}, {});
            }
        }

        // Per-step timing breakdown (gated behind executionTimingEnabled only — print every step)
        // Note: "preSyncGpu" = argmax + KV scatter + mask/posId updates (all before sync).
        //       "syncOnly" = just the cudaStreamSynchronize wait.
        //       "postSync" = embed lookup + input_ids update + GGUF scalars.
        if (stepTimingEnabled) {
            auto tLoopEnd = std::chrono::high_resolution_clock::now();
            auto wireUs = std::chrono::duration_cast<std::chrono::microseconds>(tWireEnd - stepStart).count();
            auto planUs = std::chrono::duration_cast<std::chrono::microseconds>(tPlanEnd - tWireEnd).count();
            auto preSyncGpuUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncStart - tPlanEnd).count();
            auto syncOnlyUs = std::chrono::duration_cast<std::chrono::microseconds>(tSyncEnd - tSyncStart).count();
            auto postSyncUs = std::chrono::duration_cast<std::chrono::microseconds>(tLoopEnd - tSyncEnd).count();
            auto totalStepUs = std::chrono::duration_cast<std::chrono::microseconds>(tLoopEnd - stepStart).count();
            DSP_DIAG(KV_CACHE,
                     "DECODE_STEP_TIMING step=%d path=SCALAR total=%lldus wire=%lldus plan=%lldus "
                     "preSyncGpu=%lldus syncOnly=%lldus postSync=%lldus",
                     step, totalStepUs, wireUs, planUs,
                     preSyncGpuUs, syncOnlyUs, postSyncUs);
        }
    }

    // ── Final sync ──
    cudaStreamSynchronize(*stream);
    emitCommittedStateSamples(maxNewTokens - 1);
    emitPreExecStateSamples(maxNewTokens - 1);

    // Free pinned memory (Tier 1c)
    if (pinnedTokenId != nullptr) {
        cudaFreeHost(pinnedTokenId);
        pinnedTokenId = nullptr;
    }
    if (pinnedPlanOutputSamples != nullptr) {
        cudaFreeHost(pinnedPlanOutputSamples);
        pinnedPlanOutputSamples = nullptr;
    }
    if (pinnedCommittedStateSamples != nullptr) {
        cudaFreeHost(pinnedCommittedStateSamples);
        pinnedCommittedStateSamples = nullptr;
    }
    if (pinnedPreExecStateSamples != nullptr) {
        cudaFreeHost(pinnedPreExecStateSamples);
        pinnedPreExecStateSamples = nullptr;
    }

    // ── ADR 0106 Phase 2: free speculative decode resources ──
    if (pinnedArgmax != nullptr) {
        cudaFreeHost(pinnedArgmax);
        pinnedArgmax = nullptr;
    }
    if (pinnedDraftIds != nullptr) {
        cudaFreeHost(pinnedDraftIds);
        pinnedDraftIds = nullptr;
    }
    if (specArgmaxDevice != nullptr) {
        delete specArgmaxDevice;
        specArgmaxDevice = nullptr;
    }
    if (mtpDraftDevice != nullptr) {
        delete mtpDraftDevice;
        mtpDraftDevice = nullptr;
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
        // DSP warmup takes ~60 steps to converge to true steady-state.
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
            // Not enough steps — fall back to overall
            timingInfo->p(5, static_cast<float>(tokPerSec));
            timingInfo->p(6, static_cast<float>(avgMs));
        }
    }
    if (config->nativeFinishReason == 1 && timingInfo->lengthOf() > 6) {
        timingInfo->p(6, -1.0f);
    }

    // ── Cleanup internal allocations ──
    // decodeEmbedding is NOT deleted — it's prefillEmbeddings, owned by the caller.
    delete sampledToken;
    if (internalMask != nullptr) {
        delete internalMask;
    }
    if (internalPosIds != nullptr) {
        delete internalPosIds;
    }
    // ADR 0106 Phase 2: free internally-allocated window tensors (ONNX speculative path).
    // Only free if we allocated them above; externally-provided tensors are caller-owned.
    if (internalWindowGridMask != nullptr) {
        config->windowGridMask = nullptr;   // clear pointer so caller doesn't double-free
        delete internalWindowGridMask;
    }
    if (internalWindowPositionGrid != nullptr) {
        config->windowPositionGrid = nullptr;
        delete internalWindowPositionGrid;
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
