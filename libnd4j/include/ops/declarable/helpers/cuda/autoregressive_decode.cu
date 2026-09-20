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
#include <execution/LaunchContext.h>
#include <graph/Context.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspPhaseUtils.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <vector>

#include "execution/cuda/LaunchDims.h"

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

// --- CUDA Kernels ------------------------------------------------------------

/**
 * CUDA kernel: look up a single row from the embedding table.
 *
 * Given embeddingTable [vocabSize, hidden] and a token ID, copies
 * embeddingTable[tokenId, :] into outputEmbed [1, 1, hidden].
 *
 * One block, blockDim.x threads - each thread copies hidden/blockDim.x elements.
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
 * Launcher for embedLookupKernel - called via BUILD_SINGLE_SELECTOR.
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
 * Launcher for updateAttentionMaskKernel - called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void updateAttentionMaskLauncher(const cudaStream_t* stream,
                                         void* vMask,
                                         LongType position,
                                         LongType maxKvLen) {
    updateAttentionMaskKernel<T><<<1, 1, 0, *stream>>>(vMask, position, maxKvLen);
}

/**
 * HOST-SIDE sample of one element from an already-downloaded raw buffer into
 * a float, converted through the buffer's own dtype (review round 5, finding
 * 3): the GDN-state NaN probe reads host bytes after the single batched D2H
 * and must interpret them in the state's dtype - a raw memcpy into float[]
 * misreads BF16/FP16 (packing) and FP64 (halves), and for narrow dtypes read
 * PAST the tensor's storage. Mirrors the CPU helper's sampleFirstRowValueCpu.
 */
template <typename T>
static void sampleRawHostValue(const void* valuePtr, void* out) {
    *static_cast<float*>(out) = static_cast<float>(*reinterpret_cast<const T*>(valuePtr));
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
 * Launcher for updateCausalMaskKernel - called via BUILD_SINGLE_SELECTOR.
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
 * write only ever advances row 0 - draft rows would stay stuck at the freeze
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
static SD_KERNEL void refillRepairMaskKernel(void* vMask,
                                             LongType wMax,
                                             LongType rowLen,
                                             LongType predictorBase,
                                             float maskFill) {
    const LongType totalElems = wMax * rowLen;
    auto mask = reinterpret_cast<T*>(vMask);
    for (LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < totalElems;
         idx += static_cast<LongType>(gridDim.x) * blockDim.x) {
        const LongType row = idx / rowLen;
        const LongType col = idx % rowLen;
        // Rows at/beyond the active prefix see the committed prefix and their
        // own slot so their softmax stays finite (outputs never scattered).
        // Rows inside the active prefix behave identically: the intended
        // K/V-only repair graph prunes attention entirely, so mask content
        // must not feed the returned K/V. If a future repair graph retains
        // attention, this refill is NOT a valid causal chain and the graph
        // must qualify its own mask semantics before use.
        mask[idx] = (col < predictorBase || col == predictorBase + row)
            ? static_cast<T>(0.0f) : static_cast<T>(maskFill);
    }
}

template <typename T>
static void refillRepairMaskLauncher(const cudaStream_t* stream,
                                     void* vMask,
                                     LongType wMax,
                                     LongType rowLen,
                                     LongType predictorBase) {
    const float maskFill = (sizeof(T) == 2) ? -65504.0f : -1e9f;
    const LongType totalElems = wMax * rowLen;
    const int threads = 256;
    const int blocks = static_cast<int>((totalElems + threads - 1) / threads);
    refillRepairMaskKernel<T><<<blocks, threads, 0, *stream>>>(
        vMask, wMax, rowLen, predictorBase, maskFill);
}

template <typename T>
static void refillWindowCausalMaskLauncher(const cudaStream_t* stream,
                                           void* vMask,
                                           LongType wMax,
                                           LongType maxKvLen,
                                           LongType currentPos) {
    // Match DecoderInputBuilder.buildInGraphWindowMask fill values: -65504 for
    // 2-byte float types (half/bfloat16 - exp() underflows to 0 either way),
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

// --- ADR 0106 Phase 1: Window substrate CUDA kernels -------------------------

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
            if (w >= activeWindow) {
                // The asl=1 rerun consumes row 0 only. Rows past the active
                // window must present the width-1 scalar geometry to the frozen
                // plan (fully-masked readout is what the greedy scalar pass
                // sees), not the verification causal band: fused attention /
                // GEMM reduction order over the verification band is what
                // produced the W-vs-greedy row-0 flip at step 99 (1536 vs 5218).
                val = maskFill;
            } else if (k <= currentPos + w) {
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

// --- Argmax helper (greedy decode) -------------------------------------------

/**
 * SHARED REDUCTION CONTRACT (review round 6, finding 1/2 — one comparison
 * helper for BOTH the single-row and multi-row argmax kernels so the two
 * selectors can never diverge again):
 *
 *   1. A candidate whose index is inside [0, vocabSize) is VALID; a thread
 *      that saw no element carries the INVALID candidate (idx = vocabSize)
 *      and can never win — its synthetic value must not beat real logits
 *      (the old -1e30 init let an absent thread return vocabSize for
 *      very-negative finite rows).
 *   2. Among valid candidates: larger value wins.
 *   3. On EXACT ties: the SMALLER vocabulary index wins (CPU lowest-index
 *      contract; NaN compares false under both > and <, so an all-NaN chunk
 *      keeps its entry without propagating NaN — the per-row validity flag
 *      reports NaN content separately).
 *
 * CUDA max-reduction identity: the absent candidate uses -inf as its value
 * (NVIDIA reduction convention) plus the explicit invalid-index exclusion,
 * making the empty-thread and all--inf cases well-defined.
 */
template <typename T>
static SD_DEVICE inline bool argmaxTakeOther(T currentVal, LongType currentIdx,
                                             T otherVal, LongType otherIdx,
                                             LongType vocabSize) {
    const bool currentValid = currentIdx < vocabSize;
    const bool otherValid = otherIdx < vocabSize;
    if (!otherValid) return false;             // absent candidate never wins
    if (!currentValid) return true;            // a real candidate beats absent
    if (otherVal > currentVal) return true;    // larger value wins
    // Exact tie: smaller vocabulary index wins (CPU lowest-index contract).
    // NaN compares false under both > and ==, so an all-NaN chunk keeps its
    // entry without propagating NaN; the validity flag reports NaN content.
    return otherVal == currentVal && otherIdx < currentIdx;
}

/**
 * CUDA kernel: find argmax over a float/half row [vocabSize].
 * Writes the index to output[0] as INT64.
 *
 * Block-level reduction using shared memory. With kWriteValidity the kernel
 * additionally writes output[1] = 1 when ANY value in the FULL row is NaN
 * (device-side reduction via __syncthreads_or; dtype-portable self-inequality
 * check, so the guard covers every float dtype the kernel is instantiated
 * for - finding 5). The two-slot output is opt-in because the MTP draft
 * argmax writes into a single slot at draftSlot offset. The validity
 * tracking work is likewise opt-in (kTrackValidity), so the plain draft
 * argmax does not pay the NaN scan or the cooperative OR.
 *
 * REDUCTION CONTRACT: see the shared ArgmaxCandidate block above (round 6:
 * valid-candidate rule fixes the vocabSize escape for very-negative finite
 * rows; ties resolve to the LOWEST index for CPU parity).
 */
template <typename T, bool kWriteValidity = false>
static SD_KERNEL void argmaxKernel(const void* vLogits, void* vOutput, LongType vocabSize) {
    extern __shared__ char smem[];
    auto sMaxVal = reinterpret_cast<T*>(smem);
    auto sMaxIdx = reinterpret_cast<LongType*>(smem + blockDim.x * sizeof(T));

    auto logits = reinterpret_cast<const T*>(vLogits);
    auto output = reinterpret_cast<LongType*>(vOutput);

    T localMax;
    LongType localIdx;
    bool localNan = false;
    if (threadIdx.x < vocabSize) {
        localMax = logits[threadIdx.x];
        localIdx = threadIdx.x;
        if (kWriteValidity && localMax != localMax) localNan = true;
        for (LongType i = threadIdx.x + blockDim.x; i < vocabSize; i += blockDim.x) {
            T val = logits[i];
            if (kWriteValidity && !localNan && val != val) localNan = true;
            // Strict > keeps the LOWEST index on ties (CPU parity).
            if (val > localMax) {
                localMax = val;
                localIdx = i;
            }
        }
    } else {
        // Thread saw no element: the ABSENT candidate. -inf is the max-reduction
        // identity (NVIDIA convention) and idx = vocabSize marks it invalid so
        // it can never win the reduction (round 6, finding 2: the finite -1e30
        // init beat very-negative finite rows and returned vocabSize as a token).
        localMax = -DataTypeUtils::infOrMax<T>();
        localIdx = vocabSize;
    }
    // (Validity NaN detection is FUSED into the max scan above — round 6 perf
    // note: no second traversal of the logits row.)

    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    // Reduction: shared contract via argmaxTakeOther (round 6 — one helper for
    // both argmax kernels so the selectors cannot diverge).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (argmaxTakeOther(sMaxVal[threadIdx.x], sMaxIdx[threadIdx.x],
                                sMaxVal[threadIdx.x + stride], sMaxIdx[threadIdx.x + stride],
                                vocabSize)) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Full-row validity reduction: uniform cooperative call, no shared memory.
    unsigned blockNan = kWriteValidity
        ? static_cast<unsigned>(__syncthreads_or(localNan ? 1 : 0)) : 0;

    if (threadIdx.x == 0) {
        output[0] = sMaxIdx[0];
        if (kWriteValidity) output[1] = blockNan ? 1L : 0L;
    }
}

/**
 * Launcher for argmaxKernel - called via BUILD_SINGLE_SELECTOR.
 */
template <typename T>
static void argmaxLauncher(const cudaStream_t* stream, const void* logitsPtr,
                            void* outputPtr, LongType vocabSize) {
    int threads = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxKernel<T><<<1, threads, smemSize, *stream>>>(logitsPtr, outputPtr, vocabSize);
}

/**
 * Launcher for the two-slot validity variant (finding 5): output[0] = argmax,
 * output[1] = full-row NaN flag. Same single kernel launch and sync as the
 * plain argmax - the NaN detection adds NO extra host wait; the caller drains
 * both slots (and any diagnostics/state samples) in one readback.
 */
template <typename T>
static void argmaxValidityLauncher(const cudaStream_t* stream, const void* logitsPtr,
                                   void* outputPtr, LongType vocabSize) {
    int threads = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxKernel<T, true><<<1, threads, smemSize, *stream>>>(logitsPtr, outputPtr, vocabSize);
}

// --- ADR 0106 Phase 2: n-gram speculative decoding kernels -------------------

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
 *
 * ROUND 6 (finding 1/2): this is the SPECULATIVE VERIFIER's selector — the
 * accept rule and bonus token come from these rows. It previously kept the
 * OLD reduction (no tie rule, finite -1e30 init), so verification could
 * select a different token than the corrected single-row/CPU selectors from
 * identical logits (tied bonus row: CPU 1, multi-row 256). The kernel now
 * shares the EXACT argmaxTakeOther contract with argmaxKernel: absent
 * threads carry the invalid (-inf, vocabSize) candidate, larger value wins,
 * exact ties resolve to the smaller index. The per-row validity flags are
 * written to a parallel [numRows] INT64 buffer when validityPtr != null
 * (round 6 finding 3: active-row NaN coverage at the verifier boundary,
 * incl. the fully-accepted batch where the rerun guard never runs).
 */
template <typename T>
static SD_KERNEL void argmaxMultiRowKernel(const void* vLogits, void* vOutput,
                                            LongType numRows, LongType vocabSize,
                                            void* validityPtr) {
    extern __shared__ char smem[];
    auto sMaxVal = reinterpret_cast<T*>(smem);
    auto sMaxIdx = reinterpret_cast<LongType*>(smem + blockDim.x * sizeof(T));

    LongType row = blockIdx.x;
    if (row >= numRows) return;

    auto logits = reinterpret_cast<const T*>(vLogits) + row * vocabSize;
    auto output = reinterpret_cast<LongType*>(vOutput);
    auto validity = validityPtr != nullptr
        ? reinterpret_cast<LongType*>(validityPtr) + row : nullptr;

    T localMax;
    LongType localIdx;
    bool localNan = false;
    if (threadIdx.x < vocabSize) {
        localMax = logits[threadIdx.x];
        localIdx = threadIdx.x;
        if (validity != nullptr && localMax != localMax) localNan = true;
        for (LongType i = threadIdx.x + blockDim.x; i < vocabSize; i += blockDim.x) {
            T val = logits[i];
            if (validity != nullptr && !localNan && val != val) localNan = true;
            // Strict > keeps the LOWEST index on ties (CPU parity).
            if (val > localMax) {
                localMax = val;
                localIdx = i;
            }
        }
    } else {
        // ABSENT candidate: -inf identity + invalid index; never wins.
        localMax = -DataTypeUtils::infOrMax<T>();
        localIdx = vocabSize;
    }

    sMaxVal[threadIdx.x] = localMax;
    sMaxIdx[threadIdx.x] = localIdx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            if (argmaxTakeOther(sMaxVal[threadIdx.x], sMaxIdx[threadIdx.x],
                                sMaxVal[threadIdx.x + stride], sMaxIdx[threadIdx.x + stride],
                                vocabSize)) {
                sMaxVal[threadIdx.x] = sMaxVal[threadIdx.x + stride];
                sMaxIdx[threadIdx.x] = sMaxIdx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    // ROUND 7 (review finding 1): __syncthreads_or is a BLOCK-WIDE collective
    // - every thread in the block must execute it for the reduction to be
    // valid (NVIDIA: conditional execution must be uniform across the block).
    // The previous placement inside `if (threadIdx.x == 0)` meant only thread
    // zero participated: the result reflected one thread's predicate (the
    // all-NaN test row could not distinguish this because thread zero's own
    // elements were NaN) and the call is undefined behavior in divergent code.
    // validity is UNIFORM across the block (same pointer on every thread), so
    // the collective is evaluated unconditionally and only the STORE is
    // restricted to thread zero.
    unsigned rowNan = validity != nullptr
        ? static_cast<unsigned>(__syncthreads_or(localNan ? 1 : 0)) : 0;
    if (threadIdx.x == 0) {
        output[row] = sMaxIdx[0];
        if (validity != nullptr) {
            validity[0] = rowNan ? 1L : 0L;
        }
    }
}

/**
 * Launcher for argmaxMultiRowKernel.
 * numRows blocks, 256 threads per block with smem for (maxVal, maxIdx) pairs.
 * validityPtr: optional [numRows] INT64 buffer receiving per-row NaN flags
 * (nullptr = skip validity work entirely).
 */
template <typename T>
static void argmaxMultiRowLauncher(const cudaStream_t* stream, const void* logitsPtr,
                                   void* outputPtr, LongType numRows, LongType vocabSize,
                                   void* validityPtr) {
    if (numRows <= 0) return;
    int threads  = 256;
    int smemSize = threads * (sizeof(T) + sizeof(LongType));
    argmaxMultiRowKernel<T><<<static_cast<int>(numRows), threads, smemSize, *stream>>>(
        logitsPtr, outputPtr, numRows, vocabSize, validityPtr);
}

// --- Main Implementation -----------------------------------------------------

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

    // Validate that we have a plan to execute - hard error, not silent return.
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

    DSP_DIAG(KV_CACHE,
             "AUTOREGRESSIVE_DECODE_CUDA entered plan=%p maxNewTokens=%d prefillSeqLen=%d "
             "embExtIdx=%d maskExtIdx=%d posExtIdx=%d idsExtIdx=%d causalExtIdx=%d "
             "attnReformatExtIdx=%d numKvPairs=%d logitsOutIdx=%d",
             plan, maxNewTokens, prefillSeqLen,
             config->embeddingsExtIdx, config->maskExtIdx, config->posIdsExtIdx,
             config->inputIdsExtIdx, config->causalMaskExtIdx,
             config->attnMaskReformatExtIdx, numKvPairs, config->logitsOutputIdx);

    // -- Timing --
    std::vector<double> stepTimesMs;
    std::vector<int> stepTokenCounts;
    stepTimesMs.reserve(maxNewTokens);
    stepTokenCounts.reserve(maxNewTokens);
    auto loopStart = std::chrono::high_resolution_clock::now();

    // -- Internal state --
    LongType currentPosition = static_cast<LongType>(prefillSeqLen);
    auto hidden = embeddingTable->sizeAt(1);
    auto vocabSize = embeddingTable->sizeAt(0);
    auto embTableRowStride = embeddingTable->strideAt(0);

    // -- Build internal attention mask if not provided --
    // Shape: [1, 1, 1, maxKvLen] - single-token decode mask
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

    // -- Build internal position_ids if not provided --
    // Shape: [1, 1] - single-token decode
    NDArray* internalPosIds = nullptr;
    if (positionIds == nullptr) {
        std::vector<LongType> posShape = {1, 1};
        internalPosIds = NDArrayFactory::create('c', posShape, DataType::INT64, context);
        internalPosIds->p(0, static_cast<LongType>(prefillSeqLen));
        positionIds = internalPosIds;
    }

    // -- Working buffers --
    // Reuse prefillEmbeddings (Java's decodeEmbeddings [1,1,hidden]) for embed lookup.
    // CRITICAL: Do NOT allocate a new NDArray - the CUDA graph was captured with
    // prefillEmbeddings' device address as the embeddings ext input. Using a new
    // allocation would change the address, causing externalAddrsMatch() to fail,
    // which forces fallback to phaseReplay (broken ext input sync -> degenerate output).
    NDArray* decodeEmbedding = prefillEmbeddings;

    // Token sample output: single INT64 scalar
    std::vector<LongType> sampleShape = {1};
    NDArray* sampledToken = NDArrayFactory::create('c', sampleShape, DataType::INT64, context);

    // Logits slice: [vocabSize] - last-position logits from plan output
    // We'll point into the plan's output buffer directly when possible

    int tokensGenerated = 0;

    // -- Get plan's external inputs from the persistent OpaqueContext --
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

    // -- Extract causal mask from ext inputs (if present) --
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
    // bonus at row 0 every step - greedy (always row 0) and speculative rows >= 1
    // then compute different hidden states for the SAME token, breaking lossless
    // speculative equivalence (divergence compounds through the attention stack).
    // When the two masks share a buffer, the causal-mask maintenance owns every
    // update and the 0/1 update must not run.
    const bool attnMaskAliasesCausal = attentionMask != nullptr && causalMask != nullptr
        && attentionMask->dataBuffer() == causalMask->dataBuffer();

    // -- Extract attn_mask_reformat from ext inputs (if present) --
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
    const bool useScalarTarget = config->scalarPlanHandle != nullptr;
    std::vector<NDArray*> scalarInputs(config->scalarNumPlanExternalInputs, nullptr);
    std::vector<NDArray*> scalarOutputs(config->scalarNumPlanOutputs, nullptr);
    if (useScalarTarget) {
        auto* scalarContext = reinterpret_cast<graph::Context*>(config->scalarExtInputContext);
        // Build the reverse map (window ext idx -> scalar ext idx) once. The W plan marks
        // shared KV as variable+device-managed (lines ~1885) precisely so the plan stages
        // them WITHOUT a writable-copy migration; the scalar plan must treat shared KV,
        // weights and derived inputs the same way or FLOAT8 quantized KV triggers a
        // memcpyWithT migration that has no FLOAT8 case.
        std::vector<char> scalarIsDeviceManaged(config->scalarNumPlanExternalInputs, 0);
        for (int i = 0; i < config->scalarNumPlanExternalInputs; ++i) {
            int ti = config->scalarInputToTarget[i];
            NDArray* scalarArr = scalarContext->array(i);
            // Buffer-identical inputs (shared KV) are the same storage the W plan
            // registers as device-managed; mirror that registration on the scalar plan.
            bool shared = ti >= 0 && ti < numExtInputs && extInputs[ti] != nullptr
                          && scalarArr != nullptr
                          && scalarArr->dataBuffer() == extInputs[ti]->dataBuffer();
            for (int s = 0; s < config->numGdnStatePairs && !shared; ++s) {
                shared = config->gdnStateExtIndices != nullptr
                         && ti == config->gdnStateExtIndices[s];
            }
            for (int s = 0; s < config->numConvStatePairs && !shared; ++s) {
                shared = config->convStateExtIndices != nullptr
                         && ti == config->convStateExtIndices[s];
            }
            bool geometry = i == config->scalarInputIdsExtIdx || i == config->scalarCausalMaskExtIdx
                || i == config->scalarPositionOffsetExtIdx || i == config->scalarCachePositionExtIdx
                || i == config->scalarActualSequenceLengthExtIdx;
            if (shared || geometry) {
                scalarIsDeviceManaged[i] = 1;
                config->scalarPlanHandle->markExternalInputVariable(i);
                if (shared && ti >= 0 && ti < numExtInputs && extInputs[ti] != nullptr) {
                    config->scalarPlanHandle->registerDeviceManagedExternalInput(scalarArr);
                } else if (scalarArr != nullptr) {
                    // geometry inputs are device-written in place each step (op kernels)
                    config->scalarPlanHandle->registerDeviceManagedExternalInput(scalarArr);
                }
            }
            scalarInputs[i] = scalarArr;
        }
    }
    // Snapshot BEFORE the window forward. Private recurrent inputs protect the prefix even
    // when a verifier op mutates its input. KV rows are shared and overwritten at the same slot.
    auto prepareScalarTarget = [&]() {
        for (int i = 0; i < config->scalarNumPlanExternalInputs; ++i) {
            NDArray* dst = scalarInputs[i];
            NDArray* src = extInputs[config->scalarInputToTarget[i]];
            if (dst->dataBuffer() == src->dataBuffer()) continue;
            bool geometry = i == config->scalarInputIdsExtIdx || i == config->scalarCausalMaskExtIdx
                || i == config->scalarPositionOffsetExtIdx || i == config->scalarCachePositionExtIdx
                || i == config->scalarActualSequenceLengthExtIdx;
            // Recurrent decision in the TARGET index domain (CPU-mirror): scalar
            // input i maps to target index ti; recurrent iff ti equals a target-
            // domain GDN/conv state index. Never re-map gdn/conv indices through
            // the scalar-indexed vector.
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
            // Geometry and recurrent snapshots are refreshed from the window plan each
            // call. Shared KV is buffer-identical (skipped above). Everything else -
            // graph weights and derived plan-internal inputs - keeps the CAPTURED value:
            // weights are immutable and derived inputs are width-specific.
            if (!geometry && !recurrent) continue;
            REQUIRE_TRUE(dst->lengthOf() <= src->lengthOf(), 0,
                         "autoregressive_decode: scalar source is smaller than captured input");
            NDArray::prepareSpecialUse({dst}, {src});
            auto error = cudaMemcpyAsync(dst->specialBuffer(), src->specialBuffer(),
                dst->lengthOf() * dst->sizeOfT(), cudaMemcpyDeviceToDevice, *stream);
            REQUIRE_TRUE(error == cudaSuccess, 0, "autoregressive_decode: scalar input copy failed: %s",
                         cudaGetErrorString(error));
            NDArray::registerSpecialUse({dst}, {src});
        }
        NDArray* active = scalarInputs[config->scalarActualSequenceLengthExtIdx];
        NDArray::prepareSpecialUse({active}, {});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(active->specialBuffer(), 1);
        NDArray::registerSpecialUse({active}, {});
    };
    auto executeScalarTarget = [&]() {
        // Call-scoped invocation counter for the bounded full-scan audit below.
        static LongType scalarTargetAuditCall = 0;
        const LongType auditCall = scalarTargetAuditCall++;
        DSP_DIAG(KV_CACHE, "SCALAR_TARGET_SELECTED plan=%p idsWidth=1 maskRows=1 position=%lld inputs=%d outputs=%d",
                 config->scalarPlanHandle, static_cast<long long>(currentPosition),
                 config->scalarNumPlanExternalInputs, config->scalarNumPlanOutputs);
        // SCALAR-INPUT FINITENESS AUDIT (gates 4-6 discriminator): probe the
        // PRIVATE scalar arrays executeScalarTarget consumes. Earlier probes
        // covered only the first 8 of 550 inputs; gate-6 proved the poison is
        // NOT in shared KV rows (restore had no effect). The remaining
        // unprobed layer is the DERIVED (non-geometry, non-recurrent) inputs
        // that prepareScalarTarget deliberately does NOT refresh - if any of
        // them shares storage with a W-plan buffer the verify pass mutates,
        // the scalar rerun reads post-verify garbage. Scan ALL float inputs
        // (bounded: first 16 bytes each, names+indices reported for the first
        // NaN). Cost is bounded at ~550 small D2H probes on a FAILING run
        // only - but that is still too many per-step syncs, so: scan on the
        // step AFTER the first speculative step only (the failing boundary),
        // and stop at the first hit. Invocation counter: this lambda is built
        // before the step loop; use a call-scoped monotonic counter so the
        // scan runs only from the second scalar-target invocation onward.
        if (DSP_DIAG_ENABLED(KV_CACHE) && auditCall >= 1) {
            bool scalarInNan = false;
            int poisonIdx = -1;
            int probedCount = 0;
            for (int i = 0;
                 i < config->scalarNumPlanExternalInputs && !scalarInNan; ++i) {
                NDArray* arr = scalarInputs[i];
                if (arr == nullptr || arr->dataType() != DataType::FLOAT32
                        || arr->lengthOf() < 4) continue;
                ++probedCount;
                NDArray::prepareSpecialUse({}, {arr});
                float probe[4] = {};
                cudaMemcpyAsync(probe, arr->specialBuffer(),
                                sizeof(probe), cudaMemcpyDeviceToHost, *stream);
                cudaError_t probeSync = cudaStreamSynchronize(*stream);
                REQUIRE_TRUE(probeSync == cudaSuccess, 0,
                             "autoregressive_decode: scalar-input probe sync failed: %s",
                             cudaGetErrorString(probeSync));
                NDArray::registerSpecialUse({}, {arr});
                for (int j = 0; j < 4; ++j) {
                    if (std::isnan(probe[j])) {
                        scalarInNan = true;
                        poisonIdx = i;
                        break;
                    }
                }
            }
            DSP_DIAG(KV_CACHE,
                     "SCALAR_INPUT_AUDIT pos=%lld inNaN=%d idx=%d probed=%d total=%d",
                     static_cast<long long>(currentPosition),
                     scalarInNan ? 1 : 0, poisonIdx, probedCount,
                     config->scalarNumPlanExternalInputs);
        }
        Status status = config->scalarPlanHandle->executeSteadyState(
            scalarInputs.data(), config->scalarNumPlanExternalInputs,
            scalarOutputs.data(), config->scalarNumPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
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
            // RETRY DISCRIMINATOR: on all-NaN logits from the scalar plan, run
            // the SAME plan a second time on the SAME inputs. finite retry =>
            // stale/uninitialized internal staging (first replay consumed it,
            // second is clean); NaN retry => deterministically poisoned capture
            // (captured weights/KV pointers read garbage independent of staging
            // age). One extra execution only inside a failing diagnostic run.
            if (DSP_DIAG_ENABLED(KV_CACHE) && logits->dataType() == DataType::FLOAT32
                    && logits->lengthOf() >= 8) {
                NDArray::prepareSpecialUse({}, {logits});
                float pre[8] = {};
                cudaMemcpyAsync(pre, logits->specialBuffer(),
                                sizeof(pre), cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);
                NDArray::registerSpecialUse({}, {logits});
                bool firstNan = false;
                for (int j = 0; j < 8; j++) firstNan = firstNan || std::isnan(pre[j]);
                if (firstNan) {
                    DSP_DIAG(KV_CACHE,
                             "SCALAR_RETRY pos=%lld firstPass=NaN - re-executing "
                             "the same scalar plan on identical inputs",
                             static_cast<long long>(currentPosition));
                    Status retryStatus = config->scalarPlanHandle->executeSteadyState(
                        scalarInputs.data(), config->scalarNumPlanExternalInputs,
                        scalarOutputs.data(), config->scalarNumPlanOutputs,
                        reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
                    if (retryStatus == Status::OK) {
                        for (int i = 0; i < numPlanOutputs; ++i) {
                            planOutputs[i] = scalarOutputs[config->targetOutputToScalar[i]];
                        }
                        logits = scalarOutputs[config->scalarLogitsOutputIdx];
                        NDArray::prepareSpecialUse({}, {logits});
                        float post[8] = {};
                        cudaMemcpyAsync(post, logits->specialBuffer(),
                                        sizeof(post), cudaMemcpyDeviceToHost, *stream);
                        cudaStreamSynchronize(*stream);
                        NDArray::registerSpecialUse({}, {logits});
                        bool retryNan = false;
                        for (int j = 0; j < 8; j++) retryNan = retryNan || std::isnan(post[j]);
                        DSP_DIAG(KV_CACHE,
                                 "SCALAR_RETRY pos=%lld retry=%s first8=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]",
                                 static_cast<long long>(currentPosition),
                                 retryNan ? "NaN" : "finite",
                                 post[0], post[1], post[2], post[3],
                                 post[4], post[5], post[6], post[7]);
                    }
                }
            }
        }
        return status;
    };

    REQUIRE_TRUE(extCtx != nullptr || config->planExternalInputs != nullptr, 0,
                 "autoregressive_decode: no external input source. "
                 "Either extInputContext (OpaqueContext*) or planExternalInputs (NDArray**) "
                 "must be non-null. Both are null - cannot wire plan inputs.");

    bool stepTimingEnabled = plan->isExecutionTimingEnabled();

    // ADR 0106 Phase 1: window substrate flag.
    // When activeWindow > 1 and the pre-allocated window tensors are present,
    // we use the fixed [1,1,W_max,past+W_max] mask + [1,W_max] position grid
    // instead of the 1-wide tensors. Addresses are stable - kernels update in-place.
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
        LongType kLen = static_cast<LongType>(maxKvLen);  // past+W_max ? maxKvLen
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

    // Tier 1c: Pinned memory for D2H token readback - enables true async DMA
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
    // the speculative and scalar paths so the two are directly comparable -
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

    // -- ADR 0106 Phase 2: speculative decode state -------------------------
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
    // ROUND 6 (finding 3): per-row NaN validity flags from the verifier
    // reduction. Device buffer sized [specK+1]; host readback rides the
    // acceptance path's existing sync (no new synchronization boundary).
    // validity[r] = 1 means logits row r contains NaN somewhere.
    NDArray* specValidityDevice = nullptr;
    LongType* pinnedValidity = nullptr;
    LongType stackValidity[33] = {};
    if (useSpeculative) {
        std::vector<LongType> validityShape = {static_cast<LongType>(specK + 1)};
        specValidityDevice = NDArrayFactory::create('c', validityShape, DataType::INT64, context);
        cudaError_t validityPinErr = cudaMallocHost(&pinnedValidity, (specK + 1) * sizeof(LongType));
        if (validityPinErr != cudaSuccess) pinnedValidity = nullptr;
    }

    // Stable device buffers for target argmax rows and scalar MTP drafts.
    NDArray* specArgmaxDevice = nullptr;
    NDArray* mtpDraftDevice = nullptr;
    // Stage 2: dedicated rerun-refresh scratch (one INT64 slot). NEVER aliases
    // specArgmaxDevice, whose committed token sequence the predictor repair
    // loop and pending-input publication consume.
    NDArray* mtpRerunScratch = nullptr;
    // Pre-verification recurrent-snapshot storage for accepted-prefix state reruns
    // when NO scalar binding exists (bindingless window models): without this,
    // a window-geometry rerun executes from whatever post-verification state the
    // live ext inputs hold - the exact double-advance NaN mechanism fixed for the
    // scalar rerun. Snapshots are refreshed pre-verify and restored pre-rerun.
    std::vector<NDArray*> unboundStateSnapshots;
    std::vector<int> unboundStateSnapshotExtIdx;
    // DEEP PRE-VERIFICATION RECURRENT SNAPSHOTS (scalar-binding aliasing fix,
    // mtp-fix-gate2 NaN GUARD step=1 geometry=scalar-width-1): prepareScalarTarget's
    // recurrent copy is skipped for every scalar input whose DataBuffer is identical to
    // the target window ext input (shared-buffer "private" arrays), so with such a
    // binding there was NO private snapshot at all - the verify pass mutated the live
    // window state in place, the rerun restore skipped the same buffer, and the scalar
    // rerun double-advanced through the rejected draft rows into NaN. These dedicated
    // owned scratch arrays are captured D2D on the decode stream immediately BEFORE
    // each verification execution (never read from at capture time, so buffer
    // identity cannot disable them) and are the single restore source for BOTH rerun
    // geometries. Slot layout matches the bindingless arrays above: [0, numGdnStatePairs)
    // are GDN pairs, [numGdnStatePairs, +numConvStatePairs) are conv pairs, each paired
    // with its TARGET-domain ext input index. Allocated lazily once per decode call and
    // freed in the cleanup section with the other speculative resources.
    std::vector<NDArray*> stateSnapshotArrays;
    std::vector<int> stateSnapshotExtIdx;
    // SHARED-KV ROW SNAPSHOTS (verdict-c fix, gate-5 SCALAR_RETRY evidence):
    // the static KV buffers are SHARED between the window and scalar plans
    // (in-graph KV writes), so the W-wide verification pass writes rows
    // [base, base+K) with draft-conditioned K/V before any scalar rerun. The
    // captured scalar graph reads those SAME rows deterministically - a retry
    // on identical inputs stays NaN (gate-5: SCALAR_RETRY retry=NaN), because
    // the poison lives in the shared buffer rows, not in any ext input
    // (RERUN_INPUT_AUDIT extNaN=0) nor the scalar staging arrays
    // (SCALAR_INPUT_AUDIT inNaN=0). Snapshot rows [base, base+K) of every
    // static KV buffer immediately before the verification pass and restore
    // them before the rerun; the rerun then reads exactly the rows the W=1
    // scalar pass wrote - the greedy-identical state it must consume. One
    // snapshot buffer per KV pair per step, allocated lazily, freed in the
    // cleanup section with the other speculative resources.
    std::vector<NDArray*> kvRowSnapshots;
    std::vector<LongType> kvRowSnapshotRows;  // snapshot rows per KV buffer
    std::vector<NDArray*> kvRowSnapshotSources;  // source static KV buffer
    LongType kvRowSnapshotBase = -1;              // base row captured
    // Capture the complete verification write range of every static KV buffer.
    // Called immediately before the verification execution.
    auto capturePreVerificationKvRows = [&](LongType base, int rows) {
        if (!config->planOwnsKvScatter || rows <= 0
                || config->kvInputExtIndices == nullptr || numKvPairs <= 0) return;
        kvRowSnapshotBase = base;
        // Indices contain all keys followed by all values. Both halves must
        // roll back before re-executing an accepted prefix.
        const int numKvBuffers = 2 * numKvPairs;
        kvRowSnapshotRows.assign(numKvBuffers, rows);
        for (int kv = 0; kv < numKvBuffers; kv++) {
            int extIdx = config->kvInputExtIndices[kv];
            NDArray* src = (extIdx >= 0 && extIdx < numExtInputs)
                ? extInputs[extIdx] : nullptr;
            if (src == nullptr || src->rankOf() != 4) continue;
            const LongType kvSeq = src->sizeAt(1);
            const LongType heads = src->sizeAt(2);
            const LongType dim = src->sizeAt(3);
            if (base < 0 || base + rows > kvSeq) continue;
            if (static_cast<int>(kvRowSnapshots.size()) <= kv) {
                kvRowSnapshots.resize(kv + 1, nullptr);
                kvRowSnapshotSources.resize(kv + 1, nullptr);
            }
            if (kvRowSnapshots[kv] == nullptr
                    || kvRowSnapshots[kv]->dataType() != src->dataType()
                    || kvRowSnapshots[kv]->lengthOf() != rows * heads * dim) {
                delete kvRowSnapshots[kv];
                std::vector<LongType> snapShape{1, rows, heads, dim};
                kvRowSnapshots[kv] = NDArrayFactory::create(
                    'c', snapShape, src->dataType(), context);
            }
            kvRowSnapshotSources[kv] = src;
            NDArray* snap = kvRowSnapshots[kv];
            const size_t rowBytes = static_cast<size_t>(heads) * dim * src->sizeOfT();
            const char* srcBase = static_cast<const char*>(src->specialBuffer())
                                  + static_cast<size_t>(base) * rowBytes;
            NDArray::prepareSpecialUse({snap}, {src});
            cudaMemcpyAsync(snap->specialBuffer(), srcBase,
                            rowBytes * rows, cudaMemcpyDeviceToDevice, *stream);
            p0.snapshotBytes += static_cast<std::uint64_t>(rowBytes * rows);
            NDArray::registerSpecialUse({snap}, {src});
        }
    };
    // Restore the captured KV rows back into the shared static buffers.
    auto restorePreVerificationKvRows = [&]() {
        if (kvRowSnapshotBase < 0) return;
        for (size_t kv = 0; kv < kvRowSnapshots.size(); ++kv) {
            NDArray* snap = kvRowSnapshots[kv];
            NDArray* dst = (kv < kvRowSnapshotSources.size())
                ? kvRowSnapshotSources[kv] : nullptr;
            if (snap == nullptr || dst == nullptr) continue;
            const int rows = static_cast<int>(kvRowSnapshotRows[kv]);
            const LongType heads = snap->sizeAt(2);
            const LongType dim = snap->sizeAt(3);
            const size_t rowBytes = static_cast<size_t>(heads) * dim * dst->sizeOfT();
            char* dstBase = static_cast<char*>(dst->specialBuffer())
                            + static_cast<size_t>(kvRowSnapshotBase) * rowBytes;
            NDArray::prepareSpecialUse({dst}, {snap});
            auto restoreErr = cudaMemcpyAsync(dstBase, snap->specialBuffer(),
                rowBytes * rows, cudaMemcpyDeviceToDevice, *stream);
            p0.restoreBytes += static_cast<std::uint64_t>(rowBytes * rows);
            REQUIRE_TRUE(restoreErr == cudaSuccess, 0,
                "autoregressive_decode: shared-KV row restore failed: %s",
                cudaGetErrorString(restoreErr));
            NDArray::registerSpecialUse({dst}, {snap});
        }
        // Keep the pre-verification snapshot valid for another rollback in
        // this transaction (e.g. a disagreeing rerun shortened to one row).
        // The next committed step retires it before capturing a new snapshot.
    };
    // Capture every recurrent state ext input (GDN + conv pairs) into the dedicated
    // owned snapshot arrays. Called immediately before the plan execution that may
    // mutate the live ext inputs (the verification pass), so the snapshot is genuinely
    // PRE-verification regardless of whether the scalar plan's "private" arrays share
    // buffers with the window ext inputs.
    auto capturePreVerificationState = [&]() {
        for (int s = 0; s < config->numGdnStatePairs; s++) {
            int extIdx = config->gdnStateExtIndices != nullptr
                ? config->gdnStateExtIndices[s] : -1;
            NDArray* src = (extIdx >= 0 && extIdx < numExtInputs) ? extInputs[extIdx] : nullptr;
            if (src == nullptr) continue;
            if (static_cast<int>(stateSnapshotArrays.size()) <= s) {
                stateSnapshotArrays.resize(s + 1, nullptr);
                stateSnapshotExtIdx.resize(s + 1, -1);
            }
            if (stateSnapshotArrays[s] == nullptr
                    || stateSnapshotArrays[s]->dataType() != src->dataType()
                    || stateSnapshotArrays[s]->lengthOf() != src->lengthOf()) {
                // Own allocation - never aliases the live ext input, so the snapshot
                // survives any in-place mutation the plan applies to ext inputs.
                delete stateSnapshotArrays[s];
                std::vector<LongType> snapShape;
                snapShape.reserve(src->rankOf());
                for (int d = 0; d < src->rankOf(); d++) snapShape.push_back(src->sizeAt(d));
                stateSnapshotArrays[s] = NDArrayFactory::create(
                    'c', snapShape, src->dataType(), context);
                stateSnapshotExtIdx[s] = extIdx;
            }
            NDArray* snap = stateSnapshotArrays[s];
            NDArray::prepareSpecialUse({snap}, {src});
            cudaMemcpyAsync(snap->specialBuffer(), src->specialBuffer(),
                            src->lengthOf() * src->sizeOfT(),
                            cudaMemcpyDeviceToDevice, *stream);
            p0.snapshotBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            NDArray::registerSpecialUse({snap}, {src});
        }
        for (int s = 0; s < config->numConvStatePairs; s++) {
            int extIdx = config->convStateExtIndices != nullptr
                ? config->convStateExtIndices[s] : -1;
            int slot = config->numGdnStatePairs + s;
            NDArray* src = (extIdx >= 0 && extIdx < numExtInputs) ? extInputs[extIdx] : nullptr;
            if (src == nullptr) continue;
            if (static_cast<int>(stateSnapshotArrays.size()) <= slot) {
                stateSnapshotArrays.resize(slot + 1, nullptr);
                stateSnapshotExtIdx.resize(slot + 1, -1);
            }
            if (stateSnapshotArrays[slot] == nullptr
                    || stateSnapshotArrays[slot]->dataType() != src->dataType()
                    || stateSnapshotArrays[slot]->lengthOf() != src->lengthOf()) {
                delete stateSnapshotArrays[slot];
                std::vector<LongType> snapShape;
                snapShape.reserve(src->rankOf());
                for (int d = 0; d < src->rankOf(); d++) snapShape.push_back(src->sizeAt(d));
                stateSnapshotArrays[slot] = NDArrayFactory::create(
                    'c', snapShape, src->dataType(), context);
                stateSnapshotExtIdx[slot] = extIdx;
            }
            NDArray* snap = stateSnapshotArrays[slot];
            NDArray::prepareSpecialUse({snap}, {src});
            cudaMemcpyAsync(snap->specialBuffer(), src->specialBuffer(),
                            src->lengthOf() * src->sizeOfT(),
                            cudaMemcpyDeviceToDevice, *stream);
            p0.snapshotBytes += static_cast<std::uint64_t>(src->lengthOf() * src->sizeOfT());
            NDArray::registerSpecialUse({snap}, {src});
        }
    };
    // Restore the deep pre-verification snapshots into the LIVE window ext inputs the
    // window plan reads (both rerun geometries read this storage: the window plan
    // directly, and the scalar plan indirectly - prepareScalarTarget, re-run by the
    // caller after this restore, re-stages its width-1 arrays FROM the live ext
    // inputs). Cost: 23 GDN pairs * state size + conv pairs, a per-step D2D copy that
    // mirrors the existing window-restore machinery.
    auto restorePreVerificationState = [&]() {
        for (size_t s = 0; s < stateSnapshotArrays.size(); ++s) {
            NDArray* snap = stateSnapshotArrays[s];
            int ti = stateSnapshotExtIdx[s];
            NDArray* windowArr = (ti >= 0 && ti < numExtInputs) ? extInputs[ti] : nullptr;
            if (snap == nullptr || windowArr == nullptr
                    || snap->dataType() != windowArr->dataType()
                    || snap->lengthOf() != windowArr->lengthOf()) continue;
            NDArray::prepareSpecialUse({windowArr}, {snap});
            auto restoreErr = cudaMemcpyAsync(windowArr->specialBuffer(),
                snap->specialBuffer(), snap->lengthOf() * snap->sizeOfT(),
                cudaMemcpyDeviceToDevice, *stream);
            p0.restoreBytes += static_cast<std::uint64_t>(snap->lengthOf() * snap->sizeOfT());
            REQUIRE_TRUE(restoreErr == cudaSuccess, 0,
                "autoregressive_decode: recurrent state restore failed: %s",
                cudaGetErrorString(restoreErr));
            NDArray::registerSpecialUse({windowArr}, {snap});
        }
    };
    // Allocate speculative scratch whenever speculation OR the K=0
    // maintenance path can run. Gate-8 evidence: with effK=0 (adaptive drop)
    // the epilogue's predictor maintenance forward (draftSlot=0,
    // writeTargetRow=false) still needs mtpDraftDevice/specArgmaxDevice, but
    // the old useSpeculative-only gate skipped allocation -> REQUIRE_TRUE
    // crash on the first scalar-only step. MTP metadata presence (plan +
    // context) is the correct allocation condition; specK=0 yields a 1-slot
    // shape, exactly what the maintenance forward writes.
    const bool mtpMetadataReady0 = config->mtpPlanHandle != nullptr
                                   && config->mtpExtInputContext != nullptr;
    if (useSpeculative || mtpMetadataReady0) {
        std::vector<LongType> argmaxShape = {static_cast<LongType>(specK + 1)};
        specArgmaxDevice = NDArrayFactory::create('c', argmaxShape, DataType::INT64, context);
        if (useMtp || (mtpMetadataReady0 && config->speculatorType == 2)) {
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

    // -- Qwen3.5 bundled MTP predictor plan --------------------------------
    // MTP METADATA vs MTP DRAFTING (gate-9 SIGSEGV root fix): the maintenance
    // forward in the epilogue (executeMtpCuda with draftSlot=0/writeTargetRow=
    // false) must run under effK=0 too - it keeps the predictor KV hole-free
    // across scalar-only stretches so K re-raise resumes with a complete
    // attention context. Therefore the predictor RESOURCES (plan, context,
    // ext-input wiring) are gated on metadata presence, NOT on useMtp (which
    // additionally requires specK>0). Gate-9 evidence: with the wiring gated
    // on useMtp, effK=0 left mtpPlan==null and the maintenance forward
    // dereferenced null -> dumpPlanPhaseState SIGSEGV (si_addr=0x2b0).
    const bool mtpMetadataReady = config->mtpPlanHandle != nullptr
                                  && config->mtpExtInputContext != nullptr;
    graph::NativeDynamicShapePlan* mtpPlan = (useMtp || mtpMetadataReady)
                                                 ? config->mtpPlanHandle : nullptr;
    auto* mtpContext = (useMtp || mtpMetadataReady)
        ? reinterpret_cast<graph::Context*>(config->mtpExtInputContext) : nullptr;
    std::vector<NDArray*> mtpExtInputsVec;
    std::vector<NDArray*> mtpPlanOutputsVec;
    std::vector<NDArray*> mtpRepairExtInputsVec;
    std::vector<NDArray*> mtpRepairPlanOutputsVec;
    std::vector<NDArray*> mtpRepairBatchExtInputsVec;
    std::vector<NDArray*> mtpRepairBatchPlanOutputsVec;
    NDArray** mtpExtInputs = nullptr;
    NDArray** mtpPlanOutputs = nullptr;
    NDArray** mtpRepairExtInputs = nullptr;
    NDArray** mtpRepairPlanOutputs = nullptr;
    NDArray** mtpRepairBatchExtInputs = nullptr;
    NDArray** mtpRepairBatchPlanOutputs = nullptr;
    int mtpNumExtInputs = 0;
    int mtpNumOutputs = 0;
    LongType mtpMaskLen = 0;

    if (mtpPlan != nullptr && mtpContext != nullptr) {
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
        // MTP carry arrays are mutated in-place between chained predictor calls
        // (writeMtpCarry / setMtpTargetCarryCuda / setMtpNextInputCuda). They
        // deliberately stay on the GENERIC VARIABLE path (staging + per-call D2D
        // refresh): the captured predictor graph reads the plan's STAGING buffers,
        // and ensureAndSyncStagingBuffers copies live -> staging before every
        // replay. Registering them as device-managed made the passthrough skip
        // that refresh (baked-addr identity recorded the live address at capture,
        // so no drift was ever detected) while the graph kept reading staging -
        // staging held warmup-era garbage forever (proven by dspt staging-vs-live
        // diff: staging carry byte-identical across all calls while live carry
        // advanced every step), which was the true acceptance-collapse mechanism.
        // KV buffers DO use device-managed passthrough: the graph bakes their
        // addresses and attention writes rows in place (no per-call copy needed).
    }

    const bool mtpRepairReady = config->mtpRepairPlanHandle != nullptr
        && config->mtpRepairExtInputContext != nullptr;
    if (mtpRepairReady) {
        auto* repairContext = reinterpret_cast<graph::Context*>(config->mtpRepairExtInputContext);
        REQUIRE_TRUE(config->mtpRepairNumPlanExternalInputs > 0
                         && config->mtpRepairNumPlanOutputs > 0,
                     0, "autoregressive_decode: invalid MTP repair plan dimensions");
        mtpRepairExtInputsVec.resize(config->mtpRepairNumPlanExternalInputs);
        for (int i = 0; i < config->mtpRepairNumPlanExternalInputs; i++) {
            mtpRepairExtInputsVec[i] = repairContext->array(i);
        }
        auto setRepairInput = [&](int idx, NDArray* array) {
            if (idx >= 0 && idx < static_cast<int>(mtpRepairExtInputsVec.size()))
                mtpRepairExtInputsVec[idx] = array;
        };
        setRepairInput(config->mtpRepairInputIdsExtIdx, config->mtpInputIds);
        setRepairInput(config->mtpRepairTargetHiddenExtIdx, config->mtpTargetHidden);
        setRepairInput(config->mtpRepairCausalMaskExtIdx, config->mtpCausalMask);
        setRepairInput(config->mtpRepairPositionOffsetExtIdx, config->mtpPositionOffset);
        setRepairInput(config->mtpRepairCachePositionExtIdx, config->mtpCachePosition);
        setRepairInput(config->mtpRepairKvInputExtIndices[0], config->mtpKvBuffers[0]);
        setRepairInput(config->mtpRepairKvInputExtIndices[1], config->mtpKvBuffers[1]);
        mtpRepairExtInputs = mtpRepairExtInputsVec.data();
        mtpRepairPlanOutputsVec.resize(config->mtpRepairNumPlanOutputs, nullptr);
        mtpRepairPlanOutputs = mtpRepairPlanOutputsVec.data();
        auto markRepairVariable = [&](int idx) {
            if (idx >= 0) config->mtpRepairPlanHandle->markExternalInputVariable(idx);
        };
        markRepairVariable(config->mtpRepairInputIdsExtIdx);
        markRepairVariable(config->mtpRepairTargetHiddenExtIdx);
        markRepairVariable(config->mtpRepairCausalMaskExtIdx);
        markRepairVariable(config->mtpRepairPositionOffsetExtIdx);
        markRepairVariable(config->mtpRepairCachePositionExtIdx);
        markRepairVariable(config->mtpRepairKvInputExtIndices[0]);
        markRepairVariable(config->mtpRepairKvInputExtIndices[1]);
    }

    const bool mtpRepairBatchReady = config->mtpRepairBatchPlanHandle != nullptr
        && config->mtpRepairBatchExtInputContext != nullptr
        && config->mtpRepairBatchWidth > 0;
    if (mtpRepairBatchReady) {
        auto* batchRepairContext = reinterpret_cast<graph::Context*>(config->mtpRepairBatchExtInputContext);
        REQUIRE_TRUE(config->mtpRepairBatchNumPlanExternalInputs > 0
                         && config->mtpRepairBatchNumPlanOutputs > 0,
                     0, "autoregressive_decode: invalid batched MTP repair plan dimensions");
        REQUIRE_TRUE(config->mtpRepairBatchInputIds != nullptr
                         && config->mtpRepairBatchTargetHidden != nullptr
                         && config->mtpRepairBatchCausalMask != nullptr
                         && config->mtpRepairBatchPositionOffset != nullptr
                         && config->mtpRepairBatchCachePosition != nullptr,
                     0, "autoregressive_decode: batched MTP repair arrays are null");
        REQUIRE_TRUE(config->mtpRepairBatchInputIds->rankOf() == 2
                         && config->mtpRepairBatchInputIds->sizeAt(0) == 1
                         && config->mtpRepairBatchInputIds->sizeAt(1) == config->mtpRepairBatchWidth
                         && config->mtpRepairBatchInputIds->dataType() == DataType::INT64
                         && config->mtpRepairBatchTargetHidden->rankOf() == 3
                         && config->mtpRepairBatchTargetHidden->sizeAt(0) == 1
                         && config->mtpRepairBatchTargetHidden->sizeAt(1) == config->mtpRepairBatchWidth
                         && config->mtpRepairBatchCausalMask->rankOf() == 4
                         && config->mtpRepairBatchCausalMask->sizeAt(0) == 1
                         && config->mtpRepairBatchCausalMask->sizeAt(1) == 1
                         && config->mtpRepairBatchCausalMask->sizeAt(2) == config->mtpRepairBatchWidth
                         && config->mtpRepairBatchCausalMask->sizeAt(3) > 0
                         && config->mtpRepairBatchPositionOffset->lengthOf() == 1
                         && config->mtpRepairBatchCachePosition->lengthOf() == 1,
                     0, "autoregressive_decode: invalid batched MTP repair array geometry");
        mtpRepairBatchExtInputsVec.resize(config->mtpRepairBatchNumPlanExternalInputs);
        for (int i = 0; i < config->mtpRepairBatchNumPlanExternalInputs; i++) {
            mtpRepairBatchExtInputsVec[i] = batchRepairContext->array(i);
        }
        auto setBatchRepairInput = [&](int idx, NDArray* array) {
            if (idx >= 0 && idx < static_cast<int>(mtpRepairBatchExtInputsVec.size()))
                mtpRepairBatchExtInputsVec[idx] = array;
        };
        setBatchRepairInput(config->mtpRepairBatchInputIdsExtIdx, config->mtpRepairBatchInputIds);
        setBatchRepairInput(config->mtpRepairBatchTargetHiddenExtIdx, config->mtpRepairBatchTargetHidden);
        setBatchRepairInput(config->mtpRepairBatchCausalMaskExtIdx, config->mtpRepairBatchCausalMask);
        setBatchRepairInput(config->mtpRepairBatchPositionOffsetExtIdx, config->mtpRepairBatchPositionOffset);
        setBatchRepairInput(config->mtpRepairBatchCachePositionExtIdx, config->mtpRepairBatchCachePosition);
        setBatchRepairInput(config->mtpRepairBatchKvInputExtIndices[0], config->mtpKvBuffers[0]);
        setBatchRepairInput(config->mtpRepairBatchKvInputExtIndices[1], config->mtpKvBuffers[1]);
        mtpRepairBatchExtInputs = mtpRepairBatchExtInputsVec.data();
        mtpRepairBatchPlanOutputsVec.resize(config->mtpRepairBatchNumPlanOutputs, nullptr);
        mtpRepairBatchPlanOutputs = mtpRepairBatchPlanOutputsVec.data();
        auto markBatchRepairVariable = [&](int idx) {
            if (idx >= 0) config->mtpRepairBatchPlanHandle->markExternalInputVariable(idx);
        };
        markBatchRepairVariable(config->mtpRepairBatchInputIdsExtIdx);
        markBatchRepairVariable(config->mtpRepairBatchTargetHiddenExtIdx);
        markBatchRepairVariable(config->mtpRepairBatchCausalMaskExtIdx);
        markBatchRepairVariable(config->mtpRepairBatchPositionOffsetExtIdx);
        markBatchRepairVariable(config->mtpRepairBatchCachePositionExtIdx);
        markBatchRepairVariable(config->mtpRepairBatchKvInputExtIndices[0]);
        markBatchRepairVariable(config->mtpRepairBatchKvInputExtIndices[1]);
        config->mtpRepairBatchPlanHandle->registerDeviceManagedExternalInput(config->mtpKvBuffers[0]);
        config->mtpRepairBatchPlanHandle->registerDeviceManagedExternalInput(config->mtpKvBuffers[1]);
    }

    // KV_CACHE-gated chain probe: per chain exec, sample the carry-in hidden,
    // input token, and hidden-out (async D2H on the exec stream, drained by the
    // acceptance path's existing sync - no new sync points). Diagnoses whether
    // the draft chain's hidden/token carry is visible to the predictor plan.
    // Raw samples preserve HALF/BFLOAT16/FLOAT/DOUBLE storage bits equally.
    uint64_t mtpChainCarryIn[33][2] = {};
    uint64_t mtpChainHidOut[33][2] = {};
    size_t mtpChainSampleBytes[33] = {};
    LongType mtpChainTok[33] = {};
    int mtpChainSampled = 0;

    // KV self-row write-visibility probe (loop scope so the speculative accept
    // block drains it): slot-0 arm records the position; the accept block re-reads
    // the same K row after executeSteadyState and compares.
    LongType kvSelfRowAfterPos = -1;
    NDArray* kvSelfRowAfterBuf = nullptr;

    // Adaptive MTP chain-depth cap. Recursive drafting feeds the predictor its
    // OWN output hidden - out-of-distribution for heads trained only on trunk
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

    // Explicit artifact request only: no ALL/KV_CACHE/debug activation. One
    // process-wide bounded owner, independent of plan epochs and chain slots.
    auto& tensorDiagnostics = graph::DspDiagnostics::getInstance();
    const bool captureMtpInputs = useMtp && tensorDiagnostics.tensorSnapshotRequested()
        && tensorDiagnostics.beginTensorSnapshot("executeMtpCuda/pre-executeSteadyState",
                                                  reinterpret_cast<void*>(*stream));
    int mtpSnapshotCall = 0;

    // -- Write carry to the live array -------------------------------------
    // The MTP carry arrays are registered as device-managed, so the predictor
    // plan's CUDA graph reads them directly (no staging indirection). Writing
    // to the live array on the decode stream is sufficient: the graph replay
    // is stream-ordered after the write.
    auto writeMtpCarry = [&](NDArray* liveArray, int extIdx,
                              const void* src, size_t bytes) {
        NDArray::prepareSpecialUse({liveArray}, {});
        cudaMemcpyAsync(liveArray->specialBuffer(), src, bytes,
                        cudaMemcpyDeviceToDevice, *stream);
        NDArray::registerSpecialUse({liveArray}, {});
    };

    // Execute the optional KV-only repair plan and scatter its BSHD outputs into
    // the predictor cache. This path intentionally has no logits/hidden output
    // publication, so it cannot execute the predictor LM head or recursive carry.
    auto executeMtpRepairCuda = [&](LongType targetTokenPosition) {
        REQUIRE_TRUE(mtpRepairReady, 0,
                     "autoregressive_decode: MTP KV-only repair plan is unavailable");
        const LongType predictorRow = targetTokenPosition - 1;
        REQUIRE_TRUE(targetTokenPosition >= 1 && predictorRow >= 0, 0,
                     "autoregressive_decode: invalid KV-only repair target position");
        NDArray::prepareSpecialUse({config->mtpPositionOffset, config->mtpCachePosition}, {});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpPositionOffset->specialBuffer(), predictorRow);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpCachePosition->specialBuffer(), predictorRow);
        if (config->mtpRepairCausalMaskExtIdx >= 0) {
            NDArray::prepareSpecialUse({config->mtpCausalMask}, {});
            BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskLauncher,
                                  (stream, config->mtpCausalMask->specialBuffer(),
                                   predictorRow, mtpMaskLen), SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({config->mtpCausalMask}, {});
        }
        NDArray::registerSpecialUse({config->mtpPositionOffset, config->mtpCachePosition}, {});
        // The retained carry/token writes above run on the decode caller stream,
        // while the independent repair plan may replay on the DSP execution
        // stream. Preserve the same device-side happens-before contract as the
        // full predictor path; without it, the repair plan can consume stale
        // staging/carry data only when host timing is fast enough to expose it.
        void* repairCarryEvent = sd::graph::dspCreateEvent();
        REQUIRE_TRUE(repairCarryEvent != nullptr, 0,
                     "autoregressive_decode: failed to create CUDA MTP repair carry event");
        sd::graph::dspEventRecord(repairCarryEvent, *stream);
        void* repairDspStream = sd::graph::dspGetExecutionStream();
        if (repairDspStream != nullptr
                && repairDspStream != static_cast<void*>(*stream)) {
            sd::graph::dspStreamWaitEvent(repairDspStream, repairCarryEvent);
        }
        sd::graph::dspDestroyEvent(repairCarryEvent);
        const auto repairPhaseBefore = config->mtpRepairPlanHandle->getPlanPhase();
        Status repairStatus = config->mtpRepairPlanHandle->executeSteadyState(
            mtpRepairExtInputs, config->mtpRepairNumPlanExternalInputs,
            mtpRepairPlanOutputs, config->mtpRepairNumPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
        const auto repairPhaseAfter = config->mtpRepairPlanHandle->getPlanPhase();
        if (repairPhaseBefore != repairPhaseAfter) p0.planPhaseTransitions++;
        if (repairPhaseAfter == graph::PlanPhase::REPLAYING) p0.planReplayForwards++;
        else p0.planWarmupForwards++;
        REQUIRE_TRUE(repairStatus == Status::OK, 0,
                     "autoregressive_decode: KV-only repair plan failed at row %lld",
                     (long long)predictorRow);
        REQUIRE_TRUE(config->mtpRepairKeyOutputIdx >= 0
                         && config->mtpRepairKeyOutputIdx < config->mtpRepairNumPlanOutputs
                         && config->mtpRepairValueOutputIdx >= 0
                         && config->mtpRepairValueOutputIdx < config->mtpRepairNumPlanOutputs,
                     0, "autoregressive_decode: KV-only repair output indices are invalid");
        NDArray* key = mtpRepairPlanOutputs[config->mtpRepairKeyOutputIdx];
        NDArray* value = mtpRepairPlanOutputs[config->mtpRepairValueOutputIdx];
        REQUIRE_TRUE(key != nullptr && value != nullptr && key->rankOf() == 4 && value->rankOf() == 4,
                     0, "autoregressive_decode: KV-only repair outputs are invalid");
        NDArray::prepareSpecialUse({config->mtpKvBuffers[0], config->mtpKvBuffers[1]}, {key, value});
        ops::helpers::kvInPlaceWriteBSHD(config->mtpKvBuffers[0], key,
                                         config->mtpCachePosition->specialBuffer(), context);
        ops::helpers::kvInPlaceWriteBSHD(config->mtpKvBuffers[1], value,
                                         config->mtpCachePosition->specialBuffer(), context);
        NDArray::registerSpecialUse({config->mtpKvBuffers[0], config->mtpKvBuffers[1]}, {key, value});
        // The scatter is queued on the caller stream, while the next predictor
        // replay may use the DSP execution stream. Publish the repaired cache
        // row across that boundary before returning to the decode loop.
        void* repairScatterEvent = sd::graph::dspCreateEvent();
        REQUIRE_TRUE(repairScatterEvent != nullptr, 0,
                     "autoregressive_decode: failed to create CUDA MTP repair scatter event");
        sd::graph::dspEventRecord(repairScatterEvent, *stream);
        void* nextDspStream = sd::graph::dspGetExecutionStream();
        if (nextDspStream != nullptr
                && nextDspStream != static_cast<void*>(*stream)) {
            sd::graph::dspStreamWaitEvent(nextDspStream, repairScatterEvent);
        }
        sd::graph::dspDestroyEvent(repairScatterEvent);
        p0.predictorRepairForwards++;
    };

    // Execute the fixed-width B=1 repair plan once for the contiguous accepted
    // prefix. The five stable arrays are overwritten in place; inactive rows
    // remain masked and are never scattered into the predictor cache.
    auto executeMtpRepairBatchCuda = [&](int activeRows, LongType predictorBase) {
        REQUIRE_TRUE(mtpRepairBatchReady, 0,
                     "autoregressive_decode: batched MTP KV-only repair plan is unavailable");
        REQUIRE_TRUE(activeRows >= 1 && activeRows <= config->mtpRepairBatchWidth,
                     0, "autoregressive_decode: invalid active batched repair rows %d/%d",
                     activeRows, config->mtpRepairBatchWidth);
        REQUIRE_TRUE(predictorBase >= 0, 0,
                     "autoregressive_decode: invalid batched repair predictor base %lld",
                     (long long)predictorBase);
        NDArray* ids = config->mtpRepairBatchInputIds;
        NDArray* hidden = config->mtpRepairBatchTargetHidden;
        NDArray* mask = config->mtpRepairBatchCausalMask;
        NDArray* position = config->mtpRepairBatchPositionOffset;
        NDArray* cachePosition = config->mtpRepairBatchCachePosition;
        REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                         && config->targetHiddenOutputIdx < numPlanOutputs,
                     0, "autoregressive_decode: batched repair target hidden output index is invalid");
        NDArray* targetHiddenRows = planOutputs[config->targetHiddenOutputIdx];
        REQUIRE_TRUE(targetHiddenRows != nullptr && targetHiddenRows->rankOf() == 3
                         && targetHiddenRows->sizeAt(0) == 1
                         && targetHiddenRows->sizeAt(1) >= activeRows
                         && targetHiddenRows->sizeAt(2) == hidden->sizeAt(2)
                         && targetHiddenRows->dataType() == hidden->dataType()
                         && targetHiddenRows->ordering() == 'c'
                         && shape::strideDescendingCAscendingF(targetHiddenRows->shapeInfo()),
                     0, "autoregressive_decode: batched repair target hidden rows are not contiguous [1,W,H]");
        REQUIRE_TRUE(ids->ordering() == 'c' && shape::strideDescendingCAscendingF(ids->shapeInfo())
                         && hidden->ordering() == 'c' && shape::strideDescendingCAscendingF(hidden->shapeInfo()),
                     0, "autoregressive_decode: batched repair inputs must use contiguous C layout");
        const LongType maskLen = mask->sizeAt(3);
        // Only the active prefix is written and scattered; inactive rows are
        // masked finite and their outputs are ignored, so requiring the whole
        // physical width inside the mask would reject valid near-capacity
        // repairs. The scatter capacity check below covers the active rows.
        REQUIRE_TRUE(predictorBase + activeRows <= maskLen,
                     0, "autoregressive_decode: batched repair active prefix exceeds mask capacity");
        REQUIRE_TRUE(config->mtpKvBuffers[0] != nullptr && config->mtpKvBuffers[1] != nullptr
                         && config->mtpKvBuffers[0]->rankOf() == 4
                         && config->mtpKvBuffers[1]->rankOf() == 4
                         && config->mtpKvBuffers[0]->sizeAt(0) == 1
                         && config->mtpKvBuffers[1]->sizeAt(0) == 1
                         && config->mtpKvBuffers[0]->sizeAt(2) == config->mtpKvBuffers[1]->sizeAt(2)
                         && config->mtpKvBuffers[0]->sizeAt(3) == config->mtpKvBuffers[1]->sizeAt(3)
                         && config->mtpKvBuffers[0]->dataType() == config->mtpKvBuffers[1]->dataType(),
                     0, "autoregressive_decode: batched repair KV caches must be matching BSHD arrays");

        const LongType hiddenRowBytes = hidden->sizeAt(2) * hidden->sizeOfT();
        // The graph may prune its mask input. Do not refill or publish a
        // fabricated mask write when no repair-plan consumer exists.
        NDArray::prepareSpecialUse({ids, hidden, position, cachePosition},
                                   {specArgmaxDevice, targetHiddenRows});
        cudaMemcpyAsync(ids->specialBuffer(), specArgmaxDevice->specialBuffer(),
                        static_cast<size_t>(activeRows) * sizeof(LongType),
                        cudaMemcpyDeviceToDevice, *stream);
        cudaMemcpyAsync(hidden->specialBuffer(), targetHiddenRows->specialBuffer(),
                        static_cast<size_t>(activeRows) * hiddenRowBytes,
                        cudaMemcpyDeviceToDevice, *stream);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(position->specialBuffer(), predictorBase);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(cachePosition->specialBuffer(), predictorBase);
        if (config->mtpRepairBatchCausalMaskExtIdx >= 0) {
            NDArray::prepareSpecialUse({mask}, {});
            BUILD_SINGLE_SELECTOR(mask->dataType(), refillRepairMaskLauncher,
                                  (stream, mask->specialBuffer(),
                                   static_cast<LongType>(config->mtpRepairBatchWidth), maskLen,
                                   predictorBase), SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({mask}, {});
        }
        NDArray::registerSpecialUse({ids, hidden, position, cachePosition},
                                    {specArgmaxDevice, targetHiddenRows});

        void* repairCarryEvent = sd::graph::dspCreateEvent();
        REQUIRE_TRUE(repairCarryEvent != nullptr, 0,
                     "autoregressive_decode: failed to create CUDA batched repair event");
        sd::graph::dspEventRecord(repairCarryEvent, *stream);
        void* repairDspStream = sd::graph::dspGetExecutionStream();
        if (repairDspStream != nullptr && repairDspStream != static_cast<void*>(*stream)) {
            sd::graph::dspStreamWaitEvent(repairDspStream, repairCarryEvent);
        }
        sd::graph::dspDestroyEvent(repairCarryEvent);
        const auto repairPhaseBefore = config->mtpRepairBatchPlanHandle->getPlanPhase();
        Status repairStatus = config->mtpRepairBatchPlanHandle->executeSteadyState(
            mtpRepairBatchExtInputs, config->mtpRepairBatchNumPlanExternalInputs,
            mtpRepairBatchPlanOutputs, config->mtpRepairBatchNumPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
        const auto repairPhaseAfter = config->mtpRepairBatchPlanHandle->getPlanPhase();
        if (repairPhaseBefore != repairPhaseAfter) p0.planPhaseTransitions++;
        if (repairPhaseAfter == graph::PlanPhase::REPLAYING) p0.planReplayForwards++;
        else p0.planWarmupForwards++;
        REQUIRE_TRUE(repairStatus == Status::OK, 0,
                     "autoregressive_decode: batched MTP K/V-only repair plan failed at base row %lld",
                     (long long)predictorBase);
        REQUIRE_TRUE(config->mtpRepairBatchKeyOutputIdx >= 0
                         && config->mtpRepairBatchKeyOutputIdx < config->mtpRepairBatchNumPlanOutputs
                         && config->mtpRepairBatchValueOutputIdx >= 0
                         && config->mtpRepairBatchValueOutputIdx < config->mtpRepairBatchNumPlanOutputs,
                     0, "autoregressive_decode: batched repair output indices are invalid");
        NDArray* key = mtpRepairBatchPlanOutputs[config->mtpRepairBatchKeyOutputIdx];
        NDArray* value = mtpRepairBatchPlanOutputs[config->mtpRepairBatchValueOutputIdx];
        // Match the floating source/destination families supported by the
        // stride-aware BSHD scatter. It converts values while writing; requiring
        // identical source/cache dtypes rejects valid FLOAT -> HALF repair.
        const auto repairScatterTypeSupported = [](DataType dtype) {
            return dtype == DataType::HALF || dtype == DataType::BFLOAT16
                || dtype == DataType::FLOAT32 || dtype == DataType::DOUBLE;
        };
        REQUIRE_TRUE(key != nullptr && value != nullptr && key->rankOf() == 4 && value->rankOf() == 4
                         && key->sizeAt(0) == 1 && value->sizeAt(0) == 1
                         && key->sizeAt(1) >= activeRows && value->sizeAt(1) >= activeRows
                         && key->sizeAt(2) == config->mtpKvBuffers[0]->sizeAt(2)
                         && value->sizeAt(2) == config->mtpKvBuffers[1]->sizeAt(2)
                         && key->sizeAt(3) == config->mtpKvBuffers[0]->sizeAt(3)
                         && value->sizeAt(3) == config->mtpKvBuffers[1]->sizeAt(3)
                         && repairScatterTypeSupported(key->dataType())
                         && repairScatterTypeSupported(value->dataType())
                         && repairScatterTypeSupported(config->mtpKvBuffers[0]->dataType())
                         && repairScatterTypeSupported(config->mtpKvBuffers[1]->dataType())
                         && predictorBase + activeRows <= config->mtpKvBuffers[0]->sizeAt(1)
                         && predictorBase + activeRows <= config->mtpKvBuffers[1]->sizeAt(1),
                     0, "autoregressive_decode: batched repair outputs/caches violate BSHD capacity contract");
        std::vector<LongType> prefix{0, 1, 0, activeRows, 0, key->sizeAt(2), 0, key->sizeAt(3)};
        NDArray* keyPrefix = (*key)(prefix, true);
        prefix[4] = 0;
        prefix[6] = 0;
        NDArray* valuePrefix = (*value)(prefix, true);
        NDArray::prepareSpecialUse({config->mtpKvBuffers[0], config->mtpKvBuffers[1]},
                                   {keyPrefix, valuePrefix});
        ops::helpers::kvInPlaceWriteBSHD(config->mtpKvBuffers[0], keyPrefix,
                                         cachePosition->specialBuffer(), context);
        ops::helpers::kvInPlaceWriteBSHD(config->mtpKvBuffers[1], valuePrefix,
                                         cachePosition->specialBuffer(), context);
        NDArray::registerSpecialUse({config->mtpKvBuffers[0], config->mtpKvBuffers[1]},
                                    {keyPrefix, valuePrefix});
        delete keyPrefix;
        delete valuePrefix;
        void* repairScatterEvent = sd::graph::dspCreateEvent();
        REQUIRE_TRUE(repairScatterEvent != nullptr, 0,
                     "autoregressive_decode: failed to create CUDA batched repair scatter event");
        sd::graph::dspEventRecord(repairScatterEvent, *stream);
        void* nextDspStream = sd::graph::dspGetExecutionStream();
        if (nextDspStream != nullptr && nextDspStream != static_cast<void*>(*stream)) {
            sd::graph::dspStreamWaitEvent(nextDspStream, repairScatterEvent);
        }
        sd::graph::dspDestroyEvent(repairScatterEvent);
        p0.predictorRepairForwards++;
    };

    auto executeMtpCuda = [&](LongType targetTokenPosition, int draftSlot, bool writeTargetRow) {
        // PREDICTOR ROW MAPPING (review-ruled convention, packet 2): the argument
        // is a TARGET input-token position P; the predictor consumes the pair
        // (x_(P+1), h_P) at predictor row r = P - 1, with predictor RoPE = r and
        // predictor KV slot = r. Callers keep target coordinates; this boundary
        // converts exactly once. Bounds: P >= 1 so r >= 0 (a negative row is a
        // caller bug, not a clamp candidate).
        REQUIRE_TRUE(targetTokenPosition >= 1, 0,
                     "autoregressive_decode: CUDA MTP target position %lld maps to "
                     "negative predictor row",
                     (long long)targetTokenPosition);
        const LongType predictorRow = targetTokenPosition - 1;

        // P02 adaptive-K (review finding 1): MTP RESOURCES vs MTP DRAFTING.
        // Maintenance forwards (K=0 scalar-only steps, draftSlot==0,
        // writeTargetRow==false) consume the freshly published carry/token to
        // keep the predictor KV hole-free; only DRAFT production is gated on
        // active speculation (targetWindowReady).
        REQUIRE_TRUE(config->mtpPlanHandle != nullptr && config->mtpExtInputContext != nullptr
                         && mtpDraftDevice != nullptr,
                     0, "autoregressive_decode: attempted CUDA MTP execution while disabled");
        REQUIRE_TRUE((useMtp || (draftSlot == 0 && !writeTargetRow))
                         && draftSlot >= 0 && draftSlot <= specK, 0,
                     "autoregressive_decode: CUDA MTP draft slot %d outside [0,%d]",
                     draftSlot, specK);

        const bool chainProbe = DSP_DIAG_ENABLED(KV_CACHE) && draftSlot < 33
            && config->mtpTargetHidden->lengthOf() > 0;
        if (chainProbe) {
            mtpChainSampleBytes[draftSlot] = std::min(sizeof(mtpChainCarryIn[draftSlot]),
                static_cast<size_t>(config->mtpTargetHidden->lengthOf()) * config->mtpTargetHidden->sizeOfT());
            cudaMemcpyAsync(mtpChainCarryIn[draftSlot],
                            config->mtpTargetHidden->specialBuffer(),
                            mtpChainSampleBytes[draftSlot], cudaMemcpyDeviceToHost, *stream);
            cudaMemcpyAsync(&mtpChainTok[draftSlot],
                            config->mtpInputIds->specialBuffer(),
                            sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
            if (draftSlot + 1 > mtpChainSampled) mtpChainSampled = draftSlot + 1;
        }

        // Capacity gate (packet C1 / review round 4, finding E): the converted
        // row must fit the predictor mask and both KV buffers BEFORE any
        // predictor-cache indexing. CACHE LAYOUT CONTRACT: the MTP predictor
        // KV cache is BSHD [batch, maxSeqLen, heads, dim] (kv_scatter.h:148,
        // kvInPlaceWriteBSHD reads cacheMaxSeqLen = sizeAt(1)) - the SEQUENCE
        // dimension is dim 1, unambiguously. A rank-4 cache with the wrong
        // dimension order is a fixture/contract bug and fails loudly here.
        REQUIRE_TRUE(config->mtpKvBuffers[0] != nullptr && config->mtpKvBuffers[1] != nullptr,
                     0, "autoregressive_decode: CUDA MTP predictor KV buffers are unavailable");
        REQUIRE_TRUE(config->mtpKvBuffers[0]->rankOf() == 4 && config->mtpKvBuffers[1]->rankOf() == 4,
                     0, "autoregressive_decode: CUDA MTP predictor KV buffers must be rank 4 "
                        "[batch, maxSeqLen, heads, dim]");
        const LongType kvRows0 = config->mtpKvBuffers[0]->sizeAt(1);
        const LongType kvRows1 = config->mtpKvBuffers[1]->sizeAt(1);
        REQUIRE_TRUE(
            predictorRow < mtpMaskLen
                && predictorRow < kvRows0
                && predictorRow < kvRows1,
            0,
            "autoregressive_decode: CUDA MTP predictor row %lld (target position %lld) "
            "is outside cache/mask capacity (seq capacity %lld/%lld, mask %lld)",
            (long long)predictorRow, (long long)targetTokenPosition,
            (long long)kvRows0, (long long)kvRows1, (long long)mtpMaskLen);

        NDArray::prepareSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition, config->mtpCausalMask}, {});
        // Predictor RoPE position AND KV write slot are BOTH predictorRow:
        // row r consumes x_(r+1) at rope=r, slot=r (packet 2). The previous
        // code wrote rope=targetP and cache=targetP (or, in the WIP,
        // cache=targetP-1) - either split put the KV row at a slot that did
        // not match the row the prefill convention established, duplicating
        // the tail pair and shifting every draft row one past its input.
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpPositionOffset->specialBuffer(), predictorRow);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpCachePosition->specialBuffer(), predictorRow);
        BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(), updateCausalMaskLauncher,
                              (stream, config->mtpCausalMask->specialBuffer(),
                               predictorRow, mtpMaskLen),
                              SD_FLOAT_TYPES);
        NDArray::registerSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition, config->mtpCausalMask}, {});

        // KV self-row visibility probe: sample the predictor K row at THIS call's
        // predictor row BEFORE execution (must be zero/masked or stale prior draft) and
        // gate whether the plan's in-graph write actually lands where attention
        // will read it. Byte-identical between calls would mean the predictor plan
        // never writes its own KV row (degenerate self-attention -> uniform logits
        // -> 1-accept-per-run collapse signature).
        if (DSP_DIAG_ENABLED(KV_CACHE) && draftSlot == 0
                && config->mtpKvInputExtIndices != nullptr) {
            NDArray* kBuf = config->mtpKvBuffers[0];
            const LongType heads = kBuf->sizeAt(2);
            const LongType dim = kBuf->sizeAt(3);
            const LongType rowElems = heads * dim;
            if (predictorRow >= 0 && predictorRow < kBuf->sizeAt(1)) {
                std::vector<float> kSample(std::min<LongType>(8, rowElems));
                const void* rowPtr = static_cast<const char*>(kBuf->specialBuffer())
                                     + predictorRow * rowElems * kBuf->sizeOfT();
                // HALF/BF16 need conversion; sample raw bytes then expand via Nd4j-free path.
                std::vector<uint8_t> raw(kSample.size() * kBuf->sizeOfT());
                cudaMemcpyAsync(raw.data(), rowPtr, raw.size(),
                                cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);
                float vals[8] = {};
                const int n = static_cast<int>(kSample.size());
                if (kBuf->dataType() == DataType::FLOAT32) {
                    std::memcpy(vals, raw.data(), n * 4);
                } else {
                    for (int i = 0; i < n; i++) {
                        unsigned h = raw[i * 2] | (raw[i * 2 + 1] << 8);
                        unsigned sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
                        float v = exp == 0 ? (man == 0 ? 0.0f : std::ldexp((float)man, -24))
                                  : exp == 0x1F ? std::numeric_limits<float>::quiet_NaN()
                                  : std::ldexp(1.0f + man / 1024.0f, (int)exp - 15);
                        vals[i] = sign ? -v : v;
                    }
                }
                DSP_DIAG(KV_CACHE,
                         "MTP_KV_SELFROW row=%lld targetPos=%lld dtype=%d before=[%.4f,%.4f,%.4f,%.4f]",
                         (long long)predictorRow, (long long)targetTokenPosition, (int)kBuf->dataType(),
                         vals[0], vals[1], vals[2], vals[3]);
            }
        }
        // Post-write visibility sample handle: re-reads the same row AFTER the plan
        // executes (drained right before the accept rule), closing the
        // write-visibility question: if before != after, the in-graph KV write
        // landed; if identical, the write never reaches the row attention reads.
        // NOTE: declared at decode-loop scope (see kvSelfRowAfterBuf below) so the
        // speculative accept block can drain it - executeMtpCuda may run several
        // times per step and only slot 0 arms the probe.
        if (DSP_DIAG_ENABLED(KV_CACHE) && draftSlot == 0
                && config->mtpKvInputExtIndices != nullptr) {
            NDArray* kBuf = config->mtpKvBuffers[0];
            if (predictorRow >= 0 && predictorRow < kBuf->sizeAt(1)) {
                kvSelfRowAfterPos = predictorRow;
                kvSelfRowAfterBuf = kBuf;
            }
        }

        // Save admission for this invocation: the counter remains 3 on later calls.
        const bool captureThisCall = captureMtpInputs && mtpSnapshotCall < 3;
        if (captureThisCall) {
            tensorDiagnostics.enqueueTensorSnapshot(++mtpSnapshotCall, predictorRow,
                reinterpret_cast<void*>(*stream),
                {"mtp_input_ids", "mtp_target_hidden_states", "mtp_position_offset",
                 "mtp_cache_position", "mtp_causal_mask", "mtp_past_key_values.0.key",
                 "mtp_past_key_values.0.value"},
                {config->mtpInputIds, config->mtpTargetHidden, config->mtpPositionOffset,
                 config->mtpCachePosition, config->mtpCausalMask, config->mtpKvBuffers[0],
                 config->mtpKvBuffers[1]});
        }

        if (writeTargetRow) {
            p0.predictorProposalForwards++;
        } else if (p0RepairActive) {
            p0.predictorRepairForwards++;
        } else {
            p0.predictorMaintenanceForwards++;
        }
        const auto mtpPhaseBefore = mtpPlan->getPlanPhase();
        Status mtpStatus = mtpPlan->executeSteadyState(
            mtpExtInputs, mtpNumExtInputs,
            mtpPlanOutputs, mtpNumOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
        const auto mtpPhaseAfter = mtpPlan->getPlanPhase();
        if (mtpPhaseBefore != mtpPhaseAfter) p0.planPhaseTransitions++;
        if (mtpPhaseAfter == graph::PlanPhase::REPLAYING) p0.planReplayForwards++;
        else p0.planWarmupForwards++;
        if (!writeTargetRow && p0RepairActive && config->mtpLogitsOutputIdx >= 0)
            p0.predictorRepairLmHeadForwards++;
        // Post-execution staging audit: the VARIABLE slots must have been D2D
        // refreshed from the live arrays above (performPreReplaySync step 3).
        // Snapshotting the plan's own staging buffers on the same callIndex as
        // phase 1 (live inputs) makes staging-vs-live divergence directly
        // visible in the .dspt artifact: staging != live at this point proves
        // the captured graph consumed stale bytes; staging == live while drafts
        // stay frozen localizes the defect to the captured graph itself.
        if (captureThisCall) {
            std::vector<std::string> stagingNames;
            std::vector<NDArray*> stagingArrays;
            const int stagingSlots[] = {
                config->mtpInputIdsExtIdx, config->mtpTargetHiddenExtIdx,
                config->mtpCausalMaskExtIdx, config->mtpPositionOffsetExtIdx,
                config->mtpCachePositionExtIdx,
                config->mtpKvInputExtIndices != nullptr ? config->mtpKvInputExtIndices[0] : -1,
                config->mtpKvInputExtIndices != nullptr ? config->mtpKvInputExtIndices[1] : -1};
            const char* const stagingLabels[] = {
                "staging/mtp_input_ids", "staging/mtp_target_hidden_states",
                "staging/mtp_causal_mask", "staging/mtp_position_offset",
                "staging/mtp_cache_position",
                "staging/mtp_past_key_values.0.key", "staging/mtp_past_key_values.0.value"};
            for (int s = 0; s < 7; s++) {
                NDArray* staging = (stagingSlots[s] >= 0 && stagingSlots[s] < mtpNumExtInputs)
                    ? mtpPlan->getStagingBufferArray(stagingSlots[s]) : nullptr;
                if (staging == nullptr) continue;
                stagingNames.emplace_back(stagingLabels[s]);
                stagingArrays.push_back(staging);
            }
            if (!stagingArrays.empty()) {
                tensorDiagnostics.enqueueTensorSnapshot(mtpSnapshotCall, predictorRow,
                    reinterpret_cast<void*>(*stream), stagingNames, stagingArrays);
            }
        }
        std::string mtpFailureDetail;
        if (mtpStatus != Status::OK) mtpFailureDetail = nestedPlanFailureDetail();
        REQUIRE_TRUE(mtpStatus == Status::OK, 0,
                     "%s [autoregressive_decode nested CUDA MTP plan targetPos=%lld predictorRow=%lld, status=%s (%d)]",
                     mtpFailureDetail.c_str(), (long long)targetTokenPosition, (long long)predictorRow,
                     graph::dsp::dspStatusName(mtpStatus), static_cast<int>(mtpStatus));
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
        if (captureThisCall) {
            tensorDiagnostics.enqueueTensorSnapshot(mtpSnapshotCall, predictorRow,
                reinterpret_cast<void*>(*stream),
                {"output/pre-argmax/mtp_logits", "output/pre-argmax/mtp_hidden"},
                {mtpLogits, mtpHidden});
            // Post-exec live-KV audit: after executeSteadyState returns, the current
            // row must be present in the live buffer (the in-plan in-place write) and
            // every prior row must be unchanged vs the pre-exec snapshot. A missing
            // current row or a moved prior row directly exposes the in-call
            // write-vs-refresh ordering defect.
            tensorDiagnostics.enqueueTensorSnapshot(mtpSnapshotCall, predictorRow,
                reinterpret_cast<void*>(*stream),
                {"postexec/mtp_past_key_values.0.key", "postexec/mtp_past_key_values.0.value",
                 "postexec/mtp_causal_mask", "postexec/mtp_cache_position",
                 "postexec/mtp_input_ids"},
                {config->mtpKvBuffers[0], config->mtpKvBuffers[1],
                 config->mtpCausalMask, config->mtpCachePosition, config->mtpInputIds});
        }
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
        if (DSP_DIAG_ENABLED(KV_CACHE)) {
            // MTP_ARGMAX_IMMEDIATE: D2H the argmax + first logits value RIGHT AFTER the
            // argmax kernel, before the target verification pass (or any other plan) has
            // a chance to touch mtpDraftDevice or the logits output staging. Compare with
            // the draft later consumed by MTP_POS_STATS: equal => logits were fresh and
            // the draft is genuinely the predictor's answer (input/weight issue);
            // different => the draft was CLOBBERED between argmax and consumption
            // (aliasing/clobbering defect on the output path).
            {
                LongType immDraft[1] = {};
                uint32_t immLogit[1] = {};
                cudaMemcpyAsync(immDraft, draftPtr, sizeof(LongType),
                                cudaMemcpyDeviceToHost, *stream);
                cudaMemcpyAsync(immLogit, mtpLogits->specialBuffer(),
                                std::min<size_t>(sizeof(uint32_t),
                                                 static_cast<size_t>(mtpLogits->sizeOfT())),
                                cudaMemcpyDeviceToHost, *stream);
                // DRAFT-QUALITY DISCRIMINATOR (endgame goal: >=50% acceptance):
                // true global top-5 draft tokens + their raw logits, and the
                // argmax rank. Interpretation: target top-1 absent from draft
                // top-100 => conditioning broken (carry row / KV position /
                // input token wiring on the native 27B path);
                // present-but-lower-ranked => calibration/quantization
                // interaction. The ranking must span the ENTIRE vocabulary:
                // a winner above ID 128 never shows up in a bounded first-128
                // probe, so the whole logits row is copied D2H once and the
                // host pass ranks every element with dtype-correct indexing.
                // The argmax over the FULL vocab is still the kernel's job.
                // D2H BUDGET: the row is copied element-granularly in D2H
                // transactions of at most 4MB. Real vocab rows (~150k => 300KB
                // BF16/FP16, 600KB FP32) fit a single transaction; a row above
                // the 4MB single-copy budget is chunked instead of skipped, so
                // the top-5 stays diagnostic-only with bounded transfers.
                LongType immVocab = mtpVocab;
                LongType immTop5[5] = {};
                const DataType immDtype = mtpLogits->dataType();
                const size_t immElemSize = mtpLogits->sizeOfT();
                const char* immDtypeName =
                    immDtype == DataType::BFLOAT16 ? "BF16"
                    : immDtype == DataType::HALF   ? "FP16"
                    : immDtype == DataType::FLOAT32 ? "FP32"
                    : immDtype == DataType::DOUBLE  ? "FP64"
                                                    : "OTHER";
                const size_t immRowBytes =
                    static_cast<size_t>(immVocab) * immElemSize;
                constexpr size_t immMaxRowBytes = 4u * 1024u * 1024u; // 4MB cap
                // Full-row copy budget: issue the element-granular D2H in
                // transactions bounded by 4MB, then keep the probe's single
                // stream sync - every host read below (top-5 path and
                // fallback) observes completed data.
                std::vector<uint8_t> immRaw;
                const bool fullRowReady =
                    immVocab > 0 && immElemSize > 0;
                const size_t immChunkElems =
                    fullRowReady ? std::max<size_t>(1, immMaxRowBytes / immElemSize)
                                 : 1;
                if (fullRowReady) {
                    // One element-granular D2H of the full logits row.
                    // Byte layout: element i lives at offset i*elemSize, exactly
                    // the linear layout argmaxKernel scores (logits[i]).
                    immRaw.resize(immRowBytes);
                    for (LongType chunkStart = 0; chunkStart < immVocab;
                         chunkStart += static_cast<LongType>(immChunkElems)) {
                        const LongType chunkLen =
                            std::min<LongType>(static_cast<LongType>(immChunkElems),
                                               immVocab - chunkStart);
                        cudaMemcpyAsync(
                            immRaw.data()
                                + static_cast<size_t>(chunkStart) * immElemSize,
                            static_cast<const char*>(mtpLogits->specialBuffer())
                                + static_cast<size_t>(chunkStart) * immElemSize,
                            static_cast<size_t>(chunkLen) * immElemSize,
                            cudaMemcpyDeviceToHost, *stream);
                    }
                }
                cudaError_t immErr = cudaStreamSynchronize(*stream);
                if (fullRowReady) {
                    // Decode one raw element by dtype into float. Bit-level
                    // decode keeps the reported scores the raw device bits
                    // (BF16/FP16/FP32 exact); DOUBLE is narrowed to the
                    // shared float score field, tagged dtype=FP64.
                    auto decodeRaw = [&](const uint8_t* elem) -> float {
                        if (immDtype == DataType::BFLOAT16) {
                            // 2B element: widen to FP32 by shifting into the
                            // high 16 bits (exact, subnormals included).
                            uint16_t v = 0;
                            std::memcpy(&v, elem, sizeof(v));
                            uint32_t f = static_cast<uint32_t>(v) << 16;
                            float out;
                            std::memcpy(&out, &f, sizeof(out));
                            return out;
                        }
                        if (immDtype == DataType::HALF) {
                            // 2B element: FP16 -> FP32 by bit fields, matching
                            // the decode used by the other probes in this file
                            // (MTP_TARGET_CARRY_CONTENT). Subnormals (exp==0,
                            // man!=0) decode to their true value man*2^-24 via
                            // ldexp instead of collapsing to zero; Inf/NaN are
                            // preserved explicitly.
                            uint16_t v = 0;
                            std::memcpy(&v, elem, sizeof(v));
                            uint32_t sign = (v & 0x8000u) >> 15;
                            uint32_t exp = (v & 0x7C00u) >> 10;
                            uint32_t man = v & 0x03FFu;
                            float mag;
                            if (exp == 0) {
                                mag = man == 0
                                          ? 0.0f
                                          : std::ldexp(static_cast<float>(man), -24);
                            } else if (exp == 0x1F) {
                                mag = man == 0
                                          ? std::numeric_limits<float>::infinity()
                                          : std::numeric_limits<float>::quiet_NaN();
                            } else {
                                mag = std::ldexp(
                                    1.0f + static_cast<float>(man) / 1024.0f,
                                    static_cast<int>(exp) - 15);
                            }
                            return sign != 0u ? -mag : mag;
                        }
                        if (immDtype == DataType::FLOAT32) {
                            float out;
                            std::memcpy(&out, elem, sizeof(out));
                            return out;
                        }
                        if (immDtype == DataType::DOUBLE) {
                            // DOUBLE scores are reported narrowed to the
                            // shared float score field; the dtype=FP64 tag on
                            // the event identifies the source precision.
                            double wide;
                            std::memcpy(&wide, elem, sizeof(wide));
                            return static_cast<float>(wide);
                        }
                        // Non-float row: report the integer value directly.
                        // Selector list is SD_FLOAT_TYPES, so this is only a
                        // defensive branch for unexpected metadata.
                        LongType iv = 0;
                        std::memcpy(&iv, elem,
                                    std::min<size_t>(sizeof(iv), immElemSize));
                        return static_cast<float>(iv);
                    };
                    // scoredCount = min(5, vocab): never index more scored
                    // entries than the vocabulary contains.
                    const size_t scoredCount =
                        static_cast<size_t>(std::min<LongType>(5, immVocab));
                    std::vector<std::pair<float, LongType>> scored;
                    scored.reserve(static_cast<size_t>(immVocab));
                    for (LongType t = 0; t < immVocab; t++) {
                        scored.emplace_back(
                            decodeRaw(immRaw.data() +
                                      static_cast<size_t>(t) * immElemSize),
                            t);
                    }
                    std::partial_sort(scored.begin(), scored.begin() + scoredCount,
                                      scored.end(),
                                      [](const auto& a, const auto& b) {
                                          return a.first > b.first;
                                      });
                    for (size_t r = 0; r < scoredCount; r++) {
                        immTop5[r] = scored[r].second;
                    }
                    // Report all five format slots; when vocab < 5 the slots
                    // beyond scoredCount stay at their zero-initialized
                    // sentinel "0:0.0000" and are not backed by scored data.
                    DSP_DIAG(KV_CACHE,
                             "MTP_DRAFT_TOP5 pos=%lld slot=%d draft=%lld "
                             "dtype=%s vocab=%lld "
                             "top5=[%lld:%.4f %lld:%.4f %lld:%.4f %lld:%.4f %lld:%.4f] "
                             "err=%d",
                             (long long)targetTokenPosition, draftSlot, (long long)immDraft[0],
                             immDtypeName, (long long)immVocab,
                             (long long)immTop5[0],
                             scoredCount > 0 ? scored[0].first : 0.0f,
                             (long long)immTop5[1],
                             scoredCount > 1 ? scored[1].first : 0.0f,
                             (long long)immTop5[2],
                             scoredCount > 2 ? scored[2].first : 0.0f,
                             (long long)immTop5[3],
                             scoredCount > 3 ? scored[3].first : 0.0f,
                             (long long)immTop5[4],
                             scoredCount > 4 ? scored[4].first : 0.0f,
                             static_cast<int>(immErr));
                } else {
                    DSP_DIAG(KV_CACHE,
                             "MTP_ARGMAX_IMMEDIATE targetPos=%lld slot=%d draft=%lld "
                             "logit0_raw=0x%08x err=%d",
                             (long long)targetTokenPosition, draftSlot, (long long)immDraft[0],
                             static_cast<unsigned>(immLogit[0]),
                             static_cast<int>(immErr));
                }
            }
        }
        if (captureThisCall) {
            // Borrow just the selected INT64 element, not the unwritten draft slots.
            // Metadata dies here; the arena owns the asynchronous host destination.
            REQUIRE_TRUE(config->mtpCachePosition->rankOf() == 0
                             && config->mtpCachePosition->dataType() == mtpDraftDevice->dataType(),
                         0, "DSP tensor snapshot requires the existing INT64 scalar shape");
            // Reuse already-resident scalar shape metadata: no shape upload/allocation
            // between argmax and its snapshot. This constructor borrows the buffer.
            NDArray selected(mtpDraftDevice->dataBuffer(), config->mtpCachePosition->shapeInfo(),
                             mtpDraftDevice->getContext(), mtpDraftDevice->offset() + draftSlot);
            tensorDiagnostics.enqueueTensorSnapshot(mtpSnapshotCall, predictorRow,
                reinterpret_cast<void*>(*stream), {"output/post-argmax/draft_id"}, {&selected});
        }

        // -- Hidden carry: unconditional self-carry + stream ordering --------
        // Upstream Qwen3.5 MTP: EVERY predictor call's output hidden feeds the
        // NEXT chained call as the hnorm input. The epilogue's
        // setMtpTargetCarryCuda overrides this for the NEXT step's slot 0,
        // installing the target trunk hidden at the newly committed position.
        // The write must be UNCONDITIONAL: both pre- and post-368a274c60 27B
        // captures prove slot 1 must chain slot 0's self-hidden. With this
        // gated to draftSlot > 0, call2's carry input was byte-identical to
        // call1's INPUT (0/5120 elements differ, norm 148.4919) instead of
        // call1's OUTPUT — the slot0->slot1 chain was broken and every
        // slot>=1 draft consumed a mismatched (token, hidden) pair. Capture
        // replay (TestQwenMtpPredictorLifecycle, 27B artifacts, proc-070/072)
        // agrees: fresh and compiled graphs are bit-exact with each other at
        // every call but diverge from production's captured native outputs at
        // exactly the chained calls 2..K.
        //
        // STREAM ORDERING (P01): the carry write is issued on the caller's
        // stream, but the predictor plan's graph launch may execute on a
        // different thread-local DSP stream (tl_dspExecutionStream). Instead
        // of a host-blocking cudaStreamSynchronize, record an event on the
        // caller's stream AFTER both carry writes and make the DSP execution
        // stream wait on it - device-side cross-stream ordering with no host
        // wait. The event is created per call (small, pool-backed) and
        // destroyed after the wait is enqueued; enqueue order still
        // guarantees the copy precedes any later graph launch.
        {
            REQUIRE_TRUE(mtpHidden->lengthOf() == config->mtpTargetHidden->lengthOf()
                             && mtpHidden->dataType() == config->mtpTargetHidden->dataType(),
                         0, "autoregressive_decode: CUDA MTP hidden carry shape/type mismatch");
            writeMtpCarry(config->mtpTargetHidden, config->mtpTargetHiddenExtIdx,
                          mtpHidden->specialBuffer(),
                          static_cast<size_t>(mtpHidden->lengthOf()) * mtpHidden->sizeOfT());
        }

        if (chainProbe) {
            cudaMemcpyAsync(mtpChainHidOut[draftSlot], mtpHidden->specialBuffer(),
                            mtpChainSampleBytes[draftSlot], cudaMemcpyDeviceToHost, *stream);
        }

        writeMtpCarry(config->mtpInputIds, config->mtpInputIdsExtIdx,
                      draftPtr, sizeof(LongType));
        // P01: single device-side barrier for BOTH carry writes (see the
        // STREAM ORDERING comment above). The predictor graph launch waits
        // on this event via the DSP execution stream; the host does not wait.
        {
            void* carryEvt = sd::graph::dspCreateEvent();
            REQUIRE_TRUE(carryEvt != nullptr, 0,
                         "autoregressive_decode: failed to create CUDA MTP carry event");
            sd::graph::dspEventRecord(carryEvt, *stream);
            void* dspExecStream = sd::graph::dspGetExecutionStream();
            if (dspExecStream != nullptr
                    && dspExecStream != static_cast<void*>(*stream)) {
                sd::graph::dspStreamWaitEvent(dspExecStream, carryEvt);
            }
            sd::graph::dspDestroyEvent(carryEvt);
        }
        DSP_DIAG(KV_CACHE,
                 "MTP_CALL row=%lld targetPos=%lld slot=%d - predictor invoked (chained input token; "
                 "carry = previous call self-hidden, epilogue overrides for next step's slot 0)",
                 (long long)predictorRow, (long long)targetTokenPosition, draftSlot);

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
        // P02 adaptive-K (review finding 1): MTP RESOURCES vs MTP DRAFTING.
        // useMtp (targetWindowReady && mtp configured) describes drafting only;
        // carry maintenance must stay valid for scalar-only steps while the
        // MTP metadata exists, so re-enabling speculation cannot resume from
        // a stale carry.
        REQUIRE_TRUE(config->mtpPlanHandle != nullptr && config->mtpExtInputContext != nullptr
                         && targetHiddenRows != nullptr
                         && targetHiddenRows->rankOf() == 3,
                     0, "autoregressive_decode: target hidden output must be rank 3 for CUDA MTP");
        REQUIRE_TRUE(row >= 0 && row < targetHiddenRows->sizeAt(1)
                         && targetHiddenRows->strideAt(2) == 1
                         && config->mtpTargetHidden->lengthOf() == targetHiddenRows->sizeAt(2)
                         && config->mtpTargetHidden->dataType() == targetHiddenRows->dataType(),
                     0, "autoregressive_decode: CUDA MTP target carry row/shape/type mismatch");
        DSP_DIAG(KV_CACHE,
                 "MTP_TARGET_CARRY shape=[%lld,%lld,%lld] row=%d pos=%p - installing "
                 "target trunk hidden into predictor carry",
                 (long long)targetHiddenRows->sizeAt(0),
                 (long long)targetHiddenRows->sizeAt(1),
                 (long long)targetHiddenRows->sizeAt(2),
                 row, targetHiddenRows->specialBuffer());
        // Dump the first 8 floats of the carry source row for diagnostics:
        // compares step-to-step carry content stability. If the same context
        // position produces different carry bytes across steps, the target's
        // verification output hidden is corrupted or misindexed.
        // GATED (P01): the copies + sync + formatting below must not execute
        // when KV_CACHE diagnostics are off - disabled diagnostics must not
        // enqueue work.
        if (DSP_DIAG_ENABLED(KV_CACHE)) {
            // Read as uint16 to handle both BF16 (2 bytes) and FP32 (4 bytes)
            // correctly: dump raw bytes and interpret by the array's actual dtype.
            const size_t elemSize = targetHiddenRows->sizeOfT();
            const size_t dumpElems = std::min<size_t>(8, targetHiddenRows->sizeAt(2));
            std::vector<uint8_t> carryDump(dumpElems * elemSize);
            const void* dumpSrc = static_cast<const char*>(targetHiddenRows->specialBuffer())
                                  + static_cast<size_t>(row)
                                        * targetHiddenRows->strideAt(1)
                                        * targetHiddenRows->sizeOfT();
            cudaMemcpyAsync(carryDump.data(), dumpSrc, carryDump.size(),
                            cudaMemcpyDeviceToHost, *stream);
            cudaStreamSynchronize(*stream);
            if (targetHiddenRows->dataType() == DataType::BFLOAT16) {
                auto* vals = reinterpret_cast<uint16_t*>(carryDump.data());
                DSP_DIAG(KV_CACHE,
                         "MTP_TARGET_CARRY_CONTENT row=%d dtype=BF16 first8_hex=[%04x %04x %04x %04x %04x %04x %04x %04x] "
                         "first8_as_u16=[%u %u %u %u %u %u %u %u]",
                         row,
                         vals[0], vals[1], vals[2], vals[3],
                         vals[4], vals[5], vals[6], vals[7],
                         vals[0], vals[1], vals[2], vals[3],
                         vals[4], vals[5], vals[6], vals[7]);
            } else {
                auto* vals = reinterpret_cast<float*>(carryDump.data());
                DSP_DIAG(KV_CACHE,
                         "MTP_TARGET_CARRY_CONTENT row=%d dtype=FP32 first8=[%.6f, %.6f, %.6f, %.6f, "
                         "%.6f, %.6f, %.6f, %.6f]",
                         row, vals[0], vals[1], vals[2], vals[3],
                         vals[4], vals[5], vals[6], vals[7]);
            }
        }
        // Dump a per-row magnitude scan across ALL window rows: if the W-wide
        // verification forward wrote every row, every row's RMS must be ~O(1e-2..1)
        // and CONSISTENT between rows. Rows near zero while logits stay correct
        // means the logits path and the carry source disagree about the buffer.
        // GATED (P01): W per-row D2H copies + W syncs + host RMS loops must not
        // execute when diagnostics are off.
        if (DSP_DIAG_ENABLED(KV_CACHE)) {
            const int W = static_cast<int>(targetHiddenRows->sizeAt(1));
            const int H = static_cast<int>(targetHiddenRows->sizeAt(2));
            std::vector<float> rowRms(W);
            std::vector<uint8_t> scratch(static_cast<size_t>(H) * targetHiddenRows->sizeOfT());
            std::vector<double> sq(H);
            for (int r = 0; r < W; r++) {
                const void* rowSrc = static_cast<const char*>(targetHiddenRows->specialBuffer())
                                     + static_cast<size_t>(r) * targetHiddenRows->strideAt(1)
                                       * targetHiddenRows->sizeOfT();
                cudaMemcpyAsync(scratch.data(), rowSrc, scratch.size(),
                                cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);
                if (targetHiddenRows->dataType() == DataType::BFLOAT16) {
                    auto* h = reinterpret_cast<uint16_t*>(scratch.data());
                    for (int i = 0; i < H; i++) {
                        unsigned sign = (h[i] >> 15) & 1u;
                        unsigned exp  = (h[i] >> 7) & 0xFFu;
                        unsigned man  = h[i] & 0x7Fu;
                        float v;
                        if (exp == 0) v = (man == 0) ? 0.0f
                                : std::ldexp(static_cast<float>(man), -7-126);
                        else if (exp == 0xFF) v = std::numeric_limits<float>::quiet_NaN();
                        else v = std::ldexp(1.0f + static_cast<float>(man) / 128.0f,
                                            static_cast<int>(exp) - 127);
                        sq[i] = static_cast<double>(sign ? -v : v) * (sign ? -v : v);
                    }
                } else if (targetHiddenRows->dataType() == DataType::HALF) {
                    auto* h = reinterpret_cast<uint16_t*>(scratch.data());
                    for (int i = 0; i < H; i++) {
                        // IEEE 754 binary16 -> float: 1 sign, 5 exp (bias 15), 10 mantissa.
                        unsigned sign = (h[i] >> 15) & 1u;
                        unsigned exp  = (h[i] >> 10) & 0x1Fu;
                        unsigned man  = h[i] & 0x3FFu;
                        float v;
                        if (exp == 0) v = (man == 0) ? 0.0f
                                : std::ldexp(static_cast<float>(man), -10-14);
                        else if (exp == 0x1F) v = (man == 0)
                                ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
                        else v = std::ldexp(1.0f + static_cast<float>(man) / 1024.0f,
                                            static_cast<int>(exp) - 15);
                        sq[i] = static_cast<double>(sign ? -v : v) * (sign ? -v : v);
                    }
                } else {
                    auto* f = reinterpret_cast<float*>(scratch.data());
                    for (int i = 0; i < H; i++) {
                        sq[i] = static_cast<double>(f[i]) * f[i];
                    }
                }
                double acc = 0.0;
                for (int i = 0; i < H; i++) acc += sq[i];
                rowRms[r] = static_cast<float>(std::sqrt(acc / std::max(1, H)));
            }
            char rmsBuf[128];
            {
                int off = 0;
                for (int r = 0; r < W; r++) {
                    off += snprintf(rmsBuf + off, sizeof(rmsBuf) - off, "%s%.4g",
                                    r ? "," : "", static_cast<double>(rowRms[r]));
                }
            }
            DSP_DIAG(KV_CACHE,
                     "MTP_TARGET_CARRY_RMS rows=%d rms=[%s] installRow=%d",
                     W, rmsBuf, row);
        }
        size_t rowBytes = static_cast<size_t>(targetHiddenRows->sizeAt(2))
                          * targetHiddenRows->sizeOfT();
        const void* source = static_cast<const char*>(targetHiddenRows->specialBuffer())
                             + static_cast<size_t>(row)
                                   * targetHiddenRows->strideAt(1)
                                   * targetHiddenRows->sizeOfT();
        writeMtpCarry(config->mtpTargetHidden, config->mtpTargetHiddenExtIdx,
                      source, rowBytes);
    };

    auto setMtpNextInputCuda = [&](NDArray* tokenSource,
                                   LongType tokenIndex,
                                   LongType nextTargetPosition) {
        // PREDICTOR ROW MAPPING (packet 2, same convention as executeMtpCuda):
        // nextTargetPosition P denotes the TARGET position of the NEXT predictor
        // call's input token; that call consumes it at predictor row r = P - 1
        // (rope = r, slot = r). The token stored here is x_(P+1) - the input the
        // row r = P - 1 consumes - so the retained pending pair describes rope
        // P-1, next write slot P-1. Bounds: P >= 1 (negative row = caller bug).
        REQUIRE_TRUE(nextTargetPosition >= 1, 0,
                     "autoregressive_decode: CUDA MTP pending target position %lld maps "
                     "to negative predictor row",
                     (long long)nextTargetPosition);
        const LongType nextPredictorRow = nextTargetPosition - 1;
        // P02 adaptive-K (review finding 1): same resources-vs-drafting split
        // as setMtpTargetCarryCuda - pending-input maintenance stays valid for
        // scalar-only steps while MTP metadata exists.
        REQUIRE_TRUE(config->mtpPlanHandle != nullptr && config->mtpExtInputContext != nullptr
                         && tokenSource != nullptr
                         && tokenSource->dataType() == DataType::INT64
                         && tokenIndex >= 0 && tokenIndex < tokenSource->lengthOf(),
                     0, "autoregressive_decode: invalid CUDA MTP next-token source");
        const void* tokenPtr = static_cast<const LongType*>(tokenSource->specialBuffer())
                               + tokenIndex;
        writeMtpCarry(config->mtpInputIds, config->mtpInputIdsExtIdx,
                      tokenPtr, sizeof(LongType));
        NDArray::prepareSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition},
            {tokenSource});
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpPositionOffset->specialBuffer(), nextPredictorRow);
        updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
            config->mtpCachePosition->specialBuffer(), nextPredictorRow);
        NDArray::registerSpecialUse(
            {config->mtpPositionOffset, config->mtpCachePosition},
            {tokenSource});
    };

    // -- Mark decode-loop-modified ext inputs as VARIABLE --------------------
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
    // via registerSpecialUse({...}). They are NOT host-fed - Java never writes them
    // per step in the native decode loop. They must therefore be VARIABLE (protected +
    // address-stable + staging D2D-refreshed each step), exactly like the GDN/conv/KV
    // inputs below - NOT PLACEHOLDER.
    //
    // PLACEHOLDER means "host-written -> force H2D" (externalInputIsPlaceholder_ ==
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
    // (must NOT H2D - device buffer is authoritative, host buffer is stale).
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
        // A rollback snapshot belongs to one commit transaction, not one restore.
        kvRowSnapshotBase = -1;
        // Cancellation is observed only at a committed step boundary. This
        // keeps KV/recurrent state coherent for a later continuation.
        if (config->cancelCallback != nullptr &&
                config->cancelCallback(config->callbackUserData)) {
            break;
        }
        // Multi-token speculative steps advance tokensGenerated faster than the
        // step counter - without this check the next step writes past the
        // generatedTokenIds buffer (maxNewTokens-sized) and over-reports count.
        if (tokensGenerated >= maxNewTokens) break;
        const int tokensBeforeStep = tokensGenerated;
        auto stepStart = std::chrono::high_resolution_clock::now();

        // -- Step 1: Update plan external inputs for this decode step --
        // decodeEmbedding IS prefillEmbeddings (same NDArray, same device address).
        // The embed lookup kernel writes into it in-place each step, keeping the
        // device address stable for CUDA graph replay (externalAddrsMatch).
        if (config->embeddingsExtIdx >= 0 && config->embeddingsExtIdx < numExtInputs) {
            extInputs[config->embeddingsExtIdx] = decodeEmbedding;
        }

        // -- ADR 0106 Phase 2: build proposals for this step -----------------
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

        // -- Step 1b: Pre-unmask the CURRENT position in causal mask --
        // GGUF only (planOwnsKvScatter == true): the dotProductAttentionV2 op writes
        // KV at cache_position = currentPosition in-place, then attends to the full
        // buffer including that position. Pre-unmasking currentPosition is required so
        // the token can attend to its own newly-written KV entry.
        //
        // ONNX/external-scatter path (planOwnsKvScatter == false): KV scatter happens
        // AFTER execution via kvScatterBatched. Position currentPosition in the static
        // KV buffer is EMPTY during plan execution - attending to it reads zeros, giving
        // wrong logits. The post-execution mask update unmasks kvJustWritten (the PREVIOUS
        // position that was just written) for the NEXT step; the current query position
        // is always exposed via mask[totalSeqLen-1] (padded layout set by Java warmup).
        // Step 1b pre-unmask of currentPosition - GATED on planOwnsKvScatter (verified correct
        // by experiment: removing the gate gives step-7 native=87 vs java=2008). For the
        // external-scatter path (planOwnsKvScatter==false, ONNX/SmolDocling) the current token's
        // K/V is NOT in the cache at currentPosition during plan execution (scatter is post-exec)
        // - it is provided at the PADDED query slot (mask[totalSeqLen-1]). Pre-unmasking
        // currentPosition would attend an EMPTY cache slot -> wrong logits. GGUF in-graph scatter
        // DOES have the current K/V at currentPosition, so it pre-unmasks.
        if (config->planOwnsKvScatter) {
            if (causalMask != nullptr && currentPosition >= 0 && currentPosition < causalMaskLen) {
                if (causalMask->rankOf() == 4 && causalMask->sizeAt(2) > 1) {
                    // W-wide window mask: the per-row causal band moves with
                    // currentPosition every step - a single-column unmask only
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

        // -- Step 2: Execute plan --
        // Use executeSteadyState() for the hot decode path. For step >= 4 in
        // REPLAYING phase, this eliminates ~200ms/step of CPU overhead (slot
        // scans, lifecycle checks, shape validation). For earlier steps or
        // pre-REPLAYING phase, it automatically falls back to full execute().
        //
        // ALL decode-loop-written ext inputs (embeddings, attn/causal/reformat masks,
        // position_ids, input_ids, position_offset, GDN/conv state, KV cache) are marked
        // VARIABLE - device-authoritative, never placeholder. This op's kernels write
        // them in-place and registerSpecialUse leaves the device buffer authoritative,
        // so performPreReplaySync respects actuality (isPrimaryActual) and skips H2D - a
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
        if (useSpeculative && proposedCount > 0) {
            // DEEP pre-verification snapshot (scalar-binding aliasing fix): copy the
            // recurrent state ext inputs into DEDICATED owned arrays so a rerun can
            // advance consumedCount rows from the pre-step state instead of the
            // post-verification state this step's verify pass leaves in place.
            // Gated on proposedCount > 0: with no proposals the state commit happens
            // inline (no rerun fires), so no snapshot is consumed and the ext inputs
            // already hold the authoritative committed state.
            // Runs for BOTH bindings: without a scalar binding the window-geometry
            // rerun needs it (previous behavior); WITH a binding the shared-buffer
            // aliasing means prepareScalarTarget copied nothing and the scalar rerun
            // executed from post-verify state (mtp-fix-gate2 NaN guard, step=1).
            // Called BEFORE prepareScalarTarget and BEFORE the verification plan
            // execution - the snapshot is genuinely pre-verify.
            capturePreVerificationState();
            // Snapshot the complete verification write range. The W-wide verify
            // executes with activeWindow = 1 + proposedCount and therefore writes
            // the current row plus every draft row; omitting the final correction
            // row leaves stale K/V visible to a later prefix rerun.
            capturePreVerificationKvRows(currentPosition, 1 + proposedCount);
        }
        if (useScalarTarget) prepareScalarTarget();
        // SCALAR-PLAN RETIREMENT (gates 8-10 evidence chain): the captured
        // scalar plan is poisoned by ANY interleaved plan execution on the
        // shared session. Gate 10 proved the predictor maintenance forward
        // ALONE suffices (K=0 gen: step-0 scalar call clean after capture,
        // one maintenance forward, step-1 scalar call all-NaN with all ext
        // inputs finite and retry deterministically NaN). The window plan
        // tolerates interleaving (gate-10 gen 1: 57 clean steps interleaved
        // with predictor calls; gate-8: 57 bypass reruns clean). Routing the
        // non-spec step through the window plan at activeWindow=1 - the same
        // teacher-forced-proven geometry as the rerun bypass - preserves
        // token parity while eliminating the poisoned capture. activeWindow
        // is already 1 here (no proposals), so no refill changes are needed.
        if (proposedCount > 0) p0.targetVerificationForwards++;
        const auto targetPhaseBefore = plan->getPlanPhase();
        Status planStatus = plan->executeSteadyState(
            extInputs, numExtInputs,
            planOutputs, numPlanOutputs,
            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
        const auto targetPhaseAfter = plan->getPlanPhase();
        if (targetPhaseBefore != targetPhaseAfter) p0.planPhaseTransitions++;
        if (targetPhaseAfter == graph::PlanPhase::REPLAYING) p0.planReplayForwards++;
        else p0.planWarmupForwards++;

        // Clear the scale registry immediately after plan execution (no stale refs).
        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr) {
            clearKvScaleRegistry();
        }

        // Validate plan output every step - these are O(1) pointer/flag checks,
        // negligible cost compared to the plan execution itself.
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

        {
            NDArray* logitsArr = planOutputs[config->logitsOutputIdx];
            auto* logitsDb = logitsArr->dataBuffer();
            REQUIRE_TRUE(logitsDb != nullptr, 0,
                         "autoregressive_decode: logits DataBuffer is null at step %d. "
                         "Output array exists but has no backing buffer - likely a stale slot.",
                         step);
            REQUIRE_TRUE(!logitsDb->isClosed(), 0,
                         "autoregressive_decode: logits DataBuffer is CLOSED at step %d. "
                         "The plan reused a freed buffer - stale slot reuse bug.",
                         step);
            REQUIRE_TRUE(logitsArr->specialBuffer() != nullptr, 0,
                         "autoregressive_decode: logits specialBuffer (device ptr) is null at step %d. "
                         "Buffer exists but has no device allocation - missing syncToDevice or stale buffer.",
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

        // -- Step 2b: GDN/conv recurrent state feedback --
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
                            p0.stateCommitBytes += static_cast<std::uint64_t>(bytes);
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
                            p0.stateCommitBytes += static_cast<std::uint64_t>(bytes);
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

        // -- Step 3: Token sampling --
        // Get logits from plan output at config->logitsOutputIdx
        NDArray* logitsOutput = planOutputs[config->logitsOutputIdx];

        // Validate logits rank before accessing shape dimensions.
        // Expected: rank 2 [batch, vocabSize] or rank 3 [batch, seqLen, vocabSize].
        // A rank-0 (scalar) output means the plan returned a wrong/stale output slot.
        auto logitsRank = logitsOutput->rankOf();
        REQUIRE_TRUE(logitsRank >= 2 && logitsRank <= 3, 0,
                     "autoregressive_decode: logitsOutput rank is %lld (expected 2 or 3) at step %d. "
                     "lengthOf=%lld, logitsOutputIdx=%d, numPlanOutputs=%d. "
                     "The plan output at this index is not logits - check logitsOutputIdx mapping.",
                     (long long)logitsRank, step,
                     (long long)logitsOutput->lengthOf(),
                     config->logitsOutputIdx, numPlanOutputs);

        // logitsOutput shape: [batch, seqLen, vocabSize] (rank 3) or [batch, vocabSize] (rank 2)
        // For rank 3: decode steps have seqLen=1 -> [1, 1, vocabSize], prefill -> [1, N, vocabSize]
        // For rank 2: always [batch, vocabSize] - treat as seqLen=1
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

        // -- ADR 0106 Phase 2 speculative path OR Phase 1 scalar path ------------
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
            // -- Speculative multi-row argmax --------------------------------------
            // logitsOutput shape: [1, W_max, vocab]. Rows 0..proposedCount are the
            // active positions filled by this step's forward (activeWindow=1+proposedCount).
            int numRows = 1 + proposedCount;
            // The contiguous device ptr for rows 0..numRows-1 is logitsOutput->specialBuffer()
            // (batch=1, so offset 0 IS row 0). Rows are stride-vocabVocab apart (contiguous).
            // ROUND 6 (finding 3): the same kernel writes per-row NaN flags so
            // EVERY active verification row is validity-checked at the
            // acceptance boundary — including a fully accepted batch, where
            // the rerun/recovery guard never executes.
            NDArray::prepareSpecialUse({specArgmaxDevice, specValidityDevice}, {logitsOutput});
            BUILD_SINGLE_SELECTOR(logitsOutput->dataType(), argmaxMultiRowLauncher,
                                  (stream, logitsOutput->specialBuffer(),
                                   specArgmaxDevice->specialBuffer(),
                                   static_cast<LongType>(numRows),
                                   logitsVocab,
                                   specValidityDevice->specialBuffer()),
                                  SD_FLOAT_TYPES);
            NDArray::registerSpecialUse({specArgmaxDevice, specValidityDevice}, {logitsOutput});

            // D2H: target rows and MTP drafts share the acceptance path's
            // existing synchronization. No predictor-side host boundary is added.
            LongType* argmaxDst = pinnedArgmax ? pinnedArgmax : stackArgmax;
            cudaMemcpyAsync(argmaxDst, specArgmaxDevice->specialBuffer(),
                            numRows * sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
            LongType* validityDst = pinnedValidity ? pinnedValidity : stackValidity;
            cudaMemcpyAsync(validityDst, specValidityDevice->specialBuffer(),
                            numRows * sizeof(LongType), cudaMemcpyDeviceToHost, *stream);
            LongType* mtpDraftDst = pinnedDraftIds ? pinnedDraftIds : stackDraftIds;
            if (useMtp) {
                cudaMemcpyAsync(mtpDraftDst, mtpDraftDevice->specialBuffer(),
                                proposedCount * sizeof(LongType),
                                cudaMemcpyDeviceToHost, *stream);
            }

            // Gated diagnostic D2H (rides the existing sync below - no new sync
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

            // -- D2H sync: wait for the existing async argmax/draft copies --
            p0.hostWaitBoundaries++;
            p0.hostReadbackBytes += static_cast<std::uint64_t>(numRows * sizeof(LongType) * 2
                                                               + proposedCount * sizeof(LongType));
            const auto acceptanceSync = cudaStreamSynchronize(*stream);
            // ROUND 7 (review finding 3): the acceptance decision consumes the
            // argmax, validity, and draft readbacks - they are only trustworthy
            // after a SUCCESSFUL synchronization, regardless of whether tensor
            // capture is enabled. The check moved out of the capture-only
            // branch; no new synchronization is added (this inspects the return
            // value of the boundary that already exists).
            REQUIRE_TRUE(acceptanceSync == cudaSuccess, 0,
                         "autoregressive_decode: acceptance readback failed: %s",
                         cudaGetErrorString(acceptanceSync));
            if (captureMtpInputs) {
                tensorDiagnostics.drainTensorSnapshots(reinterpret_cast<void*>(*stream));
            }
            emitCommittedStateSamples(step - 1);
            emitPreExecStateSamples(step);
            dumpStepInputSlices("spec", step, basePosition);
            emitPlanOutputFingerprints();
            if (kvSelfRowAfterBuf != nullptr && kvSelfRowAfterPos >= 0) {
                const LongType heads = kvSelfRowAfterBuf->sizeAt(2);
                const LongType dim = kvSelfRowAfterBuf->sizeAt(3);
                const LongType rowElems = heads * dim;
                const void* rowPtr = static_cast<const char*>(kvSelfRowAfterBuf->specialBuffer())
                                     + kvSelfRowAfterPos * rowElems * kvSelfRowAfterBuf->sizeOfT();
                std::vector<uint8_t> raw(4 * kvSelfRowAfterBuf->sizeOfT());
                cudaMemcpyAsync(raw.data(), rowPtr, raw.size(),
                                cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);
                float kAfter[4] = {};
                if (kvSelfRowAfterBuf->dataType() == DataType::FLOAT32) {
                    std::memcpy(kAfter, raw.data(), 16);
                } else {
                    for (int i = 0; i < 4; i++) {
                        unsigned h = raw[i * 2] | (raw[i * 2 + 1] << 8);
                        unsigned sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
                        float v = exp == 0 ? (man == 0 ? 0.0f : std::ldexp((float)man, -24))
                                  : exp == 0x1F ? std::numeric_limits<float>::quiet_NaN()
                                  : std::ldexp(1.0f + man / 1024.0f, (int)exp - 15);
                        kAfter[i] = sign ? -v : v;
                    }
                }
                DSP_DIAG(KV_CACHE,
                         "MTP_KV_SELFROW_AFTER pos=%lld after=[%.4f,%.4f,%.4f,%.4f]",
                         (long long)kvSelfRowAfterPos, kAfter[0], kAfter[1], kAfter[2], kAfter[3]);
                kvSelfRowAfterBuf = nullptr;
                kvSelfRowAfterPos = -1;
            }
            if (useMtp) {
                std::copy(mtpDraftDst, mtpDraftDst + proposedCount, draftIds);
            }
            if (useMtp && DSP_DIAG_ENABLED(KV_CACHE)) {
                // Chain-probe drain: samples were queued on the exec stream during
                // each executeMtpCuda; the synchronize above completed them.
                for (int cp = 0; cp < mtpChainSampled; cp++) {
                    DSP_DIAG(KV_CACHE,
                             "MTP_CHAIN_PROBE step=%d slot=%d tok=%lld dtype=%d bytes=%zu "
                             "carryInBits=[%016llx,%016llx] hidOutBits=[%016llx,%016llx]",
                             step, cp, (long long)mtpChainTok[cp],
                             static_cast<int>(config->mtpTargetHidden->dataType()), mtpChainSampleBytes[cp],
                             (unsigned long long)mtpChainCarryIn[cp][0], (unsigned long long)mtpChainCarryIn[cp][1],
                             (unsigned long long)mtpChainHidOut[cp][0], (unsigned long long)mtpChainHidOut[cp][1]);
                }
                mtpChainSampled = 0;
            }

            // -- Apply lossless accept rule -----------------------------------------
            // Input row i contains draftIds[i - 1] for i > 0, so target logits row i
            // predicts the token after that input. Therefore row 0 validates draft 0,
            // row 1 validates draft 1, etc. On the first mismatch at j, emit accepted
            // drafts [0,j) followed by target argmax[j] as the correction token. If
            // every draft matches, argmax[proposedCount] is the bonus token.
            // Snapshot raw target argmaxes before the emission rewrite below -
            // consumed by the gated KV_CACHE diagnostic event (host data already
            // synced by the D2H above; no additional synchronization).
            LongType argmaxRaw[8] = {};
            if (DSP_DIAG_ENABLED(KV_CACHE)) {
                for (int i = 0; i < 8 && i <= proposedCount; i++) argmaxRaw[i] = argmaxDst[i];
            }

            // ROUND 6 (finding 3) — VERIFIER VALIDITY GATE: every ACTIVE row
            // whose result feeds an acceptance decision or an emitted token
            // must be NaN-free. The old multi-row kernel returned token 0 for
            // an all-NaN row (never-replaced init), so an invalid bonus row
            // could reach emission through a fully accepted batch, where the
            // rerun/recovery guard never runs. This gate closes that path;
            // it runs BEFORE any acceptance decision, state commit, callback,
            // or metric. Rows beyond the active prefix (physical padding) are
            // NOT validated — padding never feeds acceptance.
            {
                bool anyInvalid = false;
                int firstInvalidRow = -1;
                for (int r = 0; r < numRows; r++) {
                    if (validityDst[r] != 0) { anyInvalid = true; firstInvalidRow = r; break; }
                }
                REQUIRE_TRUE(!anyInvalid, 0,
                             "autoregressive_decode: SPEC VERIFY VALIDITY GUARD step=%d "
                             "rows=%d proposed=%d - verification logits row %d contains "
                             "NaN; refusing to accept or emit from invalid results "
                             "(cause requires a dedicated trace)",
                             step, numRows, proposedCount, firstInvalidRow);
            }

            // Cross-check payloads: dump the verification input row tokens plus the
            // per-row native logits argmaxes so the window4 Java-parity harness can
            // be fed IDENTICAL ids rows and its row argmaxes compared apples-to-apples.
            LongType inputRows[8] = {};
            if (DSP_DIAG_ENABLED(KV_CACHE) && config->planOwnsKvScatter
                    && inputIds != nullptr && inputIds->dataType() == DataType::INT64
                    && inputIds->lengthOf() >= 1) {
                NDArray::prepareSpecialUse({inputIds}, {});
                std::vector<uint8_t> rowBytes(
                    std::min<LongType>(8, inputIds->lengthOf()) * sizeof(LongType));
                cudaMemcpyAsync(rowBytes.data(), inputIds->specialBuffer(), rowBytes.size(),
                                cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);
                for (int i = 0; i < (int)(rowBytes.size() / sizeof(LongType)); i++) {
                    std::memcpy(&inputRows[i], rowBytes.data() + i * sizeof(LongType),
                                sizeof(LongType));
                }
                NDArray::registerSpecialUse({inputIds}, {});
                DSP_DIAG(KV_CACHE,
                         "MTP_VERIFY_INPUTS step=%d basePos=%lld rows=[%lld,%lld,%lld,%lld,%lld] "
                         "nativeRowArgmax=[%lld,%lld,%lld,%lld,%lld]",
                         step, (long long)basePosition,
                         inputRows[0], inputRows[1], inputRows[2], inputRows[3], inputRows[4],
                         argmaxRaw[0], argmaxRaw[1], argmaxRaw[2], argmaxRaw[3], argmaxRaw[4]);
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
            // sequential - this only feeds the cap statistic). Unconditional
            // counting reaches MIN_EVALS in MIN_EVALS steps instead of waiting
            // for earlier positions to hit.
            if (useMtp) {
                for (int p = 0; p < proposedCount && p < 33; p++) {
                    mtpPosEvaluated[p]++;
                    if (argmaxDst[p] == draftIds[p]) mtpPosAccepted[p]++;
                }
                DSP_DIAG(KV_CACHE,
                         "MTP_POS_STATS step=%d proposed=%d draft0=%lld argmax0=%lld "
                         "accept=[%d/%d,%d/%d,%d/%d,%d/%d] argmaxRaw0=%lld argmaxRaw1=%lld",
                         step, proposedCount,
                         (long long)draftIds[0], (long long)argmaxDst[0],
                         mtpPosAccepted[0], mtpPosEvaluated[0],
                         mtpPosAccepted[1], mtpPosEvaluated[1],
                         mtpPosAccepted[2], mtpPosEvaluated[2],
                         mtpPosAccepted[3], mtpPosEvaluated[3],
                         (long long)argmaxRaw[0], (long long)argmaxRaw[1]);
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

            // -- ADR 0106 Phase 2b exit: multi-token commit restored ----------------
            // Token-exact parity was proven for the single-token commit
            // (milestone bc3f5c2a, emissionDeltas 0/251 with the dual-plan scalar
            // rerun). The multi-token contract now returns: a step commits
            // acceptedDrafts + 1 tokens (the accepted drafts plus the
            // correction/bonus), all from the accepted prefix. The state rerun
            // advances the trunk through exactly those tokens. Emission for the
            // committed prefix reconstructs the lossless verify sequence; rows
            // beyond the committed prefix are never trusted (the W-row graph
            // divergence is documented and measured). The carry comes from the
            // last committed row. The scalar width-1 plan can only serve a
            // single-row commit, so multi-row reruns route through the WINDOW
            // plan with activeWindow=consumedCount.
            // Multi-token commit (Phase 2b exit): consume the accepted prefix
            // [0, acceptedDrafts] row by row, feeding the stop matcher
            // PROVISIONALLY against a pre-step snapshot. No persistent matcher
            // mutation survives past the finalize step below - if the rerun
            // invalidates any part of the provisional sequence, restore(snapshot)
            // re-establishes the exact pre-step state (including evicted
            // history), which rollback(n) cannot do for a bounded suffix.
            // Scalar-bound production participates in multi-token: when drafts
            // are accepted the state rerun routes through the WINDOW plan
            // (activeWindow=consumedCount) below; the scalar width-1 plan serves
            // only single-row commits (acceptedDrafts == 0). The window path's
            // equivalence is asserted by the teacher-forced contract suites and
            // the real-model gate's n>1 EMITTED_STEP evidence.
            int consumedCount = 0;
            bool shouldStop = false;
            auto matcherSnapshot = stopMatcher.snapshot();
            // COMMIT POLICY (allowMultiRowCommit): false (shipped default) caps
            // the consume at one row - the scalar width-1 plan then owns the
            // state rerun and emission stays bit-exact with greedy. true
            // (experimental) consumes the full accepted prefix and routes the
            // rerun through the window plan.
            const int commitCap = config->allowMultiRowCommit ? 1 + acceptedDrafts : 1;
            while (consumedCount < commitCap
                    && tokensGenerated + consumedCount < maxNewTokens) {
                LongType token = argmaxDst[consumedCount];
                consumedCount++;
                bool matchedStop = stopMatcher.accept(token);
                shouldStop = matchedStop
                    && stopTerminationAllowed(config, tokensGenerated + consumedCount);
                if (shouldStop) break;
            }
            if (consumedCount == 0) {
                // Budget exhausted BEFORE the first row: consume nothing, emit
                // nothing, leave matcher untouched, and TERMINATE - no token was
                // committed, so acceptance statistics must not count this step's
                // proposal as emitted output, and no further speculative work
                // follows (the outer loop condition can only re-reach this same
                // state). Do not force consumedCount to 1 to hide a control-flow
                // error.
                config->activeWindow = 1;
                NDArray::registerSpecialUse({sampledToken}, {logitsOutput});
                break;
            }
            // Finalized emission truncation note: carryRow is re-derived below
            // (inside the rerun-transaction block) from the FINALIZED
            // consumedCount - a truncated commit must not leave the predictor
            // carry or the state commit position sample describing tokens that
            // were never emitted.

            // -- ADR 0106 Phase 2 / Phase 2b: authoritative state commit ------------
            // The W-wide verification forward advanced GDN/conv state through ALL
            // proposed rows. The commit must advance state through exactly the
            // committed token(s): re-execute with actual_sequence_length =
            // consumedCount (always 1 under the single-token commit) so the trunk
            // recurrent state, KV rows, and positions equal what greedy decoding
            // would hold after the same token. The re-run sees the same step
            // inputs (input_ids/mask/positions unchanged; in-graph KV writes are
            // idempotent for the same step). Next step's pre-exec update rewrites
            // actual_sequence_length, so no restore needed.
            // NOTE (proc-149): with the old multi-row emission this rerun fired
            // every step; under the single-token commit it fires whenever the
            // window is wider than the commit (proposedCount > 0), which is every
            // speculative step. Semantics unchanged, label updated for honesty.
            // RERUN-REFRESHED EMISSION: rerunRefreshedToken carries the rerun's
            // scalar-logits argmax back to the emission rewrite below; -1 means
            // no rerun fired and the verification argmax stays authoritative.
            // SCRATCH DOMAIN (Stage 2): the refresh argmax is written to the
            // dedicated rerunScratch buffer, NEVER to specArgmaxDevice - the
            // predictor repair loop and pending-input publication read the
            // COMMITTED token sequence from specArgmaxDevice, which must not be
            // clobbered by a diagnostic rerun.
            LongType rerunRefreshedToken = -1;
            NDArray* rerunScratch = nullptr;
            // SCRATCH DOMAIN: dedicated rerun-refresh scratch, shared by BOTH
            // speculators (useSpeculative = useNgram || useMtp both enter the
            // rerun block below). The committed specArgmaxDevice sequence must
            // never be overwritten by a diagnostic rerun.
            if (useSpeculative) {
                // Two INT64 slots, allocated once per decode call, stable address:
                // slot 0 = rerun argmax, slot 1 = full-row NaN validity flag
                // written by the argmax kernel's validity variant (finding 5).
                if (mtpRerunScratch == nullptr) {
                    std::vector<LongType> scratchShape{2};
                    mtpRerunScratch = NDArrayFactory::create_('c', scratchShape, DataType::INT64);
                }
                rerunScratch = mtpRerunScratch;
            }
            if (consumedCount < 1 + proposedCount
                    && config->actualSequenceLengthExtIdx >= 0
                    && config->actualSequenceLengthExtIdx < numExtInputs
                    && extInputs[config->actualSequenceLengthExtIdx] != nullptr) {
                // T3b parity fix (step-99 flip, 1536 vs 5218): the rerun must be
                // width-1 in GEOMETRY, not just in recurrent row count. asl only
                // gates GDN/conv; attention/GEMM/softmax otherwise run the frozen
                // W-substrate geometry, whose row-0 numerics equal the verify
                // pass's row 0 (both produced 1536 where greedy produced 5218).
                // Refill the window tensors to activeWindow=1 - exactly what the
                // greedy scalar path presents - before re-executing, then restore
                // the verification window below. Costs nothing: a second pass
                // already runs every step.
                // PLAN SELECTION FIRST (Stage 3): choose the executing geometry,
                // then refill the window tensors ONCE for that geometry.
                //  - scalarRerun: the validated width-1 scalar plan executes
                //    (greedy-identical geometry, proven parity).
                //  - otherwise (multiRowRerun, experimental): the WINDOW plan
                //    executes with its logical active prefix set to consumedCount
                //    REGARDLESS of scalar-binding availability.
                const bool multiRowRerun = consumedCount > 1;
                const bool scalarRerun = useScalarTarget && !multiRowRerun;
                const int rerunActiveWindow = multiRowRerun ? consumedCount : 1;
                config->activeWindow = rerunActiveWindow;
                if (useWindowSubstrate && config->windowMax > 1) {
                    NDArray* wMask = config->windowGridMask;
                    NDArray* wPos  = config->windowPositionGrid;
                    LongType wMax  = static_cast<LongType>(config->windowMax);
                    LongType aW    = static_cast<LongType>(rerunActiveWindow);
                    LongType rowLen = wMask->sizeAt(3);
                    // ACCESS BOOKKEEPING (Stage 3): refill kernels write ONLY
                    // wMask/wPos; inputIds is not a fabricated output.
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
                    }
                    // inputIds is NOT written here - the refill kernels only touch
                    // wMask/wPos. No manual synchronization: inputIds' device buffer
                    // is already current (host-written once at handoff,
                    // device-authoritative since); the plan reads it as device-resident.
                    // For a width-one rerun the frozen plan still runs W-wide but
                    // consumes only row 0; for a multi-row rerun the plan consumes
                    // rows [0, consumedCount-1] - both keyed off activeWindow.
                    if (wPos != nullptr) NDArray::registerSpecialUse({wMask, wPos}, {});
                    else NDArray::registerSpecialUse({wMask}, {});
                }
                NDArray* aslArr = extInputs[config->actualSequenceLengthExtIdx];
                NDArray::prepareSpecialUse({aslArr}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                    aslArr->specialBuffer(), static_cast<LongType>(consumedCount));
                NDArray::registerSpecialUse({aslArr}, {});
                DSP_DIAG(KV_CACHE,
                         "SPEC_STATE_RERUN step=%d proposed=%d accepted=%d "
                         "consumed=%d geometry=%s - re-executing for authoritative state advance",
                         step, proposedCount, acceptedDrafts, consumedCount,
                         scalarRerun ? "scalar-width-1" : "window");
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
                // The scalar binding supplies width-one arrays for single-row commits.
                // Multi-row commits route through the WINDOW plan (activeWindow was
                // set to consumedCount above, independent of the binding); its
                // W-wide arrays are already wired.
                // STATE RESTORE (window4 + K=1 evidence): the verify pass MUTATES
                // its recurrent ext inputs in place (conv/gdn state advanced
                // through the full W rows, including the rejected suffix). The
                // window4 teacher-forced gate proves the chained-scalar
                // equivalence holds ONLY from the pre-step state
                // (acceptedZero/partialRerunState discriminators exact), so the
                // DEEP pre-verification recurrent snapshots (capturePreVerificationState,
                // taken before this step's verify pass) must be restored into
                // whatever storage the rerun executes from - for BOTH rerun
                // geometries:
                //  - window rerun: restore into the LIVE window ext inputs the
                //    W plan reads (also the only inputs a bindingless window
                //    rerun has - there, they ARE the window plan's inputs);
                //  - scalar rerun (K=1 state-poisoning regression,
                //    /tmp/mtp-k1-diag.log: step-1 rerun FLIPPED to argmax 0 with
                //    all-NaN logits, then commitRecurrentState poisoned every
                //    later step): prepareScalarTarget copied the pre-verify
                //    state into the private width-1 arrays, but the plan stages
                //    its OWN private replay storage at executeSteadyState -
                //    whatever it last executed with. If a previous geometry ran
                //    at a different width (the K=0 maintenance forward, or the
                //    pre-verify pass itself when the scalar plan fell back to
                //    the window substrate), stale post-verify state survives
                //    into the replay and the rerun double-advances. Re-running
                //    prepareScalarTarget AFTER the window restore re-establishes
                //    the private width-1 arrays (geometry to the rerun's asl=1,
                //    recurrent to the pre-verification state) from the restored
                //    live ext inputs, and the executeScalarTarget staging right
                //    below then carries exactly the pre-step state into the
                //    private replay.
                //  - SHARED-BUFFER ALIASING FIX (mtp-fix-gate2 NaN GUARD,
                //    step=1 geometry=scalar-width-1): the old window-geometry
                //    restore sourced its memcpy from the SCALAR arrays with a
                //    buffer-identity skip, so whenever the scalar plan's
                //    recurrent NDArrays share DataBuffers with the target's
                //    window ext inputs (Qwen's 23 GDN pairs + conv pairs) the
                //    restore was skipped or copied identical bytes and the
                //    rerun executed from post-verify state. The restore source
                //    is now the DEEP pre-verification snapshots taken before
                //    this step's verification pass - owned allocations that
                //    cannot alias the live ext inputs - and the buffer-identity
                //    skip is gone from the recurrent restore path entirely.
                // The window restore covers BOTH geometries: the window plan
                // reads the live ext inputs directly, and the scalar plan reads
                // them through the prepareScalarTarget() re-stage below (its
                // recurrent copy runs whenever the scalar arrays are NOT
                // buffer-identical; when they ARE identical, the scalar arrays
                // ARE the just-restored live storage). Buffer identity therefore
                // no longer decides what the rerun sees.
                restorePreVerificationState();
                // Verdict-c fix: restore the shared KV rows the W-wide verify
                // overwrote (draft-conditioned K/V in [base, base+K)) so the
                // captured scalar graph reads the W=1 rows it expects - the
                // deterministic-NaN source identified by SCALAR_RETRY (gate 5).
                restorePreVerificationKvRows();
                if (useScalarTarget) {
                    // Scalar-geometry re-stage from the restored live ext
                    // inputs: recurrent to the pre-verification state, geometry
                    // to the rerun's asl=1 - so the executeScalarTarget staging
                    // below cannot replay stale post-verify state left by an
                    // earlier different-width run.
                    prepareScalarTarget();
                }
                // RERUN-INPUT FINITENESS AUDIT (gate-3 discriminator): after the
                // deep-snapshot restore + scalar re-stage, immediately BEFORE
                // executing the rerun, sample the first bytes of every
                // recurrent EXT input the rerun will consume (GDN + conv pairs
                // + one KV head row at the base position) and fail/loudly log
                // NaN presence. This splits the remaining hypotheses:
                //  - NaN here => the restore/stage path itself delivered
                //    poisoned bytes (snapshot capture or restore order bug).
                //  - all finite here but the rerun output is NaN => the
                //    captured scalar plan's internal staging (executeSteadyState
                //    replay buffers) is the poison vector - an in-plan defect,
                //    not an ext-input one.
                // Gated on KV_CACHE diagnostics: diagnostics-off hot path pays
                // nothing (the guard below already runs unconditionally).
                if (DSP_DIAG_ENABLED(KV_CACHE)) {
                    bool extInputNan = false;
                    const char* poisonName = "none";
                    auto probeInput = [&](NDArray* arr, const char* what) {
                        if (extInputNan || arr == nullptr
                                || arr->dataType() != DataType::FLOAT32
                                || arr->lengthOf() < 4) return;
                        NDArray::prepareSpecialUse({}, {arr});
                        float probe[4] = {};
                        cudaMemcpyAsync(probe, arr->specialBuffer(),
                                        sizeof(probe), cudaMemcpyDeviceToHost, *stream);
                        cudaError_t probeSync = cudaStreamSynchronize(*stream);
                        REQUIRE_TRUE(probeSync == cudaSuccess, 0,
                                     "autoregressive_decode: rerun-input probe sync failed: %s",
                                     cudaGetErrorString(probeSync));
                        NDArray::registerSpecialUse({}, {arr});
                        for (int i = 0; i < 4; i++) {
                            if (std::isnan(probe[i])) {
                                extInputNan = true;
                                poisonName = what;
                            }
                        }
                    };
                    for (int s = 0; s < config->numGdnStatePairs && !extInputNan; s++) {
                        int idx = config->gdnStateExtIndices != nullptr
                            ? config->gdnStateExtIndices[s] : -1;
                        probeInput((idx >= 0 && idx < numExtInputs) ? extInputs[idx] : nullptr,
                                   "gdn");
                    }
                    for (int s = 0; s < config->numConvStatePairs && !extInputNan; s++) {
                        int idx = config->convStateExtIndices != nullptr
                            ? config->convStateExtIndices[s] : -1;
                        probeInput((idx >= 0 && idx < numExtInputs) ? extInputs[idx] : nullptr,
                                   "conv");
                    }
                    DSP_DIAG(KV_CACHE,
                             "RERUN_INPUT_AUDIT step=%d geometry=%s extNaN=%d poison=%s",
                             step, scalarRerun ? "scalar-width-1" : "window",
                             extInputNan ? 1 : 0, poisonName);
                }
                p0.acceptedPrefixReruns++;
                Status rerunStatus = Status::OK;
                {
                    // Re-execute the fixed-width window plan from the restored
                    // pre-verification state. The mask, positions and actual
                    // sequence length above select the consumed prefix;
                    // physical tensor width remains config->windowMax.
                    DSP_DIAG(KV_CACHE,
                             "PREFIX_RERUN_EXECUTE step=%d activeWindow=%d physicalWidth=%d",
                             step, rerunActiveWindow, config->windowMax);
                    rerunStatus = plan->executeSteadyState(
                        extInputs, numExtInputs, planOutputs, numPlanOutputs,
                        reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
                }
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
                             static_cast<int>(rerunStatus), acceptedDrafts, proposedCount);

                // The rerun ran at actual_sequence_length=1: its logits are the
                // W=1 (greedy-geometry) continuation of the committed base token.
                // Argmax it and use it as the emitted token so emission, state,
                // carry, and KV all come from the same scalar pass.
                if (rerunStatus == Status::OK) {
                    NDArray* rerunLogits = planOutputs[config->logitsOutputIdx];
                    REQUIRE_TRUE(rerunLogits != nullptr && rerunLogits->rankOf() >= 2,
                                 0, "autoregressive_decode: rerun logits output is invalid "
                                    "at step %d (rank=%lld)",
                                 step,
                                 rerunLogits != nullptr
                                     ? static_cast<long long>(rerunLogits->rankOf()) : -1LL);
                    LongType rerunVocab = rerunLogits->sizeAt(rerunLogits->rankOf() - 1);
                    // SCRATCH DOMAIN (Stage 2): the rerun argmax writes to the
                    // dedicated rerunScratch buffer. specArgmaxDevice holds the
                    // committed token sequence; overwriting its row 0 here made
                    // predictor repair read a token absent from the authoritative
                    // emitted prefix on multi-row reruns.
                    // FINDING 5 (single-wait batch): the validity argmax writes
                    // BOTH the argmax and the FULL-ROW NaN flag on device; the
                    // argmax D2H, the diagnostics top-8 sample, and the GDN head
                    // sample below all ride ONE stream sync - the previous code
                    // waited separately for the argmax, the (FP32-only, 8-entry)
                    // logits sample, and the state sample.
                    NDArray::prepareSpecialUse({rerunScratch}, {rerunLogits});
                    BUILD_SINGLE_SELECTOR(rerunLogits->dataType(), argmaxValidityLauncher,
                                          (stream, rerunLogits->specialBuffer(),
                                           rerunScratch->specialBuffer(),
                                           rerunVocab),
                                          SD_FLOAT_TYPES);
                    struct { LongType argmax; LongType nanFlag; } rerunReadback = {};
                    cudaMemcpyAsync(&rerunReadback, rerunScratch->specialBuffer(),
                                    sizeof(rerunReadback), cudaMemcpyDeviceToHost, *stream);
                    // Diagnostics-only top-8 sample (finding 5: the unconditional
                    // FP32-only probe became gated - the NaN GUARD no longer
                    // depends on host-side samples at all).
                    const bool rerunSample = DSP_DIAG_ENABLED(KV_CACHE)
                        && rerunVocab >= 8 && rerunLogits->dataType() == DataType::FLOAT32;
                    float rerunTop8[8] = {};
                    if (rerunSample) {
                        cudaMemcpyAsync(rerunTop8, rerunLogits->specialBuffer(),
                                        8 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
                    }
                    // GDN head-of-pair-0 sample rides the same wait (same class:
                    // a NaN here means the rerun executed from mutated state).
                    // Bytes are converted per the state's own dtype after the
                    // single sync (dtype-safe: no raw float[] reinterpretation).
                    bool rerunStateNan = false;
                    NDArray* gdnOut = nullptr;
                    if (config->numGdnStatePairs > 0 && config->gdnStateOutputIndices != nullptr) {
                        // Probe the head of GDN state pair 0's OUTPUT from the
                        // rerun pass itself (planOutputs is target-domain output
                        // indexed; for a scalar rerun executeScalarTarget remapped
                        // these slots onto the scalar plan's outputs). Probing the
                        // ext input here would read the still-uncommitted pre-verify
                        // state instead of what the rerun just produced.
                        int gdnOut0 = config->gdnStateOutputIndices[0];
                        gdnOut = (gdnOut0 >= 0 && gdnOut0 < numPlanOutputs)
                            ? planOutputs[gdnOut0] : nullptr;
                    }
                    const bool gdnProbe = gdnOut != nullptr && gdnOut->lengthOf() >= 4;
                    // DTYPE-SAFE STATE SAMPLE (review round 5, finding 3): the
                    // probe copies exactly ONE element's bytes per slot and
                    // converts on the host through the state's OWN dtype. The
                    // previous code memcpy'd 16 raw bytes into float[] - for
                    // BF16/FP16 state that read PAST the tensor's storage and
                    // for FP64 it reinterpreted halves; neither was a
                    // conversion. sampleFirstRowValueCpu converts for every
                    // SD_FLOAT_TYPES dtype from a device-visible pointer? No -
                    // it reads HOST memory, so the bytes must arrive on the
                    // host first: copy 4 * sizeOfT raw bytes, then convert.
                    std::vector<uint8_t> gdnRaw;
                    if (gdnProbe) {
                        NDArray::prepareSpecialUse({}, {gdnOut});
                        gdnRaw.resize(static_cast<size_t>(4) * gdnOut->sizeOfT());
                        cudaMemcpyAsync(gdnRaw.data(), gdnOut->specialBuffer(),
                                        gdnRaw.size(), cudaMemcpyDeviceToHost, *stream);
                    }
                    NDArray::registerSpecialUse({rerunScratch}, {rerunLogits});
                    // The emission/storage path below re-reads argmaxDst from host
                    // memory only, so a stream-ordered completion of this D2H before
                    // the rewrite is required. ONE sync drains argmax + validity +
                    // diagnostics + state samples.
                    p0.hostWaitBoundaries++;
                    p0.hostReadbackBytes += sizeof(LongType) * 2;
                    cudaError_t refreshSync = cudaStreamSynchronize(*stream);
                    REQUIRE_TRUE(refreshSync == cudaSuccess, 0,
                                 "autoregressive_decode: rerun emission refresh sync "
                                 "failed at step %d: %s", step,
                                 cudaGetErrorString(refreshSync));
                    const LongType rerunRefreshedTokenReadback = rerunReadback.argmax;
                    const bool rerunNanFlag = rerunReadback.nanFlag != 0;
                    if (gdnProbe) {
                        NDArray::registerSpecialUse({}, {gdnOut});
                        const char* gdnBase = reinterpret_cast<const char*>(gdnRaw.data());
                        for (int i = 0; i < 4; i++) {
                            float sampled = 0.0f;
                            BUILD_SINGLE_SELECTOR(gdnOut->dataType(), sampleRawHostValue,
                                                  (gdnBase + i * gdnOut->sizeOfT(), &sampled),
                                                  SD_FLOAT_TYPES);
                            if (std::isnan(sampled)) { rerunStateNan = true; break; }
                        }
                    }
                    rerunRefreshedToken = rerunRefreshedTokenReadback;
                    DSP_DIAG(KV_CACHE,
                             "RERUN_EMISSION_REFRESH step=%d verifyRow0=%lld "
                             "rerunArgmax=%lld%s nanFlag=%d - emission taken from the asl=1 pass",
                             step, (long long)argmaxDst[0], (long long)rerunRefreshedToken,
                             rerunRefreshedToken != argmaxDst[0] ? " FLIPPED" : "",
                             rerunNanFlag ? 1 : 0);
                    if (rerunSample) {
                    DSP_DIAG(KV_CACHE,
                             "RERUN_TOP8 step=%d pos=%lld logits8=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]",
                             step, (long long)currentPosition,
                             rerunTop8[0], rerunTop8[1], rerunTop8[2], rerunTop8[3],
                             rerunTop8[4], rerunTop8[5], rerunTop8[6], rerunTop8[7]);
                    }

                    // FAIL-LOUD NON-FINITE GUARD (K=1 state-poisoning regression):
                    // the refresh D2H above already completed a stream sync, so
                    // the flag reflects the rerun's committed results. The DEVICE
                    // full-row flag covers EVERY float dtype and EVERY vocab
                    // entry (finding 5) - non-finite logits or a non-finite GDN
                    // state sample mean the rerun's committed results are not
                    // trustworthy; committing them would poison every later step
                    // (silent token-0 collapse, /tmp/mtp-k1-diag.log). Fail here
                    // with the observed values and execution context; the guard
                    // does NOT assert a specific cause (review round 5, finding
                    // 3) - causal attribution needs a dedicated trace.
                    const bool rerunLogitsNan = rerunNanFlag;
                    REQUIRE_TRUE(!(rerunLogitsNan || rerunStateNan), 0,
                                 "autoregressive_decode: SPEC RERUN NON-FINITE GUARD step=%d "
                                 "geometry=%s rerunArgmax=%lld verifyRow0=%lld "
                                 "rerunLogitsNaN=%d rerunGdnStateNaN=%d - the rerun "
                                 "produced non-finite committed results; refusing to "
                                 "commit them (cause requires a dedicated trace)",
                                 step, scalarRerun ? "scalar-width-1" : "window",
                                 (long long)rerunRefreshedToken, (long long)argmaxDst[0],
                                 rerunLogitsNan ? 1 : 0, rerunStateNan ? 1 : 0);
                }
            }
            // -- FINALIZED EMISSION SEQUENCE (review round 2) ---------------------
            // Reconstruct the lossless verify emission, apply the authoritative
            // scalar-refresh winner, count acceptance from the FINALIZED tokens,
            // and upload the sequence BEFORE predictor repair and pending-input
            // publication read it. setMtpNextInputCuda takes a DEVICE COPY into
            // the predictor's input array; publishing it before the refresh
            // rewrite left the OLD verification token there whenever the scalar
            // rerun disagreed - the predictor would continue a token the target
            // never emitted.
            // Reconstruction: accepted drafts restore their proposal values and
            // the correction/bonus takes the verify argmax at the first
            // unaccepted row.
            LongType correctionOrBonus = argmaxDst[acceptedDrafts];
            for (int i = 0; i < acceptedDrafts; i++) {
                argmaxDst[i] = draftIds[i];
            }
            argmaxDst[acceptedDrafts] = correctionOrBonus;
            int n = consumedCount;
            // Emission rewrite. The rerun's row-0 argmax is authoritative
            // whenever the rerun executed: it recomputed the row-0 readout from
            // the committed pre-step state at the shortened recurrence length.
            // FINALIZE-BEFORE-COMMIT TRANSACTION (K=1 state-poisoning review):
            // the emission must be FINALIZED - truncated where the authoritative
            // rerun readout invalidates the stale verification suffix - BEFORE
            // any state commit, KV scatter, or callback runs. On a rerun
            // DISAGREEMENT the old suffix (rows >= 1) was conditioned on the
            // superseded token and is INVALID: the commit is truncated to the
            // rerun-consistent row-0 prefix and the suffix is re-derived by a
            // fresh step - the new row 0 is never spliced onto the old suffix.
            // With the shipped single-row commit (commitCap == 1) the rerun
            // refresh IS the authoritative emission. The matcher snapshot is
            // re-established and each finalized token is fed exactly once.
            int carryRow = consumedCount - 1;
                if (rerunRefreshedToken >= 0 && rerunRefreshedToken != argmaxDst[0]) {
                    LongType supersededRow0 = argmaxDst[0];
                    argmaxDst[0] = rerunRefreshedToken;
                    if (consumedCount > 1) {
                        const int rerunWidthM = consumedCount;
                        DSP_DIAG(KV_CACHE,
                                 "RERUN_TRUNCATE_COMMIT step=%d basePos=%lld "
                                 "supersededRow0=%lld rerunRow0=%lld committed=%d -> n=1 - "
                                 "stale suffix re-derived by the next step",
                                 step, (long long)basePosition, (long long)supersededRow0,
                                 (long long)rerunRefreshedToken, consumedCount);
                        consumedCount = 1;
                        n = 1;
                        shouldStop = false;

                        // SHORTENED-PREFIX STATE RECOVERY (review round 3,
                        // finding 3): the multi-row rerun's planOutputs hold
                        // recurrent state AFTER m consumed inputs, but the
                        // finalized emission is now ONE token - committing that
                        // state would pair a 1-token history with m-token
                        // recurrence (returned tokens and committed state
                        // describing different histories). Re-derive BOTH from
                        // one width-1 execution: restore the pre-verify
                        // snapshots (owned arrays, never aliased by the plan,
                        // so they still hold the pre-step state), refill the
                        // window geometry to activeWindow=1 + asl=1, re-execute
                        // the window plan, and take the emission readout AND
                        // the committed state from THIS pass.
                        restorePreVerificationState();
                        restorePreVerificationKvRows();
                        config->activeWindow = 1;
                        if (useWindowSubstrate && config->windowMax > 1) {
                            NDArray* wMask = config->windowGridMask;
                            NDArray* wPos  = config->windowPositionGrid;
                            LongType wMax  = static_cast<LongType>(config->windowMax);
                            LongType rowLen = wMask->sizeAt(3);
                            if (wPos != nullptr) NDArray::prepareSpecialUse({wMask, wPos}, {});
                            else NDArray::prepareSpecialUse({wMask}, {});
                            LongType totalElems = wMax * rowLen;
                            int threads = 256;
                            int blocks = static_cast<int>((totalElems + threads - 1) / threads);
                            fillWindowMaskKernel<<<blocks, threads, 0, *stream>>>(
                                wMask->specialBuffer(), wMax, rowLen, currentPosition, 1,
                                WINDOW_MASK_FILL);
                            if (wPos != nullptr) {
                                fillWindowPositionGridKernel<<<1, static_cast<int>(wMax), 0, *stream>>>(
                                    wPos->specialBuffer(), wMax, currentPosition, 1);
                            }
                            if (wPos != nullptr) NDArray::registerSpecialUse({wMask, wPos}, {});
                            else NDArray::registerSpecialUse({wMask}, {});
                        }
                        {
                            NDArray* aslArr = extInputs[config->actualSequenceLengthExtIdx];
                            NDArray::prepareSpecialUse({aslArr}, {});
                            updatePositionIdsKernel<<<1, 1, 0, *stream>>>(
                                aslArr->specialBuffer(), static_cast<LongType>(1));
                            NDArray::registerSpecialUse({aslArr}, {});
                        }
                        if (config->kvQuantFormat > 0 && config->kvScaleBuffers != nullptr
                            && config->kvInputExtIndices != nullptr) {
                            std::vector<NDArray*> kvQuantPtrsShort(numKvPairs);
                            for (int ki = 0; ki < numKvPairs; ki++) {
                                int extIdx = config->kvInputExtIndices[ki];
                                kvQuantPtrsShort[ki] = (extIdx >= 0 && extIdx < numExtInputs)
                                    ? extInputs[extIdx] : nullptr;
                            }
                            setKvScaleRegistry(kvQuantPtrsShort.data(), config->kvScaleBuffers,
                                               numKvPairs);
                        }
                        DSP_DIAG(KV_CACHE,
                                 "RERUN_SHORTEN_REEXEC step=%d oldM=%d - width-1 "
                                 "re-execution so committed state matches the 1-token "
                                 "history",
                                 step, rerunWidthM);
                        p0.shortenedRecoveryForwards++;
                        Status shortenStatus = plan->executeSteadyState(
                            extInputs, numExtInputs, planOutputs, numPlanOutputs,
                            reinterpret_cast<void*>(const_cast<cudaStream_t*>(stream)));
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
                        // Authoritative row-0 readout from the width-1 pass -
                        // the SAME computation the committed state derives
                        // from, so emission and state are self-consistent.
                        // FINDING 5: the validity argmax carries the FULL-ROW
                        // NaN flag on device (every dtype, every entry); the
                        // old FP32-only 8-entry host probe and its extra sync
                        // are gone - one sync drains argmax + flag.
                        NDArray* shortenLogits = planOutputs[config->logitsOutputIdx];
                        REQUIRE_TRUE(shortenLogits != nullptr && shortenLogits->rankOf() >= 2, 0,
                                     "autoregressive_decode: shortened rerun logits output is "
                                     "invalid at step %d", step);
                        LongType shortenVocab = shortenLogits->sizeAt(shortenLogits->rankOf() - 1);
                        NDArray::prepareSpecialUse({rerunScratch}, {shortenLogits});
                        BUILD_SINGLE_SELECTOR(shortenLogits->dataType(), argmaxValidityLauncher,
                                              (stream, shortenLogits->specialBuffer(),
                                               rerunScratch->specialBuffer(),
                                               shortenVocab),
                                              SD_FLOAT_TYPES);
                        struct { LongType argmax; LongType nanFlag; } shortenReadback = {};
                        cudaMemcpyAsync(&shortenReadback, rerunScratch->specialBuffer(),
                                        sizeof(shortenReadback), cudaMemcpyDeviceToHost, *stream);
                        NDArray::registerSpecialUse({rerunScratch}, {shortenLogits});
                        p0.hostWaitBoundaries++;
                        p0.hostReadbackBytes += sizeof(LongType) * 2;
                        cudaError_t shortenSync = cudaStreamSynchronize(*stream);
                        REQUIRE_TRUE(shortenSync == cudaSuccess, 0,
                                     "autoregressive_decode: shortened rerun readback sync "
                                     "failed at step %d: %s", step,
                                     cudaGetErrorString(shortenSync));
                        LongType shortenedToken = shortenReadback.argmax;
                        const bool shortenNan = shortenReadback.nanFlag != 0;
                        REQUIRE_TRUE(!shortenNan, 0,
                                     "autoregressive_decode: SHORTEN REEXEC NaN step=%d - "
                                     "width-1 re-execution produced NaN logits; refusing to "
                                     "commit", step);
                        // Emission AND state now come from this pass.
                        rerunRefreshedToken = shortenedToken;
                        argmaxDst[0] = shortenedToken;
                    }
                    stopMatcher.restore(matcherSnapshot);
                bool matchedStop = false;
                carryRow = consumedCount - 1;
                for (int i = 0; i < n; i++) {
                    matchedStop = stopMatcher.accept(argmaxDst[i])
                        && stopTerminationAllowed(config, tokensGenerated + i + 1);
                    if (matchedStop) {
                        // Truncate the committed prefix at the stop boundary.
                        n = i + 1;
                        consumedCount = n;
                        break;
                    }
                }
                shouldStop = matchedStop;
            }

            // Commit recurrent state from the (possibly re-run) accepted-prefix
            // pass - only AFTER the emission sequence is finalized (the commit
            // transaction: no state commit before finalize, no token mutation
            // after state commit).
            commitRecurrentState();
            queueCommittedStateSamples(
                step, basePosition + consumedCount, true);

            totalSpeculativeProposed += proposedCount;
            // Accepted drafts ACTUALLY EMITTED: count each emitted token that
            // still equals its draft. A scalar refresh that flipped the emitted
            // token away from its draft means that draft was not emitted; the
            // verification agreement alone is not an emission fact.
            int acceptedEmitted = 0;
            for (int i = 0; i < acceptedDrafts && i < n; i++) {
                if (argmaxDst[i] == draftIds[i]) acceptedEmitted++;
            }
            totalSpeculativeAccepted += acceptedEmitted;
            speculativeStepCount++;

            // Upload the FINALIZED sequence so the D2D storage path remains
            // stream-ordered and every device consumer below - predictor repair,
            // pending-input publication, token storage - reads the authoritative
            // tokens. Nothing mutates a token after this point: the state commit
            // above was finalized against exactly this sequence, so the KV
            // scatter and callbacks below run on the same authoritative prefix.
            NDArray::prepareSpecialUse({specArgmaxDevice}, {});
            cudaMemcpyAsync(specArgmaxDevice->specialBuffer(), argmaxDst,
                            n * sizeof(LongType), cudaMemcpyHostToDevice, *stream);
            NDArray::registerSpecialUse({specArgmaxDevice}, {});

            if (useMtp) {
                REQUIRE_TRUE(config->targetHiddenOutputIdx >= 0
                                 && config->targetHiddenOutputIdx < numPlanOutputs
                                 && planOutputs[config->targetHiddenOutputIdx] != nullptr,
                             0, "autoregressive_decode: target hidden output is unavailable for CUDA MTP");

                // The proposal loop above wrote predictor KV rows
                // [base, base + proposedCount - 1] exactly: one row per
                // executeMtpCuda call, positions base..base+K-1. Chained slots
                // (p >= 1) write rows with self-propagated predictor hidden;
                // a fully accepted step additionally commits rows beyond that
                // range (the correction/bonus row), which nothing else writes.
                // There is NO case where a committed row beyond
                // base + proposedCount - 1 was already correctly written, so
                // the previous `carryRow == proposedCount` bump (which claimed
                // base + proposedCount was already processed) was wrong: it
                // suppressed the only repair iteration covering the first
                // fully-accepted committed row and left a permanent hole in the
                // predictor KV cache (observed as acceptance collapse after the
                // first full accept in K=1).
                // PREDICTOR ROW GEOMETRY (packet 2): the proposal loop passed
                // TARGET positions base..base+K-1 to executeMtpCuda, so the
                // PREDICTOR rows written are [base-1, base+K-2] (r = P - 1).
                // In predictor-row coordinates (m = consumedCount):
                //   proposal write horizon (half-open): [base-1, base+K-1)
                //   retained consumed rows:             [base-1, base+m-1)
                //   rejected rows to mask:              [base+m-1, base+K-1)
                //   pending row after publication:      base+m-1
                //
                // Predictor-side accepted-prefix repair. Chained proposal calls
                // (slot>=1) wrote predictor KV rows [base+1, base+K-1] with each
                // draft's own recursively propagated hidden as the carry input,
                // but the reference contract pairs every retained position q with
                // the target's output hidden at q-1. A fully accepted K=1 step
                // leaves the bonus row unwritten by the prefix altogether. The
                // rerun above repairs only the target plan's KV/GDN state; the
                // predictor keeps its own position-keyed cache, so rewrite each
                // committed row here as fused(committed token, target hidden at
                // q-1), mirroring the target's accepted-prefix re-execution.
                // planOutputs holds the post-rerun hidden rows keyed by position
                // - base, so row j pairs position q-1 with token q.
                // TOKEN SOURCE (Stage 2): this loop reads the COMMITTED token
                // sequence from specArgmaxDevice[j+1]. The rerun-refresh argmax
                // goes to a dedicated mtpRerunScratch buffer and never aliases
                // this sequence, so repair can only ever consume tokens from the
                // authoritative emitted prefix.
                // NOTE (vLLM contract, unconditional repair): EVERY retained row
                // must pair (x_q, target h_{q-1}) - trusting a chained slot's
                // self-carried row poisons the predictor context from the first
                // accepted step on (observed: step 0 accepts via the
                // warmup-primed carry, every later slot-0 draft then diverges ->
                // 1-accept-per-run collapse). The rewrite runs for all committed
                // rows [base+1, base+consumedCount-1]; its executeMtpCuda call
                // both READS the carry installed above and then self-carries its
                // own output hidden, but that clobber is transient: the
                // epilogue's setMtpTargetCarryCuda below runs AFTER this loop and
                // is the last carry write of the step. The next carry reader is
                // the NEXT step's slot 0, whose plan executes (consuming the
                // epilogue value) before that step's first unconditional
                // self-carry fires - executeMtpCuda reads its carry input
                // during plan execution and only writes the self-carry
                // afterwards - so the epilogue carry is never consumed stale.
                // ADR 0106 Phase 2b: with the single-token commit, consumedCount is
                // 1 and no row is retained beyond the base: this loop is a no-op and
                // EVERY proposal row is hidden below. The repair machinery stays
                // for the multi-token contract's return.
                if (mtpRepairBatchReady && consumedCount > 1) {
                    // The batched plan consumes exactly the contiguous accepted
                    // prefix. Its row j is positioned at predictorBase+j and
                    // receives token[j] plus target hidden row[j].
                    p0RepairActive = false;
                    executeMtpRepairBatchCuda(consumedCount - 1, basePosition);
                    DSP_DIAG(KV_CACHE,
                             "MTP_PREFIX_REPAIR_BATCH step=%d predictorBase=%lld activeRows=%d "
                             "carryRow=%d - one fixed-width K/V-only forward",
                             step, (long long)basePosition, consumedCount - 1, carryRow);
                } else {
                    p0RepairActive = true;
                    for (int j = 0; j < consumedCount - 1; j++) {
                        // Scalar compatibility fallback: repair one row at a
                        // time using the retained [1,1] arrays.
                        LongType repairPosition = basePosition + 1 + j;
                        setMtpTargetCarryCuda(
                            planOutputs[config->targetHiddenOutputIdx], j);
                        setMtpNextInputCuda(specArgmaxDevice, j, repairPosition);
                        if (mtpRepairReady) executeMtpRepairCuda(repairPosition);
                        else executeMtpCuda(repairPosition, 0, false);
                        DSP_DIAG(KV_CACHE,
                                 "MTP_PREFIX_REPAIR step=%d targetPos=%lld predictorRow=%lld committedRow=%d "
                                 "carryRow=%d - rewriting predictor KV row with target hidden",
                                 step, (long long)repairPosition, (long long)(repairPosition - 1), j, carryRow);
                    }
                    p0RepairActive = false;
                }

                // All predictor rows at and beyond the pending row are future
                // state after this commit. Remask the entire tail, not merely
                // the suffix proposed by this step: adaptive K can shrink, and
                // rows left unmasked by a wider prior proposal would otherwise
                // become visible in a later predictor call.
                // nextMtpPosition = base+m is the next pending TARGET token
                // position; its predictor row is nextMtpPosition - 1 = retainedEnd.
                const LongType retainedPredictorEnd =
                    basePosition + static_cast<LongType>(consumedCount) - 1;  // exclusive
                LongType nextMtpPosition = basePosition + consumedCount;
                if (retainedPredictorEnd < mtpMaskLen) {
                    REQUIRE_TRUE(retainedPredictorEnd >= 0, 0,
                                 "autoregressive_decode: CUDA MTP future-row mask start "
                                 "%lld invalid at step %d",
                                 (long long)retainedPredictorEnd, step);
                    NDArray::prepareSpecialUse({config->mtpCausalMask}, {});
                    BUILD_SINGLE_SELECTOR(config->mtpCausalMask->dataType(),
                                          maskCausalRangeLauncher,
                                          (stream, config->mtpCausalMask->specialBuffer(),
                                           retainedPredictorEnd, mtpMaskLen,
                                           mtpMaskLen),
                                          SD_FLOAT_TYPES);
                    NDArray::registerSpecialUse({config->mtpCausalMask}, {});
                }

                setMtpTargetCarryCuda(
                    planOutputs[config->targetHiddenOutputIdx], carryRow);
                // Pending-input publication reads the FINALIZED upload: when the
                // scalar rerun flipped the winner (verify A -> rerun B), the
                // predictor must continue with B - the token the target actually
                // emitted - not the superseded verification token.
                setMtpNextInputCuda(
                    specArgmaxDevice, carryRow, nextMtpPosition);
            }

            // (Finalized-emission reconstruction, acceptance counting, and the
            // finalized sequence upload were moved ABOVE the predictor repair
            // and publication block - see the FINALIZED EMISSION SEQUENCE
            // comment there.)

            // -- Store accepted tokens to generatedTokenIds ------------------------
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
            DSP_DIAG(KV_CACHE,
                     "EMITTED_STEP step=%d basePos=%lld emitted=[%lld,%lld,%lld,%lld,%lld] "
                     "(n=%d) - authoritative committed sequence for this step",
                     step, (long long)basePosition,
                     (long long)argmaxDst[0], n > 1 ? (long long)argmaxDst[1] : -1LL,
                     n > 2 ? (long long)argmaxDst[2] : -1LL,
                     n > 3 ? (long long)argmaxDst[3] : -1LL,
                     n > 4 ? (long long)argmaxDst[4] : -1LL,
                     n);

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
            // At a terminal boundary the window is no longer a verification
            // scratch mask: hide every unconsumed KV row, including accepted EOS.
            // Row zero now exposes exactly the committed prefix; publish it to
            // every window row without changing the nonterminal verifier mask.
            if ((shouldStop || tokensGenerated == maxNewTokens) && useWindowSubstrate) {
                NDArray* mask = config->windowGridMask;
                LongType rowLen = mask->sizeAt(-1);
                NDArray::prepareSpecialUse({mask}, {});
                BUILD_SINGLE_SELECTOR(mask->dataType(), maskCausalRangeLauncher,
                                      (stream, mask->specialBuffer(), currentPosition, rowLen, rowLen),
                                      SD_FLOAT_TYPES);
                size_t rowBytes = static_cast<size_t>(rowLen) * mask->sizeOfT();
                for (LongType row = 1; row < mask->lengthOf() / rowLen; row++) {
                    cudaMemcpyAsync(static_cast<char*>(mask->specialBuffer()) + row * rowBytes,
                                    mask->specialBuffer(), rowBytes, cudaMemcpyDeviceToDevice, *stream);
                }
                NDArray::registerSpecialUse({mask}, {});
            }

            // Update position IDs to the next committed position.
            if (storedCount > 0) {
                NDArray::prepareSpecialUse({positionIds}, {});
                updatePositionIdsKernel<<<1, 1, 0, *stream>>>(positionIds->specialBuffer(), currentPosition);
                NDArray::registerSpecialUse({positionIds}, {});
            }

            // -- Update n-gram tables from the verified emission sequence ---------
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
            stepTokenCounts.push_back(tokensGenerated - tokensBeforeStep);

            // Publish the terminal input/position too; the last output is pending,
            // not consumed. This is part of the same prefix commit as a live step.
            // -- Embedding lookup and input updates for next step -----------------
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
            // In the speculative path we don't use sampledToken - registerSpecialUse to
            // keep the CUDA-graph-capture bookkeeping symmetric.
            NDArray::registerSpecialUse({sampledToken}, {logitsOutput});

            if (shouldStop) break;

            // NOTE: skip the rest of the loop body - we handled everything above.
            continue;
        }

        // -- Phase 1 scalar path (W=1 or no proposals this step) -----------------
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
        // Force scalar B=1/W=1 for the selection step - the substrate runs W-wide
        // but policy selection is still scalar (Phase 2 will extend this).
        // Also reset SPECULATIVE strategy to GREEDY: in the scalar fallback path
        // (proposedCount==0 or no window substrate) we always select greedily.
        // TOKEN_SAMPLE_SPECULATIVE(3) is not handled by tokenSamplePolicy - it
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

        // -- Tier 1a: Store token via D2D copy (avoids p() hidden H2D + stream 0 sync) --
        // generatedTokenIds->p() does host write -> syncToDevice() -> cudaMemcpyAsync
        // on stream 0 + cudaStreamSynchronize(stream_0) - a hidden pipeline drain.
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

        // -- Tier 1b: Pre-sync GPU work --
        // Everything below until the cudaStreamSynchronize only depends on
        // currentPosition (CPU counter) and plan output pointers (already on
        // device). None of it needs the token ID from D2H. Launching these
        // kernels BEFORE the sync overlaps their GPU execution with the
        // async D2H copy and hides their latency behind the sync wait.
        //
        // Advance position BEFORE updates for the NEXT decode step.
        currentPosition++;
        LongType kvJustWritten = currentPosition - 1;

        // P02 K-RE-ENABLE STATE MAINTENANCE (review round 3, finding 2 -
        // corrected semantics): when drafting is OFF but MTP resources exist,
        // run the maintenance forward BEFORE the epilogue publishes the new
        // (carry, pending-input) pair. The maintenance forward must consume
        // the pair for the token the target JUST consumed at kvJustWritten:
        // (target hidden for kvJustWritten-1, token at kvJustWritten). The
        // previous implementation ran AFTER installing (h(currentPosition),
        // token@currentPosition), which (a) skipped the predictor KV row for
        // the just-consumed token and (b) left the recursive-draft side
        // effects of executeMtpCuda in place (retained carry = predictor's
        // own output, retained input = its draft) - so re-enabling K resumed
        // from (draft, predictor_hidden) instead of the authoritative
        // (token, target_hidden) pair. By running the forward on the
        // just-consumed pair FIRST, its chain mutations are then overwritten
        // by the epilogue's authoritative publication below; the predictor KV
        // gains exactly the row the next draft step would have needed, and
        // the retained pending pair is the target-conditioned one.
        if (!useMtp
                && config->mtpPlanHandle != nullptr && config->mtpExtInputContext != nullptr
                && config->targetHiddenOutputIdx >= 0
                && config->targetHiddenOutputIdx < numPlanOutputs
                && planOutputs[config->targetHiddenOutputIdx] != nullptr) {
            // CONSUME THE SAVED PRE-STEP PAIR (review round 4, finding B/2):
            // the previous step's epilogue published the pending pair
            // (inputToken@currentPosition, h(currentPosition-1)) - the exact
            // pair the target JUST consumed at currentPosition. The input
            // token is ALREADY in config->mtpInputIds; sampledToken is the
            // NEWLY EMITTED token, NOT the consumed input, so re-publishing
            // it here wrote the next token into the previous token's KV row
            // (encoded-oracle signature: key 702 where 701 is required).
            // Only the SCALARS need re-publishing (target row = kvJustWritten
            // - 1); the pair content is untouched, then the maintenance
            // forward consumes it. The epilogue below afterwards overwrites
            // the recursive-carry side effects with the authoritative
            // (h(currentPosition-1 output), token@currentPosition) pair.
            setMtpNextInputCuda(config->mtpInputIds, 0, kvJustWritten);
            executeMtpCuda(kvJustWritten, 0, false);
        }

        // P02 K-RE-ENABLE STATE PUBLICATION: publish the carry/pending input
        // whenever MTP metadata exists - NOT only while useMtp. When the
        // adaptive policy drops K to 0 (useMtp=false), this epilogue is the
        // only thing keeping the predictor's state aligned with the live
        // sequence; without it, re-raising K mid-session resumes speculation
        // from a stale (carry, pending-input, cache_position) triple - the
        // exact defect class the reviewer's P02 pass-evidence forbids
        // ("Never resume with stale predictor KV/carry"). Cost: two small D2D
        // copies per step, gated on metadata presence, not on speculating.
        if (useMtp
                || (config->mtpPlanHandle != nullptr && config->mtpExtInputContext != nullptr
                    && config->targetHiddenOutputIdx >= 0
                    && config->targetHiddenOutputIdx < numPlanOutputs
                    && planOutputs[config->targetHiddenOutputIdx] != nullptr)) {
            setMtpTargetCarryCuda(planOutputs[config->targetHiddenOutputIdx], 0);
            setMtpNextInputCuda(sampledToken, 0, currentPosition);
        }

        // -- KV scatter - copy present KV into static buffers --
        // Moved BEFORE sync: scatter only needs currentPosition (CPU counter)
        // and plan output device pointers. Both are available without the token
        // ID. Since scatter and the next plan execution are on the same stream,
        // CUDA ordering guarantees scatter completes before the next read.
        // Skip manual scatter when the plan's native KV scatter is active
        // (planOwnsKvScatter) - executeKvScatterPostExec handles it with its
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
                             "at step %d - plan did not produce this output.",
                             kv, kvOutIdx, step);
                REQUIRE_TRUE(staticBuf != nullptr, 0,
                             "autoregressive_decode: static KV buffer[%d] is null at step %d.",
                             kv, step);
                REQUIRE_TRUE(presentKv->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: KV output[%d] has null device buffer "
                             "at step %d - stale or uninitialized output.",
                             kv, step);
                REQUIRE_TRUE(staticBuf->specialBuffer() != nullptr, 0,
                             "autoregressive_decode: static KV[%d] has null device buffer "
                             "at step %d - buffer was freed or never allocated.",
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
                         "autoregressive_decode: staticKvBuffers[0] is null at step %d - "
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

        // -- D2H token readback via pinned memory --
        // Read sampled token ID back to host (single int64).
        // Issue D2H copy on the SAME stream as the argmax kernel - FIFO ordering
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
        // Gated diagnostic D2H (rides the sync below - no new sync points):
        // sample the first 8 logits of the live row for cross-pipeline value
        // comparison against the MTP rerun's top-8 probe (T3b step-99 flip).
        float scalarLogitsSample[8] = {};
        if (DSP_DIAG_ENABLED(KV_CACHE) && logitsOutput->lengthOf() >= 8
                && logitsOutput->dataType() == DataType::FLOAT32) {
            cudaMemcpyAsync(scalarLogitsSample, logitsOutput->specialBuffer(),
                            8 * sizeof(float), cudaMemcpyDeviceToHost, *stream);
        }
        cudaStreamSynchronize(*stream);
        emitCommittedStateSamples(step);
        // NOTE: mask slices here reflect the already-advanced next-step state (the
        // advance kernels launch before this sync); the KV rows are the payload -
        // they hold exactly what this step committed.
        dumpStepInputSlices("scalar", step, currentPosition - 1);
        emitPlanOutputFingerprints();
        auto tSyncEnd = stepTimingEnabled ? std::chrono::high_resolution_clock::now() : stepStart;
        LongType nextTokenId = *tokenDst;
        if (config->tokenCallback != nullptr) {
            config->tokenCallback(nextTokenId, config->callbackUserData);
        }

        // Gated diagnostic event: per-step scalar-path record (host-side counters
        // and the already-synced token only - no additional device reads or syncs).
        // Mirrors the CPU helper's SCALAR_STEP event for step-level divergence
        // localization.
        DSP_DIAG(KV_CACHE, "SCALAR_STEP step=%d pos=%lld tok=%lld proposed=%d "
                 "r0=[%.6f,%.6f,%.6f,%.6f] logits8=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]",
                 step, (long long)(currentPosition - 1), (long long)nextTokenId,
                 proposedCount,
                 scalarLogitsSample[0], scalarLogitsSample[1],
                 scalarLogitsSample[2], scalarLogitsSample[3],
                 scalarLogitsSample[0], scalarLogitsSample[1], scalarLogitsSample[2],
                 scalarLogitsSample[3], scalarLogitsSample[4], scalarLogitsSample[5],
                 scalarLogitsSample[6], scalarLogitsSample[7]);

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

        // -- Check stop condition --
        bool matchedStop = stopMatcher.accept(nextTokenId);
        bool shouldStop = matchedStop && stopTerminationAllowed(config, tokensGenerated);
        bool matchedRepetition = repetitionMatcher.accept(nextTokenId);

        auto tStopCheck = std::chrono::high_resolution_clock::now();

        // Compute step time using the stop check timestamp
        // Always measure real wall-clock step time - needed for lateSteady metric even
        // when detailed sub-step timing (stepTimingEnabled) is off.
        double stepMs = std::chrono::duration<double, std::milli>(tStopCheck - stepStart).count();
        stepTimesMs.push_back(stepMs);
        stepTokenCounts.push_back(tokensGenerated - tokensBeforeStep);

        if (shouldStop) break;
        if (matchedRepetition) {
            config->nativeFinishReason = 1;
            break;
        }

        // -- Step 6: Embedding lookup for next token --
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

        // -- Update in-graph KV cache scalars (GGUF pattern) --
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

        // Per-step timing breakdown (gated behind executionTimingEnabled only - print every step)
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

    // -- Final sync --
    p0.hostWaitBoundaries++;
    const auto finalSnapshotSync = cudaStreamSynchronize(*stream);
    if (captureMtpInputs) {
        REQUIRE_TRUE(finalSnapshotSync == cudaSuccess, 0,
                     "DSP tensor snapshot final sync failed: %s", cudaGetErrorString(finalSnapshotSync));
        tensorDiagnostics.drainTensorSnapshots(reinterpret_cast<void*>(*stream), true);
    }
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

    // -- ADR 0106 Phase 2: free speculative decode resources --
    if (pinnedArgmax != nullptr) {
        cudaFreeHost(pinnedArgmax);
        pinnedArgmax = nullptr;
    }
    if (pinnedValidity != nullptr) {
        cudaFreeHost(pinnedValidity);
        pinnedValidity = nullptr;
    }
    if (specValidityDevice != nullptr) {
        delete specValidityDevice;
        specValidityDevice = nullptr;
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
    if (mtpRerunScratch != nullptr) {
        delete mtpRerunScratch;
        mtpRerunScratch = nullptr;
    }
    // Free the bindingless pre-verification recurrent snapshots (created lazily
    // per step when no scalar binding supplies private snapshot arrays).
    for (size_t s = 0; s < unboundStateSnapshots.size(); ++s) {
        if (unboundStateSnapshots[s] != nullptr) {
            delete unboundStateSnapshots[s];
            unboundStateSnapshots[s] = nullptr;
        }
    }
    unboundStateSnapshots.clear();
    unboundStateSnapshotExtIdx.clear();
    // Free the deep pre-verification recurrent snapshots (owned scratch NDArrays
    // captured D2D before each verification execution; never alias the live ext
    // inputs, see the stateSnapshotArrays declaration above).
    for (size_t s = 0; s < stateSnapshotArrays.size(); ++s) {
        if (stateSnapshotArrays[s] != nullptr) {
            delete stateSnapshotArrays[s];
            stateSnapshotArrays[s] = nullptr;
        }
    }
    stateSnapshotArrays.clear();
    stateSnapshotExtIdx.clear();
    // Free the shared-KV row snapshots (verdict-c fix): one owned buffer per
    // KV pair, rows [base, base+K) per step.
    for (size_t kv = 0; kv < kvRowSnapshots.size(); ++kv) {
        if (kvRowSnapshots[kv] != nullptr) {
            delete kvRowSnapshots[kv];
            kvRowSnapshots[kv] = nullptr;
        }
    }
    kvRowSnapshots.clear();
    kvRowSnapshotRows.clear();
    kvRowSnapshotSources.clear();
    kvRowSnapshotBase = -1;

    // -- Write token count --
    tokenCount->p(0, static_cast<LongType>(tokensGenerated));

    // -- Compute timing stats --
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
        // DSP warmup takes ~60 steps to converge to true steady-state.
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
            // Not enough steps - fall back to overall
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

    // -- Cleanup internal allocations --
    // decodeEmbedding is NOT deleted - it's prefillEmbeddings, owned by the caller.
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
    // Emit after native scratch teardown so the summary is the final diagnostic
    // event and cannot be displaced by cleanup events in the ring buffer.
    DSP_DIAG(KV_CACHE,
             "MTP_P0_CUDA finalized=%lld proposals=%lld accepted=%lld steps=%d "
             "targetVerify=%d reruns=%d shortened=%d predictorProposal=%d "
             "predictorRepair=%d predictorMaintenance=%d repairLmHead=%d "
             "snapshotBytes=%llu restoreBytes=%llu stateCommitBytes=%llu "
             "hostReadbackBytes=%llu hostWaits=%llu phaseTransitions=%d "
             "planReplay=%d planWarmup=%d targetExec=%d targetPhase=%d "
             "multiRow=%d specK=%d windowMax=%d",
             (long long)p0.finalizedTokens, (long long)p0.proposals,
             (long long)p0.acceptedDrafts, p0.speculativeSteps,
             p0.targetVerificationForwards, p0.acceptedPrefixReruns,
             p0.shortenedRecoveryForwards, p0.predictorProposalForwards,
             p0.predictorRepairForwards, p0.predictorMaintenanceForwards,
             p0.predictorRepairLmHeadForwards,
             (unsigned long long)p0.snapshotBytes,
             (unsigned long long)p0.restoreBytes,
             (unsigned long long)p0.stateCommitBytes,
             (unsigned long long)p0.hostReadbackBytes,
             (unsigned long long)p0.hostWaitBoundaries,
             p0.planPhaseTransitions, p0.planReplayForwards,
             p0.planWarmupForwards, plan->getExecuteCount(),
             static_cast<int>(plan->getPlanPhase()),
             config->allowMultiRowCommit ? 1 : 0, specK,
             config->windowMax);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
