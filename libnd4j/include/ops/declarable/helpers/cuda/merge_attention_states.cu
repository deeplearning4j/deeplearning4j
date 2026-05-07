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

//
// Merge attention states CUDA kernel.
// Merges partial attention outputs from chunked/disaggregated prefill
// using numerically-stable log-sum-exp weighting.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <types/float16.h>
#include <ops/declarable/helpers/merge_attention_states.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Two-chunk merge kernel
// Grid: (batch*numHeads*seqLen), Block: threads over headDim
//
// For numerical stability, computes:
//   m = max(lse_a, lse_b)
//   w_a = exp(lse_a - m),  w_b = exp(lse_b - m)
//   output = (w_a * attn_a + w_b * attn_b) / (w_a + w_b)
//   lse_out = m + log(w_a + w_b)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void mergeAttentionStatesKernel(
    const T* __restrict__ attnA,      // [B, H, S, D]
    const float* __restrict__ lseA,   // [B, H, S]
    const T* __restrict__ attnB,      // [B, H, S, D]
    const float* __restrict__ lseB,   // [B, H, S]
    T* __restrict__ output,           // [B, H, S, D]
    float* __restrict__ lseOut,       // [B, H, S] (may be nullptr)
    const LongType numHeads,
    const LongType seqLen,
    const LongType headDim,
    const LongType totalPositions) {  // B * H * S

    const LongType posIdx = blockIdx.x;
    if (posIdx >= totalPositions) return;

    // Decode position into (batch, head, seq) indices
    // Layout: [B, H, S, D]
    const LongType HS = numHeads * seqLen;
    const LongType batchIdx = posIdx / HS;
    const LongType remainder = posIdx % HS;
    const LongType headIdx = remainder / seqLen;
    const LongType seqIdx = remainder % seqLen;

    // LSE layout: [B, H, S]
    const LongType lseOffset = posIdx;
    const float lA = lseA[lseOffset];
    const float lB = lseB[lseOffset];

    // Numerically stable merge weights
    const float m = fmaxf(lA, lB);
    const float wA = expf(lA - m);
    const float wB = expf(lB - m);
    const float wSum = wA + wB;
    const float invWSum = 1.0f / wSum;

    // Attention data layout: [B, H, S, D]
    const LongType dataOffset = posIdx * headDim;
    const T* aPtr = attnA + dataOffset;
    const T* bPtr = attnB + dataOffset;
    T* outPtr = output + dataOffset;

    // Merge head dimensions in parallel
    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        float valA = static_cast<float>(aPtr[d]);
        float valB = static_cast<float>(bPtr[d]);
        float merged = (wA * valA + wB * valB) * invWSum;
        outPtr[d] = static_cast<T>(merged);
    }

    // Write merged LSE if output buffer provided
    if (lseOut != nullptr && threadIdx.x == 0) {
        lseOut[lseOffset] = m + logf(wSum);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Half-precision optimized variant using half2 vectorized loads
//////////////////////////////////////////////////////////////////////////////
SD_KERNEL void mergeAttentionStatesHalf2Kernel(
    const half* __restrict__ attnA,
    const float* __restrict__ lseA,
    const half* __restrict__ attnB,
    const float* __restrict__ lseB,
    half* __restrict__ output,
    float* __restrict__ lseOut,
    const LongType numHeads,
    const LongType seqLen,
    const LongType headDim,
    const LongType totalPositions) {

    const LongType posIdx = blockIdx.x;
    if (posIdx >= totalPositions) return;

    const LongType lseOffset = posIdx;
    const float lA = lseA[lseOffset];
    const float lB = lseB[lseOffset];

    const float m = fmaxf(lA, lB);
    const float wA = expf(lA - m);
    const float wB = expf(lB - m);
    const float invWSum = 1.0f / (wA + wB);

    const LongType dataOffset = posIdx * headDim;
    const half2* aPtr = reinterpret_cast<const half2*>(attnA + dataOffset);
    const half2* bPtr = reinterpret_cast<const half2*>(attnB + dataOffset);
    half2* outPtr = reinterpret_cast<half2*>(output + dataOffset);

    const LongType halfDim = headDim / 2;

    for (LongType d = threadIdx.x; d < halfDim; d += blockDim.x) {
        half2 vA = aPtr[d];
        half2 vB = bPtr[d];

        float2 fA = __half22float2(vA);
        float2 fB = __half22float2(vB);

        float2 merged;
        merged.x = (wA * fA.x + wB * fB.x) * invWSum;
        merged.y = (wA * fA.y + wB * fB.y) * invWSum;

        outPtr[d] = __float22half2_rn(merged);
    }

    // Handle odd headDim
    if (headDim % 2 != 0 && threadIdx.x == 0) {
        LongType lastIdx = headDim - 1;
        float valA = __half2float(*(attnA + dataOffset + lastIdx));
        float valB = __half2float(*(attnB + dataOffset + lastIdx));
        float merged = (wA * valA + wB * valB) * invWSum;
        *(output + dataOffset + lastIdx) = __float2half(merged);
    }

    if (lseOut != nullptr && threadIdx.x == 0) {
        lseOut[lseOffset] = m + logf(wA + wB);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public: mergeAttentionStates (two chunks)
//////////////////////////////////////////////////////////////////////////////
void mergeAttentionStates(LaunchContext* context,
                           NDArray* attnPrefix,
                           NDArray* lsePrefix,
                           NDArray* attnSuffix,
                           NDArray* lseSuffix,
                           NDArray* output,
                           NDArray* lseOutput) {
    // attnPrefix: [B, H, S, D]
    const LongType batch = attnPrefix->sizeAt(0);
    const LongType numHeads = attnPrefix->sizeAt(1);
    const LongType seqLen = attnPrefix->sizeAt(2);
    const LongType headDim = attnPrefix->sizeAt(3);
    const LongType totalPositions = batch * numHeads * seqLen;

    NDArray::prepareSpecialUse({output}, {attnPrefix, lsePrefix, attnSuffix, lseSuffix});
    if (lseOutput != nullptr) {
        NDArray::prepareSpecialUse({lseOutput}, {});
    }

    auto stream = context->getCudaStream();
    auto dtype = attnPrefix->dataType();

    int threads = 128;
    if (headDim > 128) threads = 256;

    float* lseOutPtr = (lseOutput != nullptr) ?
        reinterpret_cast<float*>(lseOutput->specialBuffer()) : nullptr;

    if (dtype == DataType::HALF && (headDim % 2 == 0)) {
        mergeAttentionStatesHalf2Kernel<<<totalPositions, threads, 0, *stream>>>(
            reinterpret_cast<const half*>(attnPrefix->specialBuffer()),
            reinterpret_cast<const float*>(lsePrefix->specialBuffer()),
            reinterpret_cast<const half*>(attnSuffix->specialBuffer()),
            reinterpret_cast<const float*>(lseSuffix->specialBuffer()),
            reinterpret_cast<half*>(output->specialBuffer()),
            lseOutPtr,
            numHeads, seqLen, headDim, totalPositions);
    } else if (dtype == DataType::FLOAT32) {
        mergeAttentionStatesKernel<float><<<totalPositions, threads, 0, *stream>>>(
            reinterpret_cast<const float*>(attnPrefix->specialBuffer()),
            reinterpret_cast<const float*>(lsePrefix->specialBuffer()),
            reinterpret_cast<const float*>(attnSuffix->specialBuffer()),
            reinterpret_cast<const float*>(lseSuffix->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            lseOutPtr,
            numHeads, seqLen, headDim, totalPositions);
    } else if (dtype == DataType::HALF) {
        mergeAttentionStatesKernel<float16><<<totalPositions, threads, 0, *stream>>>(
            reinterpret_cast<const float16*>(attnPrefix->specialBuffer()),
            reinterpret_cast<const float*>(lsePrefix->specialBuffer()),
            reinterpret_cast<const float16*>(attnSuffix->specialBuffer()),
            reinterpret_cast<const float*>(lseSuffix->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            lseOutPtr,
            numHeads, seqLen, headDim, totalPositions);
    } else {
        THROW_EXCEPTION("mergeAttentionStates: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("mergeAttentionStates failed");
    NDArray::registerSpecialUse({output}, {attnPrefix, lsePrefix, attnSuffix, lseSuffix});
    if (lseOutput != nullptr) {
        NDArray::registerSpecialUse({lseOutput}, {});
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public: mergeAttentionStatesMulti (N chunks)
// Iteratively merges chunks using the two-chunk kernel
//////////////////////////////////////////////////////////////////////////////
void mergeAttentionStatesMulti(LaunchContext* context,
                                const std::vector<NDArray*>& attnChunks,
                                const std::vector<NDArray*>& lseChunks,
                                NDArray* output,
                                NDArray* lseOutput) {
    if (attnChunks.empty()) {
        THROW_EXCEPTION("mergeAttentionStatesMulti: empty chunk list");
    }

    if (attnChunks.size() == 1) {
        // Single chunk: just copy
        output->assign(attnChunks[0]);
        if (lseOutput != nullptr) {
            lseOutput->assign(lseChunks[0]);
        }
        return;
    }

    if (attnChunks.size() == 2) {
        mergeAttentionStates(context, attnChunks[0], lseChunks[0],
                             attnChunks[1], lseChunks[1], output, lseOutput);
        return;
    }

    // N > 2: iterative pairwise merge
    // Start with chunk 0, merge with chunk 1 into output, then merge output with chunk 2, etc.
    const auto shape4d = attnChunks[0]->shapeOf();
    const auto shape3d = lseChunks[0]->shapeOf();

    // Temporary buffers for intermediate results
    std::vector<sd::LongType> tempAttnShape = {shape4d[0], shape4d[1], shape4d[2], shape4d[3]};
    std::vector<sd::LongType> tempLseShape = {shape3d[0], shape3d[1], shape3d[2]};
    auto tempAttn = NDArrayFactory::create_(attnChunks[0]->ordering(), tempAttnShape,
                                            attnChunks[0]->dataType(), context);
    auto tempLse = NDArrayFactory::create_(lseChunks[0]->ordering(), tempLseShape,
                                           DataType::FLOAT32, context);

    // First merge: chunks[0] + chunks[1] -> temp
    mergeAttentionStates(context, attnChunks[0], lseChunks[0],
                         attnChunks[1], lseChunks[1], tempAttn, tempLse);

    // Subsequent merges: temp + chunks[i] -> output/temp
    for (size_t i = 2; i < attnChunks.size(); i++) {
        if (i == attnChunks.size() - 1) {
            // Last merge goes to final output
            mergeAttentionStates(context, tempAttn, tempLse,
                                 attnChunks[i], lseChunks[i], output, lseOutput);
        } else {
            // Intermediate merges: swap temp buffers
            std::vector<sd::LongType> tempAttn2Shape = {shape4d[0], shape4d[1], shape4d[2], shape4d[3]};
            std::vector<sd::LongType> tempLse2Shape = {shape3d[0], shape3d[1], shape3d[2]};
            auto tempAttn2 = NDArrayFactory::create_(tempAttn->ordering(), tempAttn2Shape,
                                                      tempAttn->dataType(), context);
            auto tempLse2 = NDArrayFactory::create_(tempLse->ordering(), tempLse2Shape,
                                                     DataType::FLOAT32, context);
            mergeAttentionStates(context, tempAttn, tempLse,
                                 attnChunks[i], lseChunks[i], tempAttn2, tempLse2);

            delete tempAttn;
            delete tempLse;
            tempAttn = tempAttn2;
            tempLse = tempLse2;
        }
    }

    if (attnChunks.size() == 2) {
        // Already handled above, but clean up anyway
    }

    delete tempAttn;
    delete tempLse;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
