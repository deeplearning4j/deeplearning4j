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
// Fused QK RMSNorm + RoPE CUDA kernel.
// Combines three operations into one kernel pass:
//   1. RMSNorm(Q) per head
//   2. RMSNorm(K) per head
//   3. Rotary Position Embedding on both Q and K
// Eliminates 2 intermediate global memory round-trips.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <ops/declarable/helpers/fused_qk_norm_rope.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int FQNR_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level sum reduction
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE SD_INLINE float fqnrWarpReduceSum(float val) {
    for (int offset = FQNR_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level sum reduction
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE float fqnrBlockReduceSum(float val, float* sharedMem) {
    const int lane = threadIdx.x % FQNR_WARP_SIZE;
    const int wid = threadIdx.x / FQNR_WARP_SIZE;
    const int numWarps = (blockDim.x + FQNR_WARP_SIZE - 1) / FQNR_WARP_SIZE;

    val = fqnrWarpReduceSum(val);

    if (lane == 0) {
        sharedMem[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = fqnrWarpReduceSum(val);
    }

    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Fused QK Norm + RoPE kernel
// One warp per (batch, seq, head) tuple
// Each warp:
//   1. Computes RMS of head vector
//   2. Normalizes with gamma
//   3. Applies rotary embedding
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void fusedQkNormRopeKernel(
    const T* __restrict__ input,       // [B, S, H, D]
    const T* __restrict__ gamma,       // [D]
    const T* __restrict__ cosCache,    // [maxS, D/2] or nullptr
    const T* __restrict__ sinCache,    // [maxS, D/2] or nullptr
    T* __restrict__ output,            // [B, S, H, D]
    const LongType batch,
    const LongType seqLen,
    const LongType numHeads,
    const LongType headDim,
    const float epsilon,
    const double freqBase,
    const bool isNeox) {

    // Each block handles one (batch, seq, head)
    const LongType idx = blockIdx.x;
    const LongType totalHeads = batch * seqLen * numHeads;
    if (idx >= totalHeads) return;

    const LongType b = idx / (seqLen * numHeads);
    const LongType remainder = idx % (seqLen * numHeads);
    const LongType s = remainder / numHeads;
    const LongType h = remainder % numHeads;

    // Input/output offset: [B, S, H, D] layout
    const LongType offset = ((b * seqLen + s) * numHeads + h) * headDim;
    const T* inPtr = input + offset;
    T* outPtr = output + offset;

    extern __shared__ char sharedMemRaw[];
    float* sdata = reinterpret_cast<float*>(sharedMemRaw);

    // Step 1: Compute sum of squares for RMS norm
    float sumSq = 0.0f;
    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        float val = static_cast<float>(inPtr[d]);
        sumSq += val * val;
    }

    sumSq = fqnrBlockReduceSum(sumSq, sdata);

    __shared__ float rmsInv;
    if (threadIdx.x == 0) {
        float rms = sqrtf(sumSq / static_cast<float>(headDim) + epsilon);
        rmsInv = 1.0f / rms;
    }
    __syncthreads();

    // Step 2 + 3: Normalize with gamma and apply RoPE
    const LongType halfDim = headDim / 2;

    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        // RMSNorm
        float val = static_cast<float>(inPtr[d]) * rmsInv;
        float g = static_cast<float>(gamma[d]);
        val *= g;

        // RoPE rotation
        float cosVal, sinVal;

        if (cosCache != nullptr && sinCache != nullptr) {
            // Use precomputed cos/sin
            LongType pairIdx = d / 2;
            if (pairIdx < halfDim) {
                cosVal = static_cast<float>(cosCache[s * halfDim + pairIdx]);
                sinVal = static_cast<float>(sinCache[s * halfDim + pairIdx]);
            } else {
                cosVal = 1.0f;
                sinVal = 0.0f;
            }
        } else {
            // Compute on-the-fly
            LongType pairIdx;
            if (isNeox) {
                pairIdx = d % halfDim;
            } else {
                pairIdx = d / 2;
            }
            float theta = static_cast<float>(s) * powf(static_cast<float>(freqBase),
                           -2.0f * static_cast<float>(pairIdx) / static_cast<float>(headDim));
            cosVal = cosf(theta);
            sinVal = sinf(theta);
        }

        // Apply rotation
        float rotatedVal;
        if (isNeox) {
            // NeoX style: split-half
            if (d < halfDim) {
                float partner = static_cast<float>(inPtr[d + halfDim]) * rmsInv *
                                static_cast<float>(gamma[d + halfDim]);
                rotatedVal = val * cosVal - partner * sinVal;
            } else {
                float partner = static_cast<float>(inPtr[d - halfDim]) * rmsInv *
                                static_cast<float>(gamma[d - halfDim]);
                rotatedVal = partner * sinVal + val * cosVal;
            }
        } else {
            // GPT-J style: interleaved pairs
            if (d % 2 == 0) {
                float partner = static_cast<float>(inPtr[d + 1]) * rmsInv *
                                static_cast<float>(gamma[d + 1]);
                rotatedVal = val * cosVal - partner * sinVal;
            } else {
                float partner = static_cast<float>(inPtr[d - 1]) * rmsInv *
                                static_cast<float>(gamma[d - 1]);
                rotatedVal = partner * sinVal + val * cosVal;
            }
        }

        outPtr[d] = static_cast<T>(rotatedVal);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public: fusedQkNormRope
//////////////////////////////////////////////////////////////////////////////
void fusedQkNormRope(LaunchContext* context,
                      NDArray* query,
                      NDArray* key,
                      NDArray* gammaQ,
                      NDArray* gammaK,
                      NDArray* cosCache,
                      NDArray* sinCache,
                      NDArray* queryOut,
                      NDArray* keyOut,
                      float epsilon,
                      double freqBase,
                      bool isNeox) {
    // query: [B, S, numQHeads, D]
    const LongType batch = query->sizeAt(0);
    const LongType seqLen = query->sizeAt(1);
    const LongType numQHeads = query->sizeAt(2);
    const LongType headDim = query->sizeAt(3);
    const LongType numKVHeads = key->sizeAt(2);

    NDArray::prepareSpecialUse({queryOut, keyOut}, {query, key, gammaQ, gammaK});
    if (cosCache != nullptr) NDArray::prepareSpecialUse({}, {cosCache, sinCache});

    auto stream = context->getCudaStream();
    auto dtype = query->dataType();

    int threads = 128;
    if (headDim > 128) threads = 256;
    int numWarps = (threads + FQNR_WARP_SIZE - 1) / FQNR_WARP_SIZE;
    size_t smem = numWarps * sizeof(float);

    const void* cosPtr = (cosCache != nullptr) ? cosCache->specialBuffer() : nullptr;
    const void* sinPtr = (sinCache != nullptr) ? sinCache->specialBuffer() : nullptr;

    // Process Q heads
    LongType totalQHeads = batch * seqLen * numQHeads;
    if (dtype == DataType::FLOAT32) {
        fusedQkNormRopeKernel<float><<<totalQHeads, threads, smem, *stream>>>(
            reinterpret_cast<const float*>(query->specialBuffer()),
            reinterpret_cast<const float*>(gammaQ->specialBuffer()),
            reinterpret_cast<const float*>(cosPtr),
            reinterpret_cast<const float*>(sinPtr),
            reinterpret_cast<float*>(queryOut->specialBuffer()),
            batch, seqLen, numQHeads, headDim, epsilon, freqBase, isNeox);
    } else if (dtype == DataType::HALF) {
        fusedQkNormRopeKernel<float16><<<totalQHeads, threads, smem, *stream>>>(
            reinterpret_cast<const float16*>(query->specialBuffer()),
            reinterpret_cast<const float16*>(gammaQ->specialBuffer()),
            reinterpret_cast<const float16*>(cosPtr),
            reinterpret_cast<const float16*>(sinPtr),
            reinterpret_cast<float16*>(queryOut->specialBuffer()),
            batch, seqLen, numQHeads, headDim, epsilon, freqBase, isNeox);
    } else {
        THROW_EXCEPTION("fusedQkNormRope: unsupported data type");
    }

    // Process K heads
    LongType totalKHeads = batch * seqLen * numKVHeads;
    if (dtype == DataType::FLOAT32) {
        fusedQkNormRopeKernel<float><<<totalKHeads, threads, smem, *stream>>>(
            reinterpret_cast<const float*>(key->specialBuffer()),
            reinterpret_cast<const float*>(gammaK->specialBuffer()),
            reinterpret_cast<const float*>(cosPtr),
            reinterpret_cast<const float*>(sinPtr),
            reinterpret_cast<float*>(keyOut->specialBuffer()),
            batch, seqLen, numKVHeads, headDim, epsilon, freqBase, isNeox);
    } else if (dtype == DataType::HALF) {
        fusedQkNormRopeKernel<float16><<<totalKHeads, threads, smem, *stream>>>(
            reinterpret_cast<const float16*>(key->specialBuffer()),
            reinterpret_cast<const float16*>(gammaK->specialBuffer()),
            reinterpret_cast<const float16*>(cosPtr),
            reinterpret_cast<const float16*>(sinPtr),
            reinterpret_cast<float16*>(keyOut->specialBuffer()),
            batch, seqLen, numKVHeads, headDim, epsilon, freqBase, isNeox);
    }

    DebugHelper::checkGlobalErrorCode("fusedQkNormRope failed");
    NDArray::registerSpecialUse({queryOut, keyOut}, {query, key, gammaQ, gammaK});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
