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
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

constexpr int FQNR_WARP_SIZE = 32;

// Accumulator type: use double when T=double for full precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Warp/block sum reductions come from device_primitives.cuh
// (sd::device::warpReduceSum / sd::device::blockReduceSum).

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

    using AccT = typename AccType<T>::type;

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
    AccT* sdata = reinterpret_cast<AccT*>(sharedMemRaw);

    // Step 1: Compute sum of squares for RMS norm
    AccT sumSq = static_cast<AccT>(0);
    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        AccT val = static_cast<AccT>(inPtr[d]);
        sumSq += val * val;
    }

    sumSq = sd::device::blockReduceSum(sumSq, sdata);

    __shared__ AccT rmsInv;
    if (threadIdx.x == 0) {
        AccT rms = sd::math::sd_sqrt<AccT, AccT>(sumSq / static_cast<AccT>(headDim) + static_cast<AccT>(epsilon));
        rmsInv = static_cast<AccT>(1) / rms;
    }
    __syncthreads();

    // Step 2 + 3: Normalize with gamma and apply RoPE
    const LongType halfDim = headDim / 2;

    for (LongType d = threadIdx.x; d < headDim; d += blockDim.x) {
        // RMSNorm
        AccT val = static_cast<AccT>(inPtr[d]) * rmsInv;
        AccT g = static_cast<AccT>(gamma[d]);
        val *= g;

        // RoPE rotation
        AccT cosVal, sinVal;

        if (cosCache != nullptr && sinCache != nullptr) {
            // Use precomputed cos/sin
            LongType pairIdx = d / 2;
            if (pairIdx < halfDim) {
                cosVal = static_cast<AccT>(cosCache[s * halfDim + pairIdx]);
                sinVal = static_cast<AccT>(sinCache[s * halfDim + pairIdx]);
            } else {
                cosVal = static_cast<AccT>(1);
                sinVal = static_cast<AccT>(0);
            }
        } else {
            // Compute on-the-fly
            LongType pairIdx;
            if (isNeox) {
                pairIdx = d % halfDim;
            } else {
                pairIdx = d / 2;
            }
            AccT theta = static_cast<AccT>(s) * sd::math::sd_pow<AccT, AccT, AccT>(
                static_cast<AccT>(freqBase),
                static_cast<AccT>(-2) * static_cast<AccT>(pairIdx) / static_cast<AccT>(headDim));
            cosVal = sd::math::sd_cos<AccT, AccT>(theta);
            sinVal = sd::math::sd_sin<AccT, AccT>(theta);
        }

        // Apply rotation
        AccT rotatedVal;
        if (isNeox) {
            // NeoX style: split-half
            if (d < halfDim) {
                AccT partner = static_cast<AccT>(inPtr[d + halfDim]) * rmsInv *
                               static_cast<AccT>(gamma[d + halfDim]);
                rotatedVal = val * cosVal - partner * sinVal;
            } else {
                AccT partner = static_cast<AccT>(inPtr[d - halfDim]) * rmsInv *
                               static_cast<AccT>(gamma[d - halfDim]);
                rotatedVal = partner * sinVal + val * cosVal;
            }
        } else {
            // GPT-J style: interleaved pairs
            if (d % 2 == 0) {
                AccT partner = static_cast<AccT>(inPtr[d + 1]) * rmsInv *
                               static_cast<AccT>(gamma[d + 1]);
                rotatedVal = val * cosVal - partner * sinVal;
            } else {
                AccT partner = static_cast<AccT>(inPtr[d - 1]) * rmsInv *
                               static_cast<AccT>(gamma[d - 1]);
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
    // Use sizeof AccType<T>::type: double when T=double, float otherwise.
    size_t smem = numWarps * (dtype == DataType::DOUBLE ? sizeof(double) : sizeof(float));

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
    } else if (dtype == DataType::DOUBLE) {
        fusedQkNormRopeKernel<double><<<totalQHeads, threads, smem, *stream>>>(
            reinterpret_cast<const double*>(query->specialBuffer()),
            reinterpret_cast<const double*>(gammaQ->specialBuffer()),
            reinterpret_cast<const double*>(cosPtr),
            reinterpret_cast<const double*>(sinPtr),
            reinterpret_cast<double*>(queryOut->specialBuffer()),
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
    } else if (dtype == DataType::DOUBLE) {
        fusedQkNormRopeKernel<double><<<totalKHeads, threads, smem, *stream>>>(
            reinterpret_cast<const double*>(key->specialBuffer()),
            reinterpret_cast<const double*>(gammaK->specialBuffer()),
            reinterpret_cast<const double*>(cosPtr),
            reinterpret_cast<const double*>(sinPtr),
            reinterpret_cast<double*>(keyOut->specialBuffer()),
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
