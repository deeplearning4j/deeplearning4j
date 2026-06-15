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
// CUDA implementation of per-layer KL divergence for adaptive quantization.
//
// One CUDA block per row. Each block:
//   1. Loads the row, divides by temperature, reduces for max (stable softmax).
//   2. Computes log-softmax for reference and quantized logits.
//   3. Accumulates KL contribution per element.
//   4. Reduces the KL sum for this row and writes to a per-row buffer.
// A final reduction kernel averages over rows.
//

#include <ops/declarable/helpers/kl_divergence_per_layer.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int KL_WARP_SIZE = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: per-row log-softmax + KL divergence accumulation
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
SD_KERNEL void klPerRowKernel(
    const T* __restrict__ ref,
    const T* __restrict__ quant,
    float* __restrict__ rowKL,
    const LongType numRows,
    const LongType dim,
    const float temperature) {

    const LongType r = blockIdx.x;
    if (r >= numRows) return;

    const LongType rowOffset = r * dim;

    extern __shared__ char smem[];
    float* warpBuf = reinterpret_cast<float*>(smem);

    const int lane    = threadIdx.x % KL_WARP_SIZE;
    const int wid     = threadIdx.x / KL_WARP_SIZE;
    const int numWarps = (blockDim.x + KL_WARP_SIZE - 1) / KL_WARP_SIZE;

    // ── Find row max for reference ──────────────────────────────────────────
    float refMax = -FLT_MAX;
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        float v = static_cast<float>(ref[rowOffset + c]) / temperature;
        refMax = sd::math::sd_max<float>(refMax, v);
    }
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        refMax = sd::math::sd_max<float>(refMax, __shfl_down_sync(0xffffffff, refMax, offset));
    if (lane == 0) warpBuf[wid] = refMax;
    __syncthreads();

    float globalRefMax = -FLT_MAX;
    if (threadIdx.x < numWarps) globalRefMax = warpBuf[threadIdx.x];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        globalRefMax = sd::math::sd_max<float>(globalRefMax, __shfl_down_sync(0xffffffff, globalRefMax, offset));
    __shared__ float shRefMax;
    if (threadIdx.x == 0) shRefMax = globalRefMax;
    __syncthreads();

    // ── Log-sum-exp for reference ────────────────────────────────────────────
    float refLogSumExp = 0.0f;
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        float v = static_cast<float>(ref[rowOffset + c]) / temperature - shRefMax;
        refLogSumExp += sd::math::sd_exp<float, float>(v);
    }
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        refLogSumExp += __shfl_down_sync(0xffffffff, refLogSumExp, offset);
    if (lane == 0) warpBuf[wid] = refLogSumExp;
    __syncthreads();

    float totalRefLSE = 0.0f;
    if (threadIdx.x < numWarps) totalRefLSE = warpBuf[threadIdx.x];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        totalRefLSE += __shfl_down_sync(0xffffffff, totalRefLSE, offset);
    __shared__ float shRefLogSum;
    if (threadIdx.x == 0) shRefLogSum = logf(totalRefLSE);
    __syncthreads();

    // ── Find row max for quantized ──────────────────────────────────────────
    float qMax = -FLT_MAX;
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        float v = static_cast<float>(quant[rowOffset + c]) / temperature;
        qMax = sd::math::sd_max<float>(qMax, v);
    }
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        qMax = sd::math::sd_max<float>(qMax, __shfl_down_sync(0xffffffff, qMax, offset));
    if (lane == 0) warpBuf[wid] = qMax;
    __syncthreads();

    float globalQMax = -FLT_MAX;
    if (threadIdx.x < numWarps) globalQMax = warpBuf[threadIdx.x];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        globalQMax = sd::math::sd_max<float>(globalQMax, __shfl_down_sync(0xffffffff, globalQMax, offset));
    __shared__ float shQMax;
    if (threadIdx.x == 0) shQMax = globalQMax;
    __syncthreads();

    // ── Log-sum-exp for quantized ────────────────────────────────────────────
    float qLogSumExp = 0.0f;
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        float v = static_cast<float>(quant[rowOffset + c]) / temperature - shQMax;
        qLogSumExp += sd::math::sd_exp<float, float>(v);
    }
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        qLogSumExp += __shfl_down_sync(0xffffffff, qLogSumExp, offset);
    if (lane == 0) warpBuf[wid] = qLogSumExp;
    __syncthreads();

    float totalQLSE = 0.0f;
    if (threadIdx.x < numWarps) totalQLSE = warpBuf[threadIdx.x];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        totalQLSE += __shfl_down_sync(0xffffffff, totalQLSE, offset);
    __shared__ float shQLogSum;
    if (threadIdx.x == 0) shQLogSum = logf(totalQLSE);
    __syncthreads();

    // ── Accumulate KL(P||Q) for this row ────────────────────────────────────
    // logP[c] = ref[c]/T - shRefMax - shRefLogSum
    // logQ[c] = quant[c]/T - shQMax  - shQLogSum
    // KL contribution: P[c] * (logP[c] - logQ[c])
    float klThread = 0.0f;
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        float logP = static_cast<float>(ref[rowOffset + c]) / temperature - shRefMax - shRefLogSum;
        float logQ = static_cast<float>(quant[rowOffset + c]) / temperature - shQMax - shQLogSum;
        float p = sd::math::sd_exp<float, float>(logP);
        if (p > 1e-12f) {
            klThread += p * (logP - logQ);
        }
    }
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        klThread += __shfl_down_sync(0xffffffff, klThread, offset);
    if (lane == 0) warpBuf[wid] = klThread;
    __syncthreads();

    float rowKLVal = 0.0f;
    if (threadIdx.x < numWarps) rowKLVal = warpBuf[threadIdx.x];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        rowKLVal += __shfl_down_sync(0xffffffff, rowKLVal, offset);

    if (threadIdx.x == 0)
        rowKL[r] = rowKLVal;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: reduce per-row KL to a single mean scalar
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
SD_KERNEL void klMeanKernel(
    const float* __restrict__ rowKL,
    T* __restrict__ output,
    const LongType numRows) {

    float sum = 0.0f;
    for (LongType i = threadIdx.x; i < numRows; i += blockDim.x)
        sum += rowKL[i];
    for (int offset = KL_WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
        output[0] = static_cast<T>(sum / static_cast<float>(numRows));
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed launcher
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static void klLauncher(const cudaStream_t* stream,
                        const void* vRef, const void* vQuant,
                        void* vOutput, void* vRowKL,
                        LongType numRows, LongType dim, float temperature) {
    auto ref    = reinterpret_cast<const T*>(vRef);
    auto quant  = reinterpret_cast<const T*>(vQuant);
    auto output = reinterpret_cast<T*>(vOutput);
    auto rowKL  = reinterpret_cast<float*>(vRowKL);

    int blockDim = 256;
    if (dim < 256) {
        blockDim = ((dim + KL_WARP_SIZE - 1) / KL_WARP_SIZE) * KL_WARP_SIZE;
        if (blockDim < KL_WARP_SIZE) blockDim = KL_WARP_SIZE;
    }
    int numWarps  = (blockDim + KL_WARP_SIZE - 1) / KL_WARP_SIZE;
    size_t smSize = numWarps * sizeof(float);

    klPerRowKernel<T><<<numRows, blockDim, smSize, *stream>>>(
        ref, quant, rowKL, numRows, dim, temperature);
    DebugHelper::checkGlobalErrorCode("klPerRowKernel failed");

    klMeanKernel<T><<<1, 256, 0, *stream>>>(rowKL, output, numRows);
    DebugHelper::checkGlobalErrorCode("klMeanKernel failed");
}

BUILD_SINGLE_TEMPLATE(void klLauncher,
                      (const cudaStream_t* stream,
                       const void* vRef, const void* vQuant,
                       void* vOutput, void* vRowKL,
                       LongType numRows, LongType dim, float temperature),
                      SD_FLOAT_TYPES);

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────
void klDivergencePerLayer(NDArray* referenceLogits,
                           NDArray* quantizedLogits,
                           NDArray* output,
                           double temperature,
                           LaunchContext* context) {
    LongType dim     = referenceLogits->sizeAt(referenceLogits->rankOf() - 1);
    LongType numRows = referenceLogits->lengthOf() / dim;
    if (temperature <= 0.0) temperature = 1.0;

    auto stream = context->getCudaStream();

    auto rowKL = NDArrayFactory::create('c', {numRows}, DataType::FLOAT32, context);

    NDArray::prepareSpecialUse({output}, {referenceLogits, quantizedLogits});

    BUILD_SINGLE_SELECTOR(referenceLogits->dataType(), klLauncher,
                          (stream,
                           referenceLogits->specialBuffer(), quantizedLogits->specialBuffer(),
                           output->specialBuffer(), rowKL->specialBuffer(),
                           numRows, dim, static_cast<float>(temperature)),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {referenceLogits, quantizedLogits});

    delete rowKL;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
