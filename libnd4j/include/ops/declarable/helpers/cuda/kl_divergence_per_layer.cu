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
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int KL_WARP_SIZE = 32;

// Accumulator/scratch type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: per-row log-softmax + KL divergence accumulation
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
SD_KERNEL void klPerRowKernel(
    const T* __restrict__ ref,
    const T* __restrict__ quant,
    typename AccType<T>::type* __restrict__ rowKL,
    const LongType numRows,
    const LongType dim,
    const float temperature) {

    using AccT = typename AccType<T>::type;

    const LongType r = blockIdx.x;
    if (r >= numRows) return;

    const LongType rowOffset = r * dim;

    extern __shared__ char smem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(smem);

    const int lane    = threadIdx.x % KL_WARP_SIZE;
    const int wid     = threadIdx.x / KL_WARP_SIZE;
    const int numWarps = (blockDim.x + KL_WARP_SIZE - 1) / KL_WARP_SIZE;

    // ── Find row max for reference (broadcast to all threads) ───────────────
    AccT refMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        AccT v = static_cast<AccT>(ref[rowOffset + c]) / static_cast<AccT>(temperature);
        refMax = sd::math::sd_max<AccT>(refMax, v);
    }
    AccT globalRefMax = sd::device::blockAllReduceMax(refMax, warpBuf);

    // ── Log-sum-exp for reference (broadcast to all threads) ─────────────────
    AccT refLSE = static_cast<AccT>(0);
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        AccT v = static_cast<AccT>(ref[rowOffset + c]) / static_cast<AccT>(temperature) - globalRefMax;
        refLSE += sd::math::sd_exp<AccT, AccT>(v);
    }
    AccT totalRefLSE = sd::device::blockAllReduceSum(refLSE, warpBuf);
    AccT shRefLogSum = sd::math::sd_log<AccT, AccT>(totalRefLSE);

    // ── Find row max for quantized (broadcast to all threads) ───────────────
    AccT qMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        AccT v = static_cast<AccT>(quant[rowOffset + c]) / static_cast<AccT>(temperature);
        qMax = sd::math::sd_max<AccT>(qMax, v);
    }
    AccT globalQMax = sd::device::blockAllReduceMax(qMax, warpBuf);

    // ── Log-sum-exp for quantized (broadcast to all threads) ─────────────────
    AccT qLSE = static_cast<AccT>(0);
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        AccT v = static_cast<AccT>(quant[rowOffset + c]) / static_cast<AccT>(temperature) - globalQMax;
        qLSE += sd::math::sd_exp<AccT, AccT>(v);
    }
    AccT totalQLSE = sd::device::blockAllReduceSum(qLSE, warpBuf);
    AccT shQLogSum = sd::math::sd_log<AccT, AccT>(totalQLSE);

    // ── Accumulate KL(P||Q) for this row ────────────────────────────────────
    // logP[c] = ref[c]/T - globalRefMax - shRefLogSum
    // logQ[c] = quant[c]/T - globalQMax  - shQLogSum
    // KL contribution: P[c] * (logP[c] - logQ[c])
    AccT klThread = static_cast<AccT>(0);
    for (LongType c = threadIdx.x; c < dim; c += blockDim.x) {
        AccT logP = static_cast<AccT>(ref[rowOffset + c]) / static_cast<AccT>(temperature) - globalRefMax - shRefLogSum;
        AccT logQ = static_cast<AccT>(quant[rowOffset + c]) / static_cast<AccT>(temperature) - globalQMax - shQLogSum;
        AccT p = sd::math::sd_exp<AccT, AccT>(logP);
        if (p > static_cast<AccT>(1e-12)) {
            klThread += p * (logP - logQ);
        }
    }

    // Only thread 0 needs the KL sum to write rowKL[r]
    AccT rowKLVal = sd::device::blockReduceSum(klThread, warpBuf);

    if (threadIdx.x == 0)
        rowKL[r] = rowKLVal;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: reduce per-row KL to a single mean scalar
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
SD_KERNEL void klMeanKernel(
    const typename AccType<T>::type* __restrict__ rowKL,
    T* __restrict__ output,
    const LongType numRows) {

    using AccT = typename AccType<T>::type;

    AccT sum = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < numRows; i += blockDim.x)
        sum += rowKL[i];

    // Single-warp kernel — warpReduceSum is sufficient
    sum = sd::device::warpReduceSum(sum);

    if (threadIdx.x == 0)
        output[0] = static_cast<T>(sum / static_cast<AccT>(numRows));
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
    using AccT = typename AccType<T>::type;
    auto rowKL  = reinterpret_cast<AccT*>(vRowKL);

    int blockDim = 256;
    if (dim < 256) {
        blockDim = ((dim + KL_WARP_SIZE - 1) / KL_WARP_SIZE) * KL_WARP_SIZE;
        if (blockDim < KL_WARP_SIZE) blockDim = KL_WARP_SIZE;
    }
    int numWarps  = (blockDim + KL_WARP_SIZE - 1) / KL_WARP_SIZE;
    size_t smSize = numWarps * sizeof(AccT);

    klPerRowKernel<T><<<numRows, blockDim, smSize, *stream>>>(
        ref, quant, rowKL, numRows, dim, temperature);
    DebugHelper::checkGlobalErrorCode("klPerRowKernel failed");

    klMeanKernel<T><<<1, KL_WARP_SIZE, 0, *stream>>>(rowKL, output, numRows);
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

    auto accDtype = referenceLogits->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    auto rowKL = NDArrayFactory::create('c', {numRows}, accDtype, context);

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
