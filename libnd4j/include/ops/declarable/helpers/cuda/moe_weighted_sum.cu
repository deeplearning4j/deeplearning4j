/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//
// CUDA implementation of moe_weighted_sum helpers.
//
// Dense kernel: one CUDA block per token; threads tile over D.
// Flat kernel:  one CUDA block per row; threads tile over D (scatter-add via atomicAdd).
//

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/moe_weighted_sum.h>
#include <system/op_boilerplate.h>
#include <system/selective_rendering.h>
#include <execution/cuda/LaunchDims.h>
#include <type_traits>

#if NOT_EXCLUDED(OP_moe_weighted_sum)

namespace sd {
namespace ops {
namespace helpers {

// ─────────────────────────────────────────────────────────────────────────────
// Dense layout kernel  [T, topK, D]  →  [T, D]
//
// Grid:  (T) blocks, one block per token
// Block: BLOCK_SIZE_MOE_WEIGHTED_SUM threads tiling over D
// Shmem: topK floats for the weights of this token
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
SD_KERNEL __launch_bounds__(256, 2)
static void moeWeightedSumDenseKernel(
    const T*          expertOutputs,   // [T, topK, D]
    const T*          weights,         // [T, topK]
    T*                output,          // [T, D]
    const sd::LongType T_tokens,
    const sd::LongType topK,
    const sd::LongType D,
    const sd::LongType eoStrideT,
    const sd::LongType eoStrideK,
    const sd::LongType eoStrideD,
    const sd::LongType wStrideT,
    const sd::LongType wStrideK,
    const sd::LongType outStrideT,
    const sd::LongType outStrideD) {

    const sd::LongType t = blockIdx.x;
    if (t >= T_tokens) return;

    // Load this token's topK weights into shared memory (float for fp32 accumulation)
    extern __shared__ char sharedBuf[];
    float* sWeights = reinterpret_cast<float*>(sharedBuf);

    if (threadIdx.x < topK) {
        sWeights[threadIdx.x] = static_cast<float>(weights[t * wStrideT + threadIdx.x * wStrideK]);
    }
    __syncthreads();

    // Each thread handles a contiguous tile of D elements
    for (sd::LongType d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (sd::LongType k = 0; k < topK; k++) {
            const T eoVal = expertOutputs[t * eoStrideT + k * eoStrideK + d * eoStrideD];
            acc += sWeights[k] * static_cast<float>(eoVal);
        }
        output[t * outStrideT + d * outStrideD] = static_cast<T>(acc);
    }
}

template <typename T>
static void moeWeightedSumDenseCuda_(sd::LaunchContext* context,
                                      NDArray* expertOutputs,
                                      NDArray* weights,
                                      NDArray*       output) {
    const sd::LongType T_tokens = expertOutputs->sizeAt(0);
    const sd::LongType topK     = expertOutputs->sizeAt(1);
    const sd::LongType D        = expertOutputs->sizeAt(2);

    const sd::LongType eoStrideT  = expertOutputs->strideAt(0);
    const sd::LongType eoStrideK  = expertOutputs->strideAt(1);
    const sd::LongType eoStrideD  = expertOutputs->strideAt(2);
    const sd::LongType wStrideT   = weights->strideAt(0);
    const sd::LongType wStrideK   = weights->strideAt(1);
    const sd::LongType outStrideT = output->strideAt(0);
    const sd::LongType outStrideD = output->strideAt(1);

    const dim3 dims = getLaunchDims("moe_weighted_sum");
    const int  blockSize  = static_cast<int>(dims.y);
    // Shmem: topK floats for per-token weights; block size must be >= topK for the load above
    const size_t shmem = static_cast<size_t>(topK) * sizeof(float);

    PointersManager pm(context, "moeWeightedSumDense");

    const T* eoPtr  = expertOutputs->specialBufferasT<T>();
    const T* wPtr   = weights->specialBufferasT<T>();
    T*       outPtr = output->specialBufferasT<T>();

    moeWeightedSumDenseKernel<T><<<(int)T_tokens, blockSize, shmem, *context->getCudaStream()>>>(
        eoPtr, wPtr, outPtr,
        T_tokens, topK, D,
        eoStrideT, eoStrideK, eoStrideD,
        wStrideT, wStrideK,
        outStrideT, outStrideD);

    pm.synchronize();
}

void moeWeightedSumDense(sd::LaunchContext* context,
                          NDArray* expertOutputs,
                          NDArray* weights,
                          NDArray*       output) {
    NDArray::prepareSpecialUse({output}, {expertOutputs, weights});
    BUILD_SINGLE_SELECTOR(expertOutputs->dataType(), moeWeightedSumDenseCuda_,
                          (context, expertOutputs, weights, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {expertOutputs, weights});
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorted-flat layout kernel  [totalRows, D]  →  [T, D] via rowIndex
//
// Grid:  (totalRows) blocks, one block per sorted row
// Block: BLOCK_SIZE_MOE_WEIGHTED_SUM threads tiling over D
//
// All types: scatter-accumulate into a temporary fp32 buffer, then cast to T.
// The fp32 buffer [T, D] is allocated on device; a second pass casts it back.
// This avoids half-precision atomicAdd issues (no atomicAdd(__half) pre-sm70).
// ─────────────────────────────────────────────────────────────────────────────

template <typename IdxT>
SD_KERNEL __launch_bounds__(256, 2)
static void moeWeightedSumFlatAccumKernel(
    const float*       expertOutputsFp32, // [totalRows, D] — values up-cast to fp32
    const float*       weightsFp32,       // [T, topK]
    const IdxT*        rowIndex,          // [totalRows, 2]
    float*             accumOut,          // [T, D]  fp32 accumulator
    const sd::LongType totalRows,
    const sd::LongType D,
    const sd::LongType eoStrideRow,
    const sd::LongType eoStrideD,
    const sd::LongType wStrideT,
    const sd::LongType wStrideK,
    const sd::LongType riStrideRow,
    const sd::LongType outStrideD) {

    const sd::LongType r = blockIdx.x;
    if (r >= totalRows) return;

    const sd::LongType token_idx = static_cast<sd::LongType>(rowIndex[r * riStrideRow + 0]);
    const sd::LongType k_idx     = static_cast<sd::LongType>(rowIndex[r * riStrideRow + 1]);

    const float w      = weightsFp32[token_idx * wStrideT + k_idx * wStrideK];
    const float* eo_r  = expertOutputsFp32 + r * eoStrideRow;
    // accumOut stride: contiguous [T, D], row = token_idx * D
    float* out_row = accumOut + token_idx * D;

    for (sd::LongType d = threadIdx.x; d < D; d += blockDim.x) {
        atomicAdd(out_row + d, w * eo_r[d * eoStrideD]);
    }
}

// Cast fp32 accumulator → T output kernel
template <typename T>
SD_KERNEL __launch_bounds__(256, 2)
static void moeWeightedSumFlatCastKernel(
    const float* accumOut,  // [T*D] contiguous fp32
    T*           output,    // [T, D]
    const sd::LongType T_tokens,
    const sd::LongType D,
    const sd::LongType outStrideT,
    const sd::LongType outStrideD) {

    const sd::LongType tid = blockIdx.x * blockDim.x + threadIdx.x;
    const sd::LongType total = T_tokens * D;
    if (tid >= total) return;

    const sd::LongType t = tid / D;
    const sd::LongType d = tid % D;
    output[t * outStrideT + d * outStrideD] = static_cast<T>(accumOut[tid]);
}

template <typename T, typename IdxT>
static void moeWeightedSumFlatCuda_(sd::LaunchContext* context,
                                     NDArray* expertOutputs,
                                     NDArray* weights,
                                     NDArray* rowIndex,
                                     NDArray*       output) {
    const sd::LongType totalRows = expertOutputs->sizeAt(0);
    const sd::LongType D         = expertOutputs->sizeAt(1);
    const sd::LongType T_tokens  = output->sizeAt(0);

    const sd::LongType eoStrideRow = expertOutputs->strideAt(0);
    const sd::LongType eoStrideD   = expertOutputs->strideAt(1);
    const sd::LongType wStrideT    = weights->strideAt(0);
    const sd::LongType wStrideK    = weights->strideAt(1);
    const sd::LongType riStrideRow = rowIndex->strideAt(0);
    const sd::LongType outStrideT  = output->strideAt(0);
    const sd::LongType outStrideD  = output->strideAt(1);

    const dim3 dims      = getLaunchDims("moe_weighted_sum");
    const int  blockSize = static_cast<int>(dims.y);

    // Flat mode computes the scatter-add in fp32 directly on the input buffers.
    // segment_gemm produces fp32 in practice; other element types must be cast
    // by the caller first (checked BEFORE any device allocation).
    if (!std::is_same<T, float>::value) {
        THROW_EXCEPTION("moe_weighted_sum (flat, CUDA): non-fp32 input requires fp32 arrays in flat mode. "
                        "Cast expertOutputs and weights to fp32 before calling with mode=1.");
        return;
    }

    cudaStream_t stream = *context->getCudaStream();
    PointersManager pm(context, "moeWeightedSumFlat");

    // Allocate temporary fp32 accumulator [T_tokens * D] on device
    const sd::LongType accumElems = T_tokens * D;
    float* d_accum = nullptr;
    cudaMallocAsync(&d_accum, accumElems * sizeof(float), stream);
    cudaMemsetAsync(d_accum, 0, accumElems * sizeof(float), stream);

    float* d_eoFp32 = reinterpret_cast<float*>(expertOutputs->specialBufferasT<T>());
    float* d_wFp32  = reinterpret_cast<float*>(weights->specialBufferasT<T>());

    // Scatter-accumulate into fp32 buffer
    moeWeightedSumFlatAccumKernel<IdxT><<<(int)totalRows, blockSize, 0, stream>>>(
        d_eoFp32, d_wFp32,
        rowIndex->specialBufferasT<IdxT>(),
        d_accum,
        totalRows, D,
        eoStrideRow, eoStrideD,
        wStrideT, wStrideK,
        riStrideRow, D /* accumOut D stride (contiguous) */);

    // Cast fp32 accumulator to output type
    const sd::LongType castTotal = T_tokens * D;
    const int castBlocks = static_cast<int>((castTotal + blockSize - 1) / blockSize);
    moeWeightedSumFlatCastKernel<T><<<castBlocks, blockSize, 0, stream>>>(
        d_accum, output->specialBufferasT<T>(),
        T_tokens, D, outStrideT, outStrideD);

    cudaFreeAsync(d_accum, stream);
    pm.synchronize();
}

// Outer dispatch: index type then element type
template <typename T>
static void moeWeightedSumFlatDispatchIdx_(sd::LaunchContext* context,
                                            NDArray* expertOutputs,
                                            NDArray* weights,
                                            NDArray* rowIndex,
                                            NDArray*       output) {
    if (rowIndex->dataType() == sd::DataType::INT64) {
        moeWeightedSumFlatCuda_<T, sd::LongType>(context, expertOutputs, weights, rowIndex, output);
    } else {
        moeWeightedSumFlatCuda_<T, int>(context, expertOutputs, weights, rowIndex, output);
    }
}

void moeWeightedSumFlat(sd::LaunchContext* context,
                         NDArray* expertOutputs,
                         NDArray* weights,
                         NDArray* rowIndex,
                         NDArray*       output) {
    NDArray::prepareSpecialUse({output}, {expertOutputs, weights, rowIndex});
    BUILD_SINGLE_SELECTOR(expertOutputs->dataType(), moeWeightedSumFlatDispatchIdx_,
                          (context, expertOutputs, weights, rowIndex, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {expertOutputs, weights, rowIndex});
}

// Explicit instantiations for the linker
BUILD_SINGLE_TEMPLATE(void moeWeightedSumDenseCuda_,
    (sd::LaunchContext*, NDArray*, NDArray*, NDArray*),
    SD_FLOAT_TYPES);

BUILD_SINGLE_TEMPLATE(void moeWeightedSumFlatDispatchIdx_,
    (sd::LaunchContext*, NDArray*, NDArray*, NDArray*, NDArray*),
    SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_moe_weighted_sum)
