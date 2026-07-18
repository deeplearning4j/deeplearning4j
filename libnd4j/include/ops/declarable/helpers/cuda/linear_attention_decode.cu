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
// linear_attention_decode.cu — single-token decode step for linear attention.
//
// Implements the recurrent decode step of linear attention:
//
//   decay    = exp(-decayRates[h])               (per-head scalar)
//   state   += decay * state + k ⊗ v             (rank-1 outer-product update, in-place)
//   output   = q @ state                          (matrix-vector product)
//
// Optimization patterns drawn from cuLA (la_decode.py):
//   1. State is tiled in the V-dimension; each block handles one (batch, head) pair.
//   2. Warp butterfly shuffle reduction accumulates the q·state[v,:] dot product.
//   3. __expf() is used for all exponentials (maps to ~4 ULP hardware instruction).
//   4. State is always float32 regardless of input dtype to prevent precision
//      degradation from accumulating across decode steps.
//   5. __launch_bounds__ hints to the compiler for register allocation.
//
// Layout assumptions (all C-contiguous within their logical rank):
//   query / key      : [batch, 1, numHeads, headDimK]
//   value            : [batch, 1, numHeads, headDimV]
//   decayRates       : [numHeads]  float32
//   state (in/out)   : [batch, numHeads, headDimV, headDimK]  float32
//   output           : [batch, 1, numHeads, headDimV]
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <math/templatemath.h>
#include <math/cuda_fast_math.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <types/float16.h>
#include <ops/declarable/helpers/linear_attention_decode.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

// Accumulator type: double when T=double for precision, float otherwise.
// The state buffer is always float32 by design; AccT is used for the q·state
// dot product so that double-precision inputs yield double-precision output.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Warp size — fixed by the CUDA hardware.
static constexpr int LA_WARP_SIZE = 32;

// Block size used for the decode kernel.
// 128 threads: two warps per block, each covering a contiguous K-tile.
// Chosen to allow high occupancy for the K-dimension inner loop.
static constexpr int LA_BLOCK_SIZE = 128;

// ---------------------------------------------------------------------------
// Warp/block sum reductions come from device_primitives.cuh
// (sd::device::warpReduceSum / sd::device::blockReduceSum).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main decode kernel
//
// Grid:  (batch * numHeads)  — one block per (batch, head) pair.
// Block: LA_BLOCK_SIZE threads.
//
// Each block:
//   1. Reads the per-head decay scalar and computes decay = exp(-decayRates[h]).
//   2. Iterates over the V-dimension: for each v-row of the state matrix,
//      a) updates state[v, :] = decay * state[v, :] + v_val * k[:]
//      b) accumulates output[v] += q[:] · state[v, :] via warp-reduce.
//   3. Writes output[v] to the output buffer.
//
// Shared memory layout (total = LA_BLOCK_SIZE/LA_WARP_SIZE floats):
//   float reductionBuf[numWarps]  — warp-leader partial sums for block reduce.
//
// No explicit shared memory for q/k/v vectors — they fit in registers for
// typical head dimensions (64-128) when threads are strided over the K-dim.
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL __launch_bounds__(LA_BLOCK_SIZE, 2) void linearAttentionDecodeKernel(
    const T*    __restrict__ query,       // [batch, 1, numHeads, headDimK]
    const T*    __restrict__ key,         // [batch, 1, numHeads, headDimK]
    const T*    __restrict__ value,       // [batch, 1, numHeads, headDimV]
    const float* __restrict__ decayRates, // [numHeads]
    float*      __restrict__ state,       // [batch, numHeads, headDimV, headDimK], in-place
    T*          __restrict__ output,      // [batch, 1, numHeads, headDimV]
    // Dimension sizes
    const LongType batch,
    const LongType numHeads,
    const LongType headDimK,
    const LongType headDimV,
    // query strides [batch, seq=1, heads, headDimK]
    const LongType qS0, const LongType qS1, const LongType qS2, const LongType qS3,
    // key strides [batch, seq=1, heads, headDimK]
    const LongType kS0, const LongType kS1, const LongType kS2, const LongType kS3,
    // value strides [batch, seq=1, heads, headDimV]
    const LongType vS0, const LongType vS1, const LongType vS2, const LongType vS3,
    // output strides [batch, seq=1, heads, headDimV]
    const LongType oS0, const LongType oS1, const LongType oS2, const LongType oS3) {

    // -----------------------------------------------------------------
    // Map block to (batch, head) pair.
    // -----------------------------------------------------------------
    const LongType bh = static_cast<LongType>(blockIdx.x);
    if (bh >= batch * numHeads) return;

    const LongType b = bh / numHeads;
    const LongType h = bh % numHeads;

    using AccT = typename AccType<T>::type;

    // -----------------------------------------------------------------
    // Per-head decay scalar: decay = exp(-decayRates[h]).
    // Using __expf for the fast hardware path (~4 ULP, ~5x faster).
    // State is always float32 by design; decay stays float.
    // -----------------------------------------------------------------
    const float decay = __expf(-decayRates[h]);

    // -----------------------------------------------------------------
    // Shared memory for warp-leader partial sums during block reduction.
    // AccT (double for T=double, float otherwise) for the dot product reduction.
    // -----------------------------------------------------------------
    extern __shared__ char sharedMemBuf[];
    AccT* sharedMem = reinterpret_cast<AccT*>(sharedMemBuf);

    // -----------------------------------------------------------------
    // Base pointers for this (batch, head) pair.
    // -----------------------------------------------------------------
    // query: shape [batch, 1, numHeads, headDimK], seq dimension = 0
    const T* qBase = query + b * qS0 + 0 * qS1 + h * qS2;  // + dk * qS3
    // key: same layout as query
    const T* kBase = key   + b * kS0 + 0 * kS1 + h * kS2;  // + dk * kS3
    // value: shape [batch, 1, numHeads, headDimV]
    const T* vBase = value + b * vS0 + 0 * vS1 + h * vS2;  // + dv * vS3
    // state: shape [batch, numHeads, headDimV, headDimK], always float32
    float*   sBase = state + (b * numHeads + h) * headDimV * headDimK;
    // output: shape [batch, 1, numHeads, headDimV]
    T*       oBase = output + b * oS0 + 0 * oS1 + h * oS2;  // + dv * oS3

    // -----------------------------------------------------------------
    // Main loop over the V-dimension (one row of the state matrix per iter).
    // Each iteration:
    //   a) Update the state row:  state[dv, :] = decay * state[dv, :] + v_val * k[:]
    //      (state stays float32 by design to prevent quantization drift)
    //   b) Accumulate the output: out[dv] = q[:] · state[dv, :]
    //      in AccT precision (double for T=double) using warp/block reduction.
    // -----------------------------------------------------------------
    for (LongType dv = 0; dv < headDimV; ++dv) {
        // Float32 v-value for this row (state accumulation stays float32).
        const float v_val = static_cast<float>(vBase[dv * vS3]);

        // Pointer to state row [dv, :] — headDimK elements.
        float* sRow = sBase + dv * headDimK;

        // Accumulate the dot product q · state[dv, :] in AccT.
        // Threads are strided over the K-dimension.
        AccT dot = static_cast<AccT>(0);

        for (LongType dk = static_cast<LongType>(threadIdx.x);
             dk < headDimK;
             dk += static_cast<LongType>(blockDim.x)) {

            const float k_val = static_cast<float>(kBase[dk * kS3]);
            const AccT q_val = static_cast<AccT>(qBase[dk * qS3]);

            // State update (in-place, float32): decay * s + v * k
            float s_new = decay * sRow[dk] + v_val * k_val;
            sRow[dk] = s_new;

            // Accumulate q · s_new into the dot product in AccT.
            dot += q_val * static_cast<AccT>(s_new);
        }

        // Block-wide reduction of dot products from all threads.
        dot = sd::device::blockReduceSum(dot, sharedMem);

        // Thread 0 writes the output for this v-row.
        if (threadIdx.x == 0) {
            oBase[dv * oS3] = static_cast<T>(dot);
        }

        // Ensure all threads have finished the state update for this v-row
        // before the next iteration reads back updated values.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Type-dispatching launcher.
// Allocates no temporary memory — state is updated in-place.
// ---------------------------------------------------------------------------
template <typename T>
static void launchLinearAttentionDecode(
    const T*     query,
    const T*     key,
    const T*     value,
    const float* decayRates,
    float*       state,
    T*           output,
    LongType batch, LongType numHeads, LongType headDimK, LongType headDimV,
    LongType qS0, LongType qS1, LongType qS2, LongType qS3,
    LongType kS0, LongType kS1, LongType kS2, LongType kS3,
    LongType vS0, LongType vS1, LongType vS2, LongType vS3,
    LongType oS0, LongType oS1, LongType oS2, LongType oS3,
    cudaStream_t stream) {

    const int numBlocks = static_cast<int>(batch * numHeads);

    // Shared memory: one AccT per warp for block-level reduction.
    // AccT is double for T=double, float otherwise.
    const int numWarps   = (LA_BLOCK_SIZE + LA_WARP_SIZE - 1) / LA_WARP_SIZE;
    const int sharedBytes = numWarps * static_cast<int>(sizeof(typename AccType<T>::type));

    linearAttentionDecodeKernel<T><<<numBlocks, LA_BLOCK_SIZE, sharedBytes, stream>>>(
        query, key, value, decayRates, state, output,
        batch, numHeads, headDimK, headDimV,
        qS0, qS1, qS2, qS3,
        kS0, kS1, kS2, kS3,
        vS0, vS1, vS2, vS3,
        oS0, oS1, oS2, oS3);

    DebugHelper::checkGlobalErrorCode("linearAttentionDecodeKernel failed");
}

// ---------------------------------------------------------------------------
// Kernel: convert typed input state to float32 working buffer.
// Used when the caller passes a non-float32 state array.
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL void laStateToFloat32Kernel(
    const T*   __restrict__ src,
    float*     __restrict__ dst,
    const LongType total) {
    const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: convert float32 working buffer back to typed output state.
// ---------------------------------------------------------------------------
template <typename T>
SD_KERNEL void laFloat32ToStateKernel(
    const float* __restrict__ src,
    T*           __restrict__ dst,
    const LongType total) {
    const LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = static_cast<T>(src[idx]);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void linearAttentionDecode(
    LaunchContext* context,
    NDArray* query,
    NDArray* key,
    NDArray* value,
    NDArray* decayRates,
    NDArray* state,
    NDArray* output) {

    // Dimension extraction.
    // query layout: [batch, 1, numHeads, headDimK]
    const LongType batch    = query->sizeAt(0);
    const LongType numHeads = query->sizeAt(2);
    const LongType headDimK = query->sizeAt(3);
    // value layout: [batch, 1, numHeads, headDimV]
    const LongType headDimV = value->sizeAt(3);

    // State is always float32: [batch, numHeads, headDimV, headDimK]
    // The caller is responsible for passing a float32 state array.
    // This matches gated_delta_rule and mamba2_ssm conventions.
    const LongType stateElems = batch * numHeads * headDimV * headDimK;

    NDArray::prepareSpecialUse({output, state}, {query, key, value, decayRates});

    auto stream   = context->getCudaStream();
    int  deviceId = sd::AffinityManager::currentDeviceId();

    // Determine whether we need a float32 temporary for the state.
    // If state is already FLOAT32, operate directly in-place.
    // Otherwise allocate a float32 scratch buffer and copy back at the end.
    const bool stateIsFloat32 = (state->dataType() == DataType::FLOAT32);
    float* workingState       = nullptr;

    if (stateIsFloat32) {
        workingState = reinterpret_cast<float*>(state->specialBuffer());
    } else {
        // Allocate float32 scratch via pool (stream-ordered, no sync required).
        workingState = reinterpret_cast<float*>(
            sd::memory::CudaMemoryPool::getInstance().allocate(
                stateElems * sizeof(float), deviceId, *stream));

        // Convert in-typed state to float32.
        const int initBlocks = static_cast<int>((stateElems + 255) / 256);
        auto dtype = state->dataType();
        if (dtype == DataType::HALF) {
            laStateToFloat32Kernel<float16><<<initBlocks, 256, 0, *stream>>>(
                reinterpret_cast<const float16*>(state->specialBuffer()),
                workingState, stateElems);
        } else if (dtype == DataType::DOUBLE) {
            laStateToFloat32Kernel<double><<<initBlocks, 256, 0, *stream>>>(
                reinterpret_cast<const double*>(state->specialBuffer()),
                workingState, stateElems);
        }
    }

    // Dispatch typed kernel.
    auto dtype = query->dataType();

    if (dtype == DataType::FLOAT32) {
        launchLinearAttentionDecode<float>(
            reinterpret_cast<const float*>(query->specialBuffer()),
            reinterpret_cast<const float*>(key->specialBuffer()),
            reinterpret_cast<const float*>(value->specialBuffer()),
            reinterpret_cast<const float*>(decayRates->specialBuffer()),
            workingState,
            reinterpret_cast<float*>(output->specialBuffer()),
            batch, numHeads, headDimK, headDimV,
            query->strideAt(0), query->strideAt(1), query->strideAt(2), query->strideAt(3),
            key->strideAt(0),   key->strideAt(1),   key->strideAt(2),   key->strideAt(3),
            value->strideAt(0), value->strideAt(1), value->strideAt(2), value->strideAt(3),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else if (dtype == DataType::HALF) {
        launchLinearAttentionDecode<float16>(
            reinterpret_cast<const float16*>(query->specialBuffer()),
            reinterpret_cast<const float16*>(key->specialBuffer()),
            reinterpret_cast<const float16*>(value->specialBuffer()),
            reinterpret_cast<const float*>(decayRates->specialBuffer()),
            workingState,
            reinterpret_cast<float16*>(output->specialBuffer()),
            batch, numHeads, headDimK, headDimV,
            query->strideAt(0), query->strideAt(1), query->strideAt(2), query->strideAt(3),
            key->strideAt(0),   key->strideAt(1),   key->strideAt(2),   key->strideAt(3),
            value->strideAt(0), value->strideAt(1), value->strideAt(2), value->strideAt(3),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else if (dtype == DataType::DOUBLE) {
        launchLinearAttentionDecode<double>(
            reinterpret_cast<const double*>(query->specialBuffer()),
            reinterpret_cast<const double*>(key->specialBuffer()),
            reinterpret_cast<const double*>(value->specialBuffer()),
            reinterpret_cast<const float*>(decayRates->specialBuffer()),
            workingState,
            reinterpret_cast<double*>(output->specialBuffer()),
            batch, numHeads, headDimK, headDimV,
            query->strideAt(0), query->strideAt(1), query->strideAt(2), query->strideAt(3),
            key->strideAt(0),   key->strideAt(1),   key->strideAt(2),   key->strideAt(3),
            value->strideAt(0), value->strideAt(1), value->strideAt(2), value->strideAt(3),
            output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3),
            *stream);
    } else {
        THROW_EXCEPTION("linearAttentionDecode: unsupported data type");
    }

    // If state was not float32, copy the updated float32 working buffer back.
    if (!stateIsFloat32) {
        const int copyBlocks = static_cast<int>((stateElems + 255) / 256);
        auto dtype2 = state->dataType();
        if (dtype2 == DataType::HALF) {
            laFloat32ToStateKernel<float16><<<copyBlocks, 256, 0, *stream>>>(
                workingState,
                reinterpret_cast<float16*>(state->specialBuffer()),
                stateElems);
        } else if (dtype2 == DataType::DOUBLE) {
            laFloat32ToStateKernel<double><<<copyBlocks, 256, 0, *stream>>>(
                workingState,
                reinterpret_cast<double*>(state->specialBuffer()),
                stateElems);
        }

        // Return scratch to the pool.
        sd::memory::CudaMemoryPool::getInstance().free(workingState, deviceId, *stream);
    }

    NDArray::registerSpecialUse({output, state}, {query, key, value, decayRates});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
