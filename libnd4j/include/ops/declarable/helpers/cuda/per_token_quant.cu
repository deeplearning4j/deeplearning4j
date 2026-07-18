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
// Per-token FP8 quantization/dequantization CUDA kernels.
// Quantizes activations to FP8 E4M3 with per-token scale factors.
// Companion kernel for FP8 GEMM inference pipelines.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <ops/declarable/helpers/per_token_quant.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

constexpr int PTQ_WARP_SIZE = 32;

// FP8 E4M3 max representable value: 448.0
constexpr float FP8_E4M3_MAX = 448.0f;

// Warp/block max reductions come from device_primitives.cuh
// (sd::device::warpReduceMax / sd::device::blockReduceMax).

//////////////////////////////////////////////////////////////////////////////
// Per-token FP8 quantize kernel: one block per token (row)
// Two-pass: (1) find abs-max, (2) quantize to FP8 range
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void perTokenQuantFp8Kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numTokens,
    const LongType hiddenDim) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    extern __shared__ char sharedMemRaw[];
    float* sdata = reinterpret_cast<float*>(sharedMemRaw);

    const T* tokenInput = input + token * hiddenDim;
    int8_t* tokenOut = quantized + token * hiddenDim;

    // Pass 1: find per-token absmax
    float threadMax = 0.0f;
    for (LongType i = threadIdx.x; i < hiddenDim; i += blockDim.x) {
        float val = fabsf(static_cast<float>(tokenInput[i]));
        threadMax = fmaxf(threadMax, val);
    }

    float tokenMax = sd::device::blockReduceMax(threadMax, sdata);

    __shared__ float invScale;
    __shared__ float tokenScale;

    if (threadIdx.x == 0) {
        // scale = absmax / FP8_MAX
        tokenScale = tokenMax / FP8_E4M3_MAX;
        if (tokenScale == 0.0f) tokenScale = 1.0f;
        scales[token] = tokenScale;
        invScale = 1.0f / tokenScale;
    }
    __syncthreads();

    // Pass 2: quantize to FP8 E4M3 range, stored as int8
    // FP8 E4M3 range: [-448, 448], we map to [-127, 127] for int8 storage
    // with the scale factor encoding the FP8 magnitude
    for (LongType i = threadIdx.x; i < hiddenDim; i += blockDim.x) {
        float val = static_cast<float>(tokenInput[i]) * invScale;
        // Clamp to int8 range for storage
        val = fmaxf(-127.0f, fminf(127.0f, rintf(val)));
        tokenOut[i] = static_cast<int8_t>(val);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Warp-local variant: 8 tokens per block for small batches
// More efficient when numTokens < SM count
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void perTokenQuantFp8WarpLocalKernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numTokens,
    const LongType hiddenDim) {

    // Each warp handles one token
    const int warpId = threadIdx.x / PTQ_WARP_SIZE;
    const int lane = threadIdx.x % PTQ_WARP_SIZE;
    const int warpsPerBlock = blockDim.x / PTQ_WARP_SIZE;
    const LongType token = blockIdx.x * warpsPerBlock + warpId;

    if (token >= numTokens) return;

    const T* tokenInput = input + token * hiddenDim;
    int8_t* tokenOut = quantized + token * hiddenDim;

    // Pass 1: find absmax with warp reduction
    float threadMax = 0.0f;
    for (LongType i = lane; i < hiddenDim; i += PTQ_WARP_SIZE) {
        float val = fabsf(static_cast<float>(tokenInput[i]));
        threadMax = fmaxf(threadMax, val);
    }
    float tokenMax = sd::device::warpReduceMax(threadMax);

    // Broadcast scale from lane 0
    float tokenScale = tokenMax / FP8_E4M3_MAX;
    if (tokenScale == 0.0f) tokenScale = 1.0f;
    tokenScale = __shfl_sync(0xffffffff, tokenScale, 0);
    float invScale = 1.0f / tokenScale;

    if (lane == 0) {
        scales[token] = tokenScale;
    }

    // Pass 2: quantize
    for (LongType i = lane; i < hiddenDim; i += PTQ_WARP_SIZE) {
        float val = static_cast<float>(tokenInput[i]) * invScale;
        val = fmaxf(-127.0f, fminf(127.0f, rintf(val)));
        tokenOut[i] = static_cast<int8_t>(val);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Per-token FP8 dequantize kernel
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void perTokenDequantFp8Kernel(
    const int8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    T* __restrict__ output,
    const LongType numTokens,
    const LongType hiddenDim) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    const int8_t* tokenIn = quantized + token * hiddenDim;
    T* tokenOut = output + token * hiddenDim;
    float scale = scales[token];

    for (LongType i = threadIdx.x; i < hiddenDim; i += blockDim.x) {
        tokenOut[i] = static_cast<T>(static_cast<float>(tokenIn[i]) * scale);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Per-token group quantization kernel
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void perTokenGroupQuantKernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numTokens,
    const LongType hiddenDim,
    const int groupSize,
    const float quantMax) {

    const LongType token = blockIdx.x;
    if (token >= numTokens) return;

    const int numGroups = (hiddenDim + groupSize - 1) / groupSize;
    const T* tokenInput = input + token * hiddenDim;
    int8_t* tokenOut = quantized + token * hiddenDim;
    float* tokenScales = scales + token * numGroups;

    // Process each group
    for (int g = threadIdx.x; g < numGroups; g += blockDim.x) {
        const LongType gStart = g * groupSize;
        const LongType gEnd = min(gStart + groupSize, hiddenDim);

        // Find group absmax
        float groupMax = 0.0f;
        for (LongType i = gStart; i < gEnd; i++) {
            float val = fabsf(static_cast<float>(tokenInput[i]));
            groupMax = fmaxf(groupMax, val);
        }

        float groupScale = groupMax / quantMax;
        if (groupScale == 0.0f) groupScale = 1.0f;
        tokenScales[g] = groupScale;
        float invScale = 1.0f / groupScale;

        // Quantize group
        for (LongType i = gStart; i < gEnd; i++) {
            float val = static_cast<float>(tokenInput[i]) * invScale;
            val = fmaxf(-quantMax, fminf(quantMax, rintf(val)));
            tokenOut[i] = static_cast<int8_t>(val);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher helpers
//////////////////////////////////////////////////////////////////////////////
static int ptqChooseThreads(LongType hiddenDim) {
    if (hiddenDim <= 128) return 128;
    if (hiddenDim <= 256) return 256;
    if (hiddenDim <= 512) return 512;
    return 1024;
}

//////////////////////////////////////////////////////////////////////////////
// Public: perTokenQuantFp8
//////////////////////////////////////////////////////////////////////////////
void perTokenQuantFp8(LaunchContext* context,
                      NDArray* input,
                      NDArray* quantized,
                      NDArray* scales) {
    const LongType numTokens = input->lengthOf() / input->sizeAt(-1);
    const LongType hiddenDim = input->sizeAt(-1);

    NDArray::prepareSpecialUse({quantized, scales}, {input});

    auto stream = context->getCudaStream();
    auto dtype = input->dataType();

    int threads = ptqChooseThreads(hiddenDim);
    int numWarps = (threads + PTQ_WARP_SIZE - 1) / PTQ_WARP_SIZE;
    size_t smem = numWarps * sizeof(float);

    // Auto-dispatch: warp-local for small batches, block-per-token for large
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, context->getDeviceID());
    bool useWarpLocal = (numTokens < smCount * 4) && (hiddenDim <= 1024);

    if (useWarpLocal) {
        int warpsPerBlock = 8;
        int blockThreads = warpsPerBlock * PTQ_WARP_SIZE;
        int blocks = (numTokens + warpsPerBlock - 1) / warpsPerBlock;

        if (dtype == DataType::FLOAT32) {
            perTokenQuantFp8WarpLocalKernel<float><<<blocks, blockThreads, 0, *stream>>>(
                reinterpret_cast<const float*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                reinterpret_cast<float*>(scales->specialBuffer()),
                numTokens, hiddenDim);
        } else if (dtype == DataType::HALF) {
            perTokenQuantFp8WarpLocalKernel<float16><<<blocks, blockThreads, 0, *stream>>>(
                reinterpret_cast<const float16*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                reinterpret_cast<float*>(scales->specialBuffer()),
                numTokens, hiddenDim);
        } else {
            THROW_EXCEPTION("perTokenQuantFp8: unsupported data type");
        }
    } else {
        if (dtype == DataType::FLOAT32) {
            perTokenQuantFp8Kernel<float><<<numTokens, threads, smem, *stream>>>(
                reinterpret_cast<const float*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                reinterpret_cast<float*>(scales->specialBuffer()),
                numTokens, hiddenDim);
        } else if (dtype == DataType::HALF) {
            perTokenQuantFp8Kernel<float16><<<numTokens, threads, smem, *stream>>>(
                reinterpret_cast<const float16*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                reinterpret_cast<float*>(scales->specialBuffer()),
                numTokens, hiddenDim);
        } else {
            THROW_EXCEPTION("perTokenQuantFp8: unsupported data type");
        }
    }

    DebugHelper::checkGlobalErrorCode("perTokenQuantFp8 failed");
    NDArray::registerSpecialUse({quantized, scales}, {input});
}

//////////////////////////////////////////////////////////////////////////////
// Public: perTokenDequantFp8
//////////////////////////////////////////////////////////////////////////////
void perTokenDequantFp8(LaunchContext* context,
                        NDArray* quantized,
                        NDArray* scales,
                        NDArray* output) {
    const LongType numTokens = output->lengthOf() / output->sizeAt(-1);
    const LongType hiddenDim = output->sizeAt(-1);

    NDArray::prepareSpecialUse({output}, {quantized, scales});

    auto stream = context->getCudaStream();
    auto dtype = output->dataType();
    int threads = ptqChooseThreads(hiddenDim);

    if (dtype == DataType::FLOAT32) {
        perTokenDequantFp8Kernel<float><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const int8_t*>(quantized->specialBuffer()),
            reinterpret_cast<const float*>(scales->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            numTokens, hiddenDim);
    } else if (dtype == DataType::HALF) {
        perTokenDequantFp8Kernel<float16><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const int8_t*>(quantized->specialBuffer()),
            reinterpret_cast<const float*>(scales->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            numTokens, hiddenDim);
    } else {
        THROW_EXCEPTION("perTokenDequantFp8: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("perTokenDequantFp8 failed");
    NDArray::registerSpecialUse({output}, {quantized, scales});
}

//////////////////////////////////////////////////////////////////////////////
// Public: perTokenGroupQuant
//////////////////////////////////////////////////////////////////////////////
void perTokenGroupQuant(LaunchContext* context,
                        NDArray* input,
                        NDArray* quantized,
                        NDArray* scales,
                        int groupSize,
                        bool useFp8) {
    const LongType numTokens = input->lengthOf() / input->sizeAt(-1);
    const LongType hiddenDim = input->sizeAt(-1);

    NDArray::prepareSpecialUse({quantized, scales}, {input});

    auto stream = context->getCudaStream();
    auto dtype = input->dataType();

    const int numGroups = (hiddenDim + groupSize - 1) / groupSize;
    int threads = min(numGroups, 256);
    if (threads < PTQ_WARP_SIZE) threads = PTQ_WARP_SIZE;

    float quantMax = useFp8 ? FP8_E4M3_MAX : 127.0f;

    if (dtype == DataType::FLOAT32) {
        perTokenGroupQuantKernel<float><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const float*>(input->specialBuffer()),
            reinterpret_cast<int8_t*>(quantized->specialBuffer()),
            reinterpret_cast<float*>(scales->specialBuffer()),
            numTokens, hiddenDim, groupSize, quantMax);
    } else if (dtype == DataType::HALF) {
        perTokenGroupQuantKernel<float16><<<numTokens, threads, 0, *stream>>>(
            reinterpret_cast<const float16*>(input->specialBuffer()),
            reinterpret_cast<int8_t*>(quantized->specialBuffer()),
            reinterpret_cast<float*>(scales->specialBuffer()),
            numTokens, hiddenDim, groupSize, quantMax);
    } else {
        THROW_EXCEPTION("perTokenGroupQuant: unsupported data type");
    }

    DebugHelper::checkGlobalErrorCode("perTokenGroupQuant failed");
    NDArray::registerSpecialUse({quantized, scales}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
