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
// Weight dequantization CUDA kernels for AWQ, GPTQ, and Marlin formats.
// Enables INT4/INT8 weight-only quantized model inference.
//

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <types/float16.h>
#include <ops/declarable/helpers/weight_dequant.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// AWQ INT4 group dequantization kernel
// packed_weights stores two INT4 values per byte (low nibble, high nibble).
// Each group of groupSize elements shares one scale+zero pair.
// output = (int4_val - zero) * scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void awqDequantizeKernel(
    const uint8_t* __restrict__ packedWeights,  // [outF, inF/2]
    const T* __restrict__ scales,               // [outF, numGroups]
    const T* __restrict__ zeros,                // [outF, numGroups]
    T* __restrict__ output,                     // [outF, inF]
    const LongType outFeatures,
    const LongType inFeatures,
    const int groupSize) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType totalElements = outFeatures * inFeatures;
    if (idx >= totalElements) return;

    const LongType row = idx / inFeatures;
    const LongType col = idx % inFeatures;
    const LongType numGroups = (inFeatures + groupSize - 1) / groupSize;
    const LongType group = col / groupSize;

    // Read packed byte
    const LongType packedIdx = row * (inFeatures / 2) + col / 2;
    uint8_t packed = packedWeights[packedIdx];

    // Extract INT4 value (0-15, unsigned)
    int intVal;
    if (col % 2 == 0) {
        intVal = packed & 0x0F;
    } else {
        intVal = (packed >> 4) & 0x0F;
    }

    // Dequantize: (int4 - zero) * scale
    float scale = static_cast<float>(scales[row * numGroups + group]);
    float zero = static_cast<float>(zeros[row * numGroups + group]);
    float dequant = (static_cast<float>(intVal) - zero) * scale;

    output[idx] = static_cast<T>(dequant);
}

//////////////////////////////////////////////////////////////////////////////
// GPTQ INT4 dequantization kernel with optional permutation (g_idx)
// packed_weights: [outF, inF/2] packed INT4 pairs
// scales: [numGroups, outF]
// zeros: [numGroups, outF/8] packed zero-points (8 INT4 zeros per uint32)
// g_idx: [inF] maps input column to quantization group
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void gptqDequantizeKernel(
    const uint8_t* __restrict__ packedWeights,  // [outF, inF/2]
    const T* __restrict__ scales,               // [numGroups, outF]
    const uint32_t* __restrict__ zeros,         // [numGroups, outF/8] packed
    const int32_t* __restrict__ gIdx,           // [inF] or nullptr
    T* __restrict__ output,                     // [outF, inF]
    const LongType outFeatures,
    const LongType inFeatures,
    const int groupSize,
    const int bits) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType totalElements = outFeatures * inFeatures;
    if (idx >= totalElements) return;

    const LongType row = idx / inFeatures;
    const LongType col = idx % inFeatures;

    // Determine quantization group
    int group;
    if (gIdx != nullptr) {
        group = gIdx[col];
    } else {
        group = static_cast<int>(col / groupSize);
    }

    // Read packed byte and extract value
    const LongType packedIdx = row * (inFeatures / 2) + col / 2;
    uint8_t packed = packedWeights[packedIdx];

    int intVal;
    if (bits == 4) {
        if (col % 2 == 0) {
            intVal = packed & 0x0F;
        } else {
            intVal = (packed >> 4) & 0x0F;
        }
    } else {
        // INT8 mode
        intVal = static_cast<int>(reinterpret_cast<const int8_t*>(packedWeights)[row * inFeatures + col]);
    }

    // Read scale
    float scale = static_cast<float>(scales[group * outFeatures + row]);

    // Read zero-point from packed format
    float zero = 0.0f;
    if (zeros != nullptr && bits == 4) {
        LongType zeroWordIdx = group * (outFeatures / 8) + row / 8;
        int zeroShift = (static_cast<int>(row % 8)) * 4;
        uint32_t zeroWord = zeros[zeroWordIdx];
        zero = static_cast<float>((zeroWord >> zeroShift) & 0x0F);
    }

    float dequant = (static_cast<float>(intVal) - zero) * scale;
    output[idx] = static_cast<T>(dequant);
}

//////////////////////////////////////////////////////////////////////////////
// Marlin-style fused dequantize + GEMM kernel
// This is a simplified version that dequantizes then calls cuBLAS.
// A full Marlin kernel with fused dequant+TC math requires extensive
// assembly-level optimization; this provides the correct API and
// functional baseline.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void marlinDequantizeKernel(
    const uint8_t* __restrict__ marlinWeights,  // [K/16, N*16/8] packed
    const T* __restrict__ scales,               // [numGroups, N]
    T* __restrict__ dequantized,                // [K, N] output
    const LongType K,
    const LongType N,
    const int groupSize) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    const LongType totalElements = K * N;
    if (idx >= totalElements) return;

    const LongType row = idx / N;  // K dimension
    const LongType col = idx % N;  // N dimension

    // Marlin packing: weights are stored in tiles of 16x16
    // with INT4 values packed 8 per uint32.
    // Simplified: treat as standard row-major INT4 packing for the baseline.
    const LongType packedIdx = (row * N + col) / 2;
    uint8_t packed = marlinWeights[packedIdx];

    int intVal;
    if (idx % 2 == 0) {
        intVal = packed & 0x0F;
    } else {
        intVal = (packed >> 4) & 0x0F;
    }

    // Apply scale
    int group = (groupSize > 0) ? static_cast<int>(row / groupSize) : 0;
    LongType numGroups = (groupSize > 0) ? (K + groupSize - 1) / groupSize : 1;
    float scale = static_cast<float>(scales[group * N + col]);

    // INT4 unsigned → signed: subtract 8
    float dequant = (static_cast<float>(intVal) - 8.0f) * scale;
    dequantized[idx] = static_cast<T>(dequant);
}

//////////////////////////////////////////////////////////////////////////////
// Public: awqDequantize
//////////////////////////////////////////////////////////////////////////////
void awqDequantize(LaunchContext* context,
                    NDArray* packedWeights,
                    NDArray* scales,
                    NDArray* zeros,
                    NDArray* output,
                    int groupSize) {
    const LongType outFeatures = output->sizeAt(0);
    const LongType inFeatures = output->sizeAt(1);
    const LongType totalElements = outFeatures * inFeatures;

    NDArray::prepareSpecialUse({output}, {packedWeights, scales, zeros});

    auto stream = context->getCudaStream();
    auto dtype = output->dataType();

    int threads = 256;
    int blocks = static_cast<int>((totalElements + threads - 1) / threads);

    if (dtype == DataType::FLOAT32) {
        awqDequantizeKernel<float><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(packedWeights->specialBuffer()),
            reinterpret_cast<const float*>(scales->specialBuffer()),
            reinterpret_cast<const float*>(zeros->specialBuffer()),
            reinterpret_cast<float*>(output->specialBuffer()),
            outFeatures, inFeatures, groupSize);
    } else if (dtype == DataType::HALF) {
        awqDequantizeKernel<float16><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(packedWeights->specialBuffer()),
            reinterpret_cast<const float16*>(scales->specialBuffer()),
            reinterpret_cast<const float16*>(zeros->specialBuffer()),
            reinterpret_cast<float16*>(output->specialBuffer()),
            outFeatures, inFeatures, groupSize);
    } else {
        THROW_EXCEPTION("awqDequantize: unsupported output data type");
    }

    DebugHelper::checkGlobalErrorCode("awqDequantize failed");
    NDArray::registerSpecialUse({output}, {packedWeights, scales, zeros});
}

//////////////////////////////////////////////////////////////////////////////
// Public: gptqDequantize
//////////////////////////////////////////////////////////////////////////////
void gptqDequantize(LaunchContext* context,
                     NDArray* packedWeights,
                     NDArray* scales,
                     NDArray* zeros,
                     NDArray* gIdx,
                     NDArray* output,
                     int groupSize,
                     int bits) {
    const LongType outFeatures = output->sizeAt(0);
    const LongType inFeatures = output->sizeAt(1);
    const LongType totalElements = outFeatures * inFeatures;

    NDArray::prepareSpecialUse({output}, {packedWeights, scales});
    if (zeros != nullptr) NDArray::prepareSpecialUse({}, {zeros});
    if (gIdx != nullptr) NDArray::prepareSpecialUse({}, {gIdx});

    auto stream = context->getCudaStream();
    auto dtype = output->dataType();

    int threads = 256;
    int blocks = static_cast<int>((totalElements + threads - 1) / threads);

    const uint32_t* zerosPtr = (zeros != nullptr) ?
        reinterpret_cast<const uint32_t*>(zeros->specialBuffer()) : nullptr;
    const int32_t* gIdxPtr = (gIdx != nullptr) ?
        reinterpret_cast<const int32_t*>(gIdx->specialBuffer()) : nullptr;

    if (dtype == DataType::FLOAT32) {
        gptqDequantizeKernel<float><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(packedWeights->specialBuffer()),
            reinterpret_cast<const float*>(scales->specialBuffer()),
            zerosPtr, gIdxPtr,
            reinterpret_cast<float*>(output->specialBuffer()),
            outFeatures, inFeatures, groupSize, bits);
    } else if (dtype == DataType::HALF) {
        gptqDequantizeKernel<float16><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(packedWeights->specialBuffer()),
            reinterpret_cast<const float16*>(scales->specialBuffer()),
            zerosPtr, gIdxPtr,
            reinterpret_cast<float16*>(output->specialBuffer()),
            outFeatures, inFeatures, groupSize, bits);
    } else {
        THROW_EXCEPTION("gptqDequantize: unsupported output data type");
    }

    DebugHelper::checkGlobalErrorCode("gptqDequantize failed");
    NDArray::registerSpecialUse({output}, {packedWeights, scales});
}

//////////////////////////////////////////////////////////////////////////////
// Public: marlinGemm
// Functional baseline: dequantize + cuBLAS FP16 GEMM.
// A fused kernel with dequant-in-register + TC math would be the next step.
//////////////////////////////////////////////////////////////////////////////
void marlinGemm(LaunchContext* context,
                 NDArray* input,
                 NDArray* marlinWeights,
                 NDArray* scales,
                 NDArray* output,
                 NDArray* workspace,
                 int groupSize) {
    const LongType M = input->sizeAt(0);
    const LongType K = input->sizeAt(1);
    const LongType N = output->sizeAt(1);

    NDArray::prepareSpecialUse({output}, {input, marlinWeights, scales});

    auto stream = context->getCudaStream();
    auto dtype = input->dataType();

    // Step 1: Dequantize Marlin-packed weights to FP16/FP32
    std::vector<sd::LongType> dequantShape = {K, N};
    auto dequantized = NDArrayFactory::create_(input->ordering(), dequantShape,
                                                dtype, context);
    NDArray::prepareSpecialUse({dequantized}, {marlinWeights, scales});

    int threads = 256;
    LongType totalWeightElements = K * N;
    int blocks = static_cast<int>((totalWeightElements + threads - 1) / threads);

    if (dtype == DataType::FLOAT32) {
        marlinDequantizeKernel<float><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(marlinWeights->specialBuffer()),
            reinterpret_cast<const float*>(scales->specialBuffer()),
            reinterpret_cast<float*>(dequantized->specialBuffer()),
            K, N, groupSize);
    } else if (dtype == DataType::HALF) {
        marlinDequantizeKernel<float16><<<blocks, threads, 0, *stream>>>(
            reinterpret_cast<const uint8_t*>(marlinWeights->specialBuffer()),
            reinterpret_cast<const float16*>(scales->specialBuffer()),
            reinterpret_cast<float16*>(dequantized->specialBuffer()),
            K, N, groupSize);
    } else {
        delete dequantized;
        THROW_EXCEPTION("marlinGemm: unsupported data type (use FLOAT32 or HALF)");
    }

    NDArray::registerSpecialUse({dequantized}, {marlinWeights, scales});

    // Step 2: GEMM via cuBLAS: output = input * dequantized
    NDArray::prepareSpecialUse({output}, {input, dequantized});

    cublasHandle_t handle = *reinterpret_cast<cublasHandle_t*>(context->getCublasHandle());
    cublasSetStream(handle, *stream);

    if (dtype == DataType::HALF) {
        __half alpha = __float2half(1.0f);
        __half beta = __float2half(0.0f);
        cublasHgemm(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     reinterpret_cast<const __half*>(dequantized->specialBuffer()), K,
                     reinterpret_cast<const __half*>(input->specialBuffer()), K,
                     &beta,
                     reinterpret_cast<__half*>(output->specialBuffer()), N);
    } else {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     reinterpret_cast<const float*>(dequantized->specialBuffer()), K,
                     reinterpret_cast<const float*>(input->specialBuffer()), K,
                     &beta,
                     reinterpret_cast<float*>(output->specialBuffer()), N);
    }

    NDArray::registerSpecialUse({output}, {input, dequantized});

    delete dequantized;

    DebugHelper::checkGlobalErrorCode("marlinGemm failed");
    NDArray::registerSpecialUse({output}, {input, marlinWeights, scales});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
