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
// Standalone CUDA dequantization kernels for GGML quantization formats.
// No external dependencies — this is our own baseline implementation.
//

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <types/float16.h>
#include <types/bfloat16.h>

#if NOT_EXCLUDED(OP_ggml_dequantize)

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// Device FP16 conversion
//////////////////////////////////////////////////////////////////////////
SD_DEVICE SD_INLINE float devFp16ToFloat(uint16_t h) {
    __half hVal;
    memcpy(&hVal, &h, sizeof(uint16_t));
    return __half2float(hVal);
}

//////////////////////////////////////////////////////////////////////////
// K-quant helper: get_scale_min_k4 (device)
//////////////////////////////////////////////////////////////////////////
SD_DEVICE SD_INLINE void devGetScaleMinK4(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = ((q[j + 4] & 0xF)) | (((q[j - 4] >> 6) & 3) << 4);
        m = ((q[j + 4] >> 4) & 0xF) | (((q[j] >> 6) & 3) << 4);
    }
}

//////////////////////////////////////////////////////////////////////////
// Q4_0 CUDA kernel: one thread per block of 32 elements
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void dequantize_q4_0_kernel(const uint8_t* __restrict__ data, float* __restrict__ output,
                                        LongType numElements) {
    constexpr int BLOCK_SIZE = 18;
    constexpr int QK = 32;
    LongType blockIdx_g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    LongType numBlocks = (numElements + QK - 1) / QK;
    if (blockIdx_g >= numBlocks) return;

    const uint8_t* block = data + blockIdx_g * BLOCK_SIZE;
    uint16_t dRaw;
    memcpy(&dRaw, block, 2);
    float d = devFp16ToFloat(dRaw);
    const uint8_t* qs = block + 2;
    LongType outBase = blockIdx_g * QK;

    for (int j = 0; j < QK / 2; j++) {
        int v0 = (qs[j] & 0x0F) - 8;
        int v1 = ((qs[j] >> 4) & 0x0F) - 8;
        LongType idx0 = outBase + j * 2;
        LongType idx1 = idx0 + 1;
        if (idx0 < numElements) output[idx0] = d * v0;
        if (idx1 < numElements) output[idx1] = d * v1;
    }
}

//////////////////////////////////////////////////////////////////////////
// Q4_K CUDA kernel: one thread per super-block of 256 elements
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void dequantize_q4_K_kernel(const uint8_t* __restrict__ data, float* __restrict__ output,
                                        LongType numElements) {
    constexpr int BLOCK_SIZE = 144;
    constexpr int QK = 256;
    LongType blockIdx_g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    LongType numBlocks = (numElements + QK - 1) / QK;
    if (blockIdx_g >= numBlocks) return;

    const uint8_t* block = data + blockIdx_g * BLOCK_SIZE;
    uint16_t dRaw, dminRaw;
    memcpy(&dRaw, block, 2);
    memcpy(&dminRaw, block + 2, 2);
    float d = devFp16ToFloat(dRaw);
    float dmin = devFp16ToFloat(dminRaw);
    const uint8_t* scaleBytes = block + 4;
    const uint8_t* qs = block + 16;
    LongType outBase = blockIdx_g * QK;
    LongType outIdx = outBase;

    int is = 0;
    int qIdx = 0;
    for (int j = 0; j < QK; j += 64) {
        int sc1, m1, sc2, m2;
        devGetScaleMinK4(is, scaleBytes, sc1, m1);
        float d1 = d * sc1;
        float m1f = dmin * m1;
        devGetScaleMinK4(is + 1, scaleBytes, sc2, m2);
        float d2 = d * sc2;
        float m2f = dmin * m2;

        for (int l = 0; l < 32; l++) {
            int val = qs[qIdx + l] & 0x0F;
            if (outIdx < numElements) output[outIdx] = d1 * val - m1f;
            outIdx++;
        }
        for (int l = 0; l < 32; l++) {
            int val = (qs[qIdx + l] >> 4) & 0x0F;
            if (outIdx < numElements) output[outIdx] = d2 * val - m2f;
            outIdx++;
        }
        qIdx += 32;
        is += 2;
    }
}

//////////////////////////////////////////////////////////////////////////
// Q5_K CUDA kernel
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void dequantize_q5_K_kernel(const uint8_t* __restrict__ data, float* __restrict__ output,
                                        LongType numElements) {
    constexpr int BLOCK_SIZE = 176;
    constexpr int QK = 256;
    LongType blockIdx_g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    LongType numBlocks = (numElements + QK - 1) / QK;
    if (blockIdx_g >= numBlocks) return;

    const uint8_t* block = data + blockIdx_g * BLOCK_SIZE;
    uint16_t dRaw, dminRaw;
    memcpy(&dRaw, block, 2);
    memcpy(&dminRaw, block + 2, 2);
    float d = devFp16ToFloat(dRaw);
    float dmin = devFp16ToFloat(dminRaw);
    const uint8_t* scaleBytes = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* qs = block + 48;
    LongType outIdx = blockIdx_g * QK;

    int is = 0;
    int qlOff = 0;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK; j += 64) {
        int sc1, m1, sc2, m2;
        devGetScaleMinK4(is, scaleBytes, sc1, m1);
        float d1 = d * sc1;
        float m1f = dmin * m1;
        devGetScaleMinK4(is + 1, scaleBytes, sc2, m2);
        float d2 = d * sc2;
        float m2f = dmin * m2;

        for (int l = 0; l < 32; l++) {
            int lowVal = qs[qlOff + l] & 0x0F;
            int highBit = (qh[l] & u1) ? 16 : 0;
            if (outIdx < numElements) output[outIdx] = d1 * (lowVal + highBit) - m1f;
            outIdx++;
        }
        for (int l = 0; l < 32; l++) {
            int highNibble = (qs[qlOff + l] >> 4) & 0x0F;
            int highBit = (qh[l] & u2) ? 16 : 0;
            if (outIdx < numElements) output[outIdx] = d2 * (highNibble + highBit) - m2f;
            outIdx++;
        }
        qlOff += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

//////////////////////////////////////////////////////////////////////////
// Q6_K CUDA kernel
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void dequantize_q6_K_kernel(const uint8_t* __restrict__ data, float* __restrict__ output,
                                        LongType numElements) {
    constexpr int BLOCK_SIZE = 210;
    constexpr int QK = 256;
    LongType blockIdx_g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    LongType numBlocks = (numElements + QK - 1) / QK;
    if (blockIdx_g >= numBlocks) return;

    const uint8_t* block = data + blockIdx_g * BLOCK_SIZE;
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    uint16_t dRaw;
    memcpy(&dRaw, block + 208, 2);
    float d = devFp16ToFloat(dRaw);
    LongType outBase = blockIdx_g * QK;

    int qlOff = 0;
    int qhOff = 0;
    int scOff = 0;

    for (int n = 0; n < QK; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;

            int q1 = ((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
            int q3 = (((ql[qlOff + l] >> 4) & 0xF) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
            int q4 = (((ql[qlOff + l + 32] >> 4) & 0xF) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

            LongType idx = outBase + n + l;
            if (idx < numElements) output[idx] = d * scales[scOff + is] * q1;
            idx = outBase + n + l + 32;
            if (idx < numElements) output[idx] = d * scales[scOff + is + 2] * q2;
            idx = outBase + n + l + 64;
            if (idx < numElements) output[idx] = d * scales[scOff + is + 4] * q3;
            idx = outBase + n + l + 96;
            if (idx < numElements) output[idx] = d * scales[scOff + is + 6] * q4;
        }
        qlOff += 64;
        qhOff += 32;
        scOff += 8;
    }
}

//////////////////////////////////////////////////////////////////////////
// Q8_0 CUDA kernel
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void dequantize_q8_0_kernel(const uint8_t* __restrict__ data, float* __restrict__ output,
                                        LongType numElements) {
    constexpr int BLOCK_SIZE = 34;
    constexpr int QK = 32;
    LongType blockIdx_g = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    LongType numBlocks = (numElements + QK - 1) / QK;
    if (blockIdx_g >= numBlocks) return;

    const uint8_t* block = data + blockIdx_g * BLOCK_SIZE;
    uint16_t dRaw;
    memcpy(&dRaw, block, 2);
    float d = devFp16ToFloat(dRaw);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    LongType outBase = blockIdx_g * QK;

    for (int j = 0; j < QK; j++) {
        LongType idx = outBase + j;
        if (idx < numElements) output[idx] = d * qs[j];
    }
}

//////////////////////////////////////////////////////////////////////////
// F32 -> F16 conversion kernel
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void convertF32ToF16Kernel(const float* __restrict__ input, half* __restrict__ output, LongType n) {
    LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

//////////////////////////////////////////////////////////////////////////
// F32 -> BF16 conversion kernel
//////////////////////////////////////////////////////////////////////////
SD_KERNEL void convertF32ToBF16Kernel(const float* __restrict__ input, __nv_bfloat16* __restrict__ output, LongType n) {
    LongType idx = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

//////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////
void ggmlDequantize(
    LaunchContext* context,
    NDArray* input,
    NDArray* output,
    int quantType) {

    NDArray::prepareSpecialUse({output}, {input});

    const auto* rawBytes = reinterpret_cast<const uint8_t*>(input->specialBuffer());
    LongType numElements = output->lengthOf();
    auto outputDtype = output->dataType();
    auto stream = context->getCudaStream();

    // For non-F32 output we dequantize to a temp F32 buffer on device, then convert
    float* f32Buf = nullptr;
    bool needsConvert = (outputDtype != DataType::FLOAT32);

    if (needsConvert) {
        int deviceId = sd::AffinityManager::currentDeviceId();
        f32Buf = reinterpret_cast<float*>(
            memory::CudaMemoryPool::getInstance().allocate(numElements * sizeof(float), deviceId, *stream));
    } else {
        f32Buf = reinterpret_cast<float*>(output->specialBuffer());
    }

    // Dispatch dequantization kernel
    // Block sizes: Q4_0/Q8_0 = 32 elements, K-quants = 256 elements
    int qBlockSize;
    switch (quantType) {
        case GGML_QUANT_Q4_0: case GGML_QUANT_Q4_1: case GGML_QUANT_Q5_0:
        case GGML_QUANT_Q5_1: case GGML_QUANT_Q8_0: case GGML_QUANT_Q8_1:
            qBlockSize = 32; break;
        default:
            qBlockSize = 256; break;
    }

    LongType numQBlocks = (numElements + qBlockSize - 1) / qBlockSize;
    int threadsPerBlock = 256;
    int gridSize = (numQBlocks + threadsPerBlock - 1) / threadsPerBlock;

    switch (quantType) {
        case GGML_QUANT_Q4_0:
            dequantize_q4_0_kernel<<<gridSize, threadsPerBlock, 0, *stream>>>(rawBytes, f32Buf, numElements);
            break;
        case GGML_QUANT_Q4_K:
            dequantize_q4_K_kernel<<<gridSize, threadsPerBlock, 0, *stream>>>(rawBytes, f32Buf, numElements);
            break;
        case GGML_QUANT_Q5_K:
            dequantize_q5_K_kernel<<<gridSize, threadsPerBlock, 0, *stream>>>(rawBytes, f32Buf, numElements);
            break;
        case GGML_QUANT_Q6_K:
            dequantize_q6_K_kernel<<<gridSize, threadsPerBlock, 0, *stream>>>(rawBytes, f32Buf, numElements);
            break;
        case GGML_QUANT_Q8_0:
            dequantize_q8_0_kernel<<<gridSize, threadsPerBlock, 0, *stream>>>(rawBytes, f32Buf, numElements);
            break;
        default:
            // For types without dedicated CUDA kernels, fall back to CPU-side dequant
            // Copy raw bytes to host, dequantize on CPU, copy result back
            {
                std::vector<uint8_t> hostRaw(input->lengthOf());
                cudaMemcpyAsync(hostRaw.data(), rawBytes, input->lengthOf(), cudaMemcpyDeviceToHost, *stream);
                cudaStreamSynchronize(*stream);

                std::vector<float> hostOut(numElements);
                // Use CPU dequantization for unsupported GPU types
                // Import the CPU dispatch function signature
                switch (quantType) {
                    case GGML_QUANT_Q4_1: {
                        // Q4_1: 20 bytes per 32 elements
                        LongType nBlocks = (numElements + 31) / 32;
                        LongType outIdx = 0;
                        for (LongType bl = 0; bl < nBlocks; bl++) {
                            const uint8_t* blk = hostRaw.data() + bl * 20;
                            uint16_t dR, mR;
                            memcpy(&dR, blk, 2); memcpy(&mR, blk + 2, 2);
                            // Simple FP16 decode via bit manipulation
                            union { uint32_t u; float f; } du, mu;
                            du.u = ((uint32_t)(dR & 0x8000) << 16) | ((uint32_t)((dR & 0x7C00) + 0x1C000) << 13) | ((uint32_t)(dR & 0x03FF) << 13);
                            mu.u = ((uint32_t)(mR & 0x8000) << 16) | ((uint32_t)((mR & 0x7C00) + 0x1C000) << 13) | ((uint32_t)(mR & 0x03FF) << 13);
                            float d = du.f, m = mu.f;
                            const uint8_t* qs = blk + 4;
                            for (int j = 0; j < 16 && outIdx < numElements; j++) {
                                hostOut[outIdx++] = d * (qs[j] & 0x0F) + m;
                                if (outIdx < numElements) hostOut[outIdx++] = d * ((qs[j] >> 4) & 0x0F) + m;
                            }
                        }
                        break;
                    }
                    default:
                        // Zero-fill for truly unsupported types
                        memset(hostOut.data(), 0, numElements * sizeof(float));
                        break;
                }

                cudaMemcpyAsync(f32Buf, hostOut.data(), numElements * sizeof(float), cudaMemcpyHostToDevice, *stream);
                cudaStreamSynchronize(*stream);
            }
            break;
    }

    DebugHelper::checkGlobalErrorCode("ggml_dequantize kernel failed");

    // Convert F32 to target type if needed
    if (needsConvert) {
        int convThreads = 256;
        int convGrid = (numElements + convThreads - 1) / convThreads;

        if (outputDtype == DataType::HALF) {
            convertF32ToF16Kernel<<<convGrid, convThreads, 0, *stream>>>(
                f32Buf, reinterpret_cast<half*>(output->specialBuffer()), numElements);
        } else if (outputDtype == DataType::BFLOAT16) {
            convertF32ToBF16Kernel<<<convGrid, convThreads, 0, *stream>>>(
                f32Buf, reinterpret_cast<__nv_bfloat16*>(output->specialBuffer()), numElements);
        }

        DebugHelper::checkGlobalErrorCode("ggml_dequantize type conversion failed");
        int deviceId = sd::AffinityManager::currentDeviceId();
        memory::CudaMemoryPool::getInstance().free(f32Buf, deviceId, *stream);
    }

    NDArray::registerSpecialUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_dequantize)
