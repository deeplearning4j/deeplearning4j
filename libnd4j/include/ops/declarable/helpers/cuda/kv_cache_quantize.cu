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
// KV Cache Quantization/Dequantization CUDA implementation
// Per-row absmax symmetric quantization with shared-memory reduction.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <types/float16.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>

namespace sd {
namespace ops {
namespace helpers {

constexpr int KVQ_WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level max reduction
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE SD_INLINE float kvqWarpReduceMax(float val) {
    for (int offset = KVQ_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level max reduction using shared memory
//////////////////////////////////////////////////////////////////////////////
SD_DEVICE float kvqBlockReduceMax(float val, float* sharedMem) {
    const int lane = threadIdx.x % KVQ_WARP_SIZE;
    const int wid = threadIdx.x / KVQ_WARP_SIZE;
    const int numWarps = (blockDim.x + KVQ_WARP_SIZE - 1) / KVQ_WARP_SIZE;

    val = kvqWarpReduceMax(val);

    if (lane == 0) {
        sharedMem[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = kvqWarpReduceMax(val);
    }

    return val;
}

//////////////////////////////////////////////////////////////////////////////
// INT8 Quantize Kernel: one block per row
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheQuantizeInt8Kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numRows,
    const LongType rowLen) {

    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    extern __shared__ char sharedMemRaw[];
    float* sdata = reinterpret_cast<float*>(sharedMemRaw);

    const T* inputRow = input + row * rowLen;
    int8_t* quantRow = quantized + row * rowLen;

    // Pass 1: find absmax
    float threadMax = 0.0f;
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
        float val = fabsf(static_cast<float>(inputRow[i]));
        threadMax = fmaxf(threadMax, val);
    }

    float rowMax = kvqBlockReduceMax(threadMax, sdata);

    __shared__ float invScale;
    __shared__ float rowScale;

    if (threadIdx.x == 0) {
        rowScale = rowMax / 127.0f;
        if (rowScale == 0.0f) rowScale = 1.0f;
        scales[row] = rowScale;
        invScale = 1.0f / rowScale;
    }
    __syncthreads();

    // Pass 2: quantize
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
        float val = static_cast<float>(inputRow[i]) * invScale;
        val = fmaxf(-127.0f, fminf(127.0f, val));
        quantRow[i] = static_cast<int8_t>(rintf(val));
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT8 Dequantize Kernel: one block per row
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheDequantizeInt8Kernel(
    const int8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    T* __restrict__ output,
    const LongType numRows,
    const LongType rowLen) {

    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    const int8_t* quantRow = quantized + row * rowLen;
    T* outputRow = output + row * rowLen;
    float scale = scales[row];

    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
        outputRow[i] = static_cast<T>(static_cast<float>(quantRow[i]) * scale);
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT4 Quantize Kernel: one block per row, pack 2 values per byte
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheQuantizeInt4Kernel(
    const T* __restrict__ input,
    uint8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numRows,
    const LongType rowLen) {

    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    extern __shared__ char sharedMemRaw[];
    float* sdata = reinterpret_cast<float*>(sharedMemRaw);

    const T* inputRow = input + row * rowLen;
    const LongType packedRowLen = (rowLen + 1) / 2;
    uint8_t* quantRow = quantized + row * packedRowLen;

    // Pass 1: find absmax
    float threadMax = 0.0f;
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
        float val = fabsf(static_cast<float>(inputRow[i]));
        threadMax = fmaxf(threadMax, val);
    }

    float rowMax = kvqBlockReduceMax(threadMax, sdata);

    __shared__ float invScale;
    __shared__ float rowScale;

    if (threadIdx.x == 0) {
        rowScale = rowMax / 7.0f;
        if (rowScale == 0.0f) rowScale = 1.0f;
        scales[row] = rowScale;
        invScale = 1.0f / rowScale;
    }
    __syncthreads();

    // Pass 2: quantize and pack (each thread handles a pair)
    for (LongType i = threadIdx.x * 2; i < rowLen; i += blockDim.x * 2) {
        float val0 = static_cast<float>(inputRow[i]) * invScale;
        val0 = fmaxf(-7.0f, fminf(7.0f, val0));
        int8_t q0 = static_cast<int8_t>(rintf(val0));

        int8_t q1 = 0;
        if (i + 1 < rowLen) {
            float val1 = static_cast<float>(inputRow[i + 1]) * invScale;
            val1 = fmaxf(-7.0f, fminf(7.0f, val1));
            q1 = static_cast<int8_t>(rintf(val1));
        }

        quantRow[i / 2] = static_cast<uint8_t>((q0 + 8) & 0x0F) | static_cast<uint8_t>(((q1 + 8) & 0x0F) << 4);
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT4 Dequantize Kernel
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheDequantizeInt4Kernel(
    const uint8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    T* __restrict__ output,
    const LongType numRows,
    const LongType rowLen) {

    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    const LongType packedRowLen = (rowLen + 1) / 2;
    const uint8_t* quantRow = quantized + row * packedRowLen;
    T* outputRow = output + row * rowLen;
    float scale = scales[row];

    for (LongType i = threadIdx.x * 2; i < rowLen; i += blockDim.x * 2) {
        uint8_t packed = quantRow[i / 2];
        int8_t q0 = static_cast<int8_t>((packed & 0x0F)) - 8;
        int8_t q1 = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;

        outputRow[i] = static_cast<T>(static_cast<float>(q0) * scale);
        if (i + 1 < rowLen) {
            outputRow[i + 1] = static_cast<T>(static_cast<float>(q1) * scale);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launcher helpers
//////////////////////////////////////////////////////////////////////////////
static int chooseThreads(LongType rowLen) {
    int threads = 256;
    if (rowLen > 256) threads = 512;
    if (rowLen > 512) threads = 1024;
    if (rowLen < threads) {
        threads = ((rowLen + KVQ_WARP_SIZE - 1) / KVQ_WARP_SIZE) * KVQ_WARP_SIZE;
        if (threads < KVQ_WARP_SIZE) threads = KVQ_WARP_SIZE;
    }
    return threads;
}

static size_t sharedMemSize(int threads) {
    int numWarps = (threads + KVQ_WARP_SIZE - 1) / KVQ_WARP_SIZE;
    return numWarps * sizeof(float);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface: kvCacheQuantize
//////////////////////////////////////////////////////////////////////////////
void kvCacheQuantize(NDArray* input, NDArray* quantized, NDArray* scales,
                     int quantFormat, LaunchContext* context) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    NDArray::prepareSpecialUse({quantized, scales}, {input});

    auto stream = context->getCudaStream();
    auto dtype = input->dataType();
    auto format = static_cast<KVQuantFormat>(quantFormat);

    int threads = chooseThreads(rowLen);
    size_t smem = sharedMemSize(threads);
    int blocks = static_cast<int>(numRows);

    // Scales are always float
    float* scalesBuf = reinterpret_cast<float*>(scales->specialBuffer());

    if (format == KVQuantFormat::INT4) {
        if (dtype == DataType::FLOAT32) {
            kvCacheQuantizeInt4Kernel<float><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const float*>(input->specialBuffer()),
                reinterpret_cast<uint8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else if (dtype == DataType::DOUBLE) {
            kvCacheQuantizeInt4Kernel<double><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const double*>(input->specialBuffer()),
                reinterpret_cast<uint8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else if (dtype == DataType::HALF) {
            kvCacheQuantizeInt4Kernel<float16><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const float16*>(input->specialBuffer()),
                reinterpret_cast<uint8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else {
            THROW_EXCEPTION("kvCacheQuantize: unsupported data type");
        }
    } else {
        // INT8, FP8_E4M3, FP8_E5M2 all use INT8 kernel on GPU
        if (dtype == DataType::FLOAT32) {
            kvCacheQuantizeInt8Kernel<float><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const float*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else if (dtype == DataType::DOUBLE) {
            kvCacheQuantizeInt8Kernel<double><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const double*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else if (dtype == DataType::HALF) {
            kvCacheQuantizeInt8Kernel<float16><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const float16*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen);
        } else {
            THROW_EXCEPTION("kvCacheQuantize: unsupported data type");
        }
    }

    DebugHelper::checkGlobalErrorCode("kvCacheQuantize failed");
    NDArray::registerSpecialUse({quantized, scales}, {input});
}

//////////////////////////////////////////////////////////////////////////////
// Public interface: kvCacheDequantize
//////////////////////////////////////////////////////////////////////////////
void kvCacheDequantize(NDArray* quantized, NDArray* scales, NDArray* output,
                       int quantFormat, LaunchContext* context) {
    const LongType numRows = output->lengthOf() / output->sizeAt(-1);
    const LongType rowLen = output->sizeAt(-1);

    NDArray::prepareSpecialUse({output}, {quantized, scales});

    auto stream = context->getCudaStream();
    auto dtype = output->dataType();
    auto format = static_cast<KVQuantFormat>(quantFormat);

    int threads = chooseThreads(rowLen);
    int blocks = static_cast<int>(numRows);

    const float* scalesBuf = reinterpret_cast<const float*>(scales->specialBuffer());

    if (format == KVQuantFormat::INT4) {
        if (dtype == DataType::FLOAT32) {
            kvCacheDequantizeInt4Kernel<float><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const uint8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<float*>(output->specialBuffer()),
                numRows, rowLen);
        } else if (dtype == DataType::DOUBLE) {
            kvCacheDequantizeInt4Kernel<double><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const uint8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<double*>(output->specialBuffer()),
                numRows, rowLen);
        } else if (dtype == DataType::HALF) {
            kvCacheDequantizeInt4Kernel<float16><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const uint8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<float16*>(output->specialBuffer()),
                numRows, rowLen);
        } else {
            THROW_EXCEPTION("kvCacheDequantize: unsupported data type");
        }
    } else {
        // INT8, FP8_E4M3, FP8_E5M2
        if (dtype == DataType::FLOAT32) {
            kvCacheDequantizeInt8Kernel<float><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const int8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<float*>(output->specialBuffer()),
                numRows, rowLen);
        } else if (dtype == DataType::DOUBLE) {
            kvCacheDequantizeInt8Kernel<double><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const int8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<double*>(output->specialBuffer()),
                numRows, rowLen);
        } else if (dtype == DataType::HALF) {
            kvCacheDequantizeInt8Kernel<float16><<<blocks, threads, 0, *stream>>>(
                reinterpret_cast<const int8_t*>(quantized->specialBuffer()),
                scalesBuf, reinterpret_cast<float16*>(output->specialBuffer()),
                numRows, rowLen);
        } else {
            THROW_EXCEPTION("kvCacheDequantize: unsupported data type");
        }
    }

    DebugHelper::checkGlobalErrorCode("kvCacheDequantize failed");
    NDArray::registerSpecialUse({output}, {quantized, scales});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
