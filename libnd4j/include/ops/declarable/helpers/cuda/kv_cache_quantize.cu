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
// ADR 0107 V2 ROW-INLINE: when `scales` is null, each output row is `outRowPitch` = rowLen+4
// int8 elements — rowLen quantized values followed by the row's float32 scale (little-endian,
// 4-aligned since rowLen % 4 == 0 for KV head dims). The scale lives INSIDE the logical tensor,
// so DSP ext-input staging/copies preserve it. When `scales` is non-null (legacy separate-scale
// mode), outRowPitch == rowLen and the scale is written to scales[row].
template <typename T>
SD_KERNEL void kvCacheQuantizeInt8Kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    const LongType numRows,
    const LongType rowLen,
    const LongType outRowPitch) {

    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    extern __shared__ char sharedMemRaw[];
    float* sdata = reinterpret_cast<float*>(sharedMemRaw);

    const T* inputRow = input + row * rowLen;
    int8_t* quantRow = quantized + row * outRowPitch;
    float* scaleDst = scales != nullptr
        ? &scales[row]
        : reinterpret_cast<float*>(quantRow + rowLen);

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
        *scaleDst = rowScale;
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

    // ADR 0107 V2 ROW-INLINE: when scales is null, `quantized` is a row-inline tensor whose last
    // dimension is rowLen+4 — each row holds rowLen int8 values followed by that row's float32
    // scale. The scale rides INSIDE the logical tensor so DSP staging/copies preserve it.
    const bool inlineScale = (scales == nullptr);
    if (inlineScale) {
        NDArray::prepareSpecialUse({quantized}, {input});
    } else {
        NDArray::prepareSpecialUse({quantized, scales}, {input});
    }

    auto stream = context->getCudaStream();
    auto dtype = input->dataType();
    auto format = static_cast<KVQuantFormat>(quantFormat);

    int threads = chooseThreads(rowLen);
    size_t smem = sharedMemSize(threads);
    int blocks = static_cast<int>(numRows);

    const LongType outRowPitch = inlineScale ? rowLen + 4 : rowLen;
    float* scalesBuf = inlineScale ? nullptr : reinterpret_cast<float*>(scales->specialBuffer());

    if (inlineScale && format == KVQuantFormat::INT4) {
        THROW_EXCEPTION("kvCacheQuantize: row-inline scale mode supports INT8 only, not INT4");
    }

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
                scalesBuf, numRows, rowLen, outRowPitch);
        } else if (dtype == DataType::DOUBLE) {
            kvCacheQuantizeInt8Kernel<double><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const double*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen, outRowPitch);
        } else if (dtype == DataType::HALF) {
            kvCacheQuantizeInt8Kernel<float16><<<blocks, threads, smem, *stream>>>(
                reinterpret_cast<const float16*>(input->specialBuffer()),
                reinterpret_cast<int8_t*>(quantized->specialBuffer()),
                scalesBuf, numRows, rowLen, outRowPitch);
        } else {
            THROW_EXCEPTION("kvCacheQuantize: unsupported data type");
        }
    }

    DebugHelper::checkGlobalErrorCode("kvCacheQuantize failed");
    if (inlineScale) {
        NDArray::registerSpecialUse({quantized}, {input});
    } else {
        NDArray::registerSpecialUse({quantized, scales}, {input});
    }
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


//////////////////////////////////////////////////////////////////////////////
// V2: kvInPlaceWriteQuantisedBSHDKernel
//
// One block per (batch, kvHead) pair.  Within each block:
//   1. Per-warp absmax reduction over headDim elements (pass 1).
//   2. Write invScale to shared memory (broadcast).
//   3. INT8 scatter write to quantCache[b, cachePos, kvHead, :] (pass 2).
//   4. Write float scale to scaleCache[b, cachePos, kvHead].
//
// CUDA-graph safe:
//   - cachePos is read from a device-side LongType pointer at kernel runtime.
//   - No cudaMalloc / dynamic allocation.
//   - All output addresses are fixed pre-allocated buffers.
//   - Per-row warp reduction with NO cross-row atomics.
//////////////////////////////////////////////////////////////////////////////
// ADR 0107 V2 ROW-INLINE: when `scaleCache` is null the cache is a row-inline tensor
// [batch, maxKvLen, kvHeads, rowPitch] with rowPitch = headDim+4 — each row holds headDim int8
// values followed by that row's float32 scale (written at dstRow+headDim). When `scaleCache` is
// non-null (legacy separate-scale mode), rowPitch == headDim and the scale goes to
// scaleCache[batch, cachePos, kvHead].
SD_KERNEL void kvInPlaceWriteQuantisedBSHDKernel(
    const float* __restrict__ newKv,      // [batch, 1, kvHeads, headDim]
    int8_t*      __restrict__ quantCache, // [batch, maxKvLen, kvHeads, rowPitch]
    float*       __restrict__ scaleCache, // [batch, maxKvLen, kvHeads] or null (row-inline)
    const LongType* __restrict__ cachePosPtr,
    const LongType batch,
    const LongType kvHeads,
    const LongType headDim,
    const LongType maxKvLen,
    const LongType rowPitch) {

    const LongType cachePos = *cachePosPtr;
    if (cachePos < 0 || cachePos >= maxKvLen) return;

    // Grid: (blockIdx.x = kvHead, blockIdx.y = batch)
    const LongType kvHead   = blockIdx.x;
    const LongType batchIdx = blockIdx.y;
    if (batchIdx >= batch || kvHead >= kvHeads) return;

    // Source row: newKv[batchIdx, 0, kvHead, :]
    // Layout: [batch, 1, kvHeads, headDim] — contiguous
    const float* srcRow = newKv + batchIdx * (kvHeads * headDim) + kvHead * headDim;

    // Destination row: quantCache[batchIdx, cachePos, kvHead, :] (rowPitch-elements per row)
    const LongType qcRowOffset = batchIdx * (maxKvLen * kvHeads * rowPitch)
                                + cachePos * (kvHeads * rowPitch)
                                + kvHead * rowPitch;
    int8_t* dstRow = quantCache + qcRowOffset;

    // Scale destination: separate cache [batch, cachePos, kvHead], or inline at row end.
    float* scaleDst = scaleCache != nullptr
        ? scaleCache + (batchIdx * (maxKvLen * kvHeads) + cachePos * kvHeads + kvHead)
        : reinterpret_cast<float*>(dstRow + headDim);

    // Shared memory for block-level max reduction
    extern __shared__ char sharedMemQ[];
    float* sdata = reinterpret_cast<float*>(sharedMemQ);

    // Pass 1: find absmax over headDim
    float threadMax = 0.0f;
    for (LongType i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = srcRow[i];
        float av = fabsf(v);
        if (av > threadMax) threadMax = av;
    }
    float rowMax = kvqBlockReduceMax(threadMax, sdata);

    __shared__ float invScale;
    __shared__ float rowScale;
    if (threadIdx.x == 0) {
        rowScale = rowMax / 127.0f;
        if (rowScale == 0.0f) rowScale = 1.0f;
        *scaleDst = rowScale;
        invScale = 1.0f / rowScale;
    }
    __syncthreads();

    // Pass 2: quantize + scatter
    for (LongType i = threadIdx.x; i < headDim; i += blockDim.x) {
        float v = srcRow[i] * invScale;
        v = fmaxf(-127.0f, fminf(127.0f, v));
        dstRow[i] = static_cast<int8_t>(rintf(v));
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public: kvInPlaceWriteQuantisedBSHD (CUDA)
//////////////////////////////////////////////////////////////////////////////
void kvInPlaceWriteQuantisedBSHD(
    NDArray* quantCache,
    NDArray* scaleCache,
    NDArray* newKv,
    const void* cachePosPtr,
    LaunchContext* context) {

    const LongType batch    = newKv->sizeAt(0);
    const LongType kvHeads  = newKv->sizeAt(2);
    const LongType headDim  = newKv->sizeAt(3);
    const LongType maxKvLen = quantCache->sizeAt(1);

    // ADR 0107 V2 ROW-INLINE: when scaleCache is null the cache is [batch,maxKvLen,kvHeads,headDim+4]
    // — the per-row float32 scale lives at each row's tail (INSIDE the logical tensor). Legacy
    // separate-scale mode keeps the cache at [.,.,.,headDim] with scaleCache [batch,maxKvLen,kvHeads].
    const bool inlineScale = (scaleCache == nullptr);
    const LongType rowPitch = quantCache->sizeAt(3);
    if (inlineScale) {
        if (rowPitch != headDim + 4) {
            THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: row-inline cache last dim must equal headDim+4");
        }
        NDArray::prepareSpecialUse({quantCache}, {newKv});
    } else {
        if (rowPitch != headDim) {
            THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: separate-scale cache last dim must equal headDim");
        }
        NDArray::prepareSpecialUse({quantCache, scaleCache}, {newKv});
    }

    auto stream = context->getCudaStream();

    int threads = chooseThreads(headDim);
    size_t smem = sharedMemSize(threads);

    // Grid: one block per (kvHead, batch) pair
    dim3 grid(static_cast<unsigned>(kvHeads), static_cast<unsigned>(batch));
    dim3 block(threads);

    kvInPlaceWriteQuantisedBSHDKernel<<<grid, block, smem, *stream>>>(
        reinterpret_cast<const float*>(newKv->specialBuffer()),
        reinterpret_cast<int8_t*>(quantCache->specialBuffer()),
        inlineScale ? nullptr : reinterpret_cast<float*>(scaleCache->specialBuffer()),
        reinterpret_cast<const LongType*>(cachePosPtr),
        batch, kvHeads, headDim, maxKvLen, rowPitch);

    DebugHelper::checkGlobalErrorCode("kvInPlaceWriteQuantisedBSHD failed");
    if (inlineScale) {
        NDArray::registerSpecialUse({quantCache}, {newKv});
    } else {
        NDArray::registerSpecialUse({quantCache, scaleCache}, {newKv});
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
