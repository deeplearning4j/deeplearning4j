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

#include <system/op_boilerplate.h>
#include <cuda_runtime.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <math/templatemath.h>
#include <ops/op_types.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

#include <vector>

// The quantize/dequantize helpers back kv_cache_quantize/kv_cache_dequantize; the in-place INT8
// write backs dot_product_attention_v2's INT8 KV cache path.
#if NOT_EXCLUDED(OP_kv_cache_quantize) || NOT_EXCLUDED(OP_kv_cache_dequantize) || \
    NOT_EXCLUDED(OP_dot_product_attention_v2)
namespace sd {
namespace ops {
namespace helpers {

constexpr int KVQ_WARP_SIZE = 32;
// A block holds at most 1024 threads, so a block reduction has at most 32 warp partials.
constexpr int KVQ_MAX_WARPS = 1024 / KVQ_WARP_SIZE;

// Warp/block max reductions come from device_primitives.cuh
// (sd::device::blockAllReduceMax / sd::device::blockReduceMax).

//////////////////////////////////////////////////////////////////////////////
// Rows run along the last dimension (layout contract in kv_cache_quantize.h). Row r is TAD r
// along that dimension in every operand, whatever its order or strides: each kernel takes the
// operand's device TAD offsets plus its last-dimension element stride, and a block walks rows
// with a block-stride loop. The scale of row r is the C-order element r of `scales`.
//////////////////////////////////////////////////////////////////////////////
static std::shared_ptr<TadPack> kvRows(NDArray* array) {
    // tadForDimensions treats -1 as "the whole array", so pass the last dimension explicitly.
    return ConstantTadHelper::getInstance().tadForDimensions(array->shapeInfo(),
                                                             static_cast<LongType>(array->rankOf() - 1));
}

//////////////////////////////////////////////////////////////////////////////
// INT8 quantize: scale = max(abs(row)) / 127
//////////////////////////////////////////////////////////////////////////////
// ADR 0107 V2 ROW-INLINE: when `scales` is null each quantized row is rowLen+4 unit-stride int8
// elements — rowLen quantized values followed by the row's float32 scale. The scale lives INSIDE
// the logical tensor, so DSP ext-input staging/copies preserve it. The input keeps its dtype T;
// the absmax and the quantization run in AccT.
template <typename T>
SD_KERNEL void kvCacheQuantizeInt8Kernel(
    const T* input, const LongType* inputOffsets, const LongType inputStride,
    int8_t* quantized, const LongType* quantOffsets, const LongType quantStride,
    float* scales, const LongType* scalesShapeInfo,
    const LongType numRows, const LongType rowLen) {
    using AccT = typename simdOps::AggregateType<T>::type;

    __shared__ AccT reduceScratch[KVQ_MAX_WARPS];

    for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
        const T* inputRow = input + inputOffsets[row];
        int8_t* quantRow = quantized + quantOffsets[row];

        // Pass 1: absmax. blockAllReduceMax hands the row max to every thread and ends with a
        // barrier, so the scratch is free for the next row.
        AccT threadMax = static_cast<AccT>(0);
        for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
            threadMax = sd::math::sd_max<AccT>(threadMax,
                                               sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(inputRow[i * inputStride])));
        }
        const AccT rowMax = sd::device::blockAllReduceMax(threadMax, reduceScratch);

        // The scale is stored as FLOAT32, so every thread quantizes against the stored value.
        float rowScale = static_cast<float>(rowMax / static_cast<AccT>(127));
        if (rowScale == 0.0f) rowScale = 1.0f;
        if (threadIdx.x == 0) {
            if (scales != nullptr) {
                scales[kvRowScaleOffset(scalesShapeInfo, row)] = rowScale;
            } else {
                // The inline slot starts rowLen bytes into the row, which is not float-aligned in general.
                memcpy(quantRow + rowLen, &rowScale, sizeof(float));
            }
        }
        const AccT invScale = static_cast<AccT>(1) / static_cast<AccT>(rowScale);

        // Pass 2: quantize
        for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
            AccT val = static_cast<AccT>(inputRow[i * inputStride]) * invScale;
            val = sd::math::sd_max<AccT>(static_cast<AccT>(-127), sd::math::sd_min<AccT>(static_cast<AccT>(127), val));
            quantRow[i * quantStride] = static_cast<int8_t>(sd::math::sd_rint<AccT, AccT>(val));
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT8 dequantize: output = quantized * scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheDequantizeInt8Kernel(
    const int8_t* quantized, const LongType* quantOffsets, const LongType quantStride,
    const float* scales, const LongType* scalesShapeInfo,
    T* output, const LongType* outputOffsets, const LongType outputStride,
    const LongType numRows, const LongType rowLen) {
    using AccT = typename simdOps::AggregateType<T>::type;

    for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
        const int8_t* quantRow = quantized + quantOffsets[row];
        T* outputRow = output + outputOffsets[row];
        const AccT scale = static_cast<AccT>(scales[kvRowScaleOffset(scalesShapeInfo, row)]);

        for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
            outputRow[i * outputStride] = static_cast<T>(static_cast<AccT>(quantRow[i * quantStride]) * scale);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT4 quantize: scale = max(abs(row)) / 7, two values per byte. Element pair (2j, 2j+1) of a row
// packs into byte j of that same row; bytes [ceil(rowLen/2), rowLen) of the row are zeroed so the
// output is fully written.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheQuantizeInt4Kernel(
    const T* input, const LongType* inputOffsets, const LongType inputStride,
    uint8_t* quantized, const LongType* quantOffsets, const LongType quantStride,
    float* scales, const LongType* scalesShapeInfo,
    const LongType numRows, const LongType rowLen) {
    using AccT = typename simdOps::AggregateType<T>::type;

    __shared__ AccT reduceScratch[KVQ_MAX_WARPS];
    const LongType packedRowLen = (rowLen + 1) / 2;

    for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
        const T* inputRow = input + inputOffsets[row];
        uint8_t* quantRow = quantized + quantOffsets[row];

        // Pass 1: absmax, broadcast to every thread (the reduction ends with a barrier).
        AccT threadMax = static_cast<AccT>(0);
        for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
            threadMax = sd::math::sd_max<AccT>(threadMax,
                                               sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(inputRow[i * inputStride])));
        }
        const AccT rowMax = sd::device::blockAllReduceMax(threadMax, reduceScratch);

        // The scale is stored as FLOAT32, so every thread quantizes against the stored value.
        float rowScale = static_cast<float>(rowMax / static_cast<AccT>(7));
        if (rowScale == 0.0f) rowScale = 1.0f;
        if (threadIdx.x == 0) scales[kvRowScaleOffset(scalesShapeInfo, row)] = rowScale;
        const AccT invScale = static_cast<AccT>(1) / static_cast<AccT>(rowScale);

        // Pass 2: each thread packs whole bytes (low nibble = even element + 8, high = odd + 8).
        for (LongType j = threadIdx.x; j < rowLen; j += blockDim.x) {
            uint8_t packed = 0;
            if (j < packedRowLen) {
                const LongType i = 2 * j;
                const int q0 = kvQuantizeInt4Value<AccT>(static_cast<AccT>(inputRow[i * inputStride]) * invScale);
                const int q1 = i + 1 < rowLen
                                   ? kvQuantizeInt4Value<AccT>(static_cast<AccT>(inputRow[(i + 1) * inputStride]) * invScale)
                                   : 0;
                packed = static_cast<uint8_t>(((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4));
            }
            quantRow[j * quantStride] = packed;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// INT4 dequantize: byte j of each row unpacks into elements 2j and 2j+1
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void kvCacheDequantizeInt4Kernel(
    const uint8_t* quantized, const LongType* quantOffsets, const LongType quantStride,
    const float* scales, const LongType* scalesShapeInfo,
    T* output, const LongType* outputOffsets, const LongType outputStride,
    const LongType numRows, const LongType rowLen) {
    using AccT = typename simdOps::AggregateType<T>::type;

    const LongType packedRowLen = (rowLen + 1) / 2;
    for (LongType row = blockIdx.x; row < numRows; row += gridDim.x) {
        const uint8_t* quantRow = quantized + quantOffsets[row];
        T* outputRow = output + outputOffsets[row];
        const AccT scale = static_cast<AccT>(scales[kvRowScaleOffset(scalesShapeInfo, row)]);

        for (LongType j = threadIdx.x; j < packedRowLen; j += blockDim.x) {
            const uint8_t packed = quantRow[j * quantStride];
            const LongType i = 2 * j;
            outputRow[i * outputStride] = static_cast<T>(static_cast<AccT>(static_cast<int>(packed & 0x0F) - 8) * scale);
            if (i + 1 < rowLen) {
                outputRow[(i + 1) * outputStride] =
                    static_cast<T>(static_cast<AccT>(static_cast<int>((packed >> 4) & 0x0F) - 8) * scale);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Launchers
//////////////////////////////////////////////////////////////////////////////
// INT8, FP8_E4M3 and FP8_E5M2 all store INT8 on the GPU; only INT4 packs nibbles.
template <typename T>
static void kvCacheQuantizeLauncher(const dim3& launchDims, const cudaStream_t* stream, const bool int4,
                                    const void* vInput, const LongType* inputOffsets, const LongType inputStride,
                                    void* vQuantized, const LongType* quantOffsets, const LongType quantStride,
                                    float* scales, const LongType* scalesShapeInfo,
                                    const LongType numRows, const LongType rowLen) {
    auto input = reinterpret_cast<const T*>(vInput);
    if (int4) {
        kvCacheQuantizeInt4Kernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            input, inputOffsets, inputStride, reinterpret_cast<uint8_t*>(vQuantized), quantOffsets, quantStride,
            scales, scalesShapeInfo, numRows, rowLen);
    } else {
        kvCacheQuantizeInt8Kernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            input, inputOffsets, inputStride, reinterpret_cast<int8_t*>(vQuantized), quantOffsets, quantStride,
            scales, scalesShapeInfo, numRows, rowLen);
    }
}

template <typename T>
static void kvCacheDequantizeLauncher(const dim3& launchDims, const cudaStream_t* stream, const bool int4,
                                      const void* vQuantized, const LongType* quantOffsets, const LongType quantStride,
                                      const float* scales, const LongType* scalesShapeInfo,
                                      void* vOutput, const LongType* outputOffsets, const LongType outputStride,
                                      const LongType numRows, const LongType rowLen) {
    auto output = reinterpret_cast<T*>(vOutput);
    if (int4) {
        kvCacheDequantizeInt4Kernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            reinterpret_cast<const uint8_t*>(vQuantized), quantOffsets, quantStride, scales, scalesShapeInfo,
            output, outputOffsets, outputStride, numRows, rowLen);
    } else {
        kvCacheDequantizeInt8Kernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            reinterpret_cast<const int8_t*>(vQuantized), quantOffsets, quantStride, scales, scalesShapeInfo,
            output, outputOffsets, outputStride, numRows, rowLen);
    }
}

static void validateKvQuantFormat(const KVQuantFormat format, const char* message) {
    if (format != KVQuantFormat::INT8 && format != KVQuantFormat::FP8_E4M3 &&
        format != KVQuantFormat::FP8_E5M2 && format != KVQuantFormat::INT4) {
        THROW_EXCEPTION(message);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Public interface: kvCacheQuantize
//////////////////////////////////////////////////////////////////////////////
void kvCacheQuantize(NDArray* input, NDArray* quantized, NDArray* scales,
                     int quantFormat, LaunchContext* context) {
    const auto format = static_cast<KVQuantFormat>(quantFormat);
    validateKvQuantFormat(format, "kvCacheQuantize: unsupported quantization format");

    // ADR 0107 V2 ROW-INLINE: when scales is null, `quantized` is a row-inline tensor whose last
    // dimension is rowLen+4 — each row holds rowLen int8 values followed by that row's float32
    // scale. The scale rides INSIDE the logical tensor so DSP staging/copies preserve it.
    const bool inlineScale = (scales == nullptr);
    if (inlineScale && format == KVQuantFormat::INT4) {
        THROW_EXCEPTION("kvCacheQuantize: row-inline scale mode supports INT8 only, not INT4");
    }

    // No rows (or only empty rows) to quantize.
    if (input->lengthOf() == 0) return;

    const LongType rowLen = input->sizeAt(-1);
    auto inRows = kvRows(input);
    auto quantRows = kvRows(quantized);
    const LongType numRows = inRows->numberOfTads();

    if (inlineScale) {
        NDArray::prepareSpecialUse({quantized}, {input});
    } else {
        NDArray::prepareSpecialUse({quantized, scales}, {input});
    }

    const dim3 launchDims = getKvCacheQuantizeDims(numRows, rowLen);
    float* scalesBuf = inlineScale ? nullptr : reinterpret_cast<float*>(scales->specialBuffer());
    const LongType* scalesShapeInfo = inlineScale ? nullptr : scales->specialShapeInfo();

    BUILD_SINGLE_SELECTOR(input->dataType(), kvCacheQuantizeLauncher,
                          (launchDims, context->getCudaStream(), format == KVQuantFormat::INT4,
                           input->specialBuffer(), inRows->specialOffsets(), input->strideAt(-1),
                           quantized->specialBuffer(), quantRows->specialOffsets(), quantized->strideAt(-1),
                           scalesBuf, scalesShapeInfo, numRows, rowLen),
                          SD_FLOAT_TYPES);

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
    const auto format = static_cast<KVQuantFormat>(quantFormat);
    validateKvQuantFormat(format, "kvCacheDequantize: unsupported quantization format");

    if (output->lengthOf() == 0) return;

    const LongType rowLen = output->sizeAt(-1);
    auto quantRows = kvRows(quantized);
    auto outRows = kvRows(output);
    const LongType numRows = outRows->numberOfTads();

    NDArray::prepareSpecialUse({output}, {quantized, scales});

    const dim3 launchDims = getKvCacheQuantizeDims(numRows, rowLen);

    BUILD_SINGLE_SELECTOR(output->dataType(), kvCacheDequantizeLauncher,
                          (launchDims, context->getCudaStream(), format == KVQuantFormat::INT4,
                           quantized->specialBuffer(), quantRows->specialOffsets(), quantized->strideAt(-1),
                           reinterpret_cast<const float*>(scales->specialBuffer()), scales->specialShapeInfo(),
                           output->specialBuffer(), outRows->specialOffsets(), output->strideAt(-1),
                           numRows, rowLen),
                          SD_FLOAT_TYPES);

    DebugHelper::checkGlobalErrorCode("kvCacheDequantize failed");
    NDArray::registerSpecialUse({output}, {quantized, scales});
}


//////////////////////////////////////////////////////////////////////////////
// V2: kvInPlaceWriteQuantisedBSHDKernel
//
// One block per (batch, kvHead) pair.  Within each block:
//   1. Block absmax reduction over headDim elements in AccT (pass 1).
//   2. Thread 0 stores the FLOAT32 row scale and broadcasts invScale through shared memory.
//   3. INT8 scatter write to quantCache[b, cachePos, kvHead, :] (pass 2).
//
// newKv keeps the model dtype T; every operand is addressed through its own strides.
//
// CUDA-graph safe:
//   - cachePos is read from a device-side LongType pointer at kernel runtime.
//   - No cudaMalloc / dynamic allocation.
//   - All output addresses are fixed pre-allocated buffers.
//   - Per-row block reduction with NO cross-row atomics.
//////////////////////////////////////////////////////////////////////////////
// ADR 0107 V2 ROW-INLINE: when `scaleCache` is null the cache is a row-inline tensor
// [batch, maxKvLen, kvHeads, headDim+4] — each unit-stride row holds headDim int8 values followed
// by that row's float32 scale (written at dstRow+headDim). When `scaleCache` is non-null (legacy
// separate-scale mode), the cache last dim is headDim and the scale goes to
// scaleCache[batch, cachePos, kvHead].
template <typename T>
SD_KERNEL void kvInPlaceWriteQuantisedBSHDKernel(
    const T* newKv,                 // [batch, 1, kvHeads, headDim]
    int8_t* quantCache,             // [batch, maxKvLen, kvHeads, headDim(+4)]
    float* scaleCache,              // [batch, maxKvLen, kvHeads] or null (row-inline)
    const LongType* cachePosPtr,
    const LongType batch,
    const LongType kvHeads,
    const LongType headDim,
    const LongType maxKvLen,
    const LongType nkStride0, const LongType nkStride2, const LongType nkStride3,
    const LongType qcStride0, const LongType qcStride1, const LongType qcStride2, const LongType qcStride3,
    const LongType scStride0, const LongType scStride1, const LongType scStride2) {
    using AccT = typename simdOps::AggregateType<T>::type;

    const LongType cachePos = *cachePosPtr;
    if (cachePos < 0 || cachePos >= maxKvLen) return;

    // Grid: (blockIdx.x = kvHead, blockIdx.y = batch)
    const LongType kvHead   = blockIdx.x;
    const LongType batchIdx = blockIdx.y;
    if (batchIdx >= batch || kvHead >= kvHeads) return;

    __shared__ AccT reduceScratch[KVQ_MAX_WARPS];
    __shared__ AccT invScale;

    // Source row: newKv[batchIdx, 0, kvHead, :]
    const T* srcRow = newKv + batchIdx * nkStride0 + kvHead * nkStride2;
    // Destination row: quantCache[batchIdx, cachePos, kvHead, :]
    int8_t* dstRow = quantCache + batchIdx * qcStride0 + cachePos * qcStride1 + kvHead * qcStride2;

    // Pass 1: find absmax over headDim
    AccT threadMax = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < headDim; i += blockDim.x) {
        threadMax = sd::math::sd_max<AccT>(threadMax,
                                           sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(srcRow[i * nkStride3])));
    }
    const AccT rowMax = sd::device::blockReduceMax(threadMax, reduceScratch);

    if (threadIdx.x == 0) {
        // The scale is stored as FLOAT32, so quantize against the stored value.
        float rowScale = static_cast<float>(rowMax / static_cast<AccT>(127));
        if (rowScale == 0.0f) rowScale = 1.0f;
        if (scaleCache != nullptr) {
            scaleCache[batchIdx * scStride0 + cachePos * scStride1 + kvHead * scStride2] = rowScale;
        } else {
            // The inline slot starts headDim bytes into the row, which is not float-aligned in general.
            memcpy(dstRow + headDim, &rowScale, sizeof(float));
        }
        invScale = static_cast<AccT>(1) / static_cast<AccT>(rowScale);
    }
    __syncthreads();

    // Pass 2: quantize + scatter
    for (LongType i = threadIdx.x; i < headDim; i += blockDim.x) {
        AccT v = static_cast<AccT>(srcRow[i * nkStride3]) * invScale;
        v = sd::math::sd_max<AccT>(static_cast<AccT>(-127), sd::math::sd_min<AccT>(static_cast<AccT>(127), v));
        dstRow[i * qcStride3] = static_cast<int8_t>(sd::math::sd_rint<AccT, AccT>(v));
    }
}

template <typename T>
static void kvInPlaceWriteQuantisedBSHDLauncher(const dim3& grid, const int threads, const cudaStream_t* stream,
                                                NDArray* newKv, NDArray* quantCache, NDArray* scaleCache,
                                                const void* cachePosPtr) {
    const bool inlineScale = (scaleCache == nullptr);
    kvInPlaceWriteQuantisedBSHDKernel<T><<<grid, threads, 0, *stream>>>(
        reinterpret_cast<const T*>(newKv->specialBuffer()),
        reinterpret_cast<int8_t*>(quantCache->specialBuffer()),
        inlineScale ? nullptr : reinterpret_cast<float*>(scaleCache->specialBuffer()),
        reinterpret_cast<const LongType*>(cachePosPtr),
        newKv->sizeAt(0), newKv->sizeAt(2), newKv->sizeAt(3), quantCache->sizeAt(1),
        newKv->strideAt(0), newKv->strideAt(2), newKv->strideAt(3),
        quantCache->strideAt(0), quantCache->strideAt(1), quantCache->strideAt(2), quantCache->strideAt(3),
        inlineScale ? 0 : scaleCache->strideAt(0),
        inlineScale ? 0 : scaleCache->strideAt(1),
        inlineScale ? 0 : scaleCache->strideAt(2));
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

    if (quantCache->dataType() != DataType::INT8) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: quantCache must be INT8");
    }
    if (scaleCache != nullptr && scaleCache->dataType() != DataType::FLOAT32) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: scaleCache must be FLOAT32");
    }
    if (newKv->rankOf() != 4 || quantCache->rankOf() != 4 || newKv->sizeAt(1) != 1) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: expected newKv [batch, 1, kvHeads, headDim] and a rank-4 cache");
    }

    const LongType batch    = newKv->sizeAt(0);
    const LongType kvHeads  = newKv->sizeAt(2);
    const LongType headDim  = newKv->sizeAt(3);
    const LongType maxKvLen = quantCache->sizeAt(1);
    if (quantCache->sizeAt(0) != batch || quantCache->sizeAt(2) != kvHeads) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: cache batch/kvHeads must match newKv");
    }

    // ADR 0107 V2 ROW-INLINE: null scaleCache → cache last dim is headDim+4 and each row's
    // float32 scale sits at dstRow+headDim (inside the logical tensor). Non-null = legacy separate.
    const bool inlineScale = (scaleCache == nullptr);
    const LongType rowPitch = quantCache->sizeAt(3);
    if (inlineScale && (rowPitch != headDim + 4 || quantCache->strideAt(3) != 1)) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: row-inline cache must be [batch, maxKvLen, kvHeads, headDim+4] "
                        "with a unit last-dimension stride");
    }
    if (!inlineScale) {
        if (rowPitch != headDim) {
            THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: separate-scale cache last dim must equal headDim");
        }
        if (scaleCache->rankOf() != 3 || scaleCache->sizeAt(0) != batch || scaleCache->sizeAt(1) != maxKvLen ||
            scaleCache->sizeAt(2) != kvHeads) {
            THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: scaleCache must be [batch, maxKvLen, kvHeads]");
        }
    }

    if (batch == 0 || kvHeads == 0 || headDim == 0 || maxKvLen == 0) return;
    if (cachePosPtr == nullptr) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: cache position pointer is null");
    }

    std::vector<NDArray*> outputs = {quantCache};
    if (!inlineScale) outputs.push_back(scaleCache);
    NDArray::prepareSpecialUse(outputs, {newKv});

    // Grid: one block per (kvHead, batch) pair. The kernel reads the device-resident cache
    // position at run time and writes nothing when it is out of range.
    const int threads = static_cast<int>(getKvCacheQuantizeDims(1, headDim).y);
    const dim3 grid(static_cast<unsigned>(kvHeads), static_cast<unsigned>(batch));
    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWriteQuantisedBSHDLauncher,
                          (grid, threads, context->getCudaStream(), newKv, quantCache, scaleCache, cachePosPtr),
                          SD_FLOAT_TYPES);

    DebugHelper::checkGlobalErrorCode("kvInPlaceWriteQuantisedBSHD failed");
    NDArray::registerSpecialUse(outputs, {newKv});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif  // NOT_EXCLUDED(OP_kv_cache_quantize) || NOT_EXCLUDED(OP_kv_cache_dequantize) || NOT_EXCLUDED(OP_dot_product_attention_v2)
