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
// KV Cache Quantization/Dequantization CPU implementation
// Supports INT8, FP8_E4M3, FP8_E5M2, and INT4 quantization formats.
// Per-row absmax symmetric quantization.
//

#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <math/templatemath.h>
#include <ops/op_types.h>

#include <cstring>
#include <vector>

// The quantize/dequantize helpers back kv_cache_quantize/kv_cache_dequantize; the in-place INT8
// write and the quantised decode read back dot_product_attention_v2's INT8 KV cache path.
#if NOT_EXCLUDED(OP_kv_cache_quantize) || NOT_EXCLUDED(OP_kv_cache_dequantize) || \
    NOT_EXCLUDED(OP_dot_product_attention_v2)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// Rows run along the last dimension (layout contract in kv_cache_quantize.h). TAD r along that
// dimension is the same logical row in every operand whatever its order or strides, and each
// element is reached through the operand's own last-dimension stride. tadForDimensions treats -1
// as "the whole array", so the last dimension is passed as a non-negative index.
//////////////////////////////////////////////////////////////////////////////
static std::shared_ptr<TadPack> kvRows(NDArray* array) {
    return ConstantTadHelper::getInstance().tadForDimensions(array->shapeInfo(),
                                                             static_cast<LongType>(array->rankOf() - 1));
}

//////////////////////////////////////////////////////////////////////////////
// INT8 quantization: scale = max(abs(row)) / 127.0
//////////////////////////////////////////////////////////////////////////////
// ADR 0107 V2 ROW-INLINE: when `scales` is null the output is a row-inline tensor whose rows are
// rowLen+4 int8 elements — rowLen quantized values followed by that row's float32 scale (inside
// the logical tensor, so staging/copies preserve it). Non-null scales = legacy separate mode.
template <typename T>
static void kvCacheQuantizeInt8Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    using AccT = typename simdOps::AggregateType<T>::type;
    const LongType rowLen = input->sizeAt(-1);
    const bool inlineScale = (scales == nullptr);

    auto inRows = kvRows(input);
    auto quantRows = kvRows(quantized);
    const LongType* xOffsets = inRows->primaryOffsets();
    const LongType* qOffsets = quantRows->primaryOffsets();
    const LongType xStride = input->strideAt(-1);
    // Row-inline rows have a unit stride (kv_cache_quantize validates it), so the scale bytes
    // directly follow the rowLen values.
    const LongType qStride = quantized->strideAt(-1);

    const T* x = input->bufferAsT<T>();
    int8_t* q = reinterpret_cast<int8_t*>(quantized->buffer());
    // Scales are FLOAT32 for every input dtype (kv_cache_quantize output 1 and the inline slot).
    float* s = inlineScale ? nullptr : scales->bufferAsT<float>();
    const LongType* sShapeInfo = inlineScale ? nullptr : scales->shapeInfo();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + xOffsets[row];
            int8_t* qRow = q + qOffsets[row];

            // Pass 1: find absmax
            AccT absMax = static_cast<AccT>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                const AccT absVal = sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(xRow[i * xStride]));
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale. It is stored as FLOAT32, so quantize against the stored value.
            float scale = static_cast<float>(absMax / static_cast<AccT>(127));
            if (scale == 0.0f) scale = 1.0f;
            if (inlineScale) {
                // The slot starts rowLen bytes into the row, which is not float-aligned in general.
                std::memcpy(qRow + rowLen, &scale, sizeof(float));
            } else {
                s[kvRowScaleOffset(sShapeInfo, row)] = scale;
            }

            const AccT invScale = static_cast<AccT>(1) / static_cast<AccT>(scale);

            // Pass 2: quantize
            for (LongType i = 0; i < rowLen; ++i) {
                AccT val = static_cast<AccT>(xRow[i * xStride]) * invScale;
                // Clamp to [-127, 127] and round
                val = sd::math::sd_max<AccT>(static_cast<AccT>(-127), sd::math::sd_min<AccT>(static_cast<AccT>(127), val));
                qRow[i * qStride] = static_cast<int8_t>(sd::math::sd_rint<AccT, AccT>(val));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, inRows->numberOfTads());
}

//////////////////////////////////////////////////////////////////////////////
// INT8 dequantization: output = quantized * scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheDequantizeInt8Cpu_(NDArray* quantized, NDArray* scales, NDArray* output) {
    using AccT = typename simdOps::AggregateType<T>::type;
    const LongType rowLen = output->sizeAt(-1);

    auto quantRows = kvRows(quantized);
    auto outRows = kvRows(output);
    const LongType* qOffsets = quantRows->primaryOffsets();
    const LongType* zOffsets = outRows->primaryOffsets();
    const LongType qStride = quantized->strideAt(-1);
    const LongType zStride = output->strideAt(-1);

    const int8_t* q = reinterpret_cast<const int8_t*>(quantized->buffer());
    const float* s = scales->bufferAsT<float>();
    const LongType* sShapeInfo = scales->shapeInfo();
    T* z = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const int8_t* qRow = q + qOffsets[row];
            T* zRow = z + zOffsets[row];
            const AccT scale = static_cast<AccT>(s[kvRowScaleOffset(sShapeInfo, row)]);

            for (LongType i = 0; i < rowLen; ++i) {
                zRow[i * zStride] = static_cast<T>(static_cast<AccT>(qRow[i * qStride]) * scale);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, outRows->numberOfTads());
}

//////////////////////////////////////////////////////////////////////////////
// INT4 quantization: scale = max(abs(row)) / 7.0, pack 2 values per byte
// Element pair (2j, 2j+1) of a row packs into byte j of that same row; bytes
// [packedRowLen, rowLen) of the row are zeroed so the output is fully written.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheQuantizeInt4Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    using AccT = typename simdOps::AggregateType<T>::type;
    const LongType rowLen = input->sizeAt(-1);
    // Packed row length: ceil(rowLen / 2)
    const LongType packedRowLen = (rowLen + 1) / 2;

    auto inRows = kvRows(input);
    auto quantRows = kvRows(quantized);
    const LongType* xOffsets = inRows->primaryOffsets();
    const LongType* qOffsets = quantRows->primaryOffsets();
    const LongType xStride = input->strideAt(-1);
    const LongType qStride = quantized->strideAt(-1);

    const T* x = input->bufferAsT<T>();
    uint8_t* q = reinterpret_cast<uint8_t*>(quantized->buffer());
    float* s = scales->bufferAsT<float>();
    const LongType* sShapeInfo = scales->shapeInfo();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + xOffsets[row];
            uint8_t* qRow = q + qOffsets[row];

            // Pass 1: find absmax
            AccT absMax = static_cast<AccT>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                const AccT absVal = sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(xRow[i * xStride]));
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale. It is stored as FLOAT32, so quantize against the stored value.
            float scale = static_cast<float>(absMax / static_cast<AccT>(7));
            if (scale == 0.0f) scale = 1.0f;
            s[kvRowScaleOffset(sShapeInfo, row)] = scale;

            const AccT invScale = static_cast<AccT>(1) / static_cast<AccT>(scale);

            // Pass 2: quantize and pack 2 values per byte
            for (LongType j = 0; j < packedRowLen; ++j) {
                const LongType i = 2 * j;
                const int q0 = kvQuantizeInt4Value<AccT>(static_cast<AccT>(xRow[i * xStride]) * invScale);
                const int q1 = i + 1 < rowLen
                                   ? kvQuantizeInt4Value<AccT>(static_cast<AccT>(xRow[(i + 1) * xStride]) * invScale)
                                   : 0;
                // Pack: low nibble = q0 + 8 (offset to unsigned), high nibble = q1 + 8
                qRow[j * qStride] = static_cast<uint8_t>(((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4));
            }
            for (LongType j = packedRowLen; j < rowLen; ++j) {
                qRow[j * qStride] = 0;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, inRows->numberOfTads());
}

//////////////////////////////////////////////////////////////////////////////
// INT4 dequantization: unpack byte j of each row into elements 2j, 2j+1 and multiply by scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheDequantizeInt4Cpu_(NDArray* quantized, NDArray* scales, NDArray* output) {
    using AccT = typename simdOps::AggregateType<T>::type;
    const LongType rowLen = output->sizeAt(-1);
    const LongType packedRowLen = (rowLen + 1) / 2;

    auto quantRows = kvRows(quantized);
    auto outRows = kvRows(output);
    const LongType* qOffsets = quantRows->primaryOffsets();
    const LongType* zOffsets = outRows->primaryOffsets();
    const LongType qStride = quantized->strideAt(-1);
    const LongType zStride = output->strideAt(-1);

    const uint8_t* q = reinterpret_cast<const uint8_t*>(quantized->buffer());
    const float* s = scales->bufferAsT<float>();
    const LongType* sShapeInfo = scales->shapeInfo();
    T* z = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const uint8_t* qRow = q + qOffsets[row];
            T* zRow = z + zOffsets[row];
            const AccT scale = static_cast<AccT>(s[kvRowScaleOffset(sShapeInfo, row)]);

            for (LongType j = 0; j < packedRowLen; ++j) {
                const uint8_t packed = qRow[j * qStride];
                const LongType i = 2 * j;
                zRow[i * zStride] = static_cast<T>(static_cast<AccT>(static_cast<int>(packed & 0x0F) - 8) * scale);
                if (i + 1 < rowLen) {
                    zRow[(i + 1) * zStride] =
                        static_cast<T>(static_cast<AccT>(static_cast<int>((packed >> 4) & 0x0F) - 8) * scale);
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, outRows->numberOfTads());
}

//////////////////////////////////////////////////////////////////////////////
// FP8 quantization stubs — treated as INT8 with different scale range
// FP8_E4M3 range: [-448, 448], FP8_E5M2 range: [-57344, 57344]
// For CPU fallback, we use symmetric INT8 quantization with
// appropriate clamping.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheQuantizeFp8Cpu_(NDArray* input, NDArray* quantized, NDArray* scales, int format) {
    // FP8 is effectively the same as INT8 on CPU (no HW fp8 support)
    // The scale range differs:
    //   E4M3: max representable = 448, so we clamp to [-127, 127] and scale accordingly
    //   E5M2: max representable = 57344
    // On CPU we store as INT8 bytes with the same absmax/127 scheme.
    kvCacheQuantizeInt8Cpu_<T>(input, quantized, scales);
}

template <typename T>
static void kvCacheDequantizeFp8Cpu_(NDArray* quantized, NDArray* scales, NDArray* output, int format) {
    kvCacheDequantizeInt8Cpu_<T>(quantized, scales, output);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void kvCacheQuantize(NDArray* input, NDArray* quantized, NDArray* scales, int quantFormat,
                     LaunchContext* /*context*/) {
    auto format = static_cast<KVQuantFormat>(quantFormat);

    // ADR 0107 V2 ROW-INLINE: null scales = row-inline mode — the per-row float32 scale is written
    // inside each rowLen+4 output row by kvCacheQuantizeInt8Cpu_. FP8 formats store INT8 here as
    // they do on CUDA, so only INT4 (packed nibbles) cannot carry the inline layout.
    if (scales == nullptr && format == KVQuantFormat::INT4) {
        THROW_EXCEPTION("kvCacheQuantize: row-inline scale mode supports INT8 only, not INT4");
    }

    // No rows (or only empty rows) to quantize.
    if (input->lengthOf() == 0) return;

    switch (format) {
        case KVQuantFormat::INT8:
            BUILD_SINGLE_SELECTOR(input->dataType(), kvCacheQuantizeInt8Cpu_, (input, quantized, scales), SD_FLOAT_TYPES);
            break;
        case KVQuantFormat::FP8_E4M3:
        case KVQuantFormat::FP8_E5M2:
            BUILD_SINGLE_SELECTOR(input->dataType(), kvCacheQuantizeFp8Cpu_, (input, quantized, scales, quantFormat), SD_FLOAT_TYPES);
            break;
        case KVQuantFormat::INT4:
            BUILD_SINGLE_SELECTOR(input->dataType(), kvCacheQuantizeInt4Cpu_, (input, quantized, scales), SD_FLOAT_TYPES);
            break;
        default:
            THROW_EXCEPTION("kvCacheQuantize: unsupported quantization format");
    }

    quantized->tickWriteHost();
    quantized->syncToDevice();
    if (scales != nullptr) {
        scales->tickWriteHost();
        scales->syncToDevice();
    }
}

void kvCacheDequantize(NDArray* quantized, NDArray* scales, NDArray* output, int quantFormat,
                       LaunchContext* /*context*/) {
    auto format = static_cast<KVQuantFormat>(quantFormat);

    if (output->lengthOf() == 0) return;

    switch (format) {
        case KVQuantFormat::INT8:
            BUILD_SINGLE_SELECTOR(output->dataType(), kvCacheDequantizeInt8Cpu_, (quantized, scales, output), SD_FLOAT_TYPES);
            break;
        case KVQuantFormat::FP8_E4M3:
        case KVQuantFormat::FP8_E5M2:
            BUILD_SINGLE_SELECTOR(output->dataType(), kvCacheDequantizeFp8Cpu_, (quantized, scales, output, quantFormat), SD_FLOAT_TYPES);
            break;
        case KVQuantFormat::INT4:
            BUILD_SINGLE_SELECTOR(output->dataType(), kvCacheDequantizeInt4Cpu_, (quantized, scales, output), SD_FLOAT_TYPES);
            break;
        default:
            THROW_EXCEPTION("kvCacheDequantize: unsupported quantization format");
    }

    output->tickWriteHost();
    output->syncToDevice();
}

//////////////////////////////////////////////////////////////////////////////
// V2: kvInPlaceWriteQuantisedBSHD (CPU)
// Quantize one decode-step K or V row per (batch, kvHead) into a pre-allocated INT8 cache at
// cachePos and store that row's FLOAT32 scale. newKv keeps the model dtype T; the absmax and the
// quantization run in AccT. All operands are addressed through their own strides.
//
// Layout (INT8_KV):
//   newKv      : T      [batch, 1,        kvHeads, headDim]
//   quantCache : INT8   [batch, maxKvLen, kvHeads, headDim+4] row-inline (scale at row+headDim)
//              | INT8   [batch, maxKvLen, kvHeads, headDim]   with a separate scaleCache
//   scaleCache : FLOAT32 [batch, maxKvLen, kvHeads] or nullptr for row-inline
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvInPlaceWriteQuantisedBSHD_(NDArray* quantCache, NDArray* scaleCache, NDArray* newKv,
                                         const LongType cachePos) {
    using AccT = typename simdOps::AggregateType<T>::type;

    const LongType batch   = newKv->sizeAt(0);
    const LongType kvHeads = newKv->sizeAt(2);
    const LongType headDim = newKv->sizeAt(3);

    const T* src = newKv->bufferAsT<T>();
    const LongType nkS0 = newKv->strideAt(0), nkS2 = newKv->strideAt(2), nkS3 = newKv->strideAt(3);

    int8_t* qBuf = reinterpret_cast<int8_t*>(quantCache->buffer());
    const LongType qcS0 = quantCache->strideAt(0), qcS1 = quantCache->strideAt(1);
    const LongType qcS2 = quantCache->strideAt(2), qcS3 = quantCache->strideAt(3);

    // Row-inline: the float32 scale occupies bytes [headDim, headDim+4) of the unit-stride row.
    const bool inlineScale = (scaleCache == nullptr);
    float* scaleBuf = inlineScale ? nullptr : scaleCache->bufferAsT<float>();
    const LongType scS0 = inlineScale ? 0 : scaleCache->strideAt(0);
    const LongType scS1 = inlineScale ? 0 : scaleCache->strideAt(1);
    const LongType scS2 = inlineScale ? 0 : scaleCache->strideAt(2);

    auto func = PRAGMA_THREADS_FOR {
        for (auto bh = start; bh < stop; ++bh) {
            const LongType b = bh / kvHeads;
            const LongType h = bh % kvHeads;

            const T* srcRow = src + b * nkS0 + h * nkS2;
            int8_t* qRow = qBuf + b * qcS0 + cachePos * qcS1 + h * qcS2;

            // Pass 1: absmax
            AccT absMax = static_cast<AccT>(0);
            for (LongType d = 0; d < headDim; ++d) {
                const AccT av = sd::math::sd_abs<AccT, AccT>(static_cast<AccT>(srcRow[d * nkS3]));
                if (av > absMax) absMax = av;
            }

            // The scale is stored as FLOAT32, so quantize against the stored value.
            float rowScale = static_cast<float>(absMax / static_cast<AccT>(127));
            if (rowScale == 0.0f) rowScale = 1.0f;
            if (inlineScale) {
                std::memcpy(qRow + headDim, &rowScale, sizeof(float));
            } else {
                scaleBuf[b * scS0 + cachePos * scS1 + h * scS2] = rowScale;
            }
            const AccT invScale = static_cast<AccT>(1) / static_cast<AccT>(rowScale);

            // Pass 2: quantize + scatter
            for (LongType d = 0; d < headDim; ++d) {
                AccT v = static_cast<AccT>(srcRow[d * nkS3]) * invScale;
                v = sd::math::sd_max<AccT>(static_cast<AccT>(-127), sd::math::sd_min<AccT>(static_cast<AccT>(127), v));
                qRow[d * qcS3] = static_cast<int8_t>(sd::math::sd_rint<AccT, AccT>(v));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, batch * kvHeads);
}

void kvInPlaceWriteQuantisedBSHD(
    NDArray* quantCache,
    NDArray* scaleCache,
    NDArray* newKv,
    const void* cachePosPtr,
    LaunchContext* /*context*/) {

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
    // float32 scale sits at qRow+headDim (inside the logical tensor). Non-null = legacy separate.
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

    // Host pointer on CPU. An out-of-range position writes nothing, like kvInPlaceWriteBSHD.
    const LongType cachePos = *reinterpret_cast<const LongType*>(cachePosPtr);
    if (cachePos < 0 || cachePos >= maxKvLen) return;

    BUILD_SINGLE_SELECTOR(newKv->dataType(), kvInPlaceWriteQuantisedBSHD_,
                          (quantCache, scaleCache, newKv, cachePos), SD_FLOAT_TYPES);

    quantCache->tickWriteHost();
    if (scaleCache != nullptr) scaleCache->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// V2: fusedGQADecodeQuantisedCpu — CPU GQA decode (seqQ == 1) over INT8 K/V.
// Same contract as the CUDA fusedGQADecodeQuantisedKernel: query, the current window, the bias
// and the output are the model dtype T; each cache row is dequantized with its FLOAT32 scale
// folded into the dot product (K) or the softmax weight (V); scores, softmax and the V sum run in
// AccT. Cache rows inside the current window [currentStart, currentStart + currentSeq) are read
// from the window instead, and attention stops at the end of that window. The optional score and
// logit outputs receive the softmax weights and pre-softmax scores of that same computation.
//////////////////////////////////////////////////////////////////////////////
static inline float quantisedRowScale(const int8_t* row, LongType headDim, const float* separateScale) {
    if (separateScale != nullptr) return *separateScale;
    float s;
    std::memcpy(&s, row + headDim, sizeof(float));
    return s;
}

template <typename T>
static void fusedGQADecodeQuantisedCpu_(NDArray* query, NDArray* quantKeyCache, NDArray* keyScaleCache,
                                        NDArray* quantValCache, NDArray* valScaleCache, NDArray* output,
                                        const double scale, NDArray* attentionBias,
                                        NDArray* currentKeyWindow, NDArray* currentValueWindow,
                                        const void* currentKvPosition, NDArray* attentionScores,
                                        NDArray* attentionLogits) {
    using AccT = typename simdOps::AggregateType<T>::type;

    const LongType batch          = query->sizeAt(0);
    const LongType numQHeads      = query->sizeAt(2);
    const LongType headDim        = query->sizeAt(3);
    const LongType seqKV          = quantKeyCache->sizeAt(1);
    const LongType numKvHeads     = quantKeyCache->sizeAt(2);
    const LongType headsPerKvHead = numQHeads / numKvHeads;

    const T* qBuf = query->bufferAsT<T>();
    const LongType qS0 = query->strideAt(0), qS2 = query->strideAt(2), qS3 = query->strideAt(3);
    T* oBuf = output->bufferAsT<T>();
    const LongType oS0 = output->strideAt(0), oS2 = output->strideAt(2), oS3 = output->strideAt(3);

    // Optional [batch, qHeads, 1, seqKV] outputs; seqQ == 1, so dim 2 never contributes an offset.
    T* scoresBuf = attentionScores != nullptr ? attentionScores->bufferAsT<T>() : nullptr;
    T* logitsBuf = attentionLogits != nullptr ? attentionLogits->bufferAsT<T>() : nullptr;
    const LongType sS0 = scoresBuf != nullptr ? attentionScores->strideAt(0) : 0;
    const LongType sS1 = scoresBuf != nullptr ? attentionScores->strideAt(1) : 0;
    const LongType sS3 = scoresBuf != nullptr ? attentionScores->strideAt(3) : 0;
    const LongType lS0 = logitsBuf != nullptr ? attentionLogits->strideAt(0) : 0;
    const LongType lS1 = logitsBuf != nullptr ? attentionLogits->strideAt(1) : 0;
    const LongType lS3 = logitsBuf != nullptr ? attentionLogits->strideAt(3) : 0;

    const int8_t* kBuf = reinterpret_cast<const int8_t*>(quantKeyCache->buffer());
    const int8_t* vBuf = reinterpret_cast<const int8_t*>(quantValCache->buffer());
    const LongType kS0 = quantKeyCache->strideAt(0), kS1 = quantKeyCache->strideAt(1);
    const LongType kS2 = quantKeyCache->strideAt(2), kS3 = quantKeyCache->strideAt(3);
    const LongType vS0 = quantValCache->strideAt(0), vS1 = quantValCache->strideAt(1);
    const LongType vS2 = quantValCache->strideAt(2), vS3 = quantValCache->strideAt(3);

    // ADR 0107 V2 ROW-INLINE: null scale caches → each cache row carries its float32 scale at
    // row+headDim (inside the logical tensor). Non-null = legacy separate [batch, seqKV, kvHeads].
    const bool inlineScales = (keyScaleCache == nullptr);
    const float* ksBuf = inlineScales ? nullptr : keyScaleCache->bufferAsT<float>();
    const float* vsBuf = inlineScales ? nullptr : valScaleCache->bufferAsT<float>();
    const LongType ksS0 = inlineScales ? 0 : keyScaleCache->strideAt(0);
    const LongType ksS1 = inlineScales ? 0 : keyScaleCache->strideAt(1);
    const LongType ksS2 = inlineScales ? 0 : keyScaleCache->strideAt(2);
    const LongType vsS0 = inlineScales ? 0 : valScaleCache->strideAt(0);
    const LongType vsS1 = inlineScales ? 0 : valScaleCache->strideAt(1);
    const LongType vsS2 = inlineScales ? 0 : valScaleCache->strideAt(2);

    // Additive bias, broadcast through zero strides on size-1 dims. seqQ == 1, so the query
    // dimension never contributes an offset. Rank 3 is [batch, seqQ, seqKV], rank 2 [seqQ, seqKV].
    const T* biasBuf = nullptr;
    LongType bS0 = 0, bS1 = 0, bS3 = 0;
    if (attentionBias != nullptr && !attentionBias->isEmpty()) {
        biasBuf = attentionBias->bufferAsT<T>();
        auto strideIfWide = [attentionBias](int dim) -> LongType {
            return attentionBias->sizeAt(dim) > 1 ? attentionBias->strideAt(dim) : 0;
        };
        const int biasRank = attentionBias->rankOf();
        if (biasRank == 4) {
            bS0 = strideIfWide(0);
            bS1 = strideIfWide(1);
            bS3 = strideIfWide(3);
        } else if (biasRank == 3) {
            bS0 = strideIfWide(0);
            bS3 = strideIfWide(2);
        } else if (biasRank == 2) {
            bS3 = strideIfWide(1);
        } else {
            bS3 = attentionBias->lengthOf() > 1 ? attentionBias->strideAt(0) : 0;
        }
    }

    // "Current window" contract (mirrors fusedGQADecodeKernel): rows [currentStart, currentStart +
    // currentSeq) come from the pre-quantization window, so this call never reads back the INT8
    // rows it just wrote and the current token is not quantized twice.
    const bool hasCurrentWindow = currentKeyWindow != nullptr && currentValueWindow != nullptr &&
                                  currentKvPosition != nullptr && currentKeyWindow->sizeAt(1) > 0;
    const LongType currentSeq = hasCurrentWindow ? currentKeyWindow->sizeAt(1) : 0;
    const LongType currentStart = hasCurrentWindow ? *reinterpret_cast<const LongType*>(currentKvPosition) : -1;
    const bool validCurrentWindow = hasCurrentWindow && currentStart >= 0 && currentStart < seqKV;
    const T* curKBuf = validCurrentWindow ? currentKeyWindow->bufferAsT<T>() : nullptr;
    const T* curVBuf = validCurrentWindow ? currentValueWindow->bufferAsT<T>() : nullptr;
    const LongType cK0 = validCurrentWindow ? currentKeyWindow->strideAt(0) : 0;
    const LongType cK1 = validCurrentWindow ? currentKeyWindow->strideAt(1) : 0;
    const LongType cK2 = validCurrentWindow ? currentKeyWindow->strideAt(2) : 0;
    const LongType cK3 = validCurrentWindow ? currentKeyWindow->strideAt(3) : 0;
    const LongType cV0 = validCurrentWindow ? currentValueWindow->strideAt(0) : 0;
    const LongType cV1 = validCurrentWindow ? currentValueWindow->strideAt(1) : 0;
    const LongType cV2 = validCurrentWindow ? currentValueWindow->strideAt(2) : 0;
    const LongType cV3 = validCurrentWindow ? currentValueWindow->strideAt(3) : 0;

    // Attend only over the rows written so far. With a current window the cache prefix ends at
    // currentStart + currentSeq, which for the single decode query is also its causal bound, so
    // unwritten or stale rows past it are never read whatever the bias holds. Without a window the
    // query is the last cache row and the bias alone masks.
    const LongType maxKV = validCurrentWindow ? sd::math::sd_min<LongType>(currentStart + currentSeq, seqKV) : seqKV;

    const AccT masked = -DataTypeUtils::infOrMax<AccT>();
    const AccT scaleAcc = static_cast<AccT>(scale);

    auto func = PRAGMA_THREADS_FOR {
        std::vector<AccT> weights(maxKV);
        std::vector<AccT> acc(headDim);
        for (auto bq = start; bq < stop; ++bq) {
            const LongType b = bq / numQHeads;
            const LongType qHead = bq % numQHeads;
            const LongType kvHead = qHead / headsPerKvHead;

            const T* Q = qBuf + b * qS0 + qHead * qS2;
            const T* biasRow = biasBuf != nullptr ? biasBuf + b * bS0 + qHead * bS1 : nullptr;
            T* scoresRow = scoresBuf != nullptr ? scoresBuf + b * sS0 + qHead * sS1 : nullptr;
            T* logitsRow = logitsBuf != nullptr ? logitsBuf + b * lS0 + qHead * lS1 : nullptr;

            // Scores: Q·K * scale + bias
            AccT maxScore = masked;
            for (LongType kvIdx = 0; kvIdx < maxKV; ++kvIdx) {
                const LongType currentIndex = kvIdx - currentStart;
                const bool useCurrent = validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
                AccT dot = static_cast<AccT>(0);
                if (useCurrent) {
                    const T* kRow = curKBuf + b * cK0 + currentIndex * cK1 + kvHead * cK2;
                    for (LongType d = 0; d < headDim; ++d) {
                        dot += static_cast<AccT>(Q[d * qS3]) * static_cast<AccT>(kRow[d * cK3]);
                    }
                } else {
                    const int8_t* kRow = kBuf + b * kS0 + kvIdx * kS1 + kvHead * kS2;
                    for (LongType d = 0; d < headDim; ++d) {
                        dot += static_cast<AccT>(Q[d * qS3]) * static_cast<AccT>(kRow[d * kS3]);
                    }
                    dot *= static_cast<AccT>(quantisedRowScale(
                        kRow, headDim, inlineScales ? nullptr : ksBuf + b * ksS0 + kvIdx * ksS1 + kvHead * ksS2));
                }
                AccT score = dot * scaleAcc;
                if (biasRow != nullptr) score += static_cast<AccT>(biasRow[kvIdx * bS3]);
                weights[kvIdx] = score;
                if (logitsRow != nullptr) logitsRow[kvIdx * lS3] = static_cast<T>(score);
                if (score > maxScore) maxScore = score;
            }

            // Softmax numerators. A score equal to the mask sentinel (a -inf bias) gets weight 0,
            // so a fully masked row produces zeros instead of NaN.
            AccT sum = static_cast<AccT>(0);
            for (LongType kvIdx = 0; kvIdx < maxKV; ++kvIdx) {
                const AccT w = weights[kvIdx] == masked
                                   ? static_cast<AccT>(0)
                                   : sd::math::sd_exp_unclamped<AccT, AccT>(weights[kvIdx] - maxScore);
                weights[kvIdx] = w;
                sum += w;
            }
            const AccT invSum = sum > static_cast<AccT>(0) ? static_cast<AccT>(1) / sum : static_cast<AccT>(0);

            if (scoresRow != nullptr) {
                for (LongType kvIdx = 0; kvIdx < maxKV; ++kvIdx) {
                    scoresRow[kvIdx * sS3] = static_cast<T>(weights[kvIdx] * invSum);
                }
            }
            // Rows past the written prefix are never attended.
            for (LongType kvIdx = maxKV; kvIdx < seqKV; ++kvIdx) {
                if (logitsRow != nullptr) logitsRow[kvIdx * lS3] = static_cast<T>(masked);
                if (scoresRow != nullptr) scoresRow[kvIdx * sS3] = static_cast<T>(0);
            }

            // Weighted sum over V; the V row scale folds into the weight.
            for (LongType d = 0; d < headDim; ++d) acc[d] = static_cast<AccT>(0);
            for (LongType kvIdx = 0; kvIdx < maxKV; ++kvIdx) {
                const LongType currentIndex = kvIdx - currentStart;
                const bool useCurrent = validCurrentWindow && currentIndex >= 0 && currentIndex < currentSeq;
                if (useCurrent) {
                    const T* vRow = curVBuf + b * cV0 + currentIndex * cV1 + kvHead * cV2;
                    const AccT w = weights[kvIdx];
                    for (LongType d = 0; d < headDim; ++d) {
                        acc[d] += w * static_cast<AccT>(vRow[d * cV3]);
                    }
                } else {
                    const int8_t* vRow = vBuf + b * vS0 + kvIdx * vS1 + kvHead * vS2;
                    const AccT w = weights[kvIdx] * static_cast<AccT>(quantisedRowScale(
                        vRow, headDim, inlineScales ? nullptr : vsBuf + b * vsS0 + kvIdx * vsS1 + kvHead * vsS2));
                    for (LongType d = 0; d < headDim; ++d) {
                        acc[d] += w * static_cast<AccT>(vRow[d * vS3]);
                    }
                }
            }

            T* O = oBuf + b * oS0 + qHead * oS2;
            for (LongType d = 0; d < headDim; ++d) {
                O[d * oS3] = static_cast<T>(acc[d] * invSum);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, batch * numQHeads);
}

void fusedGQADecodeQuantisedCpu(
    NDArray* query,
    NDArray* quantKeyCache,
    NDArray* keyScaleCache,
    NDArray* quantValCache,
    NDArray* valScaleCache,
    NDArray* output,
    double scale,
    NDArray* attentionBias,
    LaunchContext* /*context*/,
    NDArray* currentKeyWindow,
    NDArray* currentValueWindow,
    const void* currentKvPosition,
    NDArray* attentionScores,
    NDArray* attentionLogits) {

    const DataType dtype = query->dataType();
    if (query->rankOf() != 4 || query->sizeAt(1) != 1) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: query must be [batch, 1, qHeads, headDim]");
    }
    const LongType batch     = query->sizeAt(0);
    const LongType numQHeads = query->sizeAt(2);
    const LongType headDim   = query->sizeAt(3);
    if (output->dataType() != dtype || output->rankOf() != 4 || output->sizeAt(0) != batch ||
        output->sizeAt(1) != 1 || output->sizeAt(2) != numQHeads || output->sizeAt(3) != headDim) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: output must match the query shape and dtype");
    }
    if (quantKeyCache->dataType() != DataType::INT8 || quantValCache->dataType() != DataType::INT8 ||
        quantKeyCache->rankOf() != 4 || !quantKeyCache->isSameShape(quantValCache)) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: key/value caches must be INT8 rank-4 with the same shape");
    }
    if (quantKeyCache->sizeAt(0) != batch) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: cache batch must match the query batch");
    }
    const LongType seqKV      = quantKeyCache->sizeAt(1);
    const LongType numKvHeads = quantKeyCache->sizeAt(2);
    if (numKvHeads <= 0 || numQHeads % numKvHeads != 0) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: qHeads must be a multiple of kvHeads");
    }

    const bool inlineScales = (keyScaleCache == nullptr);
    if (inlineScales != (valScaleCache == nullptr)) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: key and value scale caches must both be set or both be null");
    }
    if (inlineScales) {
        if (quantKeyCache->sizeAt(3) != headDim + 4 || quantKeyCache->strideAt(3) != 1 ||
            quantValCache->strideAt(3) != 1) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: row-inline caches must be [batch, seqKV, kvHeads, headDim+4] "
                            "with a unit last-dimension stride");
        }
    } else {
        if (quantKeyCache->sizeAt(3) != headDim) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: separate-scale cache last dim must equal headDim");
        }
        const std::vector<LongType> scaleShape = {batch, seqKV, numKvHeads};
        if (keyScaleCache->dataType() != DataType::FLOAT32 || valScaleCache->dataType() != DataType::FLOAT32 ||
            !keyScaleCache->isSameShape(scaleShape) || !valScaleCache->isSameShape(scaleShape)) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: scale caches must be FLOAT32 [batch, seqKV, kvHeads]");
        }
    }

    const bool hasWindow = currentKeyWindow != nullptr && currentValueWindow != nullptr;
    if (hasWindow) {
        if (currentKeyWindow->dataType() != dtype || currentValueWindow->dataType() != dtype ||
            currentKeyWindow->rankOf() != 4 || !currentKeyWindow->isSameShape(currentValueWindow) ||
            currentKeyWindow->sizeAt(0) != batch || currentKeyWindow->sizeAt(2) != numKvHeads ||
            currentKeyWindow->sizeAt(3) != headDim) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: current K/V windows must be [batch, seq, kvHeads, headDim] "
                            "in the query dtype");
        }
    }
    const bool hasBias = attentionBias != nullptr && !attentionBias->isEmpty();
    if (hasBias && attentionBias->dataType() != dtype) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: attentionBias must be in the query dtype");
    }
    // Same bias contract as fusedGQADecodeQuantisedCuda: a wide last dim must cover the whole
    // cache; the other dims broadcast or match exactly.
    if (hasBias) {
        const int biasRank = attentionBias->rankOf();
        auto broadcastsTo = [attentionBias](int dim, LongType full) {
            return attentionBias->sizeAt(dim) == 1 || attentionBias->sizeAt(dim) == full;
        };
        const LongType biasKv = biasRank == 0 ? 1 : attentionBias->sizeAt(biasRank - 1);
        bool biasFits = biasRank <= 4 && (biasKv == 1 || biasKv >= seqKV);
        if (biasRank == 4) {
            biasFits = biasFits && broadcastsTo(0, batch) && broadcastsTo(1, numQHeads) &&
                       attentionBias->sizeAt(2) == 1;
        } else if (biasRank == 3) {
            biasFits = biasFits && broadcastsTo(0, batch) && attentionBias->sizeAt(1) == 1;
        } else if (biasRank == 2) {
            biasFits = biasFits && attentionBias->sizeAt(0) == 1;
        }
        if (!biasFits) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: attentionBias must broadcast to [batch, qHeads, 1, seqKV]");
        }
    }
    // Null or empty aux outputs are not requested (the DSP executor passes empty placeholders for
    // dead outputs); requested ones must be the dpa_v2 score layout in the query dtype.
    NDArray* scoresOut = attentionScores != nullptr && !attentionScores->isEmpty() ? attentionScores : nullptr;
    NDArray* logitsOut = attentionLogits != nullptr && !attentionLogits->isEmpty() ? attentionLogits : nullptr;
    const std::vector<LongType> auxShape = {batch, numQHeads, 1, seqKV};
    for (NDArray* aux : {scoresOut, logitsOut}) {
        if (aux != nullptr && (aux->dataType() != dtype || !aux->isSameShape(auxShape))) {
            THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: attention scores/logits must be [batch, qHeads, 1, seqKV] "
                            "in the query dtype");
        }
    }

    if (batch == 0 || numQHeads == 0 || headDim == 0) return;

    BUILD_SINGLE_SELECTOR(dtype, fusedGQADecodeQuantisedCpu_,
                          (query, quantKeyCache, keyScaleCache, quantValCache, valScaleCache, output, scale,
                           attentionBias, hasWindow ? currentKeyWindow : nullptr,
                           hasWindow ? currentValueWindow : nullptr, currentKvPosition, scoresOut, logitsOut),
                          SD_FLOAT_TYPES);

    output->tickWriteHost();
    if (scoresOut != nullptr) scoresOut->tickWriteHost();
    if (logitsOut != nullptr) logitsOut->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif  // NOT_EXCLUDED(OP_kv_cache_quantize) || NOT_EXCLUDED(OP_kv_cache_dequantize) || NOT_EXCLUDED(OP_dot_product_attention_v2)
