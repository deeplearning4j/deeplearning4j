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
#include <ops/declarable/helpers/kv_cache_quantize.h>
#include <math/templatemath.h>

#if NOT_EXCLUDED(OP_kv_cache_quantize)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// INT8 quantization: scale = max(abs(row)) / 127.0
//////////////////////////////////////////////////////////////////////////////
// ADR 0107 V2 ROW-INLINE: when `scales` is null the output is a row-inline tensor whose rows are
// rowLen+4 int8 elements — rowLen quantized values followed by that row's float32 scale (inside
// the logical tensor, so staging/copies preserve it). Non-null scales = legacy separate mode.
template <typename T>
static void kvCacheQuantizeInt8Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);
    const bool inlineScale = (scales == nullptr);
    const LongType outRowPitch = inlineScale ? rowLen + 4 : rowLen;

    const T* x = input->bufferAsT<T>();
    int8_t* q = reinterpret_cast<int8_t*>(quantized->buffer());
    T* s = inlineScale ? nullptr : scales->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            int8_t* qRow = q + row * outRowPitch;

            // Pass 1: find absmax
            T absMax = static_cast<T>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                T absVal = sd::math::sd_abs<T, T>(xRow[i]);
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale
            T scale = absMax / static_cast<T>(127);
            if (scale == static_cast<T>(0)) scale = static_cast<T>(1);
            if (inlineScale) {
                *reinterpret_cast<float*>(qRow + rowLen) = static_cast<float>(scale);
            } else {
                s[row] = scale;
            }

            T invScale = static_cast<T>(1) / scale;

            // Pass 2: quantize
            for (LongType i = 0; i < rowLen; ++i) {
                T val = xRow[i] * invScale;
                // Clamp to [-127, 127] and round
                val = sd::math::sd_max<T>(static_cast<T>(-127), sd::math::sd_min<T>(static_cast<T>(127), val));
                qRow[i] = static_cast<int8_t>(sd::math::sd_round<T, int>(val));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

//////////////////////////////////////////////////////////////////////////////
// INT8 dequantization: output = quantized * scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheDequantizeInt8Cpu_(NDArray* quantized, NDArray* scales, NDArray* output) {
    const LongType numRows = output->lengthOf() / output->sizeAt(-1);
    const LongType rowLen = output->sizeAt(-1);

    const int8_t* q = reinterpret_cast<const int8_t*>(quantized->buffer());
    const T* s = scales->bufferAsT<T>();
    T* z = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const int8_t* qRow = q + row * rowLen;
            T* zRow = z + row * rowLen;
            T scale = s[row];

            for (LongType i = 0; i < rowLen; ++i) {
                zRow[i] = static_cast<T>(qRow[i]) * scale;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

//////////////////////////////////////////////////////////////////////////////
// INT4 quantization: scale = max(abs(row)) / 7.0, pack 2 values per byte
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheQuantizeInt4Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    const T* x = input->bufferAsT<T>();
    uint8_t* q = reinterpret_cast<uint8_t*>(quantized->buffer());
    T* s = scales->bufferAsT<T>();

    // Packed row length: ceil(rowLen / 2)
    const LongType packedRowLen = (rowLen + 1) / 2;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            uint8_t* qRow = q + row * packedRowLen;

            // Pass 1: find absmax
            T absMax = static_cast<T>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                T absVal = sd::math::sd_abs<T, T>(xRow[i]);
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale
            T scale = absMax / static_cast<T>(7);
            if (scale == static_cast<T>(0)) scale = static_cast<T>(1);
            s[row] = scale;

            T invScale = static_cast<T>(1) / scale;

            // Pass 2: quantize and pack 2 values per byte
            for (LongType i = 0; i < rowLen; i += 2) {
                T val0 = xRow[i] * invScale;
                val0 = sd::math::sd_max<T>(static_cast<T>(-7), sd::math::sd_min<T>(static_cast<T>(7), val0));
                int8_t q0 = static_cast<int8_t>(sd::math::sd_round<T, int>(val0));

                int8_t q1 = 0;
                if (i + 1 < rowLen) {
                    T val1 = xRow[i + 1] * invScale;
                    val1 = sd::math::sd_max<T>(static_cast<T>(-7), sd::math::sd_min<T>(static_cast<T>(7), val1));
                    q1 = static_cast<int8_t>(sd::math::sd_round<T, int>(val1));
                }

                // Pack: low nibble = q0 + 8 (offset to unsigned), high nibble = q1 + 8
                qRow[i / 2] = static_cast<uint8_t>((q0 + 8) & 0x0F) | static_cast<uint8_t>(((q1 + 8) & 0x0F) << 4);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
}

//////////////////////////////////////////////////////////////////////////////
// INT4 dequantization: unpack and multiply by scale
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void kvCacheDequantizeInt4Cpu_(NDArray* quantized, NDArray* scales, NDArray* output) {
    const LongType numRows = output->lengthOf() / output->sizeAt(-1);
    const LongType rowLen = output->sizeAt(-1);

    const uint8_t* q = reinterpret_cast<const uint8_t*>(quantized->buffer());
    const T* s = scales->bufferAsT<T>();
    T* z = output->bufferAsT<T>();

    const LongType packedRowLen = (rowLen + 1) / 2;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const uint8_t* qRow = q + row * packedRowLen;
            T* zRow = z + row * rowLen;
            T scale = s[row];

            for (LongType i = 0; i < rowLen; i += 2) {
                uint8_t packed = qRow[i / 2];
                int8_t q0 = static_cast<int8_t>((packed & 0x0F)) - 8;
                int8_t q1 = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;

                zRow[i] = static_cast<T>(q0) * scale;
                if (i + 1 < rowLen) {
                    zRow[i + 1] = static_cast<T>(q1) * scale;
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numRows);
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

    // ADR 0107 V2 ROW-INLINE: null scales = row-inline mode (INT8 only) — the per-row float32
    // scale is written inside each rowLen+4 output row by kvCacheQuantizeInt8Cpu_.
    if (scales == nullptr && format != KVQuantFormat::INT8) {
        THROW_EXCEPTION("kvCacheQuantize: row-inline scale mode supports INT8 only");
    }

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
// Append one decode-step float K or V tensor into a pre-allocated INT8 cache
// buffer and write the per-token-per-head scale.
//
// Layout (INT8_KV):
//   newKv      : float  [batch, 1,        kvHeads, headDim]
//   quantCache : INT8   [batch, maxKvLen,  kvHeads, headDim]
//   scaleCache : float  [batch, maxKvLen,  kvHeads]
//   cachePosPtr: pointer to int64 write position (host ptr on CPU)
//
// Per-row (per KV head) absmax INT8 quantize + scatter into quantCache + scaleCache.
//////////////////////////////////////////////////////////////////////////////
void kvInPlaceWriteQuantisedBSHD(
    NDArray* quantCache,
    NDArray* scaleCache,
    NDArray* newKv,
    const void* cachePosPtr,
    LaunchContext* /*context*/) {

    // Read the cache position from the host pointer
    const LongType cachePos = *reinterpret_cast<const LongType*>(cachePosPtr);

    // newKv shape: [batch, 1, kvHeads, headDim]
    const LongType batch    = newKv->sizeAt(0);
    const LongType kvHeads  = newKv->sizeAt(2);
    const LongType headDim  = newKv->sizeAt(3);
    const LongType maxKvLen = quantCache->sizeAt(1);

    if (cachePos < 0 || cachePos >= maxKvLen) return;

    // ADR 0107 V2 ROW-INLINE: null scaleCache → cache last dim is headDim+4 and each row's
    // float32 scale sits at qRow+headDim (inside the logical tensor). Non-null = legacy separate.
    const bool inlineScale = (scaleCache == nullptr);
    const LongType rowPitch = quantCache->sizeAt(3);
    if (inlineScale && rowPitch != headDim + 4) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: row-inline cache last dim must equal headDim+4");
    }
    if (!inlineScale && rowPitch != headDim) {
        THROW_EXCEPTION("kvInPlaceWriteQuantisedBSHD: separate-scale cache last dim must equal headDim");
    }

    const float* srcBuf  = newKv->bufferAsT<float>();
    int8_t* qBuf         = reinterpret_cast<int8_t*>(quantCache->buffer());
    float* scaleBuf      = inlineScale ? nullptr : scaleCache->bufferAsT<float>();

    // Strides for quantCache [batch, maxKvLen, kvHeads, rowPitch]
    const LongType qcStride0 = maxKvLen * kvHeads * rowPitch;
    const LongType qcStride1 = kvHeads * rowPitch;
    const LongType qcStride2 = rowPitch;

    // Strides for scaleCache [batch, maxKvLen, kvHeads]
    const LongType scStride0 = maxKvLen * kvHeads;
    const LongType scStride1 = kvHeads;

    // Strides for newKv [batch, 1, kvHeads, headDim] — contiguous
    const LongType nkStride0 = kvHeads * headDim;
    const LongType nkStride2 = headDim;

    auto func = PRAGMA_THREADS_FOR {
        for (auto bh = start; bh < stop; ++bh) {
            const LongType b  = bh / kvHeads;
            const LongType h  = bh % kvHeads;

            const float* srcRow = srcBuf + b * nkStride0 + h * nkStride2;
            int8_t* qRow = qBuf + b * qcStride0 + cachePos * qcStride1 + h * qcStride2;
            float* sPtr  = inlineScale
                ? reinterpret_cast<float*>(qRow + headDim)
                : scaleBuf + b * scStride0 + cachePos * scStride1 + h;

            // Pass 1: absmax
            float absMax = 0.0f;
            for (LongType d = 0; d < headDim; ++d) {
                float v = srcRow[d];
                float av = v < 0.0f ? -v : v;
                if (av > absMax) absMax = av;
            }

            float scale = absMax / 127.0f;
            if (scale == 0.0f) scale = 1.0f;
            *sPtr = scale;
            float invScale = 1.0f / scale;

            // Pass 2: quantize + scatter
            for (LongType d = 0; d < headDim; ++d) {
                float v = srcRow[d] * invScale;
                if (v > 127.0f) v = 127.0f;
                if (v < -127.0f) v = -127.0f;
                qRow[d] = static_cast<int8_t>(v + (v >= 0.0f ? 0.5f : -0.5f));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, batch * kvHeads);

    quantCache->tickWriteHost();
    if (scaleCache != nullptr) scaleCache->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// V2: fusedGQADecodeQuantisedCpu — CPU reference GQA decode with INT8 K/V
// Dequantises on-the-fly: float(int8) * scale, then standard Q@K^T+softmax+V.
// Used as correctness reference; not the performance path (no SIMD optimisation).
//////////////////////////////////////////////////////////////////////////////
void fusedGQADecodeQuantisedCpu(
    NDArray* query,
    NDArray* quantKeyCache,
    NDArray* keyScaleCache,
    NDArray* quantValCache,
    NDArray* valScaleCache,
    NDArray* output,
    double scale,
    NDArray* attentionBias,
    LaunchContext* /*context*/) {

    // query   : float [batch, 1, qHeads, headDim]
    // quantKC : INT8  [batch, seqKV, kvHeads, headDim]
    // keyScale: float [batch, seqKV, kvHeads]
    // quantVC : INT8  [batch, seqKV, kvHeads, headDim]
    // valScale: float [batch, seqKV, kvHeads]
    // output  : float [batch, 1, qHeads, headDim]

    const LongType batch       = query->sizeAt(0);
    const LongType qHeads      = query->sizeAt(2);
    const LongType headDim     = query->sizeAt(3);
    const LongType seqKV       = quantKeyCache->sizeAt(1);
    const LongType kvHeads     = quantKeyCache->sizeAt(2);
    const LongType headsPerKvH = qHeads / kvHeads;

    // ADR 0107 V2 ROW-INLINE: null scale caches → the INT8 caches are [batch,seqKV,kvHeads,headDim+4]
    // with each row's float32 scale at row+headDim (inside the logical tensor). Non-null = legacy.
    const bool inlineScales = (keyScaleCache == nullptr);
    const LongType rowPitch = quantKeyCache->sizeAt(3);
    if (inlineScales && rowPitch != headDim + 4) {
        THROW_EXCEPTION("fusedGQADecodeQuantisedCpu: row-inline cache last dim must equal headDim+4");
    }

    const float*  qBuf    = query->bufferAsT<float>();
    const int8_t* kBuf    = reinterpret_cast<const int8_t*>(quantKeyCache->buffer());
    const float*  kScale  = inlineScales ? nullptr : keyScaleCache->bufferAsT<float>();
    const int8_t* vBuf    = reinterpret_cast<const int8_t*>(quantValCache->buffer());
    const float*  vScale  = inlineScales ? nullptr : valScaleCache->bufferAsT<float>();
    float*        oBuf    = output->bufferAsT<float>();
    const float*  biasBuf = (attentionBias != nullptr && !attentionBias->isEmpty())
                            ? attentionBias->bufferAsT<float>() : nullptr;

    // strides (K/V rows are rowPitch elements apart; values occupy [0, headDim) of each row)
    const LongType qS0 = qHeads * headDim,  qS2 = headDim;
    const LongType kS0 = seqKV * kvHeads * rowPitch, kS1 = kvHeads * rowPitch, kS2 = rowPitch;
    const LongType ksS0 = seqKV * kvHeads, ksS1 = kvHeads;
    const LongType oS0 = qHeads * headDim,  oS2 = headDim;

    // bias strides [batch, 1_or_qH, 1_or_seqKV_unused, seqKV]
    LongType biasS0 = 0, biasS1 = 0, biasS3 = 1;
    if (biasBuf != nullptr) {
        biasS0 = attentionBias->sizeAt(0) > 1 ? attentionBias->strideAt(0) : 0;
        biasS1 = attentionBias->sizeAt(1) > 1 ? attentionBias->strideAt(1) : 0;
        biasS3 = attentionBias->strideAt(attentionBias->rankOf() - 1);
    }

    // Temporary buffers for scores
    std::vector<float> scores(seqKV);

    for (LongType b = 0; b < batch; ++b) {
        for (LongType qh = 0; qh < qHeads; ++qh) {
            const LongType kvh = qh / headsPerKvH;

            const float* Q  = qBuf + b * qS0 + qh * qS2;
            float*       O  = oBuf + b * oS0 + qh * oS2;
            const float* biasRow = biasBuf ? (biasBuf + b * biasS0 + qh * biasS1) : nullptr;

            // Compute Q @ K^T scores
            float maxScore = -1e38f;
            for (LongType t = 0; t < seqKV; ++t) {
                const int8_t* Krow  = kBuf   + b * kS0 + t * kS1 + kvh * kS2;
                const float   ksc   = kScale != nullptr
                    ? kScale[b * ksS0 + t * ksS1 + kvh]
                    : *reinterpret_cast<const float*>(Krow + headDim);
                float dot = 0.0f;
                for (LongType d = 0; d < headDim; ++d) {
                    float kval = static_cast<float>(Krow[d]) * ksc;
                    dot += Q[d] * kval;
                }
                dot *= static_cast<float>(scale);
                if (biasRow != nullptr) dot += biasRow[t * biasS3];
                scores[t] = dot;
                if (dot > maxScore) maxScore = dot;
            }

            // Softmax
            float sumExp = 0.0f;
            for (LongType t = 0; t < seqKV; ++t) {
                float e = expf(scores[t] - maxScore);
                scores[t] = e;
                sumExp += e;
            }
            float invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;

            // Weighted sum over V
            for (LongType d = 0; d < headDim; ++d) O[d] = 0.0f;
            for (LongType t = 0; t < seqKV; ++t) {
                float w = scores[t] * invSum;
                const int8_t* Vrow = vBuf   + b * kS0 + t * kS1 + kvh * kS2;
                const float   vsc  = vScale != nullptr
                    ? vScale[b * ksS0 + t * ksS1 + kvh]
                    : *reinterpret_cast<const float*>(Vrow + headDim);
                for (LongType d = 0; d < headDim; ++d) {
                    float vval = static_cast<float>(Vrow[d]) * vsc;
                    O[d] += w * vval;
                }
            }
        }
    }

    output->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif  // NOT_EXCLUDED(OP_kv_cache_quantize)

// NOTE: the KvScaleRegistry (setKvScaleRegistry / clearKvScaleRegistry /
// lookupKvScaleByCache + tl_kvScaleRegistry) is defined INLINE in
// helpers/kv_cache_quantize.h so it links into both the CPU and CUDA backends.
// It previously lived here (CPU-only TU), which left the CUDA link with undefined
// references from autoregressive_decode.cu / dot_product_attention_v2.cpp.
