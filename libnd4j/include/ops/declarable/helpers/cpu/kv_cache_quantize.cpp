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
template <typename T>
static void kvCacheQuantizeInt8Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    const LongType numRows = input->lengthOf() / input->sizeAt(-1);
    const LongType rowLen = input->sizeAt(-1);

    const T* x = input->bufferAsT<T>();
    int8_t* q = reinterpret_cast<int8_t*>(quantized->buffer());
    T* s = scales->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * rowLen;
            int8_t* qRow = q + row * rowLen;

            // Pass 1: find absmax
            T absMax = static_cast<T>(0);
            for (LongType i = 0; i < rowLen; ++i) {
                T absVal = sd::math::sd_abs<T, T>(xRow[i]);
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale
            T scale = absMax / static_cast<T>(127);
            if (scale == static_cast<T>(0)) scale = static_cast<T>(1);
            s[row] = scale;

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
void kvCacheQuantizeCpu(NDArray* input, NDArray* quantized, NDArray* scales, int quantFormat) {
    auto format = static_cast<KVQuantFormat>(quantFormat);

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
            THROW_EXCEPTION("kvCacheQuantizeCpu: unsupported quantization format");
    }

    quantized->tickWriteHost();
    quantized->syncToDevice();
    scales->tickWriteHost();
    scales->syncToDevice();
}

void kvCacheDequantizeCpu(NDArray* quantized, NDArray* scales, NDArray* output, int quantFormat) {
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
            THROW_EXCEPTION("kvCacheDequantizeCpu: unsupported quantization format");
    }

    output->tickWriteHost();
    output->syncToDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
