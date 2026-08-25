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
// Per-token FP8 quantization/dequantization and group quantization — CPU impl.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/per_token_quant.h>

namespace sd {
namespace ops {
namespace helpers {

static constexpr float FP8_E4M3_MAX = 448.0f;

//////////////////////////////////////////////////////////////////////////////
// Per-token FP8 quantization (CPU)
// For each row: scale = max(|row|) / FP8_MAX, quantized = clamp(row / scale)
// On CPU, FP8 values stored as int8 bytes (no HW FP8 support).
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void perTokenQuantFp8Cpu_(NDArray* input, NDArray* quantized, NDArray* scales) {
    const LongType numTokens = input->sizeAt(0);
    const LongType hiddenDim = input->sizeAt(1);

    const T* x = input->bufferAsT<T>();
    int8_t* q = reinterpret_cast<int8_t*>(quantized->buffer());
    float* s = reinterpret_cast<float*>(scales->buffer());

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * hiddenDim;
            int8_t* qRow = q + row * hiddenDim;

            // Pass 1: find absmax
            float absMax = 0.0f;
            for (LongType i = 0; i < hiddenDim; ++i) {
                float absVal = sd::math::sd_abs<float, float>(static_cast<float>(xRow[i]));
                if (absVal > absMax) absMax = absVal;
            }

            // Compute scale: scale = absMax / FP8_MAX
            float scale = absMax / FP8_E4M3_MAX;
            if (scale == 0.0f) scale = 1.0f;
            s[row] = scale;

            float invScale = 1.0f / scale;

            // Pass 2: quantize to int8 range representing FP8 values
            for (LongType i = 0; i < hiddenDim; ++i) {
                float val = static_cast<float>(xRow[i]) * invScale;
                val = sd::math::sd_max<float>(-FP8_E4M3_MAX, sd::math::sd_min<float>(FP8_E4M3_MAX, val));
                // Map to int8 range: val / FP8_MAX * 127
                float mapped = val * (SYMMETRIC_INT8_QUANT_MAX / FP8_E4M3_MAX);
                qRow[i] = static_cast<int8_t>(sd::math::sd_round<float, int>(mapped));
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numTokens);
}

//////////////////////////////////////////////////////////////////////////////
// Per-token FP8 dequantization (CPU)
// output[row] = quantized[row] * scale[row]
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void perTokenDequantFp8Cpu_(NDArray* quantized, NDArray* scales, NDArray* output) {
    const LongType numTokens = output->sizeAt(0);
    const LongType hiddenDim = output->sizeAt(1);

    const int8_t* q = reinterpret_cast<const int8_t*>(quantized->buffer());
    const float* s = reinterpret_cast<const float*>(scales->buffer());
    T* z = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const int8_t* qRow = q + row * hiddenDim;
            T* zRow = z + row * hiddenDim;
            float scale = s[row];

            for (LongType i = 0; i < hiddenDim; ++i) {
                // Reverse the mapping: int8 -> FP8 range -> original
                float fp8Val = static_cast<float>(qRow[i]) *
                               (FP8_E4M3_MAX / SYMMETRIC_INT8_QUANT_MAX);
                zRow[i] = static_cast<T>(fp8Val * scale);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numTokens);
}

//////////////////////////////////////////////////////////////////////////////
// Per-token group quantization (CPU)
// Divides hidden dim into groups, each with independent scale.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void perTokenGroupQuantCpu_(NDArray* input, NDArray* quantized, NDArray* scales,
                                    int groupSize, bool useFp8) {
    const LongType numTokens = input->sizeAt(0);
    const LongType hiddenDim = input->sizeAt(1);
    const LongType numGroups = (hiddenDim + groupSize - 1) / groupSize;

    const T* x = input->bufferAsT<T>();
    int8_t* q = reinterpret_cast<int8_t*>(quantized->buffer());
    float* s = reinterpret_cast<float*>(scales->buffer());

    const float maxRange = useFp8 ? FP8_E4M3_MAX : SYMMETRIC_INT8_QUANT_MAX;

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            const T* xRow = x + row * hiddenDim;
            int8_t* qRow = q + row * hiddenDim;
            float* sRow = s + row * numGroups;

            for (LongType g = 0; g < numGroups; ++g) {
                LongType gStart = g * groupSize;
                LongType gEnd = sd::math::sd_min<LongType>(gStart + groupSize, hiddenDim);

                // Find group absmax
                float absMax = 0.0f;
                for (LongType i = gStart; i < gEnd; ++i) {
                    float absVal = sd::math::sd_abs<float, float>(static_cast<float>(xRow[i]));
                    if (absVal > absMax) absMax = absVal;
                }

                float scale = useFp8
                                  ? (absMax > 0.0f ? absMax / maxRange : 1.0f)
                                  : symmetricInt8ScaleFromAbsMax(absMax);
                sRow[g] = scale;

                float invScale = 1.0f / scale;

                // Quantize group
                for (LongType i = gStart; i < gEnd; ++i) {
                    if (useFp8) {
                        float val = static_cast<float>(xRow[i]) * invScale;
                        val = sd::math::sd_max<float>(
                            -SYMMETRIC_INT8_QUANT_MAX,
                            sd::math::sd_min<float>(SYMMETRIC_INT8_QUANT_MAX, val));
                        qRow[i] = static_cast<int8_t>(
                            sd::math::sd_round<float, int>(val));
                    } else {
                        qRow[i] = quantizeSymmetricInt8(
                            static_cast<float>(xRow[i]), scale);
                    }
                }
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, numTokens);
}

//////////////////////////////////////////////////////////////////////////////
// Public interface
//////////////////////////////////////////////////////////////////////////////
void perTokenQuantFp8(LaunchContext* context,
                       NDArray* input,
                       NDArray* quantized,
                       NDArray* scales) {
    BUILD_SINGLE_SELECTOR(input->dataType(), perTokenQuantFp8Cpu_, (input, quantized, scales), SD_FLOAT_TYPES);
    quantized->tickWriteHost();
    scales->tickWriteHost();
}

void perTokenDequantFp8(LaunchContext* context,
                         NDArray* quantized,
                         NDArray* scales,
                         NDArray* output) {
    BUILD_SINGLE_SELECTOR(output->dataType(), perTokenDequantFp8Cpu_, (quantized, scales, output), SD_FLOAT_TYPES);
    output->tickWriteHost();
}

void perTokenGroupQuant(LaunchContext* context,
                         NDArray* input,
                         NDArray* quantized,
                         NDArray* scales,
                         int groupSize,
                         bool useFp8) {
    BUILD_SINGLE_SELECTOR(input->dataType(), perTokenGroupQuantCpu_,
                          (input, quantized, scales, groupSize, useFp8), SD_FLOAT_TYPES);
    quantized->tickWriteHost();
    scales->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
