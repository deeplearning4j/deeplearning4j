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
// Weight dequantization — CPU implementation.
// AWQ, GPTQ, and Marlin format support.
//

#include <execution/Threads.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/weight_dequant.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// AWQ INT4 group dequantization (CPU)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void awqDequantizeCpu_(NDArray* packedWeights, NDArray* scales,
                               NDArray* zeros, NDArray* output, int groupSize) {
    const LongType outFeatures = output->sizeAt(0);
    const LongType inFeatures = output->sizeAt(1);
    const LongType numGroups = (inFeatures + groupSize - 1) / groupSize;

    const uint8_t* packed = reinterpret_cast<const uint8_t*>(packedWeights->buffer());
    const T* scalesPtr = scales->bufferAsT<T>();
    const T* zerosPtr = zeros->bufferAsT<T>();
    T* outPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            for (LongType col = 0; col < inFeatures; ++col) {
                LongType group = col / groupSize;
                LongType packedIdx = row * (inFeatures / 2) + col / 2;
                uint8_t packedByte = packed[packedIdx];

                int intVal;
                if (col % 2 == 0) {
                    intVal = packedByte & 0x0F;
                } else {
                    intVal = (packedByte >> 4) & 0x0F;
                }

                float scale = static_cast<float>(scalesPtr[row * numGroups + group]);
                float zero = static_cast<float>(zerosPtr[row * numGroups + group]);
                float dequant = (static_cast<float>(intVal) - zero) * scale;

                outPtr[row * inFeatures + col] = static_cast<T>(dequant);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, outFeatures);
}

//////////////////////////////////////////////////////////////////////////////
// GPTQ dequantization (CPU)
//////////////////////////////////////////////////////////////////////////////
template <typename T>
static void gptqDequantizeCpu_(NDArray* packedWeights, NDArray* scales,
                                NDArray* zeros, NDArray* gIdx,
                                NDArray* output, int groupSize, int bits) {
    const LongType outFeatures = output->sizeAt(0);
    const LongType inFeatures = output->sizeAt(1);

    const uint8_t* packed = reinterpret_cast<const uint8_t*>(packedWeights->buffer());
    const T* scalesPtr = scales->bufferAsT<T>();
    const uint32_t* zerosPtr = (zeros != nullptr) ?
        reinterpret_cast<const uint32_t*>(zeros->buffer()) : nullptr;
    const int32_t* gIdxPtr = (gIdx != nullptr) ?
        reinterpret_cast<const int32_t*>(gIdx->buffer()) : nullptr;
    T* outPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; ++row) {
            for (LongType col = 0; col < inFeatures; ++col) {
                int group = (gIdxPtr != nullptr) ?
                    gIdxPtr[col] : static_cast<int>(col / groupSize);

                int intVal;
                if (bits == 4) {
                    LongType packedIdx = row * (inFeatures / 2) + col / 2;
                    uint8_t packedByte = packed[packedIdx];
                    if (col % 2 == 0) {
                        intVal = packedByte & 0x0F;
                    } else {
                        intVal = (packedByte >> 4) & 0x0F;
                    }
                } else {
                    intVal = static_cast<int>(
                        reinterpret_cast<const int8_t*>(packed)[row * inFeatures + col]);
                }

                float scale = static_cast<float>(scalesPtr[group * outFeatures + row]);

                float zero = 0.0f;
                if (zerosPtr != nullptr && bits == 4) {
                    LongType zeroWordIdx = group * (outFeatures / 8) + row / 8;
                    int zeroShift = static_cast<int>(row % 8) * 4;
                    zero = static_cast<float>((zerosPtr[zeroWordIdx] >> zeroShift) & 0x0F);
                }

                float dequant = (static_cast<float>(intVal) - zero) * scale;
                outPtr[row * inFeatures + col] = static_cast<T>(dequant);
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, outFeatures);
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
    BUILD_SINGLE_SELECTOR(output->dataType(), awqDequantizeCpu_,
                          (packedWeights, scales, zeros, output, groupSize),
                          SD_FLOAT_TYPES);

    output->tickWriteHost();
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
    BUILD_SINGLE_SELECTOR(output->dataType(), gptqDequantizeCpu_,
                          (packedWeights, scales, zeros, gIdx, output, groupSize, bits),
                          SD_FLOAT_TYPES);

    output->tickWriteHost();
}

//////////////////////////////////////////////////////////////////////////////
// Public: marlinGemm
// CPU reference: dequantize to FP32 then standard GEMM.
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

    const uint8_t* packed = reinterpret_cast<const uint8_t*>(marlinWeights->buffer());
    const float* scalesPtr = reinterpret_cast<const float*>(scales->buffer());
    const float* inPtr = input->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType numGroups = (groupSize > 0) ? (K + groupSize - 1) / groupSize : 1;

    // Dequantize weights and compute GEMM in one pass
    auto func = PRAGMA_THREADS_FOR {
        // Per-row of output
        for (auto m = start; m < stop; ++m) {
            for (LongType n = 0; n < N; ++n) {
                float acc = 0.0f;

                for (LongType k = 0; k < K; ++k) {
                    // Dequantize weight
                    LongType packedIdx = (k * N + n) / 2;
                    uint8_t packedByte = packed[packedIdx];
                    int intVal;
                    if ((k * N + n) % 2 == 0) {
                        intVal = packedByte & 0x0F;
                    } else {
                        intVal = (packedByte >> 4) & 0x0F;
                    }

                    int group = (groupSize > 0) ? static_cast<int>(k / groupSize) : 0;
                    float scale = scalesPtr[group * N + n];
                    float weight = (static_cast<float>(intVal) - 8.0f) * scale;

                    acc += inPtr[m * K + k] * weight;
                }

                outPtr[m * N + n] = acc;
            }
        }
    };
    samediff::Threads::parallel_tad(func, 0, M);

    output->tickWriteHost();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
