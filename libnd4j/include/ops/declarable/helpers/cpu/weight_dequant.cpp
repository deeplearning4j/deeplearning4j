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
#include <ops/op_types.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////////
// AWQ group dequantization (CPU)
//////////////////////////////////////////////////////////////////////////////
// One output row per work item. Every operand is addressed through its own
// strides, so transposed or sliced views need no copy; values are computed in
// the aggregate type of the output and converted once.
template <typename S, typename Z>
static void awqDequantizeCpu_(NDArray* packedWeights, NDArray* scales, NDArray* zeros, NDArray* output,
                              int groupSize, int numBits) {
  using AccT = typename simdOps::AggregateType<Z>::type;
  const LongType outFeatures = output->sizeAt(0);
  const LongType inFeatures = output->sizeAt(1);
  const int codesPerByte = 8 / numBits;
  const AccT midpoint = static_cast<AccT>(1 << (numBits - 1));

  const uint8_t* packed = static_cast<const uint8_t*>(packedWeights->buffer());
  const S* scale = scales->bufferAsT<S>();
  const S* zero = zeros != nullptr ? zeros->bufferAsT<S>() : nullptr;
  Z* out = output->bufferAsT<Z>();
  const LongType* packedStrides = packedWeights->stridesOf();
  const LongType* scaleStrides = scales->stridesOf();
  const LongType* zeroStrides = zeros != nullptr ? zeros->stridesOf() : nullptr;
  const LongType* outStrides = output->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto n = start; n < stop; n += increment) {
      for (LongType k = 0; k < inFeatures; ++k) {
        const LongType group = k / groupSize;
        const uint8_t byte = packed[n * packedStrides[0] + (k / codesPerByte) * packedStrides[1]];
        const AccT code = static_cast<AccT>(awqCode(byte, static_cast<int>(k % codesPerByte), numBits));
        const AccT groupScale = static_cast<AccT>(scale[n * scaleStrides[0] + group * scaleStrides[1]]);
        const AccT groupZero =
            zero != nullptr ? static_cast<AccT>(zero[n * zeroStrides[0] + group * zeroStrides[1]]) : midpoint;
        out[n * outStrides[0] + k * outStrides[1]] = static_cast<Z>((code - groupZero) * groupScale);
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, outFeatures);
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
void awqDequantize(LaunchContext* context, NDArray* packedWeights, NDArray* scales, NDArray* zeros, NDArray* output,
                   int groupSize, int numBits) {
  awqDequantizeCheck(packedWeights, scales, zeros, output, groupSize, numBits);
  if (output->isEmpty()) return;
  NDArray::preparePrimaryUse({output}, {packedWeights, scales, zeros});
  BUILD_DOUBLE_SELECTOR(scales->dataType(), output->dataType(), awqDequantizeCpu_,
                        (packedWeights, scales, zeros, output, groupSize, numBits), SD_FLOAT_TYPES, SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({output}, {packedWeights, scales, zeros});
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

    if (input->dataType() != DataType::FLOAT32) {
        THROW_EXCEPTION("marlinGemm (CPU): input must be FLOAT32 — got a different dtype; "
                        "the CPU reference path operates in float32 only (CUDA supports FLOAT32 and HALF)");
    }
    if (scales->dataType() != DataType::FLOAT32) {
        THROW_EXCEPTION("marlinGemm (CPU): scales must be FLOAT32 — got a different dtype");
    }
    if (output->dataType() != DataType::FLOAT32) {
        THROW_EXCEPTION("marlinGemm (CPU): output must be FLOAT32 — got a different dtype");
    }

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
