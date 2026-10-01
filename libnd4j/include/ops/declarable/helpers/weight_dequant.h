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

#ifndef LIBND4J_WEIGHT_DEQUANT_H
#define LIBND4J_WEIGHT_DEQUANT_H

#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <helpers/ShapeUtils.h>
#include <system/common.h>

#include <string>

namespace sd {
namespace ops {
namespace helpers {

/**
 * AWQ-style group dequantization of packed low-bit weights.
 *
 * AWQ (Activation-aware Weight Quantization) stores each weight as a
 * numBits-wide unsigned code with a per-group scale and zero point:
 *   output[n, k] = (code(n, k) - zero[n, k / groupSize]) * scale[n, k / groupSize]
 *
 * Each byte holds 8 / numBits codes along k: code j of byte (n, c) is element
 * k = c * (8 / numBits) + j and sits in bits [j * numBits, (j + 1) * numBits).
 * Zero points are in code units; without them the zero point is the middle of
 * the code range, 2^(numBits - 1). Every operand is addressed through its
 * strides, so transposed views need no copy. Values are computed in the
 * aggregate type of the output.
 *
 * @param context       launch context
 * @param packedWeights [outFeatures, ceil(inFeatures * numBits / 8)] one-byte integer codes
 * @param scales        [outFeatures, numGroups] floating per-group scales
 * @param zeros         [outFeatures, numGroups] per-group zero points of scales' type (may be nullptr)
 * @param output        [outFeatures, inFeatures] floating dequantized weights
 * @param groupSize     input features per group
 * @param numBits       bits per code: 1, 2, 4 or 8
 */
SD_LIB_HIDDEN void awqDequantize(LaunchContext* context,
                                  NDArray* packedWeights,
                                  NDArray* scales,
                                  NDArray* zeros,
                                  NDArray* output,
                                  int groupSize,
                                  int numBits);

// Code slot of a packed byte: bits [slot * numBits, (slot + 1) * numBits).
SD_HOST_DEVICE SD_INLINE int awqCode(uint8_t packed, int slot, int numBits) {
  return (packed >> (slot * numBits)) & ((1 << numBits) - 1);
}

// Checks the operands of awqDequantize against the layout documented above.
SD_INLINE void awqDequantizeCheck(NDArray* packedWeights, NDArray* scales, NDArray* zeros, NDArray* output,
                                  int groupSize, int numBits) {
  if (numBits <= 0 || 8 % numBits != 0 || groupSize <= 0) {
    const std::string message = "awqDequantize: numBits must divide 8 and groupSize must be positive, got numBits " +
                                std::to_string(numBits) + " and groupSize " + std::to_string(groupSize);
    THROW_EXCEPTION(message.c_str());
  }
  if (!DataTypeUtils::isZ(packedWeights->dataType()) || packedWeights->sizeOfT() != 1) {
    const std::string message = "awqDequantize: packed weights must hold one-byte integer codes, got " +
                                DataTypeUtils::asString(packedWeights->dataType());
    THROW_EXCEPTION(message.c_str());
  }
  if (!DataTypeUtils::isR(scales->dataType()) || !DataTypeUtils::isR(output->dataType()) ||
      (zeros != nullptr && zeros->dataType() != scales->dataType())) {
    const std::string message = "awqDequantize: scales and output must be floating and zeros of the scales' type, got " +
                                DataTypeUtils::asString(scales->dataType()) + ", " +
                                DataTypeUtils::asString(output->dataType()) +
                                (zeros != nullptr ? " and " + DataTypeUtils::asString(zeros->dataType()) : "");
    THROW_EXCEPTION(message.c_str());
  }
  const int codesPerByte = 8 / numBits;
  bool consistent = packedWeights->rankOf() == 2 && scales->rankOf() == 2 && output->rankOf() == 2;
  if (consistent) {
    const LongType outFeatures = output->sizeAt(0);
    const LongType inFeatures = output->sizeAt(1);
    consistent = packedWeights->sizeAt(0) == outFeatures &&
                 packedWeights->sizeAt(1) == (inFeatures + codesPerByte - 1) / codesPerByte &&
                 scales->sizeAt(0) == outFeatures && scales->sizeAt(1) == (inFeatures + groupSize - 1) / groupSize &&
                 (zeros == nullptr || zeros->isSameShape(scales));
  }
  if (!consistent) {
    const std::string message = "awqDequantize: inconsistent shapes: packed " + ShapeUtils::shapeAsString(packedWeights) +
                                ", scales " + ShapeUtils::shapeAsString(scales) +
                                (zeros != nullptr ? ", zeros " + ShapeUtils::shapeAsString(zeros) : "") + ", output " +
                                ShapeUtils::shapeAsString(output) + " for numBits " + std::to_string(numBits) +
                                " and groupSize " + std::to_string(groupSize);
    THROW_EXCEPTION(message.c_str());
  }
}

/**
 * GPTQ-style INT4 dequantization with permutation.
 *
 * GPTQ (Generative Pre-Training Quantization) uses INT4 with per-group
 * quantization and an optional column permutation (g_idx) that maps each
 * input column to its quantization group.
 *
 *   weight_fp = (packed_int4 - zero[g_idx[col]]) * scale[g_idx[col]]
 *
 * @param context       launch context
 * @param packedWeights [outFeatures, inFeatures / 2] packed INT4 pairs (uint8)
 * @param scales        [numGroups, outFeatures] per-group scales
 * @param zeros         [numGroups, outFeatures / 8] packed zero-points
 * @param gIdx          [inFeatures] group index mapping (may be nullptr for sequential)
 * @param output        [outFeatures, inFeatures] dequantized output
 * @param groupSize     elements per group (typically 128)
 * @param bits          quantization bits (4 or 8)
 */
SD_LIB_HIDDEN void gptqDequantize(LaunchContext* context,
                                   NDArray* packedWeights,
                                   NDArray* scales,
                                   NDArray* zeros,
                                   NDArray* gIdx,
                                   NDArray* output,
                                   int groupSize,
                                   int bits);

/**
 * Marlin-style INT4xFP16 fused dequantize + GEMM.
 *
 * Marlin format pre-permutes weights for efficient GPU memory access
 * and fuses dequantization with GEMM for maximum throughput.
 * Achieves near-FP16 speed at 4-bit precision by overlapping
 * dequant with tensor core math in a single kernel.
 *
 * output = input_fp16 @ dequant(marlinWeights, scales)
 *
 * @param context       launch context
 * @param input         [M, K] FP16 input activations
 * @param marlinWeights [K/16, N*16/8] Marlin-packed INT4 weights
 * @param scales        [numGroups, N] per-group scale factors
 * @param output        [M, N] FP16 output
 * @param workspace     [N/128 * 16] workspace for reduction (pre-allocated)
 * @param groupSize     elements per quantization group (-1 for per-channel)
 */
SD_LIB_HIDDEN void marlinGemm(LaunchContext* context,
                                NDArray* input,
                                NDArray* marlinWeights,
                                NDArray* scales,
                                NDArray* output,
                                NDArray* workspace,
                                int groupSize);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_WEIGHT_DEQUANT_H
