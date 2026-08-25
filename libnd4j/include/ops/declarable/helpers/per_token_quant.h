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

#ifndef LIBND4J_PER_TOKEN_QUANT_H
#define LIBND4J_PER_TOKEN_QUANT_H

#include <array/NDArray.h>
#include <math/templatemath.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

// Canonical symmetric INT8 quantization policy shared by eager quantizers and
// compiled backends. The range is a storage-format property, never a
// model/calibration parameter.
static constexpr float SYMMETRIC_INT8_QUANT_MAX = 127.0f;

static SD_HOST_DEVICE SD_INLINE float symmetricInt8ScaleFromAbsMax(
    float absoluteMaximum) {
  return absoluteMaximum > 0.0f
             ? absoluteMaximum / SYMMETRIC_INT8_QUANT_MAX
             : 1.0f;
}

static SD_HOST_DEVICE SD_INLINE int8_t quantizeSymmetricInt8(
    float value, float scale) {
  float scaled = value / scale;
  scaled = sd::math::sd_max<float>(
      -SYMMETRIC_INT8_QUANT_MAX,
      sd::math::sd_min<float>(SYMMETRIC_INT8_QUANT_MAX, scaled));
  return static_cast<int8_t>(sd::math::sd_round<float, int>(scaled));
}

/**
 * Per-token FP8 quantization for activations.
 *
 * For each token (row), computes:
 *   scale[row] = max(|row_values|) / fp8_max
 *   quantized[row] = clamp(row_values / scale[row], -fp8_max, fp8_max)
 *
 * This is the companion kernel to FP8 GEMM: quantize activations on-the-fly
 * before each matmul for W8A8/FP8 inference pipelines.
 *
 * @param input      [numTokens, hiddenDim] float/half/bf16 activations
 * @param quantized  [numTokens, hiddenDim] output FP8 E4M3 quantized values (stored as int8)
 * @param scales     [numTokens] output per-token scale factors (float32)
 */
SD_LIB_HIDDEN void perTokenQuantFp8(LaunchContext* context,
                                     NDArray* input,
                                     NDArray* quantized,
                                     NDArray* scales);

/**
 * Per-token FP8 dequantization.
 *
 * output[row] = quantized[row] * scale[row]
 *
 * @param quantized  [numTokens, hiddenDim] FP8 E4M3 quantized values (stored as int8)
 * @param scales     [numTokens] per-token scale factors (float32)
 * @param output     [numTokens, hiddenDim] output float/half/bf16
 */
SD_LIB_HIDDEN void perTokenDequantFp8(LaunchContext* context,
                                       NDArray* quantized,
                                       NDArray* scales,
                                       NDArray* output);

/**
 * Per-token group quantization (INT8 or FP8).
 *
 * Divides the hidden dimension into groups of groupSize elements.
 * For each (token, group), computes an independent scale factor and quantizes.
 *
 * @param input      [numTokens, hiddenDim] input activations
 * @param quantized  [numTokens, hiddenDim] output quantized values (int8)
 * @param scales     [numTokens, numGroups] per-token-per-group scales (float32)
 * @param groupSize  number of elements per quantization group
 * @param useFp8     if true, quantize to FP8 E4M3 range; otherwise INT8 range
 */
SD_LIB_HIDDEN void perTokenGroupQuant(LaunchContext* context,
                                       NDArray* input,
                                       NDArray* quantized,
                                       NDArray* scales,
                                       int groupSize,
                                       bool useFp8);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_PER_TOKEN_QUANT_H
