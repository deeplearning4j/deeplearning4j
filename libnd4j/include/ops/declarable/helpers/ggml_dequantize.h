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

#ifndef LIBND4J_GGML_DEQUANTIZE_H
#define LIBND4J_GGML_DEQUANTIZE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * GGML quant type enum — matches Java-side GGMLDequantize iArg values.
 * These are our own enum values, not GGML's internal type IDs.
 */
enum GgmlQuantType {
    GGML_QUANT_Q4_0 = 0,
    GGML_QUANT_Q4_1 = 1,
    GGML_QUANT_Q5_0 = 2,
    GGML_QUANT_Q5_1 = 3,
    GGML_QUANT_Q8_0 = 4,
    GGML_QUANT_Q8_1 = 5,
    GGML_QUANT_Q2_K = 6,
    GGML_QUANT_Q3_K = 7,
    GGML_QUANT_Q4_K = 8,
    GGML_QUANT_Q5_K = 9,
    GGML_QUANT_Q6_K = 10,
    GGML_QUANT_Q8_K = 11,
    GGML_QUANT_IQ2_XXS = 12,
    GGML_QUANT_IQ2_XS = 13,
    GGML_QUANT_IQ3_XXS = 14,
    GGML_QUANT_IQ1_S = 15,
    GGML_QUANT_IQ4_NL = 16,
    GGML_QUANT_IQ3_S = 17,
    GGML_QUANT_IQ2_S = 18,
    GGML_QUANT_IQ4_XS = 19,
    GGML_QUANT_IQ1_M = 20,
    GGML_QUANT_TQ1_0 = 21,
    GGML_QUANT_TQ2_0 = 22
};

/**
 * Output data type selector — matches Java-side GGMLDequantize iArg[1].
 */
enum GgmlDequantOutputType {
    GGML_DEQUANT_F32 = 0,
    GGML_DEQUANT_F16 = 1,
    GGML_DEQUANT_BF16 = 2
};

/**
 * Dequantize raw GGML quantized bytes to the target floating-point type.
 *
 * This is the standalone baseline kernel — no external dependencies.
 * When llamacpp is available, the platform impl overrides this.
 *
 * @param context     Launch context (for CUDA streams, etc.)
 * @param input       INT8 flat 1D array containing raw quantized bytes
 * @param output      Pre-allocated output array (F32, F16, or BF16) with target shape
 * @param quantType   GGML quant type (GgmlQuantType enum)
 */
SD_LIB_HIDDEN void ggmlDequantize(
    LaunchContext* context,
    NDArray* input,
    NDArray* output,
    int quantType);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GGML_DEQUANTIZE_H
