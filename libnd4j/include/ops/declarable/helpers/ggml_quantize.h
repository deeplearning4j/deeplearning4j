/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#ifndef LIBND4J_GGML_QUANTIZE_H
#define LIBND4J_GGML_QUANTIZE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Quantize a float tensor into raw GGML block bytes. The exact inverse of
 * helpers::ggmlDequantize for the supported types (Q4_0 uses this codebase's
 * adjacent-pair nibble packing, not upstream ggml's half-split).
 *
 * Supported quantType (GgmlQuantType): GGML_QUANT_Q4_0 (0), GGML_QUANT_Q8_0 (4).
 *
 * @param context    Launch context
 * @param input      FLOAT32 flat input; length must be a multiple of 32
 * @param output     Pre-allocated UINT8 1D output of the packed block bytes
 * @param quantType  GGML quant type enum value
 */
SD_LIB_HIDDEN void ggmlQuantize(LaunchContext* context, NDArray* input, NDArray* output, int quantType);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GGML_QUANTIZE_H
