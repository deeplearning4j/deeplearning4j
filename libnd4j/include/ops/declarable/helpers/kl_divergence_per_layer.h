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
// Per-layer KL divergence helper.
// Used by the adaptive GGUF quantization pipeline to measure how much a
// quantized layer output diverges from its full-precision counterpart.
//
// KL(P||Q) = sum_i P_i * log(P_i / Q_i)
// where P = softmax(referenceLogits / temperature)
//       Q = softmax(quantizedLogits / temperature)
//

#ifndef LIBND4J_HELPERS_KL_DIVERGENCE_PER_LAYER_H
#define LIBND4J_HELPERS_KL_DIVERGENCE_PER_LAYER_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Compute mean per-sample KL divergence between reference and quantized logits.
 *
 * Both inputs must have shape [batch, seq, dim] or [batch, dim] (any rank >= 2;
 * all dimensions except the last are collapsed into a batch dimension internally).
 *
 * @param referenceLogits  full-precision logits [batch, ..., dim]
 * @param quantizedLogits  quantized-then-dequantized logits [batch, ..., dim]
 * @param output           scalar output — mean KL divergence over all rows
 * @param temperature      softmax temperature (default 1.0)
 * @param context          launch context
 */
SD_LIB_HIDDEN void klDivergencePerLayer(NDArray* referenceLogits,
                                         NDArray* quantizedLogits,
                                         NDArray* output,
                                         double temperature,
                                         LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_KL_DIVERGENCE_PER_LAYER_H
