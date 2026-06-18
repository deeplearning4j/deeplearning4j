/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#ifndef LIBND4J_HELPERS_ACTIVATION_MUL_H
#define LIBND4J_HELPERS_ACTIVATION_MUL_H

#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Fused SiLU (Swish) and element-wise multiply.
 *
 * Computes: output = silu(gate) * up = (gate * sigmoid(gate)) * up
 *
 * This is the core computation in the MLP block of LLaMA, Qwen, Gemma,
 * Mistral, and other SwiGLU-based architectures. Fusing the activation
 * with the multiply saves a global memory round-trip.
 *
 * @param gate    [batch, ..., dim] gate projection output
 * @param up      [batch, ..., dim] up projection output (same shape as gate)
 * @param output  [batch, ..., dim] result
 */
SD_LIB_HIDDEN void siluAndMul(LaunchContext* context,
                               NDArray* gate, NDArray* up, NDArray* output);

/**
 * Fused GELU and element-wise multiply.
 *
 * Computes: output = gelu(gate) * up
 *
 * Uses the exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * Used in GPT-NeoX, Phi, and other GEGLU-based architectures.
 *
 * @param gate    [batch, ..., dim] gate projection output
 * @param up      [batch, ..., dim] up projection output (same shape as gate)
 * @param output  [batch, ..., dim] result
 */
SD_LIB_HIDDEN void geluAndMul(LaunchContext* context,
                               NDArray* gate, NDArray* up, NDArray* output);

/**
 * Fused GELU (tanh approximation) and element-wise multiply.
 *
 * Computes: output = gelu_tanh(gate) * up
 *
 * Uses the tanh approximation:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * @param gate    [batch, ..., dim] gate projection output
 * @param up      [batch, ..., dim] up projection output (same shape as gate)
 * @param output  [batch, ..., dim] result
 */
SD_LIB_HIDDEN void geluTanhAndMul(LaunchContext* context,
                                   NDArray* gate, NDArray* up, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_ACTIVATION_MUL_H
