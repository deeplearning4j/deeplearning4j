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

#ifndef LIBND4J_SQUARED_RELU_H
#define LIBND4J_SQUARED_RELU_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Squared ReLU activation function.
 *
 * Forward:  out = max(0, x)^2
 * Backward: dIn = dOut * 2 * x * (x > 0)
 *
 * @param context  launch context
 * @param input    input NDArray
 * @param output   output NDArray (same shape as input)
 */
SD_LIB_HIDDEN void squaredRelu(LaunchContext* context, NDArray* input, NDArray* output);

/**
 * Squared ReLU backward pass.
 *
 * @param context  launch context
 * @param input    original forward input NDArray
 * @param dLdO     gradient w.r.t. output (upstream gradient)
 * @param dLdI     gradient w.r.t. input (to be computed, same shape as input)
 */
SD_LIB_HIDDEN void squaredReluBp(LaunchContext* context, NDArray* input, NDArray* dLdO, NDArray* dLdI);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_SQUARED_RELU_H
