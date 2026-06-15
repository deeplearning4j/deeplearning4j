/* ******************************************************************************
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

#ifndef LIBND4J_LAYER_NORM_H
#define LIBND4J_LAYER_NORM_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

#if defined(SD_CUDA)
/**
 * Fused CUDA layer normalization kernel
 * Computes: output = (input - mean) / sqrt(variance + epsilon) * gain + bias
 * in a single fused kernel for optimal GPU performance
 *
 * @param input Input tensor
 * @param gain Scale parameter (gamma)
 * @param bias Shift parameter (beta), can be nullptr
 * @param output Output tensor
 * @param axis Dimensions to normalize over (must be last dimension for CUDA path)
 * @param epsilon Small constant for numerical stability
 * @param context Launch context
 */
SD_LIB_HIDDEN void layerNormCuda(
    NDArray* input,
    NDArray* gain,
    NDArray* bias,
    NDArray* output,
    const std::vector<LongType>& axis,
    float epsilon,
    LaunchContext* context);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LAYER_NORM_H
