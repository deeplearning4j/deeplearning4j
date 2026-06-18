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
// @author Adam Gibson
// Helper functions for mixed precision training
//

#ifndef LIBND4J_MIXED_PRECISION_H
#define LIBND4J_MIXED_PRECISION_H

#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Scale gradients in-place by the given scale factor.
 * Used for loss scaling in mixed precision training.
 *
 * @param context Launch context for execution
 * @param gradients Array to scale (modified in-place)
 * @param scale Scale factor to apply
 */
SD_LIB_EXPORT void scaleGradients(sd::LaunchContext* context, NDArray* gradients, double scale);

/**
 * Check if any element in the array is inf or nan.
 * Used for dynamic loss scaling to detect gradient overflow.
 *
 * @param context Launch context for execution
 * @param arr Array to check
 * @return true if any element is inf/nan, false if all elements are finite
 */
SD_LIB_EXPORT bool hasInfOrNan(sd::LaunchContext* context, const NDArray* arr);

/**
 * Cast array to target type and scale in a single pass.
 * More efficient than separate cast + multiply operations.
 *
 * @param context Launch context for execution
 * @param input Source array
 * @param output Destination array (must be pre-allocated with target type)
 * @param scale Scale factor to apply during conversion
 */
SD_LIB_EXPORT void castAndScale(sd::LaunchContext* context, const NDArray* input,
                                 NDArray* output, double scale);

/**
 * Unscale gradients (divide by loss scale) and check for overflow.
 * Combines unscaling and overflow detection for efficiency.
 *
 * @param context Launch context for execution
 * @param gradients Array to unscale (modified in-place)
 * @param lossScale The loss scale factor to divide by
 * @return true if gradients are valid (all finite after unscaling), false if overflow detected
 */
SD_LIB_EXPORT bool unscaleGradientsAndCheck(sd::LaunchContext* context, NDArray* gradients,
                                             double lossScale);

/**
 * Update master weights from FP16 gradients.
 * Performs: masterWeight -= learningRate * (gradient / lossScale)
 *
 * @param context Launch context for execution
 * @param masterWeight FP32 master weight array (modified in-place)
 * @param gradient Gradient array (may be FP16)
 * @param learningRate Learning rate to apply
 * @param lossScale Loss scale factor for gradient unscaling
 */
SD_LIB_EXPORT void updateMasterWeight(sd::LaunchContext* context, NDArray* masterWeight,
                                       const NDArray* gradient, double learningRate,
                                       double lossScale);

/**
 * Sync master weights to compute weights (FP32 -> FP16/BF16).
 * Copies master weights to compute weights with type conversion.
 *
 * @param context Launch context for execution
 * @param masterWeight FP32 master weight source
 * @param computeWeight Target array (may be FP16/BF16)
 */
SD_LIB_EXPORT void syncMasterToCompute(sd::LaunchContext* context, const NDArray* masterWeight,
                                        NDArray* computeWeight);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_MIXED_PRECISION_H
