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

#ifndef LIBND4J_RMS_NORM_H
#define LIBND4J_RMS_NORM_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Fused CPU RMS normalization
 * Computes: output = input / sqrt(mean(input^2) + epsilon) * gamma
 * using a 2-pass fused implementation with parallel_tad for optimal CPU performance.
 *
 * @param input Input tensor (must be contiguous row-major)
 * @param gamma Scale parameter (weight), can be nullptr
 * @param output Output tensor
 * @param epsilon Small constant for numerical stability
 */
SD_LIB_HIDDEN void rmsNormCpu(
    NDArray* input,
    NDArray* gamma,
    NDArray* output,
    float epsilon);

#if defined(SD_CUDA)
/**
 * Fused CUDA RMS normalization kernel
 * Computes: output = input / sqrt(mean(input^2) + epsilon) * gamma
 * in a single fused kernel for optimal GPU performance.
 *
 * @param input Input tensor
 * @param gamma Scale parameter (weight), can be nullptr
 * @param output Output tensor
 * @param epsilon Small constant for numerical stability
 * @param context Launch context
 */
SD_LIB_HIDDEN void rmsNormCuda(
    NDArray* input,
    NDArray* gamma,
    NDArray* output,
    float epsilon,
    LaunchContext* context);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_RMS_NORM_H
