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
 * Last-axis RMS normalization. Input/output share a storage dtype; gamma may
 * independently be HALF, BFLOAT16, FLOAT32 or DOUBLE. Arithmetic is at least
 * FLOAT32 (DOUBLE when a DOUBLE operand participates), with one output cast.
 * Arbitrary strides and empty tensors are supported. Keep epsilon as double
 * across the helper ABI so DOUBLE computation does not inherit a FLOAT32 epsilon.
 */
SD_LIB_HIDDEN void rmsNorm(
    LaunchContext* context,
    NDArray* input,
    NDArray* gamma,
    NDArray* output,
    double epsilon);

/**
 * Fused RMSNorm + Linear: output = matmul(rmsNorm(input, gamma, eps), weight)
 *
 * Eliminates the intermediate normalized tensor from global memory.
 * On CUDA with M=1 (decode), uses a single fused kernel that computes
 * the normalization and matrix-vector product in one pass.
 * For M>1, uses normalization plus BLAS in computation precision. Mixed
 * storage weights are converted in bounded panels, not as a full matrix.
 *
 * @param input  [M, K] input tensor
 * @param gamma  [K] RMS norm scale weights
 * @param weight [K, N] linear projection weights
 * @param output [M, N] output tensor (pre-allocated)
 * @param epsilon RMS norm epsilon
 */
SD_LIB_HIDDEN void rmsNormLinear(
    LaunchContext* context,
    NDArray* input,
    NDArray* gamma,
    NDArray* weight,
    NDArray* output,
    double epsilon);

/**
 * Fused Skip (Residual Add) + RMS Normalization:
 *   hidden = input + skip [+ bias]
 *   output = hidden * rsqrt(mean(hidden^2) + eps) * gamma
 *
 * Eliminates the separate add kernel and intermediate global memory round-trip.
 *
 * @param input      [batch, ..., features]
 * @param skip       [batch, ..., features] (residual)
 * @param gamma      [features] RMS norm scale weights
 * @param bias       [features] optional bias in input dtype (may be nullptr)
 * @param output     [batch, ..., features] normalized output
 * @param hiddenOut  [batch, ..., features] optional pre-norm hidden states (may be nullptr)
 * @param epsilon    RMS norm epsilon
 */
SD_LIB_HIDDEN void skipRmsNorm(
    LaunchContext* context,
    NDArray* input,
    NDArray* skip,
    NDArray* gamma,
    NDArray* bias,
    NDArray* output,
    NDArray* hiddenOut,
    double epsilon);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_RMS_NORM_H
