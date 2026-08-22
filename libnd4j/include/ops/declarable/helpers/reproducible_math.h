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

#ifndef LIBND4J_REPRODUCIBLE_MATH_H
#define LIBND4J_REPRODUCIBLE_MATH_H

#include <math/templatemath.h>

namespace sd {
namespace ops {
namespace helpers {
namespace reproducible {

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT multiply(AccT left, AccT right) {
  volatile AccT rounded = sd::math::sd_multiply<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT add(AccT left, AccT right) {
  volatile AccT rounded = sd::math::sd_add<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT subtract(AccT left, AccT right) {
  volatile AccT rounded = sd::math::sd_subtract<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT divide(AccT left, AccT right) {
  volatile AccT rounded = sd::math::sd_divide<AccT, AccT, AccT>(left, right);
  return rounded;
}

/**
 * Cross-target exponential for recurrent kernels whose output feeds state.
 * Platform exp implementations are not required to agree in their last bits;
 * recurrent models can amplify that drift across timesteps. This routine keeps
 * range reduction and every arithmetic operation in the caller's framework
 * accumulator type with an explicit rounding boundary after each operation.
 */
template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT exp(AccT value) {
  if (sd::math::sd_isnan<AccT>(value)) return value;
  const AccT maxExp = static_cast<AccT>(88);
  const AccT minExp = static_cast<AccT>(-88);
  const AccT clamped = value > maxExp ? maxExp : (value < minExp ? minExp : value);
  const AccT log2e = static_cast<AccT>(1.442695040888963407359924681001892137L);
  const AccT ln2 = static_cast<AccT>(0.693147180559945309417232121458176568L);
  const AccT scaled = multiply<AccT>(clamped, log2e);
  const AccT bias = scaled >= static_cast<AccT>(0)
      ? static_cast<AccT>(0.5L) : static_cast<AccT>(-0.5L);
  const int exponent = static_cast<int>(add<AccT>(scaled, bias));
  const AccT reduced = subtract<AccT>(
      clamped, multiply<AccT>(static_cast<AccT>(exponent), ln2));

  AccT term = static_cast<AccT>(1);
  AccT sum = static_cast<AccT>(1);
  for (int order = 1; order <= 18; ++order) {
    term = divide<AccT>(
        multiply<AccT>(term, reduced), static_cast<AccT>(order));
    sum = add<AccT>(sum, term);
  }

  AccT scale = static_cast<AccT>(1);
  const AccT scaleStep = exponent >= 0 ? static_cast<AccT>(2) : static_cast<AccT>(0.5L);
  const int scaleCount = exponent >= 0 ? exponent : -exponent;
  for (int index = 0; index < scaleCount; ++index) {
    scale = multiply<AccT>(scale, scaleStep);
  }
  return multiply<AccT>(sum, scale);
}

}  // namespace reproducible
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_REPRODUCIBLE_MATH_H
