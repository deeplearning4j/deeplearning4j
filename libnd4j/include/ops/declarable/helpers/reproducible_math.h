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

#include <cstdint>
#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {
namespace reproducible {

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT multiply(AccT left, AccT right) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<AccT, float>::value) return __fmul_rn(left, right);
  if constexpr (std::is_same<AccT, double>::value) return __dmul_rn(left, right);
#endif
  volatile AccT rounded = sd::math::sd_multiply<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT add(AccT left, AccT right) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<AccT, float>::value) return __fadd_rn(left, right);
  if constexpr (std::is_same<AccT, double>::value) return __dadd_rn(left, right);
#endif
  volatile AccT rounded = sd::math::sd_add<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT subtract(AccT left, AccT right) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<AccT, float>::value) return __fsub_rn(left, right);
  if constexpr (std::is_same<AccT, double>::value) return __dsub_rn(left, right);
#endif
  volatile AccT rounded = sd::math::sd_subtract<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT divide(AccT left, AccT right) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<AccT, float>::value) return __fdiv_rn(left, right);
  if constexpr (std::is_same<AccT, double>::value) return __ddiv_rn(left, right);
#endif
  volatile AccT rounded = sd::math::sd_divide<AccT, AccT, AccT>(left, right);
  return rounded;
}

template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT exactPowerOfTwo(int exponent) {
  if constexpr (std::is_same<AccT, float>::value) {
    union {
      uint32_t bits;
      float value;
    } result;
    if (exponent >= -126) {
      result.bits = static_cast<uint32_t>(exponent + 127) << 23;
    } else if (exponent >= -149) {
      result.bits = static_cast<uint32_t>(1) << (exponent + 149);
    } else {
      return static_cast<AccT>(0);
    }
    return result.value;
  } else if constexpr (std::is_same<AccT, double>::value) {
    union {
      uint64_t bits;
      double value;
    } result;
    if (exponent >= -1022) {
      result.bits = static_cast<uint64_t>(exponent + 1023) << 52;
    } else if (exponent >= -1074) {
      result.bits = static_cast<uint64_t>(1) << (exponent + 1074);
    } else {
      return static_cast<AccT>(0);
    }
    return result.value;
  } else {
    AccT result = static_cast<AccT>(1);
    const AccT factor = exponent >= 0 ? static_cast<AccT>(2) : static_cast<AccT>(0.5L);
    const int count = exponent >= 0 ? exponent : -exponent;
    for (int index = 0; index < count; ++index) result = multiply<AccT>(result, factor);
    return result;
  }
}

/**
 * Portable exponential for high-volume elementwise activation paths.
 *
 * Unlike exp(), this implementation uses a fixed Horner polynomial and an
 * exact IEEE power-of-two scale. It keeps explicit AccT rounding boundaries,
 * but avoids the 18 divisions and linear exponent-scaling loop that are too
 * expensive for per-activation use.
 */
template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT fastExp(AccT value) {
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

  AccT polynomial;
  if constexpr (std::is_same<AccT, double>::value) {
    polynomial = static_cast<AccT>(1.5619206968586226462e-16L);  // 1 / 18!
    polynomial = add<AccT>(static_cast<AccT>(2.8114572543455207632e-15L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(4.7794773323873852974e-14L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(7.6471637318198164759e-13L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(1.1470745597729724714e-11L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(1.6059043836821614599e-10L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(2.0876756987868098979e-9L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(2.5052108385441718775e-8L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(2.7557319223985890653e-7L), multiply<AccT>(reduced, polynomial));
    polynomial = add<AccT>(static_cast<AccT>(2.7557319223985892511e-6L), multiply<AccT>(reduced, polynomial));
  } else {
    polynomial = static_cast<AccT>(2.7557319223985892511e-6L);  // 1 / 9!
  }
  polynomial = add<AccT>(static_cast<AccT>(2.4801587301587301584e-5L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(1.9841269841269841270e-4L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(1.3888888888888888889e-3L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(8.3333333333333333332e-3L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(4.1666666666666666664e-2L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(1.6666666666666666667e-1L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(0.5L), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(1), multiply<AccT>(reduced, polynomial));
  polynomial = add<AccT>(static_cast<AccT>(1), multiply<AccT>(reduced, polynomial));
  return multiply<AccT>(polynomial, exactPowerOfTwo<AccT>(exponent));
}

/** Cross-target exponential shared by recurrent and high-volume activation paths. */
template <typename AccT>
SD_HOST_DEVICE SD_INLINE AccT exp(AccT value) {
  return fastExp<AccT>(value);
}

}  // namespace reproducible
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_REPRODUCIBLE_MATH_H
