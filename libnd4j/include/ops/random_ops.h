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
// @author raver119@gmail.com
//

#ifndef LIBND4J_RANDOM_OPS_H
#define LIBND4J_RANDOM_OPS_H
#include <type_traits>


// since we can't inherit/overwrite static methods - we just define default impls
#define method_idx                                                                                             \
  static SD_INLINE SD_HOST_DEVICE T op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *rng, \
                                       T *extraParams) {                                                       \
    return static_cast<T>(-1.0f);                                                                                              \
  }
#define method_X                                                                          \
  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,   \
                                       sd::graph::RandomGenerator *rng, T *extraParams) { \
    return static_cast<T>(-2.0f);                                                                         \
  }
#define method_XY                                                                                 \
  static SD_INLINE SD_HOST_DEVICE T op(T valueX, T valueY, sd::LongType idx, sd::LongType length, \
                                       sd::graph::RandomGenerator *rng, T *extraParams) {         \
    return static_cast<T>(-3.0f);                                                                                 \
  }

#define no_exec_special                                                                                     \
  static const bool requiresSpecial = false;                                                                \
  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y, \
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,    \
                               T *extraArguments) {}

#ifdef __CUDACC__
#define no_exec_special_cuda                                                                                     \
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer, \
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,              \
                                                sd::LongType const *zShapeBuffer, T *extraArguments) { printf("No special op for this method\n"); }
#else
#define no_exec_special_cuda
#endif
#include <array/DataTypeUtils.h>
#include <graph/RandomGenerator.h>
#include <helpers/helper_generator.h>

namespace randomOps {

/**
 * This Op merges two arrays per-element, if probability meets threshold
 */
template <typename T>
class ProbablisticMerge {
 public:
  no_exec_special no_exec_special_cuda

      method_idx method_X

      static SD_INLINE SD_HOST_DEVICE T
      op(T valueX, T valueY, sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper,
         T *extraParams) {
    T threshold = extraParams[0];
    T randVal = helper->relativeT<T>(idx);

    return randVal <= threshold ? valueY : valueX;
  }
};

/**
 * This Op produces random values within specified boundaries. Disribution is uniform
 */
template <typename T>
class UniformDistribution {
 public:
  no_exec_special no_exec_special_cuda

      method_XY method_X

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    return helper->relativeT<T>(idx, extraParams[0], extraParams[1]);
  }
};

/**
 * This op produces single bernoulli trial
 */
template <typename T>
class BernoulliDistribution {
 public:
  no_exec_special no_exec_special_cuda

      method_XY

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    return extraParams[0] >= helper->relativeT<T>(idx) ? (T)1.0f : (T)0.0f;
  }

  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    return valueX >= helper->relativeT<T>(idx) ? (T)1.0f : (T)0.0f;
  }
};

/**
 * This op produces single bernoulli trial
 */
template <typename T>
class ExponentialDistribution {
 public:
  no_exec_special no_exec_special_cuda

      method_XY

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T lambda = extraParams[0];
    T x = helper->relativeT<T>(idx, sd::DataTypeUtils::min_positive<T>(),
                               T(1.f) - sd::DataTypeUtils::template min_positive<T>());  // x from (0, 1) without bounds
    T xVal = -sd::math::sd_log<T, T>(x);

    return xVal <= (T)0.f ? (T)0.f : xVal / lambda;  // pow<T, T, T>((T) M_E, -(lambda * x));
  }

  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    T lambda = extraParams[0];
    return valueX <= (T)0.f ? (T)0.f : (T)(valueX / lambda);  // 1.f - sd::math::sd_exp<T,T>(-lambda * valueX); //pow<T, T, T>((T) M_E, -(lambda * valueX));
  }
};

template <typename T>
class PoissonDistribution {
 public:
  no_exec_special no_exec_special_cuda

      method_XY

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T lambda = extraParams[0];
    T x = helper->relativeT(idx, -sd::DataTypeUtils::template max<T>() / 10, sd::DataTypeUtils::template max<T>() / 10);
    return x <= (T)0.f ? (T)0.f : sd::math::sd_igammac<T, T, T>(sd::math::sd_floor<T, T>(x), lambda);
  }

  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    T lambda = extraParams[0];
    return valueX <= (T)0.f ? (T)0.f : (T)sd::math::sd_igammac<T, T, T>(sd::math::sd_floor<T, T>(valueX), lambda);
  }
};

template <typename T>
class GammaDistribution {
 public:
  no_exec_special no_exec_special_cuda

      method_XY

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T alpha = extraParams[0];
    T beta = extraParams[1];
    T x = helper->relativeT(idx, -sd::DataTypeUtils::template max<T>() / 10, sd::DataTypeUtils::template max<T>() / 10);
    return x <= (T)0.f ? (T)0.f : sd::math::sd_igamma<T, T, T>(alpha, x * beta);
  }

  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    T alpha = extraParams[0];
    T beta = extraParams[1];
    return valueX <= (T)0.f ? (T)0.f : sd::math::sd_igamma<T, T, T>(alpha, beta * valueX);
  }
};

/**
 * Basic DropOut/DropConnect Op
 */
template <typename T>
class DropOut {
 public:
  no_exec_special no_exec_special_cuda

      method_idx method_XY

      // please note: prob is chance to retain original value
      static SD_INLINE SD_HOST_DEVICE T
      op(T valueX, sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T randVal = helper->relativeT<T>(idx);
    return randVal >= extraParams[0] ? (T)0.0f : valueX;
  }
};

template <typename T>
class AlphaDropOut {
 public:
  no_exec_special no_exec_special_cuda

      method_idx method_XY

      // please note: prob is chance to retain original value
      static SD_INLINE SD_HOST_DEVICE T
      op(T valueX, sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T randVal = helper->relativeT<T>(idx);
    // extraParams[0] == p
    // [1] = a
    // [2] = b
    // [3] = alphaPrime
    return randVal >= extraParams[0] ? (T)extraParams[1] * extraParams[3] + extraParams[2]
                                     : extraParams[1] * valueX + extraParams[2];
  }
};

/**
 * Inverted DropOut implementation, used in DL4j
 */
template <typename T>
class DropOutInverted {
 public:
  no_exec_special no_exec_special_cuda

      method_idx method_XY

      // please note: prob is chance to retain original value
      static SD_INLINE SD_HOST_DEVICE T
      op(T valueX, sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T prob = extraParams[0];
    T randVal = helper->relativeT<T>(idx);
    return randVal >= prob ? (T)0.0f : valueX / prob;
  }
};

template <typename T>
class Linspace {
 public:
  no_exec_special no_exec_special_cuda

      method_X method_XY

      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    T from = extraParams[0];
    T to = extraParams[1];
    T step = extraParams[2];

    if (step == static_cast<T>(0.0f)) {
      step = (T)idx / ((T)length - (T)1.0f);
      return from * ((T)1.0f - step) + step * to;
    }
    return from + (idx * step);
  }
};

template <typename T>
class ExponentialDistributionInv {  // inverse exponential distribution
 public:
  no_exec_special no_exec_special_cuda
      method_XY
      
      static SD_INLINE SD_HOST_DEVICE T
      op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *helper, T *extraParams) {
    // For integer types, we need to use float for intermediate calculations
    // since exponential distribution requires floating point math
    if constexpr (std::is_integral<T>::value && !std::is_same<bool, T>::value) {
      float lambdaFloat = 1.0f;  // Default lambda for integer types
      if (extraParams != nullptr && sizeof(T) >= sizeof(float)) {
        lambdaFloat = static_cast<float>(extraParams[0]);
      }
      if (lambdaFloat == 0.0f) lambdaFloat = 1.0f;  // Avoid division by zero
      
      // Get a float random value in (0, 1)
      float x = helper->relativeT<float>(idx);
      // Ensure x is in the valid range for log
      if (x <= 0.0f) x = sd::DataTypeUtils::min_positive<float>();
      if (x >= 1.0f) x = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - x) / lambdaFloat;
      
      // Scale the result to fit in the integer type's range
      // Map [0, inf) to [0, max(T)]
      if (result < 0.0f) result = 0.0f;
      float maxVal = static_cast<float>(sd::DataTypeUtils::max<T>());
      if (result > maxVal) result = maxVal;
      
      return static_cast<T>(result);
    } else if constexpr (std::is_same<bool, T>::value) {
      // For bool type, use float intermediate and return based on threshold
      float x = helper->relativeT<float>(idx);
      if (x <= 0.0f) x = sd::DataTypeUtils::min_positive<float>();
      if (x >= 1.0f) x = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - x);
      // For bool, return true if result > 0.5, false otherwise
      return result > 0.5f;
    } else if constexpr (std::is_same<float16, T>::value || std::is_same<bfloat16, T>::value) {
      // For half precision types, use float for calculation
      float lambda = extraParams != nullptr ? static_cast<float>(extraParams[0]) : 1.0f;
      if (lambda == 0.0f) lambda = 1.0f;
      
      float x = helper->relativeT<float>(idx);
      if (x <= 0.0f) x = sd::DataTypeUtils::min_positive<float>();
      if (x >= 1.0f) x = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - x) / lambda;
      return static_cast<T>(result);
    } else {
      // For floating point types (float, double), use the original implementation
      T lambda = extraParams[0];
      if (lambda == static_cast<T>(0)) lambda = static_cast<T>(1);  // Avoid division by zero
      
      T x = helper->relativeT<T>(idx, 
                                  sd::DataTypeUtils::min_positive<T>(),
                                  static_cast<T>(1) - sd::DataTypeUtils::min_positive<T>());
      return -sd::math::sd_log<T, T>(static_cast<T>(1) - x) / lambda;
    }
  }
  
  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    if constexpr (std::is_integral<T>::value && !std::is_same<bool, T>::value) {
      float lambdaFloat = 1.0f;
      if (extraParams != nullptr && sizeof(T) >= sizeof(float)) {
        lambdaFloat = static_cast<float>(extraParams[0]);
      }
      if (lambdaFloat == 0.0f) lambdaFloat = 1.0f;
      
      float floatValueX = static_cast<float>(valueX) / static_cast<float>(sd::DataTypeUtils::max<T>());
      // Ensure value is in valid range
      if (floatValueX <= 0.0f) floatValueX = sd::DataTypeUtils::min_positive<float>();
      if (floatValueX >= 1.0f) floatValueX = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - floatValueX) / lambdaFloat;
      
      if (result < 0.0f) result = 0.0f;
      float maxVal = static_cast<float>(sd::DataTypeUtils::max<T>());
      if (result > maxVal) result = maxVal;
      
      return static_cast<T>(result);
    } else if constexpr (std::is_same<bool, T>::value) {
      float floatValueX = valueX ? 1.0f : 0.0f;
      if (floatValueX <= 0.0f) floatValueX = sd::DataTypeUtils::min_positive<float>();
      if (floatValueX >= 1.0f) floatValueX = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - floatValueX);
      return result > 0.5f;
    } else if constexpr (std::is_same<float16, T>::value || std::is_same<bfloat16, T>::value) {
      float lambda = extraParams != nullptr ? static_cast<float>(extraParams[0]) : 1.0f;
      if (lambda == 0.0f) lambda = 1.0f;
      
      float floatValueX = static_cast<float>(valueX);
      if (floatValueX <= 0.0f) floatValueX = sd::DataTypeUtils::min_positive<float>();
      if (floatValueX >= 1.0f) floatValueX = 1.0f - sd::DataTypeUtils::min_positive<float>();
      
      float result = -sd::math::sd_log<float, float>(1.0f - floatValueX) / lambda;
      return static_cast<T>(result);
    } else {
      T lambda = extraParams[0];
      if (lambda == static_cast<T>(0)) lambda = static_cast<T>(1);
      return -sd::math::sd_log<T, T>(static_cast<T>(1) - valueX) / lambda;
    }
  }
};
}  // namespace randomOps

#endif  // LIBND4J_RANDOM_OPS_H