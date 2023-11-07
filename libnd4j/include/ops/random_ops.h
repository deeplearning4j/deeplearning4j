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

// since we can't inherit/overwrite static methods - we just define default impls
#define method_idx                                                                                             \
  static SD_INLINE SD_HOST_DEVICE T op(sd::LongType idx, sd::LongType length, sd::graph::RandomGenerator *rng, \
                                       T *extraParams) {                                                       \
    return -1.0f;                                                                                              \
  }
#define method_X                                                                          \
  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,   \
                                       sd::graph::RandomGenerator *rng, T *extraParams) { \
    return -2.0f;                                                                         \
  }
#define method_XY                                                                                 \
  static SD_INLINE SD_HOST_DEVICE T op(T valueX, T valueY, sd::LongType idx, sd::LongType length, \
                                       sd::graph::RandomGenerator *rng, T *extraParams) {         \
    return -3.0f;                                                                                 \
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
    return valueX <= (T)0.f ? (T)0.f : (T)(valueX / lambda);  // 1.f - sd::math::sd_exp<T,T>(-lambda * valueX); //pow<T,
                                                              // T, T>((T) M_E, -(lambda * valueX));
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
    T lambda = extraParams[0];
    T x = helper->relativeT(idx, sd::DataTypeUtils::template min_positive<T>(),
                            (T)1.f - sd::DataTypeUtils::template min_positive<T>());
    return -sd::math::sd_log<T, T>((T)1.f - x) / lambda;
  }

  static SD_INLINE SD_HOST_DEVICE T op(T valueX, sd::LongType idx, sd::LongType length,
                                       sd::graph::RandomGenerator *helper, T *extraParams) {
    T lambda = extraParams[0];
    return -sd::math::sd_log<T, T>((T)1.f - valueX) / lambda;  // valueX must be within (0, 1]
  }
};

}  // namespace randomOps

#endif  // LIBND4J_RANDOM_OPS_H
