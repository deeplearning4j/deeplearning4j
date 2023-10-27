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

#pragma once
#ifndef OPS_H_
#define OPS_H_

#include <array/DataTypeUtils.h>
#include <helpers/shape.h>
#include <loops/ReduceType.h>
#include <loops/summarystatsreduce.h>
#include <system/Environment.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <vector>

#define no_op_exec_special_any                                                                                     \
  static const bool requiresSpecial = false;                                                                       \
  static void execSpecial(const X *dx, const sd::LongType *xShapeBuffer, Z *result,                                \
                          const sd::LongType *resultShapeBuffer, X *extraParams, const sd::LongType *tadShapeInfo, \
                          const sd::LongType *tadOffsets) {}
#define no_op_exec_special_bool                                                                                    \
  static const bool requiresSpecial = false;                                                                       \
  static void execSpecial(const X *dx, const sd::LongType *xShapeBuffer, Z *result,                                \
                          const sd::LongType *resultShapeBuffer, X *extraParams, const sd::LongType *tadShapeInfo, \
                          const sd::LongType *tadOffsets) {}
#define no_op_exec_special_same                                                                                    \
  static const bool requiresSpecial = false;                                                                       \
  static void execSpecial(const X *dx, const sd::LongType *xShapeBuffer, X *result,                                \
                          const sd::LongType *resultShapeBuffer, X *extraParams, const sd::LongType *tadShapeInfo, \
                          const sd::LongType *tadOffsets) {}
#define no_op_exec_special                                                                                         \
  static const bool requiresSpecial = false;                                                                       \
  static void execSpecial(const X *dx, const sd::LongType *xShapeBuffer, Z *result,                                \
                          const sd::LongType *resultShapeBuffer, Z *extraParams, const sd::LongType *tadShapeInfo, \
                          const sd::LongType *tadOffsets) {}
#define no_op_exec_special_accumulation                                                                   \
  static const bool requiresSpecialAccumulation = false;                                                  \
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, Z *extraParams, Z *result,          \
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}
#define no_op_exec_special_accumulation_long                                                              \
  static const bool requiresSpecialAccumulation = false;                                                  \
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, X *extraParams, Z *result,          \
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}
#define no_op_exec_special_accumulation_same                                                              \
  static const bool requiresSpecialAccumulation = false;                                                  \
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, X *extraParams, X *result,          \
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength, \
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}
#ifdef __CUDACC__
#define no_op_exec_special_any_cuda                                                                                    \
  static SD_DEVICE void execSpecialCuda(                                                                               \
      const X *dx, const sd::LongType *xShapeBuffer, Z *result, const sd::LongType *resultShapeBuffer, X *extraParams, \
      sd::LongType *allocationPointer, Z *reductionPointer, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) { \
  }
#define no_op_exec_special_bool_cuda                                                                                   \
  static SD_DEVICE void execSpecialCuda(                                                                               \
      const X *dx, const sd::LongType *xShapeBuffer, Z *result, const sd::LongType *resultShapeBuffer, X *extraParams, \
      sd::LongType *allocationPointer, Z *reductionPointer, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) { \
  }
#define no_op_exec_special_same_cuda                                                                                   \
  static SD_DEVICE void execSpecialCuda(                                                                               \
      const X *dx, const sd::LongType *xShapeBuffer, X *result, const sd::LongType *resultShapeBuffer, X *extraParams, \
      sd::LongType *allocationPointer, X *reductionPointer, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) { \
  }
#define no_op_exec_special_cuda                                                                                        \
  static SD_DEVICE void execSpecialCuda(                                                                               \
      const X *dx, const sd::LongType *xShapeBuffer, Z *result, const sd::LongType *resultShapeBuffer, Z *extraParams, \
      sd::LongType *allocationPointer, Z *reductionPointer, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) { \
  }
#define no_op_exec_special_accumulation_same_cuda                                                                  \
  static SD_INLINE SD_DEVICE void execSpecialCuda(                                                                 \
      const X *dx, const sd::LongType *xShapeInfo, X *extraParams, X *result, const sd::LongType *resultShapeInfo, \
      sd::LongType *dimension, sd::LongType dimensionLength, X *reductionBuffer, const sd::LongType *tadOnlyShapeInfo,               \
      const sd::LongType *tadOffsets) {}
#define no_op_exec_special_accumulation_long_cuda                                                                  \
  static SD_INLINE SD_DEVICE void execSpecialCuda(                                                                 \
      const X *dx, const sd::LongType *xShapeInfo, X *extraParams, Z *result, const sd::LongType *resultShapeInfo, \
      sd::LongType *dimension, sd::LongType dimensionLength, Z *reductionBuffer, const sd::LongType *tadOnlyShapeInfo,               \
      const sd::LongType *tadOffsets) {}
#define no_op_exec_special_accumulation_cuda                                                                       \
  static SD_INLINE SD_DEVICE void execSpecialCuda(                                                                 \
      const X *dx, const sd::LongType *xShapeInfo, Z *extraParams, Z *result, const sd::LongType *resultShapeInfo, \
      sd::LongType *dimension, sd::LongType dimensionLength, Z *reductionBuffer, const sd::LongType *tadOnlyShapeInfo,               \
      const sd::LongType *tadOffsets) {}

#else

#define no_op_exec_special_cuda
#define no_op_exec_special_accumulation_cuda
#define no_op_exec_special_accumulation_same_cuda
#define no_op_exec_special_accumulation_long_cuda
#define no_op_exec_special_any_cuda
#define no_op_exec_special_bool_cuda
#define no_op_exec_special_same_cuda
#define no_op_exec_special_accumulation_same_cuda
#endif

#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_LAMBDA 1.0507009873554804934193349852946

namespace functions {
namespace indexreduce {
template <typename T>
struct IndexValue {
  T value;
  sd::LongType index;
  SD_HOST_DEVICE IndexValue() = default;
  SD_HOST_DEVICE IndexValue(const T val, const sd::LongType ind) : index(ind), value(val) {}
};
}  // namespace indexreduce

namespace summarystats {
template <typename T>
class SummaryStatsData;
}
}  // namespace functions

namespace simdOps {
template <typename X, typename Y, typename Z>
class Add {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return d1 + d2; }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d1 + d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1 + params[0]); }

  SD_OP_DEF static X startingValue() { return static_cast<X>(0.f); }
};

template <typename X, typename Y>
class NewAdd {
 public:
  SD_OP_DEF static X op(X d1, Y d2, X *params) { return d1 + d2; }
};

template <typename X, typename Y, typename Z>
class Subtract {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d1 - d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d1 - d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1 - params[0]); }
};

template <typename X, typename Y, typename Z>
class SquaredSubtract {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto d = static_cast<Z>(d1 - d2);
    return d * d;
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto d = static_cast<Z>(d1 - d2);
    return d * d;
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    auto d = static_cast<Z>(d1 - params[0]);
    return d * d;
  }
};

template <typename X, typename Y, typename Z>
class SquaredReverseSubtract {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto d = static_cast<Z>(d2 - d1);
    return d * d;
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto d = static_cast<Z>(d2 - d1);
    return d * d;
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    auto d = static_cast<Z>(params[0] - d1);
    return d * d;
  }
};

template <typename X, typename Y, typename Z>
class ReverseSubtract {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d2 - d1); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d2 - d1); }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(params[0] - d1); }
};

template <typename X, typename Y, typename Z>
class LogPoissonLossFull {
 public:
  SD_OP_DEF static Z op(X z, Y c) {
    auto zz = static_cast<Z>(z);
    auto zc = static_cast<Z>(c);
    return (sd::math::sd_exp<Y, Z>(c) - zz * zc +
            (zz * sd::math::sd_log<X, Z>(z) - zz +
             static_cast<Z>(0.5f) * sd::math::sd_log<Z, Z>(static_cast<Z>(SD_DOUBLE_PI_X) * zz)));
  }

  SD_OP_DEF static Z op(X z, Y c, Z *params) {
    auto zz = static_cast<Z>(z);
    auto zc = static_cast<Z>(c);
    return (sd::math::sd_exp<Y, Z>(c) - zz * zc +
            (zz * sd::math::sd_log<X, Z>(z) - zz +
             static_cast<Z>(0.5f) * sd::math::sd_log<Z, Z>(static_cast<Z>(SD_DOUBLE_PI_X) * zz)));
  }

  SD_OP_DEF static Z op(X z) {
    auto zz = static_cast<Z>(z);
    return (zz * sd::math::sd_log<Y, Z>(z) - zz +
            static_cast<Z>(0.5f) * sd::math::sd_log<Z, Z>(static_cast<Z>(SD_DOUBLE_PI_X) * zz));
  }

  // op for MetaOps
  SD_OP_DEF static X op(X z, Y *params) {
    return (sd::math::sd_exp<X, X>(params[0]) - z * params[0] +
            (z * sd::math::sd_log<X, Z>(z) - z + static_cast<X>(0.5f) * sd::math::sd_log<X, Z>(SD_DOUBLE_PI_X * z)));
  }
};

template <typename X, typename Y, typename Z>
class LogPoissonLoss {
 public:
  SD_OP_DEF static Z op(X z, Y c) {
    auto zz = static_cast<Z>(z);
    auto zc = static_cast<Z>(c);
    return (sd::math::sd_exp<Y, Z>(c) - zz * zc);
  }

  SD_OP_DEF static Z op(X z, Y c, Z *params) {
    auto zz = static_cast<Z>(z);
    auto zc = static_cast<Z>(c);
    return (sd::math::sd_exp<Y, Z>(c) - zz * zc);
  }

  SD_OP_DEF static Z op(X z) { return static_cast<Z>(z); }

  // op for MetaOps
  SD_OP_DEF static Z op(X z, Y *params) {
    return (sd::math::sd_exp<Y, Z>(params[0]) - static_cast<Z>(z) * static_cast<Z>(params[0]));
  }
};

template <typename X, typename Y, typename Z>
class Multiply {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d1 * d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d1 * d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1 * params[0]); }

  SD_OP_DEF static X startingValue() { return static_cast<X>(1.f); }
};

template <typename X, typename Y, typename Z>
class Divide {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d1 / d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d1 / d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1 / params[0]); }

  SD_OP_DEF static X startingValue() { return static_cast<X>(1); }
};

template <typename X, typename Y, typename Z>
class DivideNoNan {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    if (d2 == (Y)0) return (Z)0;
    return static_cast<Z>(d1 / d2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    if (d2 == (Y)0) return (Z)0;
    return static_cast<Z>(d1 / d2);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    if (params[0] == (Y)0) return (Z)0;
    return static_cast<Z>(d1 / params[0]);
  }

  SD_OP_DEF static X startingValue() { return static_cast<X>(1); }
};

template <typename X, typename Y, typename Z>
class SafeDivide {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    if (d2 == static_cast<Y>(0)) return static_cast<Z>(0);
    return static_cast<Z>(d1 / d2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    if (d2 == static_cast<Y>(0)) return static_cast<Z>(0);
    return static_cast<Z>(d1 / d2);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    if (params[0] == static_cast<Y>(0)) return static_cast<Z>(0);
    return static_cast<Z>(d1 / params[0]);
  }
};

template <typename X, typename Y, typename Z>
class FloorDiv {
 public:
  //TODO: fix odd precision issue with rounding. Current static cast here is a workaround for int like types.
  //This is not a guaranteed fix and need to verify. The test case is -1 / 3 is -.333 which floor rounds down to -1.
  //We are currently reutrning
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto divResult = static_cast<float>(d1) / static_cast<float>(d2);
    //note: we do this because floor cast to an int can provide incorrect results
    //the test case that caused this change was -1 / 3 = -0.33 = -1 but it was zero instead.
    return static_cast<Z>(sd::math::sd_floor<float, float>(divResult));
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto divResult = static_cast<float>(d1) / static_cast<float>(d2);
    //note: we do this because floor cast to an int can provide incorrect results
    //the test case that caused this change was -1 / 3 = -0.33 = -1 but it was zero instead.
    return static_cast<Z>(sd::math::sd_floor<float, float>(divResult));
  }

  SD_OP_DEF static Z op(X d1) { return sd::math::sd_floor<Z, Z>(static_cast<Z>(d1)); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    printf("in params divide\n");
    return sd::math::sd_floor<Z, Z>(static_cast<Z>(static_cast<float>(d1) / static_cast<float>(params[0])));
  }
};

template <typename X, typename Y, typename Z>
class TruncateDiv {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(d2);
    return static_cast<Z>(i1 / i2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(d2);
    return static_cast<Z>(i1 / i2);
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(params[0]);
    return static_cast<Z>(i1 / i2);
  }
};

template <typename X, typename Y, typename Z>
class TruncateMod {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(d2);
    return static_cast<Z>(i1 % i2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(d2);
    return static_cast<Z>(i1 % i2);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(params[0]);
    return static_cast<Z>(i1 % i2);
  }
};

template <typename X, typename Y, typename Z>
class Remainder {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_remainder<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_remainder<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return sd::math::sd_remainder<X, Y, Z>(d1, params[0]); }
};

template <typename X, typename Y, typename Z>
class FMod {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_fmod<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_fmod<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return sd::math::sd_fmod<X, Y, Z>(d1, params[0]); }
};

template <typename X, typename Y, typename Z>
class FloorMod {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto m = sd::math::sd_fmod<X, Y, Z>(d1, d2);
    return (d1 < static_cast<X>(0)) == (d2 < static_cast<Y>(0))
           ? m
           : sd::math::sd_fmod<Z, Y, Z>(m + static_cast<Z>(d2), d2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto m = sd::math::sd_fmod<X, Y, Z>(d1, d2);
    return (d1 < static_cast<X>(0.0f)) == (d2 < static_cast<Y>(0))
           ? m
           : sd::math::sd_fmod<Z, Y, Z>(m + static_cast<Z>(d2), d2);
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return op(d1, params[0]); }
};

template <typename X, typename Y, typename Z>
class ReverseDivide {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d2 / d1); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d2 / d1); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(params[0] / d1); }
};

template <typename X, typename Y, typename Z>
class CopyPws {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1); }
};

template <typename X>
class Copy {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1;
  }
};

template <typename X, typename Y, typename Z>
class Copy2 {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return static_cast<Z>(d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }

  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(d1); }
};

template <typename X, typename Y, typename Z>
class Axpy {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<Z>(d2 + d1); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto alpha = params[0];
    return alpha * static_cast<Z>(d1) + static_cast<Z>(d2);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }
};

template <typename X, typename Z>
class Assign {
 public:
  no_op_exec_special_any no_op_exec_special_any_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return static_cast<Z>(d1);
  }
};

template <typename X, typename Z>
class And {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  SD_OP_DEF static Z
  op(X d1, X d2) {
    return d2 + d1;
  }

  SD_OP_DEF static Z op(X d1, X d2, X *params) {
    if (params != nullptr) {
      auto comp = params[0];
      return d1 != comp && d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
    } else {
      auto b1 = static_cast<bool>(d1);
      auto b2 = static_cast<bool>(d2);

      return (b1 && b2) ? static_cast<Z>(1) : static_cast<Z>(0);
    }
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, X *params) { return static_cast<Z>(119); }
};

template <typename X>
class IntOr {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return d2 | d1; }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class IntAnd {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return d2 & d1; }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class IntXor {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return d2 ^ d1; }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class ShiftLeft {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return d1 << d2; }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class ShiftRight {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return d1 >> d2; }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class CyclicShiftLeft {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_rotl<X>(d1, d2); }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X>
class CyclicShiftRight {
 public:
  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_rotr<X>(d1, d2); }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return op(d1, d2); }
};

template <typename X, typename Z>
class Or {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  SD_OP_DEF static Z
  op(X d1, X d2) {
    return d2 + d1;
  }

  SD_OP_DEF static Z op(X d1, X d2, X *params) {
    if (params != nullptr) {
      auto comp = params[0];

      return d1 != comp || d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
    } else {
      auto b1 = static_cast<bool>(d1);
      auto b2 = static_cast<bool>(d2);

      return b1 || b2 ? static_cast<Z>(1) : static_cast<Z>(0);
    }
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, X *params) { return static_cast<Z>(119); }
};

template <typename X, typename Z>
class Xor {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  SD_OP_DEF static Z
  op(X d1, X d2) {
    return d2 + d1;
  }

  SD_OP_DEF static Z op(X d1, X d2, X *params) {
    if (params != nullptr) {
      auto comp = params[0];

      return ((d1 == comp && d2 != comp) || (d1 != comp && d2 == comp)) ? static_cast<Z>(1) : static_cast<Z>(0);
    } else {
      auto b1 = static_cast<bool>(d1);
      auto b2 = static_cast<bool>(d2);

      return (!b1 && b2) || (b1 && !b2) ? static_cast<Z>(1) : static_cast<Z>(0);
    }
  }

  SD_OP_DEF static Z op(X d1) { return d1; }
};

template <typename X, typename Z>
class Not {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  SD_OP_DEF static Z
  op(X d1, X d2) {
    return static_cast<Z>(0);
  }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return d1 != d2 ? static_cast<Z>(1) : static_cast<Z>(0); }

  // this transform op should run only on boolean input
  SD_OP_DEF static Z op(X d1, X *params) {
    auto b1 = static_cast<bool>(d1);
    return !b1;
  }
};

template <typename X, typename Y, typename Z>
class LogicalNot {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return !((int)d1 && (int)d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    return static_cast<X>(!(static_cast<int>(d1) && static_cast<int>(d2)));
  }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<X>(119); }
};

template <typename X, typename Y, typename Z>
class LogicalXor {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto i1 = static_cast<int>(d1);
    auto i2 = static_cast<int>(d2);

    return (i1 | i2) & ~(i1 & i2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(119); }
};

template <typename X, typename Y, typename Z>
class LogicalAnd {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<int>(d1) & static_cast<int>(d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(Y d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<Z>(119); }
};

template <typename X, typename Y, typename Z>
class LogicalOr {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<int>(d1) | static_cast<int>(d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return static_cast<X>(119); }
};

template <typename X, typename Y, typename Z>
class Mod {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) {
    auto dx = static_cast<X>(d2);
    auto f = sd::math::sd_floor<X, X>(d1 / dx);
    auto r = f * dx;
    return d1 - r;
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  // op for MetaOp
  SD_OP_DEF static Z op(X d1, Y *params) { return op(d1, params[0]); }
};

template <typename X, typename Y, typename Z>
class ReverseMod {
 public:
  SD_OP_DEF static Z op(X d1, Y d2) { return static_cast<int>(d2) % static_cast<int>(d1); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  // op for MetaOp
  SD_OP_DEF static Z op(X d1, Y *params) { return op(d1, params[0]); }
};

/**
 * Whether 2 elements in an array
 * are epsilon equal
 */
template <typename X, typename Z>
class Epsilon {
 public:
  SD_OP_DEF static Z op(X d1, X d2) {
    X diff = d1 - d2;
    X absDiff = sd::math::sd_abs<X>(diff);
    if (absDiff <= static_cast<X>(SD_MIN_V)) return static_cast<Z>(1);
    return static_cast<Z>(0);
  }

  SD_OP_DEF static Z op(X d1, X d2, X *params) {
    X diff = d1 - d2;
    X absDiff = sd::math::sd_abs<X>(diff);
    if(params != nullptr && absDiff <= static_cast<X>(params[0])) {
      return static_cast<Z>(1);
    } else  if(absDiff <= static_cast<X>(1e-5)) {
      return static_cast<Z>(1);
    }
    return static_cast<Z>(0);
  }


  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class EqualTo {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 == d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class NotEqualTo {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 != d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class GreaterThanOrEqual {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 >= d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  // FIXME: this signature clashes with MetaOp stuff
  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class GreaterThan {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 > d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  // FIXME: this signature clashes with MetaOp stuff
  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class LessThan {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 < d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X, typename Z>
class LessThanOrEqual {
 public:
  SD_OP_DEF static Z op(X d1, X d2) { return d1 <= d2; }

  SD_OP_DEF static Z op(X d1, X d2, X *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, X *params) { return d1; }
};

template <typename X>
class Abs {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_abs<X>(d1);
  }
};

template <typename X>
class Ceiling {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_ceil<X, X>(d1);
  }
};

template <typename X>
class Cosine {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_cos<X, X>(d1);
  }
};

template <typename X>
class Exp {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_exp<X, X>(d1);
  }
};

template <typename X>
class HardTanhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return ((d1 >= static_cast<X>(-1.f) && d1 <= static_cast<X>(1.f)) ? static_cast<X>(1.f) : static_cast<X>(0.f));
  }
};

template <typename X>
class HardTanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    if (d1 < static_cast<X>(-1))
      return static_cast<X>(-1);
    else if (d1 > static_cast<X>(1))
      return static_cast<X>(1);
    else
      return d1;
  }
};

template <typename X>
class Floor {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_floor<X, X>(d1);
  }
};

template <typename X>
class Log {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_log<X, X>(d1);
  }
};

template <typename X>
class Log1p {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_log<X, X>(1 + d1);
  }
};

template <typename X, typename Y, typename Z>
class LogX {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_log<X, Z>(d1) / sd::math::sd_log<Y, Z>(d2); }
};

template <typename X>
class StabilizeFP16 {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    if (d1 <= static_cast<X>(0))
      return static_cast<X>(sd::DataTypeUtils::min<float16>());
    else
      return d1;
  }
};

template <typename X>
class StabilizeX {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    if (d1 <= static_cast<X>(0))
      return sd::DataTypeUtils::min<X>();
    else
      return d1;
  }
};

template <typename X>
class SpecialDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * (static_cast<X>(1.f) - d1);
  }
};

template <typename X>
class Neg {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return -d1;
  }
};

template <typename X>
class Erf {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_erf<X, X>(d1);
  }
};

template <typename X>
class Erfc {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_erfc<X, X>(d1);
  }
};

template <typename X>
class Reciprocal {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda
  SD_OP_DEF static X
  op(X d1, X *params) {
    return (static_cast<X>(1) / d1);
  }
};

template <typename X, typename Z>
class Sqr {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_pow<X, X, Z>(d1, static_cast<X>(2));
  }

  SD_OP_DEF static Z op(X d1) { return sd::math::sd_pow<X, X, Z>(d1, static_cast<X>(2)); }
};

template <typename X, typename Y, typename Z>
class RelativeError {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2) {
    return sd::math::sd_re<X>(d1, d2);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(0); }
};

template <typename X, typename Y, typename Z>
class BinaryRelativeError {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    X threshold = params[0];
    return sd::math::sd_re<X>(d1, d2) > threshold ? static_cast<Z>(1) : static_cast<Z>(0);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(0); }
};

template <typename X, typename Y, typename Z>
class BinaryMinimumAbsoluteRelativeError {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    X d2 = params[0];
    X thresholdRelative = params[1];
    X thresholdAbsolute = params[2];
    return sd::math::sd_re<X>(d1, d2) > thresholdRelative
           ? (sd::math::sd_abs<X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0)
                                                                               : static_cast<Z>(1))
           : static_cast<Z>(0);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    X thresholdRelative = params[0];
    X thresholdAbsolute = params[1];
    return sd::math::sd_re<X>(d1, d2) > thresholdRelative
           ? (sd::math::sd_abs<X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0)
                                                                               : static_cast<Z>(1))
           : static_cast<Z>(0);
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(0); }
};

template <typename X, typename Y, typename Z>
class ReversePow {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_pow<X, X, Z>(params[0], d1);
  }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_pow<X, Y, Z>(d2, d1); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_pow<X, Y, Z>(d2, d1); }

  SD_OP_DEF static Z op(X d1) { return d1; }
};

template <typename X, typename Y, typename Z>
class Pow {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_pow<X, X, Z>(d1, params[0]);
  }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_pow<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_pow<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }
};

template <typename X, typename Y, typename Z>
class PowDerivative {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return params[0] * sd::math::sd_pow<X, Z, Z>(d1, static_cast<Z>(params[0]) - static_cast<Z>(1.f));
  }

  SD_OP_DEF static Z op(X d1, Y d2) {
    return static_cast<Z>(d2) * sd::math::sd_pow<X, Z, Z>(d1, static_cast<Z>(d2) - static_cast<Z>(1.f));
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    return static_cast<Z>(d2) * sd::math::sd_pow<X, Z, Z>(d1, static_cast<Z>(d2) - static_cast<Z>(1.f));
  }

  SD_OP_DEF static Z op(X d1) { return static_cast<Z>(d1); }
};

template <typename X, typename Y, typename Z>
class IGamma {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_igamma<X, X, Z>(d1, params[0]);
  }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_igamma<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_igamma<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }
};

template <typename X, typename Y, typename Z>
class IGammac {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_igammac<X, X, Z>(d1, params[0]);
  }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_igammac<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_igammac<X, Y, Z>(d1, d2); }

  SD_OP_DEF static Z op(X d1) { return d1; }
};

template <typename X>
class Round {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_round<X, X>(d1);
  }
};

template <typename X, typename Z>
class IsNan {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return sd::math::sd_isnan(d1) ? static_cast<X>(1) : static_cast<X>(0);
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X>
class Expm1 {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_exp<X, X>(d1) - static_cast<X>(1);
  }
};

template <typename X, typename Z>
class IsPositive {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return d1 > (X)0.f;
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Z>
class IsNegative {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return d1 < (X)0.f;
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Z>
class IsInf {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return sd::math::sd_isinf<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Z>
class IsInfOrNan {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return sd::math::sd_isfin<X>(d1) ? static_cast<Z>(0) : static_cast<Z>(1);
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) {
    return opOutput == static_cast<X>(0) && old == static_cast<X>(0) ? static_cast<Z>(0) : static_cast<Z>(1);
  }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) {
    return opOutput == static_cast<X>(0) && old == static_cast<X>(0) ? static_cast<Z>(0) : static_cast<Z>(1);
  }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction != static_cast<X>(0); }
};

template <typename X, typename Z>
class IsFinite {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  op(X d1, X *params) {
    return sd::math::sd_isfin<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
  }

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(1); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) {
    return opOutput == static_cast<X>(0) || old == static_cast<X>(0) ? static_cast<Z>(0) : static_cast<Z>(1);
  }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) {
    return opOutput == static_cast<X>(0) || old == static_cast<X>(0) ? static_cast<Z>(0) : static_cast<Z>(1);
  }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction != static_cast<X>(0); }
};

template <typename X>
class ClipByValue {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    if (d1 > params[1]) return params[1];
    if (d1 < params[0]) return params[0];
    return d1;
  }
};

template <typename X, typename Y, typename Z>
class LstmClip {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    X _v = (X)d2;
    if (d1 > _v)
      return _v;
    else if (d1 < -_v)
      return -_v;
    else
      return d1;
  }
};

template <typename X>
class Swish {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * sd::math::sd_sigmoid<X, X>(d1);
  }
};

template <typename X>
class Mish {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * sd::math::sd_tanh<X, X>(sd::math::sd_softplus<X, X>(d1));
  }
};

template <typename X>
class MishDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto ex = sd::math::sd_exp<X, X>(d1);
    auto e2x = ex * ex;
    auto e3x = ex * ex * ex;

    return (ex * (4 * (d1 + 1) + 4 * e2x + e3x + ex * (4 * d1 + 6))) /
           sd::math::sd_pow<X, X, X>((2 * ex + e2x + 2), (X)2.f);
  }
};

template <typename X>
class GELU {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * sd::math::sd_sigmoid<X, X>(static_cast<X>(1.702f) * d1);
  }
};

template <typename X>
class PreciseGELU {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto sp = sd::math::sd_sqrt<X, X>(static_cast<X>(2) / static_cast<X>(M_PI));
    auto xp = d1 + sd::math::sd_pow<X, X, X>(static_cast<X>(0.044715) * d1, static_cast<X>(3));
    return (d1 / static_cast<X>(2)) * (static_cast<X>(1) + sd::math::sd_tanh<X, X>(sp * xp));
  }
};

template <typename X>
class GELUDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto x17 = static_cast<X>(1.702f) * d1;
    auto ep = sd::math::sd_pow<X, X, X>(static_cast<X>(M_E), x17);
    // (E^(1.702 x) (1. + E^(1.702 x) + 1.702 x))/(1. + E^(1.702 x))^2
    return (ep * (static_cast<X>(1.f) + ep + x17)) / sd::math::sd_pow<X, int, X>((static_cast<X>(1.f) + ep), 2);
  }
};

template <typename X>
class PreciseGELUDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto x79 = static_cast<X>(0.797885) * d1;
    auto x03 = sd::math::sd_pow<X, int, X>(static_cast<X>(0.0356774) * d1, 3);
    auto x39 = static_cast<X>(0.398942) * d1;
    auto x05 = sd::math::sd_pow<X, int, X>(static_cast<X>(0.0535161) * d1, 3);
    auto scz = sd::math::sd_sech<X, X>(x79 + x03);
    // 0.5 + (0.398942 x + 0.0535161 x^3) Sech[0.797885 x + 0.0356774 x^3]^2 + 0.5 Tanh[0.797885 x + 0.0356774 x^3]
    return static_cast<X>(0.5) + (x39 + x05) * (scz * scz) + static_cast<X>(0.5) * sd::math::sd_tanh<X, X>(x79 + x03);
  }
};

template <typename X>
class SwishDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    X ex = sd::math::sd_pow<X, X, X>(static_cast<X>(M_E), d1);
    return (ex * (d1 + ex + static_cast<X>(1.f))) /
           sd::math::sd_pow<X, X, X>((ex + static_cast<X>(1.f)), static_cast<X>(2.f));
  }
};

template <typename X>
class LogSigmoid {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_log<X, X>(sd::math::sd_sigmoid<X, X>(d1));
  }
};

template <typename X>
class LogSigmoidDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    X ex = sd::math::sd_pow<X, X, X>(M_E, d1);
    return static_cast<X>(1.f) / (ex + static_cast<X>(1.f));
  }
};

template <typename X>
class Sigmoid {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_sigmoid<X, X>(d1);
  }
};

template <typename X>
class Affine {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return params[0] * d1 + params[1];
  }
};

template <typename X>
class SigmoidDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_sigmoidderivative<X, X>(d1);
  }
};

template <typename X>
class HardSigmoid {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_min<X>(
        static_cast<X>(1), sd::math::sd_max<X>(static_cast<X>(0), (static_cast<X>(0.2f)) * d1 + static_cast<X>(0.5f)));
  }
};

template <typename X>
class HardSigmoidDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 < static_cast<X>(-2.5f) || d1 > static_cast<X>(2.5f) ? static_cast<X>(0.f) : static_cast<X>(0.2f);
  }
};

/**
 * Scale to be between a min and max
 */
template <typename X>
class SetRange {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto min = params[0];
    auto max = params[1];
    if (static_cast<X>(d1) >= min && static_cast<X>(d1) <= max) return d1;
    if (min == static_cast<X>(0) && max == static_cast<X>(1)) {
      auto val = static_cast<X>(1) / (static_cast<X>(1) + sd::math::sd_exp<X, X>(-d1));
      return (sd::math::sd_floor<X, X>(val * (max - min)) + min);
    }

    return (sd::math::sd_floor<X, X>(d1 * (max - min)) + min);
  }
};

template <typename X>
class Sin {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_sin<X, X>(d1);
  }
};

template <typename X>
class Square {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * d1;
  }
};

template <typename X, typename Z>
class Sqrt {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return sd::math::sd_sqrt<X, Z>(d1);
  }
};

template <typename X, typename Z>
class RSqrt {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Z *params) {
    return static_cast<Z>(1.0) / sd::math::sd_sqrt<X, Z>(d1);
  }
};

template <typename X>
class Rint {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_rint<X, X>(d1);
  }
};

template <typename X>
class SoftPlus {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_softplus<X, X>(d1);
  }
};

template <typename X>
class Sign {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return (d1 > static_cast<X>(0)) - (d1 < static_cast<X>(0));
  }
};

template <typename X>
class TimesOneMinus {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * (static_cast<X>(1) - d1);
  }
};

template <typename X>
class RationalTanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    // keep 2/3 as runtime variable, to match precision
    auto dis = (static_cast<X>(2) / static_cast<X>(3)) * d1;

    auto tanh = sd::math::sd_sgn<X, X>(dis) *
                (static_cast<X>(1) -
                 (static_cast<X>(1) / (static_cast<X>(1) + static_cast<X>(sd::math::sd_abs<X>(dis)) +
                                       sd::math::sd_pow<X, X, X>(dis, static_cast<X>(2)) +
                                       static_cast<X>(1.41645f) * sd::math::sd_pow<X, X, X>(dis, static_cast<X>(4)))));
    return static_cast<X>(1.7159f) * tanh;
  }
};

template <typename X>
class RationalTanhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    auto dis = (static_cast<X>(2.f) / static_cast<X>(3.f)) * d1;

    auto a = static_cast<X>(1.f) + sd::math::sd_abs<X>(dis) + sd::math::sd_pow<X, X, X>(dis, static_cast<X>(2.f)) +
             static_cast<X>(1.41645f) * sd::math::sd_pow<X, X, X>(dis, static_cast<X>(4));

    auto tDeriv =
        (static_cast<X>(1.f) + sd::math::sd_sign<X, X>(dis) * (static_cast<X>(2.f) * dis +
                                                               static_cast<X>(4.f) * static_cast<X>(1.41645f) *
                                                               sd::math::sd_pow<X, X, X>(dis, static_cast<X>(3)))) /
        (a * a);

    return static_cast<X>(1.7159f) * (static_cast<X>(2.f) / static_cast<X>(3.f)) * tDeriv;
  }
};

template <typename X>
class Tanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_tanh<X, X>(d1);
  }
};

template <typename X>
class ScaledTanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return params[0] * sd::math::sd_tanh<X, X>(params[1] * d1);
  }
};

template <typename X>
class RectifiedTanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_max<X>(static_cast<X>(0), sd::math::sd_tanh<X, X>(d1));
  }
};

template <typename X>
class RectifiedTanhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 > static_cast<X>(0.f) ? sd::math::sd_tanhderivative<X, X>(d1) : static_cast<X>(0.f);
  }
};

template <typename X>
class ATanh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_atanh<X, X>(d1);
  }
};

template <typename X>
class TanhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_tanhderivative<X, X>(d1);
  }
};

template <typename X>
class Cube {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 * d1 * d1;
  }
};

template <typename X>
class CubeDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(3) * d1 * d1;
  }
};

template <typename X>
class ACos {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_acos<X, X>(d1);
  }
};

template <typename X>
class ASinh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_asinh<X, X>(d1);
  }
};

template <typename X>
class ASinhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(1.f) /
           (sd::math::sd_sqrt<X, X>(sd::math::sd_pow<X, X, X>(d1, static_cast<X>(2.f)) + static_cast<X>(1.f)));
  }
};

template <typename X>
class ACosh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_acosh<X, X>(d1);
  }
};

template <typename X>
class ACoshDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(1.f) /
           (sd::math::sd_sqrt<X, X>(d1 - static_cast<X>(1.f)) * sd::math::sd_sqrt<X, X>(d1 + static_cast<X>(1.f)));
  }
};

template <typename X>
class Ones {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(1.0f);
  }
};

template <typename X>
class SoftSign {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_softsign<X, X>(d1);
  }
};

template <typename X>
class SoftSignDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_softsignderivative<X, X>(d1);
  }
};

template <typename X, typename Z>
class MatchConditionBool {
 public:
  no_op_exec_special_bool no_op_exec_special_bool_cuda

  // this op return 1.0 if condition met, 0.0 otherwise
  SD_OP_DEF static Z
  op(X d1, X *extraParams) {
    X compare = extraParams[0];
    X eps = extraParams[1];

    auto mode = static_cast<int>(extraParams[2]);
    sd_debug("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

    switch (mode) {
      case 0:  // equals
        return sd::math::sd_abs<X>(d1 - compare) <= eps ? true : false;
      case 1:  // not equals
        return sd::math::sd_abs<X>(d1 - compare) > eps ? true : false;
      case 2:  // less_than
        return d1 < compare ? true : false;
      case 3:  // greater_than
        return d1 > compare ? true : false;
      case 4:  // less_or_equals_than
        return d1 <= compare ? true : false;
      case 5:  // greater_or_equals_than
        return d1 >= compare ? true : false;
      case 6:  // abs_less_than
        return sd::math::sd_abs<X>(d1) < compare ? true : false;
      case 7:  // abs_greater_than
        return sd::math::sd_abs<X>(d1) > compare ? true : false;
      case 8:  // is inf
        return sd::math::sd_isinf(d1) ? true : false;
      case 9:  // is nan
        return sd::math::sd_isnan(d1) ? true : false;
      case 10:
        return (d1 == compare) ? true : false;
      case 11:
        return (d1 != compare) ? true : false;
      case 12:  // abs_greater_or_equals_than
        return sd::math::sd_abs<X>(d1) >= compare ? true : false;
      case 13:  // abs_less_or_equals_than
        return sd::math::sd_abs<X>(d1) <= compare ? true : false;
      case 14:
        // isFinite
        return !(sd::math::sd_isinf(d1) || sd::math::sd_isnan(d1));
      case 15:
        // isInfinite
        return sd::math::sd_isinf(d1) || sd::math::sd_isnan(d1);
      default:
        sd_debug("Undefined match condition: [%i]\n", mode);
    }

    return d1;
  }
};

template <typename X, typename Z>
class MatchCondition {
 public:
  no_op_exec_special no_op_exec_special_cuda

  no_op_exec_special_accumulation_long no_op_exec_special_accumulation_cuda

  SD_OP_DEF static Z
  startingValue(const X *input) {
    return static_cast<Z>(0);
  }

  SD_OP_DEF static Z merge(Z old, Z opOutput, X *extraParams) { return old + opOutput; }

  SD_OP_DEF static Z update(Z old, Z opOutput, X *extraParams) { return old + opOutput; }

  SD_OP_DEF static Z op(X d1, X compare, X eps, int mode) {
    switch (mode) {
      case 0:  // equals
        return sd::math::sd_abs<X>(d1 - compare) <= eps ? 1 : 0;
      case 1:  // not equals
        return sd::math::sd_abs<X>(d1 - compare) > eps ? 1 : 0;
      case 2:  // less_than
        return d1 < compare ? 1 : 0;
      case 3:  // greater_than
        return d1 > compare ? 1 : 0;
      case 4:  // less_or_equals_than
        return d1 <= compare ? 1 : 0;
      case 5:  // greater_or_equals_than
        return d1 >= compare ? 1 : 0;
      case 6:  // abs_less_than
        return sd::math::sd_abs<X>(d1) < compare ? 1 : 0;
      case 7:  // abs_greater_than
        return sd::math::sd_abs<X>(d1) > compare ? 1 : 0;
      case 8:  // is inf
        return sd::math::sd_isinf(d1) ? 1 : 0;
      case 9:  // is nan
        return sd::math::sd_isnan(d1) ? 1 : 0;
      case 10:
        return (d1 == compare) ? 1 : 0;
      case 11:
        return (d1 != compare) ? 1 : 0;
      case 12:  // abs_greater_or_equals_than
        return sd::math::sd_abs<X>(d1) >= compare ? 1 : 0;
      case 13:  // abs_less_or_equals_than
        return sd::math::sd_abs<X>(d1) <= compare ? 1 : 0;
      case 14:
        // isFinite
        return !(sd::math::sd_isinf(d1) || sd::math::sd_isnan(d1)) ? 1 : 0;
      case 15:
        // isInfinite
        return sd::math::sd_isinf(d1) || sd::math::sd_isnan(d1) ? 1 : 0;
      default:
        sd_printf("Undefined match condition: [%i]\n", mode);
    }

    return d1;
  }

  // this op return 1.0 if condition met, 0.0 otherwise
  SD_OP_DEF static Z op(X d1, X compare, X *extraParams) {
    X eps = extraParams[1];

    auto mode = static_cast<int>(extraParams[2]);

    return op(d1, compare, eps, mode);
  }

  // this op return 1.0 if condition met, 0.0 otherwise
  SD_OP_DEF static Z op(X d1, X *extraParams) {
    X compare = extraParams[0];
    X eps = extraParams[1];

    auto mode = static_cast<int>(extraParams[2]);

    return op(d1, compare, eps, mode);
  }

  SD_OP_DEF static Z postProcess(Z reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Y, typename Z>
class ELU {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    return sd::math::sd_elu<X, Z>(d1, static_cast<X>(d2));
  }
};

template <typename X, typename Y, typename Z>
class ELUDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    return sd::math::sd_eluderivative<X, Z>(d1, static_cast<X>(d2));
  }
};

template <typename X, typename Y, typename Z>
class RELU {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    auto xt = static_cast<Z>(d1);
    auto xf = static_cast<Z>(d2);
    return xt < xf ? xf : xt;
  }
};

template <typename X, typename Y, typename Z>
class RELUDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    auto xt = static_cast<Z>(d1);
    auto xf = static_cast<Z>(d2);
    return xt > xf ? static_cast<Z>(1.f) : static_cast<Z>(0.f);
  }
};

template <typename X, typename Y, typename Z>
class SXELogitsSmoother {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return d1 * ((X)1.f - (X)d2) + (X)(0.5f) * (X)d2; }
};

template <typename X, typename Y, typename Z>
class RELU6 {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    auto relu = simdOps::RELU<X, Y, Z>::op(d1, d2, params);
    return relu < static_cast<Z>(6) ? relu : static_cast<Z>(6);
  }
};

template <typename X, typename Y, typename Z>
class LeakyRELU {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    auto val = static_cast<Z>(d1);
    auto alpha = static_cast<Z>(d2);
    return val < 0.0f ? alpha * val : val;
  }
};

template <typename X>
class SELU {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 > static_cast<X>(0.0f)
           ? static_cast<X>(SELU_LAMBDA) * static_cast<X>(d1)
           : static_cast<X>(SELU_LAMBDA) *
             (static_cast<X>(SELU_ALPHA) * sd::math::sd_exp<X, X>(d1) - static_cast<X>(SELU_ALPHA));
  }
};

template <typename X>
class SELUDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1 > static_cast<X>(0.f)
           ? static_cast<X>(SELU_LAMBDA)
           : static_cast<X>(SELU_ALPHA) * static_cast<X>(SELU_LAMBDA) * sd::math::sd_exp<X, X>(d1);
  }
};

template <typename X, typename Y, typename Z>
class LeakyRELUDerivative {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    if (d1 >= static_cast<X>(0))
      return static_cast<Z>(1);
    else
      return static_cast<Z>(d2);
  }
};

template <typename X>
class ASin {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_asin<X, X>(d1);
  }
};

template <typename X>
class Sinh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_sinh<X, X>(d1);
  }
};

template <typename X>
class SinhDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_cosh<X, X>(d1);
  }
};

template <typename X>
class Cosh {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_cosh<X, X>(d1);
  }
};

template <typename X>
class Tan {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_tan<X, X>(d1);
  }
};

template <typename X>
class TanDerivative {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(1.f) / sd::math::sd_pow<X, X, X>(sd::math::sd_cos<X, X>(d1), static_cast<X>(2.0f));
  }
};

template <typename X>
class ATan {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return sd::math::sd_atan<X, X>(d1);
  }
};

template <typename X, typename Y, typename Z>
class Atan2 {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2) {
    return sd::math::sd_atan2<X, Z>(d2, d1);
  }

  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  // op for MetaOps
  SD_OP_DEF static Z op(X d1, Y *params) { return op(d1, params[0]); }
};

template <typename X>
class Identity {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return d1;
  }
};

template <typename X>
class Stabilize {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    X k = params[0];
    if (d1 * k > static_cast<X>(SD_MAX_CUTFOFF))
      return static_cast<X>(SD_MAX_CUTFOFF) / k;
    else if (d1 * k < static_cast<X>(SD_MIN_CUTFOFF))
      return static_cast<X>(SD_MIN_CUTFOFF) / k;
    return d1;
  }
};

template <typename X, typename Y, typename Z>
class Step {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    return (d1 > static_cast<X>(d2) ? static_cast<Z>(1) : static_cast<Z>(0));
  }
};

template <typename X>
class OneMinus {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  SD_OP_DEF static X
  op(X d1, X *params) {
    return static_cast<X>(1) - d1;
  }
};

template <typename X>
class Sum {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  SD_OP_DEF static X
  startingValue(const X *input) {
    return static_cast<X>(0.0f);
  }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static X op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X>
class ReduceSameBenchmarkOp {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0.0f); }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static X op(X d1, X *extraParams) {
    auto f1 = static_cast<float>(d1);
    return static_cast<X>(sd::math::sd_pow<float, float, float>(f1, 3) +
                          sd::math::sd_log<float, float>(f1) * sd::math::sd_sin<float, float>(f1) /
                          sd::math::sd_tanh<float, float>(static_cast<float>(M_E) * static_cast<float>(M_PI) * f1) *
                          sd::math::sd_sqrt<float, float>(static_cast<float>(M_PI) / f1) -
                          sd::math::sd_atan<float, float>(static_cast<float>(M_E) / f1));
  }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

/**
 * @brief AggregateType - helper template to use desired type for the aggregation expressions.
 *  This way we can reduce overflow and precision issues for certain types
 *
 * @tparam Z
 */
template <typename Z>
struct AggregateType {
  using type = Z;
};

template <>
struct AggregateType<float16> {
  using type = float;
};

template <typename X, typename Z>
class ShannonEntropy {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    auto p = d1;
    return static_cast<Z>(p) * sd::math::sd_log2<X, Z>(p);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) { return -reduction; }
};


template <typename X, typename Z>
class LogEntropy {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda using InterType =
      typename AggregateType<Z>::type;
  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    return static_cast<InterType>(d1) * sd::math::sd_log<X, InterType>(d1);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    // entropy is -sum(p(x) * log(p(x))); log entropy is log of this
    return sd::math::sd_log<InterType, Z>(-reduction);
  }
};

template <typename X, typename Z>
class Entropy {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    return static_cast<InterType>(d1) * sd::math::sd_log<X, InterType>(d1);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return static_cast<Z>(-reduction);  // entropy is -sum(p(x) * log(p(x)))
  }
};

template <typename X>
class ASum {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::ASUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) {
    return sd::math::sd_abs<X>(opOutput) + sd::math::sd_abs<X>(old);
  }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) {
    return sd::math::sd_abs<X>(opOutput) + sd::math::sd_abs<X>(old);
  }

  SD_OP_DEF static X op(X d1, X *extraParams) { return sd::math::sd_abs<X>(d1); }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return sd::math::sd_abs<X>(reduction); }
};

template <typename X, typename Z>
class CountNonZero {
 public:
  no_op_exec_special_accumulation_long no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::ASUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static Z startingValue(const X *input) { return static_cast<Z>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, X *extraParams) {
    return d1 == static_cast<X>(0.0f) ? static_cast<InterType>(0.0f) : static_cast<InterType>(1.0f);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, X *extraParams) {
    return static_cast<Z>(reduction);
  }
};

template <typename X, typename Z>
class CountZero {
 public:
  no_op_exec_special_accumulation_long no_op_exec_special_accumulation_cuda using InterType =
      typename AggregateType<Z>::type;
  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static Z startingValue(const X *input) { return static_cast<Z>(0.0f); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, X *extraParams) {
    return d1 == static_cast<X>(0) ? static_cast<InterType>(1) : static_cast<InterType>(0);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, X *extraParams) {
    return static_cast<Z>(reduction);
  }
};

template <typename X>
class Prod {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::PRODUCT;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(1); }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) { return opOutput * old; }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) { return opOutput * old; }

  SD_OP_DEF static X op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Z>
class Any {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda using InterType = Z;
  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0.0f); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) {
    return reduction > static_cast<X>(0) ? static_cast<Z>(1) : static_cast<Z>(0);
  }
};

template <typename X, typename Z>
class All {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda using InterType = Z;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(1); }

  SD_OP_DEF static Z merge(X old, X opOutput, X *extraParams) { return opOutput * old; }

  SD_OP_DEF static Z update(X old, X opOutput, X *extraParams) { return opOutput * old; }

  SD_OP_DEF static Z op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static Z postProcess(X reduction, sd::LongType n, X *extraParams) {
    return reduction > static_cast<X>(0) ? static_cast<Z>(1) : static_cast<Z>(0);
  }
};

template <typename X, typename Z>
class Mean {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  using InterType = typename AggregateType<Z>::type;

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) { return d1; }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return static_cast<Z>(reduction / (InterType)n);
  }
};

template <typename X, typename Z>
class ReduceFloatBenchmarkOp {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    auto f1 = static_cast<float>(d1);
    return static_cast<InterType>(
        sd::math::sd_pow<float, float, float>(f1, 3) +
        sd::math::sd_log<float, float>(f1) * sd::math::sd_sin<float, float>(f1) /
        sd::math::sd_tanh<float, float>(static_cast<float>(M_E) * static_cast<float>(M_PI) * f1) *
        sd::math::sd_sqrt<float, float>(static_cast<float>(M_PI) / f1) -
        sd::math::sd_atan<float, float>(static_cast<float>(M_E) / f1));
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return (InterType)reduction / (InterType)n;
  }
};

template <typename X, typename Z>
class AMean {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) {
    return sd::math::sd_abs<X>(opOutput) + sd::math::sd_abs<X>(old);
  }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) { return sd::math::sd_abs<InterType>(d1); }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return sd::math::sd_abs<Z>(reduction / static_cast<InterType>(n));
  }
};

template <typename X>
class Max {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::MAX;

  SD_OP_DEF static X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) { return sd::math::sd_max<X>(old, opOutput); }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) { return sd::math::sd_max<X>(opOutput, old); }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return sd::math::sd_max<X>(d1, d2); }

  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_max<X>(d1, d2); }

  // FIXME: this signature overlaps with MetaOp
  SD_OP_DEF static X op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Y, typename Z>
class AMaxPairwise {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2) {
    auto z1 = static_cast<Z>(d1);
    auto z2 = static_cast<Z>(d2);

    if (sd::math::sd_abs<Z>(z1) > sd::math::sd_abs<Z>(z2))
      return z1;
    else
      return z2;
  }
};

template <typename X, typename Y, typename Z>
class AMinPairwise {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return op(d1, d2); }

  SD_OP_DEF static Z op(X d1, Y d2) {
    auto z1 = static_cast<Z>(d1);
    auto z2 = static_cast<Z>(d2);

    if (sd::math::sd_abs<Z>(z1) < sd::math::sd_abs<Z>(z2))
      return z1;
    else
      return z2;
  }
};

template <typename X, typename Y, typename Z>
class MaxPairwise {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_max<Z>(static_cast<Z>(d1), static_cast<Z>(d2)); }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_max<Z>(static_cast<Z>(d1), static_cast<Z>(d2)); }
};

template <typename X, typename Y, typename Z>
class MinPairwise {
 public:
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) { return sd::math::sd_min<Z>(static_cast<Z>(d1), static_cast<Z>(d2)); }

  SD_OP_DEF static Z op(X d1, Y d2) { return sd::math::sd_min<Z>(static_cast<Z>(d1), static_cast<Z>(d2)); }
};

template <typename X>
class AMax {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::AMAX;

  SD_OP_DEF static X startingValue(const X *input) { return input[0]; }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) {
    return sd::math::sd_max<X>(sd::math::sd_abs<X>(old), sd::math::sd_abs<X>(opOutput));
  }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) {
    return sd::math::sd_max<X>(sd::math::sd_abs<X>(opOutput), sd::math::sd_abs<X>(old));
  }

  SD_OP_DEF static X op(X d1, X d2, X *params) {
    return sd::math::sd_max<X>(sd::math::sd_abs<X>(d1), sd::math::sd_abs<X>(d2));
  }

  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_abs<X>(d1) > sd::math::sd_abs<X>(d2) ? d1 : d2; }

  // FIXME: this signature overlaps with MetaOp
  SD_OP_DEF static X op(X d1, X *extraParams) { return sd::math::sd_abs<X>(d1); }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return sd::math::sd_abs<X>(reduction); }
};

template <typename X>
class AMin {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::AMIN;

  SD_OP_DEF static X startingValue(const X *input) { return input[0]; }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) {
    return sd::math::sd_min<X>(sd::math::sd_abs<X>(old), sd::math::sd_abs<X>(opOutput));
  }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) {
    return sd::math::sd_min<X>(sd::math::sd_abs<X>(opOutput), sd::math::sd_abs<X>(old));
  }

  SD_OP_DEF static X op(X d1, X d2, X *params) {
    return sd::math::sd_min<X>(sd::math::sd_abs<X>(d1), sd::math::sd_abs<X>(d2));
  }

  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_min<X>(sd::math::sd_abs<X>(d1), sd::math::sd_abs<X>(d2)); }

  // FIXME: this signature overlaps with MetaOp
  SD_OP_DEF static X op(X d1, X *extraParams) { return sd::math::sd_abs<X>(d1); }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return sd::math::sd_abs<X>(reduction); }
};

template <typename X>
class Min {
 public:
  no_op_exec_special_accumulation_same no_op_exec_special_accumulation_same_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::MIN;

  SD_OP_DEF static X startingValue(const X *input) { return sd::DataTypeUtils::infOrMax<X>(); }

  SD_OP_DEF static X merge(X old, X opOutput, X *extraParams) { return sd::math::sd_min<X>(old, opOutput); }

  SD_OP_DEF static X update(X old, X opOutput, X *extraParams) { return sd::math::sd_min<X>(opOutput, old); }

  SD_OP_DEF static X op(X d1, X d2, X *params) { return sd::math::sd_min<X>(d1, d2); }

  SD_OP_DEF static X op(X d1, X d2) { return sd::math::sd_min<X>(d1, d2); }

  // FIXME: this signature overlaps with MetaOp
  SD_OP_DEF static X op(X d1, X *extraParams) { return d1; }

  SD_OP_DEF static X postProcess(X reduction, sd::LongType n, X *extraParams) { return reduction; }
};

template <typename X, typename Z>
class Norm1 {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) { return static_cast<InterType>(sd::math::sd_abs<X>(d1)); }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) { return reduction; }
};

template <typename X, typename Z>
class Norm2 {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda using InterType =
      typename AggregateType<Z>::type;
  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return sd::math::sd_sqrt<InterType, Z>(reduction);
  }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    return static_cast<InterType>(d1) * static_cast<InterType>(d1);
  }
};

template <typename X, typename Z>
class SquaredNorm {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    return static_cast<InterType>(d1) * static_cast<InterType>(d1);
  }

  SD_OP_DEF static Z postProcess(Z reduction, sd::LongType n, Z *extraParams) { return reduction; }
};

template <typename X, typename Z>
class NormFrobenius {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    auto v = sd::math::sd_abs<InterType>(d1);
    return static_cast<InterType>(v * v);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return sd::math::sd_sqrt<InterType, Z>(reduction);
  }
};

template <typename X, typename Z>
class NormP {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    return sd::math::sd_pow<X, Z, InterType>(sd::math::sd_abs<X>(d1), extraParams[0]);
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return sd::math::sd_pow<InterType, Z, Z>(reduction, static_cast<Z>(1.0f) / extraParams[0]);
  }
};

template <typename X, typename Z>
class NormMax {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = Z;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0); }

  SD_OP_DEF static Z merge(Z old, Z opOutput, Z *extraParams) { return opOutput + old; }

  SD_OP_DEF static Z update(Z old, Z opOutput, Z *extraParams) {
    return sd::math::sd_max<Z>(sd::math::sd_abs<Z>(old), sd::math::sd_abs<Z>(opOutput));
  }

  SD_OP_DEF static Z op(X d1, Z *extraParams) { return static_cast<Z>(d1); }

  SD_OP_DEF static Z postProcess(Z reduction, sd::LongType n, Z *extraParams) {
    return sd::math::sd_max<Z>(sd::math::sd_abs<Z>(reduction), sd::math::sd_abs<Z>(reduction));
  }
};

template <typename X, typename Z>
class Variance {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda

  const static functions::ReduceType reduceType = functions::ReduceType::SUM;
  using InterType = typename AggregateType<Z>::type;
  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0.0f); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return old + opOutput; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return old + opOutput; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    X mean = static_cast<InterType>(extraParams[0]);
    X ret = d1 - mean;
    return ret * ret;
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    return static_cast<Z>(reduction / static_cast<InterType>(n - 1));
  }
};

/**
 * Standard deviation of a buffer
 */
template <typename X, typename Z>
class StandardDeviation {
 public:
  no_op_exec_special_accumulation no_op_exec_special_accumulation_cuda using InterType =
      typename AggregateType<Z>::type;
  const static functions::ReduceType reduceType = functions::ReduceType::SUM;

  SD_OP_DEF static X startingValue(const X *input) { return static_cast<X>(0.0f); }

  SD_OP_DEF static InterType merge(InterType old, InterType opOutput, Z *extraParams) { return old + opOutput; }

  SD_OP_DEF static InterType update(InterType old, InterType opOutput, Z *extraParams) { return old + opOutput; }

  SD_OP_DEF static InterType op(X d1, Z *extraParams) {
    InterType mean = static_cast<InterType>(extraParams[0]);
    InterType ret = d1 - mean;
    return ret * ret;
  }

  SD_OP_DEF static Z postProcess(InterType reduction, sd::LongType n, Z *extraParams) {
    auto ret = Variance<X, InterType>::postProcess(reduction, n, extraParams);
    Z sqrtRet = sd::math::sd_sqrt<InterType, Z>(ret);
    return sqrtRet;
  }
};

template <typename X, typename Y>
class CosineSimilarity {
 public:
  static const int extraParamsLen = 2;

  SD_OP_DEF static X *generateExtraParams() {
    return nullptr;
  }

  SD_OP_DEF static void finalizeExtraParams(X *extraParams) {
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParams) {
    return reduction / (sd::math::sd_sqrt<Y, Y>(extraParams[0]) * sd::math::sd_sqrt<Y, Y>(extraParams[1]));
  }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParams) {
    extraParams[0] += static_cast<Y>(d1 * d1);
    extraParams[1] += static_cast<Y>(d2 * d2);
    return static_cast<Y>(d1 * d2);
  }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
    extraParamsTotal[0] += extraParamsLocal[0];
    extraParamsTotal[1] += extraParamsLocal[1];
  }

#ifdef __CUDACC__
  static SD_DEVICE inline Y opAtomic(X d1, X d2, Y *extraParams) {
    sd::math::atomics::sd_atomicAdd(&extraParams[0], static_cast<Y>(d1 * d1));
    sd::math::atomics::sd_atomicAdd(&extraParams[1], static_cast<Y>(d2 * d2));

    return static_cast<Y>(d1 * d2);
  }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParams) { return old + opOutput; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParams) { return update(old, opOutput, extraParams); }
};

template <typename X, typename Y>
class JaccardDistance {
 public:
  static const int extraParamsLen = 2;

  SD_OP_DEF static X *generateExtraParams() {
    return nullptr;
  }

  SD_OP_DEF static void finalizeExtraParams(X *extraParams) {
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<X>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParams) {
    // num / denom
    return (static_cast<Y>(1.0f)) - (extraParams[0] / extraParams[1]);
  }

  SD_OP_DEF static Y num(X d1, X d2) { return sd::math::sd_min<X>(d1, d2); }

  SD_OP_DEF static Y denom(X d1, X d2) { return sd::math::sd_max<X>(d1, d2); }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParams) {
    extraParams[0] += static_cast<Y>(num(d1, d2));
    extraParams[1] += static_cast<Y>(denom(d1, d2));
    return static_cast<Y>(0.0f);
  }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
    extraParamsTotal[0] += extraParamsLocal[0];
    extraParamsTotal[1] += extraParamsLocal[1];
  }

#ifdef __CUDACC__
  SD_DEVICE
  static inline Y opAtomic(X d1, X d2, Y *extraParams) {
    sd::math::atomics::sd_atomicAdd(&extraParams[0], num(d1, d2));
    sd::math::atomics::sd_atomicAdd(&extraParams[1], denom(d1, d2));

    return static_cast<Y>(0.0f);
  }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParams) { return old + opOutput; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParams) { return update(old, opOutput, extraParams); }
};

template <typename X, typename Y>
class SimpleHammingDistance {
 public:
  static const int extraParamsLen = 0;

  SD_OP_DEF static X *generateExtraParams() {
    return nullptr;
  }

  SD_OP_DEF static void finalizeExtraParams(X *extraParams) {
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParams) { return static_cast<Y>(reduction / n); }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParams) { return (d1 == d2) ? static_cast<Y>(0.0f) : static_cast<Y>(1.0f); }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}

#ifdef __CUDACC__
  SD_DEVICE
  static inline Y opAtomic(X d1, X d2, Y *extraParams) { return op(d1, d2, extraParams); }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParams) { return old + opOutput; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParams) { return update(old, opOutput, extraParams); }
};

template <typename X, typename Y>
class CosineDistance {
 public:
  static const int extraParamsLen = 2;

  SD_OP_DEF static X *generateExtraParams() {
    return nullptr;
  }

  SD_OP_DEF static void finalizeExtraParams(X *extraParams) {
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParams) {
    return (static_cast<Y>(1.0f)) -
           (reduction / (sd::math::sd_sqrt<Y, Y>(extraParams[0]) * sd::math::sd_sqrt<Y, Y>(extraParams[1])));
  }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParams) {
    extraParams[0] += static_cast<Y>(sd::math::sd_abs<X>(d1) * sd::math::sd_abs<X>(d1));
    extraParams[1] += static_cast<Y>(sd::math::sd_abs<X>(d2) * sd::math::sd_abs<X>(d2));
    return (d1 * d2);
  }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {
    extraParamsTotal[0] += extraParamsLocal[0];
    extraParamsTotal[1] += extraParamsLocal[1];
  }

#ifdef __CUDACC__
  static SD_DEVICE inline Y opAtomic(X d1, X d2, Y *extraParams) {
    sd::math::atomics::sd_atomicAdd(&extraParams[0], sd::math::sd_abs<Y>(d1) * sd::math::sd_abs<Y>(d1));
    sd::math::atomics::sd_atomicAdd(&extraParams[1], sd::math::sd_abs<Y>(d2) * sd::math::sd_abs<Y>(d2));

    return (d1 * d2);
  }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParams) { return old + opOutput; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParams) { return update(old, opOutput, extraParams); }
};

/**
 * Dot product between 2 arrays
 */
template <typename X, typename Y>
class Dot {
 public:
  static const int extraParamsLen = 0;

  SD_OP_DEF static X *generateExtraParams() { return nullptr; }

  SD_OP_DEF static void finalizeExtraParams(X *extraParamsRef) {
    // no-op

  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParamsRef) { return reduction; }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParamsRef) { return static_cast<Y>(d1 * d2); }

#ifdef __CUDACC__
  SD_DEVICE
  static inline Y opAtomic(X d1, X d2, Y *extraParamsRef) { return op(d1, d2, extraParamsRef); }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParamsRef) { return opOutput + old; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParamsRef) { return update(old, opOutput, extraParamsRef); }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}
};

/**
 * Op to check equality within arrays
 */
template <typename X, typename Z>
class EqualsWithEps {
 public:
  static const int extraParamsLen = 0;

  SD_OP_DEF static X *generateExtraParams() { return nullptr; }

  SD_OP_DEF static void finalizeExtraParams(X *extraParamsRef) {
    // no-op
  }

  SD_OP_DEF static Z startingValue(const X *input) { return static_cast<Z>(0.0f); }

  SD_OP_DEF static Z postProcess(Z reduction, sd::LongType n, Z *extraParamsRef) { return reduction; }

  SD_OP_DEF static Z op(X d1, X d2, Z *extraParamsRef) {
    double eps = sd::math::sd_abs<double>(extraParamsRef[2]);
    return static_cast<Z>(!sd::math::sd_eq<X>(d1, d2, eps));
  }

#ifdef __CUDACC__
  SD_DEVICE
  static inline Z opAtomic(X d1, X d2, Z *extraParamsRef) { return op(d1, d2, extraParamsRef); }
#endif

  SD_OP_DEF static Z update(Z old, Z opOutput, Z *extraParamsRef) { return opOutput + old; }

  SD_OP_DEF static Z merge(X old, Z opOutput, Z *extraParamsRef) { return update(old, opOutput, extraParamsRef); }

  SD_OP_DEF static void aggregateExtraParams(Z *extraParamsTotal, Z *extraParamsLocal) {}
};

template <typename X, typename Y>
class EuclideanDistance {
 public:
  static const int extraParamsLen = 0;

  SD_OP_DEF static X *generateExtraParams() { return nullptr; }

  SD_OP_DEF static void finalizeExtraParams(X *extraParamsRef) {
    // no-op
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParamsRef) {
    return sd::math::sd_sqrt<Y, Y>(reduction);
  }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParamsRef) {
    X ret = d1 - d2;
    return static_cast<Y>(ret * ret);
  }

#ifdef __CUDACC__
  SD_DEVICE
  static inline Y opAtomic(X d1, X d2, Y *extraParamsRef) { return op(d1, d2, extraParamsRef); }
#endif

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParamsRef) { return opOutput + old; }

  SD_OP_DEF static Y merge(Y old, Y opOutput, Y *extraParamsRef) { return update(old, opOutput, extraParamsRef); }
  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}
};

template <typename X, typename Y>
class ManhattanDistance {
 public:
  static const int extraParamsLen = 0;

  SD_OP_DEF static X *generateExtraParams() { return nullptr; }

  SD_OP_DEF static void finalizeExtraParams(X *extraParamsRef) {
    // no-op
  }

  SD_OP_DEF static Y startingValue(const X *input) { return static_cast<Y>(0.0f); }

  SD_OP_DEF static Y postProcess(Y reduction, sd::LongType n, Y *extraParamsRef) { return reduction; }

  SD_OP_DEF static Y op(X d1, X d2, Y *extraParamsRef) { return sd::math::sd_abs<X>(d1 - d2); }

  SD_OP_DEF static Y update(Y old, Y opOutput, Y *extraParamsRef) { return old + opOutput; }

  SD_OP_DEF static void aggregateExtraParams(Y *extraParamsTotal, Y *extraParamsLocal) {}

#ifdef __CUDACC__
  SD_DEVICE
  static inline Y opAtomic(X d1, X d2, Y *extraParamsRef) { return op(d1, d2, extraParamsRef); }
#endif

  SD_OP_DEF static Y merge(X old, X opOutput, X *extraParamsRef) { return update(old, opOutput, extraParamsRef); }
};

template <typename X, typename Z>
class IndexAbsoluteMax {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return sd::math::sd_abs<X>(val);
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> update(
      functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
    opOutput.value = sd::math::sd_abs<X>(opOutput.value);
    old.value = sd::math::sd_abs<X>(old.value);
    if (opOutput.value > old.value) return opOutput;
#ifdef __CUDACC__
      // workaround for cuda race condition at merge phase
    else if (opOutput.value == old.value && opOutput.index < old.index)
      return opOutput;
#elif defined(__GNUC__)

#endif
    return old;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (sd::math::sd_abs<X>(f1.value) > sd::math::sd_abs<X>(f2.value)) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return 0; }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class FirstIndex {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }

  static SD_HOST_DEVICE functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old,
                                                                     functions::indexreduce::IndexValue<X> &opOutput,
                                                                     X *extraParams) {
#ifdef __CUDACC__
    if (opOutput.index < 0) return old;
#endif

    auto res = simdOps::MatchCondition<X, X>::op(opOutput.value, extraParams);

    if (res == static_cast<X>(0)) return old;

    if (old.index < 0) return opOutput;

    if (old.index > opOutput.index) return opOutput;

    return old;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = -1;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.index > f2.index) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
};

template <typename X, typename Z>
class LastIndex {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }

  static SD_HOST_DEVICE functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old,
                                                                     functions::indexreduce::IndexValue<X> &opOutput,
                                                                     X *extraParams) {
#ifdef __CUDACC__
    if (opOutput.index < 0) return old;
#endif

    auto res = simdOps::MatchCondition<X, X>::op(opOutput.value, extraParams);

    if (res == static_cast<X>(0)) return old;

    if (old.index < 0) return opOutput;

    if (old.index < opOutput.index) return opOutput;

    return old;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = -1;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.index < f2.index) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
};

template <typename X, typename Z>
class IndexMax {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }

  static SD_HOST_DEVICE functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old,
                                                                     functions::indexreduce::IndexValue<X> &opOutput,
                                                                     X *extraParams) {
    if (opOutput.value > old.value) {
      return opOutput;
    }
#ifdef __CUDACC__
      // workaround for cuda race condition at merge phase
    else if (opOutput.value == old.value && opOutput.index < old.index)
      return opOutput;
#elif defined(__GNUC__)

#endif
    return old;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.value > f2.value) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class IndexAbsoluteMin {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return sd::DataTypeUtils::infOrMax<X>(); }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> update(
      functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
    opOutput.value = sd::math::sd_abs<X>(opOutput.value);
    old.value = sd::math::sd_abs<X>(old.value);
    if (opOutput.value < old.value) return opOutput;

#ifdef __CUDACC__
      // workaround for cuda race condition at merge phase
    else if (opOutput.value == old.value && opOutput.index < old.index)
      return opOutput;
#elif defined(__GNUC__)

#endif
    return old;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (sd::math::sd_abs<X>(f1.value) < sd::math::sd_abs<X>(f2.value)) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class IndexMin {
 public:
  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }

  static SD_HOST_DEVICE inline X startingValue(const X *input) { return sd::DataTypeUtils::infOrMax<X>(); }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> update(
      functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
    if (opOutput.value < old.value) return opOutput;

#ifdef __CUDACC__
      // workaround for cuda race condition at merge phase
    else if (opOutput.value == old.value && opOutput.index < old.index)
      return opOutput;
#elif defined(__GNUC__)

#endif
    return old;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.value < f2.value) return f2;
    return f1;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }

  static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class SummaryStatsVariance {
 public:
  static SD_HOST_DEVICE inline Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
    if (biasCorrected) {
      Z ret = static_cast<Z>(val.varianceBiasCorrected());
      if (ret < static_cast<Z>(0.0f)) return static_cast<Z>(val.variance());
      return ret;
    }
    return static_cast<Z>(val.variance());
  }

  static SD_HOST_DEVICE inline functions::summarystats::SummaryStatsData<X> op(
      functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class SummaryStatsStandardDeviation {
 public:
  static SD_HOST_DEVICE inline Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
    if (biasCorrected) {
      auto ret = static_cast<Z>(val.varianceBiasCorrected());
      if (ret < static_cast<Z>(0.0f))
        return sd::math::sd_sqrt<double, Z>(val.variance());
      else
        return sd::math::sd_sqrt<double, Z>(ret);
    }
    return sd::math::sd_sqrt<double, Z>(val.variance());
  }

  static SD_HOST_DEVICE inline functions::summarystats::SummaryStatsData<X> op(
      functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
    return d1;
  }
};

template <typename X>
class DropOut {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  inline SD_DEVICE static X
  op(X d1, X *params) {
    X prob = params[0];

#ifdef __CUDACC__
    X length = params[1];
    X tid = blockIdx.x * blockDim.x + threadIdx.x;
    X rnd = sd::math::sd_abs<X>(sd::math::sd_cos<X>(static_cast<X>(clock64()) * static_cast<X>(tid) +
                                                    static_cast<X>(length) * static_cast<X>(tid)));
#else
    X rnd = static_cast<X>(rand() / RAND_MAX);
#endif
    return rnd >= prob ? static_cast<X>(0.0f) : d1;
  }
};

template <typename X, typename Y, typename Z>
class DropOutInverted {
 public:
  no_op_exec_special no_op_exec_special_cuda
#ifdef __CUDACC__
  SD_DEVICE
#endif
  inline static Z
  op(X d1, Y d2, Z *params) {
    Y prob = d2;
#ifdef __CUDACC__
    X length = params[1];
    X tid = blockIdx.x * blockDim.x + threadIdx.x;
    X rnd = sd::math::sd_abs<X>(sd::math::sd_cos<X>(static_cast<X>(clock64()) * static_cast<X>(tid) +
                                                    static_cast<X>(length) * static_cast<X>(tid)));
#else
    X rnd = static_cast<X>(rand() / RAND_MAX);
#endif
    return rnd >= static_cast<X>(prob) ? static_cast<Z>(0.0f) : reinterpret_cast<Z>(d1 / static_cast<X>(prob));
  }
};

template <typename X, typename Y, typename Z>
class ReplaceNans {
 public:
  no_op_exec_special no_op_exec_special_cuda

  SD_OP_DEF static Z
  op(X d1, Y d2, Z *params) {
    return sd::math::sd_isnan(d1) ? static_cast<Z>(d2) : static_cast<Z>(d1);
  }
};

// this op is used for conditional pairwise transforms only
template <typename X, typename Y, typename Z>
class CompareAndReplace {
 public:
  // op definition for PairWise Transform
  SD_OP_DEF static Z op(X d1, Y d2, Z *params) {
    auto zd1 = static_cast<Z>(d1);
    auto zd2 = static_cast<Z>(d2);
    auto compare = params[0];
    auto eps = params[2];
    int mode = (int)params[3];
    if (mode == 0)  // equals
      if (sd::math::sd_abs<Z>(zd1 - compare) <= eps)
        return zd2;
      else
        return zd1;
    else if (mode == 1)  // not equals eps
      if (sd::math::sd_abs<Z>(zd1 - compare) > eps)
        return zd2;
      else
        return zd1;
    else if (mode == 2)  // less_than eps
      if (zd1 < compare)
        return zd2;
      else
        return zd1;
    else if (mode == 3)  // greater_than
      if (zd1 > compare)
        return zd2;
      else
        return zd1;
    else if (mode == 4)  // less_or_equals_than
      if (zd1 <= compare)
        return zd2;
      else
        return zd1;
    else if (mode == 5)  // greater_or_equals_than
      if (zd1 >= compare)
        return zd2;
      else
        return zd1;
    else if (mode == 6)  // abs_less_than
      if (sd::math::sd_abs<Z>(zd1) < compare)
        return zd2;
      else
        return zd1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<Z>(zd1) > compare)
        return zd2;
      else
        return zd1;
      //equivalent case to NOT_FINITE
    else if (mode == 8 || mode == 15)  // is inf
      if (sd::math::sd_isinf(zd1))
        return zd2;
      else
        return zd1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan(zd1))
        return zd2;
      else
        return zd1;
    else if (mode == 10)
      if (zd1 == compare)
        return zd2;
      else
        return zd1;
    else if (mode == 11)
      if (zd1 != compare)
        return zd2;
      else
        return zd1;
    else if (mode == 12)  // abs_greater_or_equals_than
      if (sd::math::sd_abs<Z>(zd1) >= compare)
        return zd2;
      else
        return zd1;
    else if (mode == 13) {  // abs_less_or_equals_than
      if (sd::math::sd_abs<Z>(zd1) <= compare) return zd2;
    }
    else if (mode == 14) {  // is_inf
      if (!sd::math::sd_isinf(zd1))
        return zd2;
      else
        return zd1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return zd1;
  }
};

template <typename X, typename Y, typename Z>
class CompareAndSet {
 public:
  // op definition for PairWise Transform
  SD_OP_DEF static Z op(X dX, Y dY, Z *params) {
    auto d1 = static_cast<Z>(dX);
    auto d2 = static_cast<Z>(dY);
    auto compare = params[0];
    auto eps = params[2];
    auto mode = static_cast<int>(params[3]);
    if (mode == 0)  // equals
      if (sd::math::sd_abs<Z>(d2 - compare) <= eps)
        return d2;
      else
        return d1;
    else if (mode == 1)  // not equals
      if (sd::math::sd_abs<Z>(d2 - compare) > eps)
        return d2;
      else
        return d1;
    else if (mode == 2)  // less_than
      if (d2 < compare)
        return d2;
      else
        return d1;
    else if (mode == 3)  // greater_than
      if (d2 > compare)
        return d2;
      else
        return d1;
    else if (mode == 4)  // less_or_equals_than
      if (d2 <= compare)
        return d2;
      else
        return d1;
    else if (mode == 5)  // greater_or_equals_than
      if (d2 >= compare)
        return d2;
      else
        return d1;
    else if (mode == 6)  // abs_less_than
      if (sd::math::sd_abs<Z>(d2) < compare)
        return d2;
      else
        return d1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<Z>(d2) > compare)
        return d2;
      else
        return d1;
      //equivalent case to NOT_FINITE
    else if (mode == 8 || mode == 15)  // is inf
      if (sd::math::sd_isinf(d2))
        return d2;
      else
        return d1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan(d2))
        return d2;
      else
        return d1;
    else if (mode == 10)
      if (d2 == compare)
        return d2;
      else
        return d1;
    else if (mode == 11)
      if (d2 != compare)
        return d2;
      else
        return d1;
    else if (mode == 12)  // abs_greater_or_equals_than
      if (sd::math::sd_abs<Z>(d1) >= compare)
        return d2;
      else
        return d1;
    else if (mode == 13)  // abs_less_or_equals_than
      if (sd::math::sd_abs<Z>(d1) <= compare)
        return d2;
      else
        return d1;
    else if (mode == 14) {  // is_inf
      if (!sd::math::sd_isinf(d1))
        return d2;
      else
        return d1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return d1;
  }
};

template <typename X>
class CompareAndSetTransform {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda

  // op definition for Transform
  SD_OP_DEF static X
  op(X d1, X *params) {
    auto compare = params[0];
    auto set = params[1];
    auto eps = params[2];

    // with mode == 0 we do set if d1 equals to compare, and with mode == 1 - we go otherwise
    int mode = (int)params[3];
    if (mode == 0)  // equals
      if (sd::math::sd_abs<X>(d1 - compare) <= eps)
        return set;
      else
        return d1;
      // return sd::math::sd_abs<T>(d1 - compare) <= eps ? set : d1;
    else if (mode == 1)  // not equals
      if (sd::math::sd_abs<X>(d1 - compare) > eps)
        return set;
      else
        return d1;
      // return sd::math::sd_abs<T>(d1 - compare) > eps ? set : d1;
    else if (mode == 2)  // less_than
      if (d1 < compare)
        return set;
      else
        return d1;
    else if (mode == 3)  // greater_than
      if (d1 > compare)
        return set;
      else
        return d1;
    else if (mode == 4)  // less_or_equals_than
      if (d1 <= compare)
        return set;
      else
        return d1;
    else if (mode == 5)  // greater_or_equals_than
      if (d1 >= compare)
        return set;
      else
        return d1;
    else if (mode == 6)  // abs_less_than
      if (sd::math::sd_abs<X>(d1) < compare)
        return set;
      else
        return d1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<X>(d1) > compare)
        return set;
      else
        return d1;
    else if (mode == 8)  // is inf
      if (sd::math::sd_isinf(d1))
        return set;
      else
        return d1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan(d1))
        return set;
      else
        return d1;
    else if (mode == 10)
      if (d1 == compare)
        return set;
      else
        return d1;
    else if (mode == 11)
      if (d1 != compare)
        return set;
      else
        return d1;
    else if (mode == 12)  // abs_greater_or_equals_than
      if (sd::math::sd_abs<X>(d1) >= compare)
        return set;
      else
        return d1;
    else if (mode == 13)  // abs_less_or_equals_than
      if (sd::math::sd_abs<X>(d1) <= compare)
        return set;
      else
        return d1;
    else if (mode == 14) {  // is_inf
      if (!sd::math::sd_isinf(d1))
        return compare;
      else
        return d1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return d1;
  }
};

}  // namespace simdOps

#endif
