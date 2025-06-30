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
#include <math/templatemath.h>
#include <system/Environment.h>
#include <system/common.h>
#include <system/op_boilerplate.h>

#include <codecvt>
#include <vector>

#include "helpers/unicode.h"
#include "op_macros_meta.h"

// =============================================================================
// CONSTANTS
// =============================================================================

#define SELU_ALPHA 1.6732632423543772848170429916717
#define SELU_LAMBDA 1.0507009873554804934193349852946
#define SD_STRING_ASSIGN_TEMP_BUFFER_BYTES 256




namespace simdOps {


// =============================================================================
// BINARY ARITHMETIC OPERATIONS
// =============================================================================

DECLARE_BINARY_MATH_OP(Add, sd_add)
DECLARE_BINARY_MATH_OP(Subtract, sd_subtract)
DECLARE_BINARY_MATH_OP(Multiply, sd_multiply)
DECLARE_BINARY_MATH_OP(Divide, sd_divide)


// Reverse operations
DECLARE_REVERSE_BINARY_MATH_OP(ReverseSubtract, sd_subtract, 0.f)
DECLARE_REVERSE_BINARY_MATH_OP(ReverseDivide, sd_divide, 1)

// HardTanh - uses complex conditional macro
DECLARE_UNARY_COMPLEX_CONDITIONAL_OP(HardTanh,
                                     d1 < static_cast<X>(-1), static_cast<X>(-1),
                                     d1 > static_cast<X>(1), static_cast<X>(1),
                                     d1)

DECLARE_UNARY_CONDITIONAL_OP(RectifiedTanhDerivative,
                             d1 > static_cast<X>(0.f),
                             sd::math::sd_tanhderivative<X COMMA X>(d1),
                             static_cast<X>(0.f))



DECLARE_BINARY_MATH_OP_XZ(Atan2, sd_atan2)


DECLARE_BINARY_MATH_OP_WITH_STARTING(PowDerivative,
                                     static_cast<Z>(d2) * sd::math::sd_pow<X COMMA Z COMMA Z>(d1 COMMA static_cast<Z>(d2) - static_cast<Z>(1.f)),
                                     static_cast<Z>(d1),
                                     params[0] * sd::math::sd_pow<X COMMA Z COMMA Z>(d1 COMMA static_cast<Z>(params[0]) - static_cast<Z>(1.f)),
                                     static_cast<X>(0)
)

DECLARE_BINARY_COPY_OP(AMaxPairwise,
                       sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d1)) > sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d2)) ? static_cast<Z>(d1) : static_cast<Z>(d2),
                       sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d1)) > sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d2)) ? static_cast<Z>(d1) : static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)

DECLARE_BINARY_COPY_OP(AMinPairwise,
                       sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d1)) < sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d2)) ? static_cast<Z>(d1) : static_cast<Z>(d2),
                       sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d1)) < sd::math::sd_abs<Z COMMA Z>(static_cast<Z>(d2)) ? static_cast<Z>(d1) : static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)

DECLARE_BINARY_COPY_OP(MaxPairwise,
                       sd::math::sd_max<Z>(static_cast<Z>(d1) COMMA static_cast<Z>(d2)),
                       sd::math::sd_max<Z>(static_cast<Z>(d1) COMMA static_cast<Z>(d2)),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)

DECLARE_BINARY_COPY_OP(MinPairwise,
                       sd::math::sd_min<X COMMA Y COMMA Z>(d1 COMMA d2),
                       sd::math::sd_min<X COMMA Y COMMA Z>(d1 COMMA d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)

DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT(CosineSimilarity,
// Bool logic (for boolean input types)
                                     extraParamsRef[0] += static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d1)); \
                                     extraParamsRef[1] += static_cast<Y>(static_cast<int>(d2) * static_cast<int>(d2)); \
                                     return static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d2));,

// Normal logic (for non-boolean types)
                                     extraParamsRef[0] += static_cast<Y>(d1 * d1); \
                                     extraParamsRef[1] += static_cast<Y>(d2 * d2); \
                                     return static_cast<Y>(d1 * d2);,

                                     2,  // extraParamsLen
                                     0.0f,  // starting value

// Post process - calculate cosine similarity from dot product and norms
                                     reduction / (sd::math::sd_sqrt<Y COMMA Y>(extraParamsRef[0]) * sd::math::sd_sqrt<Y COMMA Y>(extraParamsRef[1]))
)



DECLARE_BINARY_MATH_OP(IGamma, sd_igamma)


DECLARE_BINARY_MATH_OP(IGammac, sd_igammac)



DECLARE_BINARY_COPY_OP(LogX,
                       sd::math::sd_log<X COMMA Z>(d1) / sd::math::sd_log<Y COMMA Z>(d2),
                       sd::math::sd_log<X COMMA Z>(d1) / sd::math::sd_log<Y COMMA Z>(d2),
                       static_cast<Z>(d1),
                       sd::math::sd_log<X COMMA Z>(d1) / sd::math::sd_log<Y COMMA Z>(params[0])
)




// ASinhDerivative - uses complex math expression macro
DECLARE_UNARY_COMPLEX_MATH_OP(ASinhDerivative,
                              static_cast<X>(1.f) / (sd::math::sd_sqrt<X, X>(sd::math::sd_pow<X, X, X>(d1, static_cast<X>(2.f)) + static_cast<X>(1.f))))

// ACoshDerivative - uses complex math expression macro
DECLARE_UNARY_COMPLEX_MATH_OP(ACoshDerivative,
                              static_cast<X>(1.f) / (sd::math::sd_sqrt<X, X>(d1 - static_cast<X>(1.f)) * sd::math::sd_sqrt<X, X>(d1 + static_cast<X>(1.f))))

// Power operations


DECLARE_POWER_OP(Pow, sd_pow)
DECLARE_REVERSE_BINARY_MATH_OP(ReversePow, sd_pow, 1)


DECLARE_UNARY_SIMPLE_OP(TanDerivative, static_cast<X>(1.f) / sd::math::sd_pow<X COMMA X COMMA X>(sd::math::sd_cos<X COMMA X>(d1) COMMA static_cast<X>(2.0f)))




DECLARE_SQUARED_SUBTRACT_OP(SquaredSubtract, sd_subtract)
DECLARE_SQUARED_REVERSE_SUBTRACT_OP(SquaredReverseSubtract, sd_subtract)

// =============================================================================
// COMPARISON OPERATIONS
// =============================================================================

DECLARE_COMPARISON_OP(EqualTo, ==)
DECLARE_COMPARISON_OP(NotEqualTo, !=)
DECLARE_COMPARISON_OP(GreaterThan, >)
DECLARE_COMPARISON_OP(GreaterThanOrEqual, >=)
DECLARE_COMPARISON_OP(LessThan, <)
DECLARE_COMPARISON_OP(LessThanOrEqual, <=)

// =============================================================================
// UNARY MATH OPERATIONS
// =============================================================================

DECLARE_UNARY_MATH_OP(Abs, sd_abs)
DECLARE_UNARY_MATH_OP(Ceiling, sd_ceil)
DECLARE_UNARY_MATH_OP(Cosine, sd_cos)
DECLARE_UNARY_MATH_OP(Exp, sd_exp)
DECLARE_UNARY_MATH_OP(Floor, sd_floor)
DECLARE_UNARY_MATH_OP(Log, sd_log)
DECLARE_UNARY_MATH_OP(Sin, sd_sin)
DECLARE_UNARY_MATH_OP(Tanh, sd_tanh)
DECLARE_UNARY_MATH_OP(Sigmoid, sd_sigmoid)

DECLARE_UNARY_SIMPLE_OP(Neg, -d1)
DECLARE_UNARY_SIMPLE_OP(Square, d1 * d1)
DECLARE_UNARY_SIMPLE_OP(Cube, d1 * d1 * d1)
DECLARE_UNARY_SIMPLE_OP(Identity, d1)
DECLARE_UNARY_SIMPLE_OP(OneMinus, static_cast<X>(1) - d1)
DECLARE_UNARY_SIMPLE_OP(Reciprocal, static_cast<X>(1) / d1)




// =============================================================================
// CONDITIONAL OPERATIONS
// =============================================================================


DECLARE_UNARY_CONDITIONAL_OP(Sign,
                             (d1 > static_cast<X>(0)) - (d1 < static_cast<X>(0)),
                             static_cast<X>(1), static_cast<X>(-1))


DECLARE_UNARY_CONDITIONAL_OP(HardTanhDerivative,
                             ((d1 >= static_cast<X>(-1.f) && d1 <= static_cast<X>(1.f)) ? static_cast<X>(1.f) : static_cast<X>(0.f)), d1, d1)

DECLARE_UNARY_CONDITIONAL_OP(HardSigmoidDerivative,
                             d1 < static_cast<X>(-2.5f) || d1 > static_cast<X>(2.5f) ? static_cast<X>(0.f) : static_cast<X>(0.2f), d1, d1)

DECLARE_BINARY_MATH_OP(Remainder, sd_remainder)
DECLARE_BINARY_MATH_OP(FMod, sd_fmod)


DECLARE_SAFE_DIVISION_OP(DivideNoNan, d2 == static_cast<Y>(0))
DECLARE_SAFE_DIVISION_OP(SafeDivide, d2 == static_cast<Y>(0))

// Floor division:
DECLARE_FLOOR_DIVISION_OP(FloorDiv, sd_floor)



DECLARE_BINARY_MATH_OP_WITH_STARTING(TruncateDiv,
                                     static_cast<Z>(sd::math::sd_divide<int,int,int>(static_cast<int>(d1), static_cast<int>(d2))),
                                     static_cast<Z>(d1),
                                     static_cast<Z>(sd::math::sd_divide<int,int,int>(static_cast<int>(d1), static_cast<int>(params[0]))),
                                     static_cast<X>(1)
)

DECLARE_BINARY_MATH_OP_WITH_STARTING(TruncateMod,
                                     static_cast<Z>(static_cast<int>(d1) % static_cast<int>(d2)),
                                     static_cast<Z>(d1),
                                     static_cast<Z>(static_cast<int>(d1) % static_cast<int>(params[0])),
                                     static_cast<X>(0)
)




DECLARE_UNARY_IDENTITY_OP(Copy)

template <typename X, typename Y, typename Z>
class FloorMod {
 private:
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2) {
    Z m = sd::math::sd_fmod<X,Y,Z>(d1, d2);
    return (d1 < static_cast<X>(0)) == (d2 < static_cast<Y>(0))
           ? m
           : sd::math::sd_fmod<Z,Y,Z>(m + static_cast<Z>(d2), d2);
  }
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2, Z *params) { return op_logic(d1, d2); }
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1) { return static_cast<Z>(d1); }
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y *params) {
    Z m = sd::math::sd_fmod<X,Y,Z>(d1, params[0]);
    return (d1 < static_cast<X>(0)) == (params[0] < static_cast<Y>(0))
           ? m
           : sd::math::sd_fmod<Z,Y,Z>(m + static_cast<Z>(params[0]), params[0]);
  }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1) { return op_logic(d1); }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, Y *params) { return op_logic(d1, params); }

 public:
  static SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2);
    else return op_simd(d1, d2);
  }
  static SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2, params);
    else return op_simd(d1, d2, params);
  }
  static SD_HOST_DEVICE SD_INLINE Z op(X d1) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1);
    else return op_simd(d1);
  }
  static SD_HOST_DEVICE SD_INLINE Z op(X d1, Y *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, params);
    else return op_simd(d1, params);
  }
  SD_HOST_DEVICE SD_INLINE static X startingValue() { return static_cast<X>(0); }
};

DECLARE_BINARY_COPY_OP(CopyPws,
                       static_cast<Z>(d2),
                       static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)

DECLARE_BINARY_COPY_OP(Copy2,
                       static_cast<Z>(d2),
                       static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)



DECLARE_BINARY_COPY_OP(Axpy,
                       static_cast<Z>(d2 + d1),
                       params[0] * static_cast<Z>(d1) + static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)



DECLARE_BINARY_PARAM_OP(LstmClip,
                        [&]() {
                          X _v = static_cast<X>(d2);
                          if (d1 > _v) return static_cast<Z>(_v);
                          else if (d1 < -_v) return static_cast<Z>(-_v);
                          else return static_cast<Z>(d1);
                        }(),
                        no_op_exec_special no_op_exec_special_cuda
)

DECLARE_BINARY_PARAM_OP(Step,
                        (d1 > static_cast<X>(d2) ? static_cast<Z>(1) : static_cast<Z>(0)),
                        no_op_exec_special_same no_op_exec_special_same_cuda
)

DECLARE_BINARY_PARAM_OP(SXELogitsSmoother,
                        static_cast<Z>(d1 * (static_cast<X>(1.f) - static_cast<X>(d2)) + static_cast<X>(0.5f) * static_cast<X>(d2)),
// no special boilerplate needed
)



DECLARE_UNARY_MATH_OP(Round, sd_round)
DECLARE_UNARY_MATH_OP(Rint, sd_rint)
DECLARE_UNARY_MATH_OP(Erf, sd_erf)
DECLARE_UNARY_MATH_OP(Erfc, sd_erfc)
DECLARE_UNARY_MATH_OP(ASin, sd_asin)
DECLARE_UNARY_MATH_OP(ACos, sd_acos)
DECLARE_UNARY_MATH_OP(ATan, sd_atan)
DECLARE_UNARY_MATH_OP(ATanh, sd_atanh)
DECLARE_UNARY_MATH_OP(ASinh, sd_asinh)
DECLARE_UNARY_MATH_OP(ACosh, sd_acosh)
DECLARE_UNARY_MATH_OP(Sinh, sd_sinh)
DECLARE_UNARY_MATH_OP(Cosh, sd_cosh)
DECLARE_UNARY_MATH_OP(Tan, sd_tan)
DECLARE_UNARY_MATH_OP(SoftSign, sd_softsign)

// Log1p - simple math expression
DECLARE_UNARY_SIMPLE_OP(Log1p, sd::math::sd_log<X COMMA X>(1 + d1))

// Expm1 - simple math expression
DECLARE_UNARY_SIMPLE_OP(Expm1, sd::math::sd_exp<X COMMA X>(d1) - static_cast<X>(1))

// StabilizeFP16 - conditional operation
DECLARE_UNARY_CONDITIONAL_OP(StabilizeFP16,
                             d1 <= static_cast<X>(0),
                             static_cast<X>(sd::DataTypeUtils::min<float16>()),
                             d1)
// StabilizeX - conditional operation
DECLARE_UNARY_CONDITIONAL_OP(StabilizeX,
                             d1 <= static_cast<X>(0),
                             sd::DataTypeUtils::min<X>(),
                             d1)

// SoftPlus - simple math function
DECLARE_UNARY_MATH_OP(SoftPlus, sd_softplus)

// SoftMax - complex math expression using params
DECLARE_UNARY_COMPLEX_MATH_OP(SoftMax,
                              sd::math::sd_exp<X COMMA X>(d1 - params[0]) / params[1])

// LogSoftMax - simple math expression using params
DECLARE_UNARY_SIMPLE_OP(LogSoftMax, (d1 - params[0]) - params[1])

// Sech - simple math expression (reciprocal of cosh)
DECLARE_UNARY_SIMPLE_OP(Sech, static_cast<X>(1) / sd::math::sd_cosh<X COMMA X>(d1))

// Csch - simple math expression (reciprocal of sinh)
DECLARE_UNARY_SIMPLE_OP(Csch, static_cast<X>(1) / sd::math::sd_sinh<X COMMA X>(d1))

// Coth - simple math expression (cosh/sinh)
DECLARE_UNARY_SIMPLE_OP(Coth, sd::math::sd_cosh<X COMMA X>(d1) / sd::math::sd_sinh<X COMMA X>(d1))



DECLARE_UNARY_COMPLEX_CONDITIONAL_OP(ClipByValue,
                                     d1 > params[1], params[1],
                                     d1 < params[0], params[0],
                                     d1)
template <typename X>
class LGamma {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda
      SD_HOST_DEVICE SD_INLINE static X op(X d1, X *params) {
    return sd::math::sd_lgamma<X, X>(d1);
  }
};



template <typename X>
class SetRange {
 private:
  static SD_HOST_DEVICE SD_INLINE X op_logic(X d1, X *params) {
    auto min = params[0];
    auto max = params[1];
    if (static_cast<X>(d1) >= min && static_cast<X>(d1) <= max) return d1;
    if (min == static_cast<X>(0) && max == static_cast<X>(1)) {
      auto val = static_cast<X>(1) / (static_cast<X>(1) + sd::math::sd_exp<X, X>(-d1));
      return (sd::math::sd_floor<X, X>(val * (max - min)) + min);
    }
    return (sd::math::sd_floor<X, X>(d1 * (max - min)) + min);
  }
  static SD_HOST_DEVICE SD_INLINE X op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  no_op_exec_special_same no_op_exec_special_same_cuda;
  static SD_HOST_DEVICE SD_INLINE X op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)
      return op_logic(d1, params);
    else
      return op_simd(d1, params);
  }
};


DECLARE_UNARY_SIMPLE_OP(Affine, params[0] * d1 + params[1])




template <typename X>
class Stabilize {
 private:
  static SD_HOST_DEVICE SD_INLINE X op_logic(X d1, X *params) {
    X k = params[0];
    if (d1 * k > static_cast<X>(SD_MAX_CUTFOFF))
      return static_cast<X>(SD_MAX_CUTFOFF) / k;
    else if (d1 * k < static_cast<X>(SD_MIN_CUTFOFF))
      return static_cast<X>(SD_MIN_CUTFOFF) / k;
    return d1;
  }
  static X op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  no_op_exec_special_same no_op_exec_special_same_cuda;
  static SD_HOST_DEVICE SD_INLINE X op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)
      return op_logic(d1, params);
    else
      return op_simd(d1, params);
  }
};

DECLARE_UNARY_SIMPLE_OP(Ones, static_cast<X>(1.0f))



DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT(JaccardDistance,
                                     Y num_val = static_cast<Y>(static_cast<int>(d1) & static_cast<int>(d2)); \
                                     Y denom_val = static_cast<Y>(static_cast<int>(d1) | static_cast<int>(d2)); \
                                     extraParamsRef[0] += num_val; \
                                     extraParamsRef[1] += denom_val; \
                                     return static_cast<Y>(0.0f);,
                                     Y num_val = static_cast<Y>(sd::math::sd_min<X>(d1 COMMA d2)); \
                                     Y denom_val = static_cast<Y>(sd::math::sd_max<X>(d1 COMMA d2)); \
                                     extraParamsRef[0] += num_val; \
                                     extraParamsRef[1] += denom_val; \
                                     return static_cast<Y>(0.0f);,
                                     2,
                                     0.0f,
                                     (static_cast<Y>(1.0f)) - (extraParamsRef[0] / extraParamsRef[1])
)


DECLARE_HAMMING_DISTANCE_OP_WITH_BOOL_SUPPORT(SimpleHammingDistance,
                                              (static_cast<int>(d1) == static_cast<int>(d2)) ? 0.0f : 1.0f,
                                              (d1 == d2) ? 0.0f : 1.0f,
                                              0.0f
)


DECLARE_REDUCE3_OP_WITH_BOOL_SUPPORT(CosineDistance,
                                     extraParamsRef[0] += static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d1)); \
                                     extraParamsRef[1] += static_cast<Y>(static_cast<int>(d2) * static_cast<int>(d2)); \
                                     return static_cast<Y>(static_cast<int>(d1) * static_cast<int>(d2));,
                                     extraParamsRef[0] += static_cast<Y>(sd::math::sd_abs<X COMMA X>(d1) * sd::math::sd_abs<X COMMA X>(d1)); \
                                     extraParamsRef[1] += static_cast<Y>(sd::math::sd_abs<X COMMA X>(d2) * sd::math::sd_abs<X COMMA X>(d2)); \
                                     return static_cast<Y>(d1 * d2);,
                                     2,
                                     0.0f,
                                     (static_cast<Y>(1.0f)) - (reduction / (sd::math::sd_sqrt<Y COMMA Y>(extraParamsRef[0]) * sd::math::sd_sqrt<Y COMMA Y>(extraParamsRef[1])))
)

DECLARE_DISTANCE_OP_WITH_BOOL_SUPPORT(Dot,
                                      static_cast<int>(d1) * static_cast<int>(d2),
                                      d1 * d2,
                                      0.0f
)
DECLARE_BOOLEAN_OP_WITH_TYPE_SAFETY(EqualsWithEps,
                                    sd::math::sd_eq<X COMMA X>(d1 COMMA d2 COMMA eps),
                                    1.0f
)

DECLARE_DISTANCE_OP_WITH_BOOL_SUPPORT(EuclideanDistance,
                                      static_cast<int>(d1) != static_cast<int>(d2) ? 1 : 0,
                                      (d1 - d2) * (d1 - d2),
                                      0.0f
)

DECLARE_DISTANCE_OP_WITH_BOOL_SUPPORT(ManhattanDistance,
                                      static_cast<int>(d1) != static_cast<int>(d2) ? 1 : 0,
                                      sd::math::sd_abs<X COMMA X>(d1 - d2),
                                      0.0f
)

template <typename X>
class DropOut {
 public:
  no_op_exec_special_same no_op_exec_special_same_cuda
      SD_HOST_DEVICE SD_INLINE static X op(X d1, X *params) {
    X prob = params[0];
#ifdef __CUDACC__
    X length = params[1];
   X tid = blockIdx.x * blockDim.x + threadIdx.x;
   X rnd = sd::math::sd_abs<X,X>(sd::math::sd_cos<X>( static_cast<X>(tid) +
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
          SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2, Z *params) {
    Y prob = d2;
#ifdef __CUDACC__
    X length = params[1];
   X tid = blockIdx.x * blockDim.x + threadIdx.x;
   X rnd = sd::math::sd_abs<X,X>(sd::math::sd_cos<X>( static_cast<X>(tid) +
                                                      static_cast<X>(length) * static_cast<X>(tid)));
#else
    X rnd = static_cast<X>(rand() / RAND_MAX);
#endif
    return rnd >= static_cast<X>(prob) ? static_cast<Z>(0.0f) : reinterpret_cast<Z>(d1 / static_cast<X>(prob));
  }
};

DECLARE_BINARY_COPY_OP(ReplaceNans,
                       sd::math::sd_isnan(d1) ? static_cast<Z>(d2) : static_cast<Z>(d1),
                       sd::math::sd_isnan(d1) ? static_cast<Z>(d2) : static_cast<Z>(d1),
                       static_cast<Z>(d1),
                       static_cast<Z>(d1)
)


template <typename X, typename Y, typename Z>
class CompareAndReplace {
 private:
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, Y d2, Z *params) {
    auto zd1 = static_cast<Z>(d1);
    auto zd2 = static_cast<Z>(d2);
    auto compare = params[0];
    auto eps = params[2];
    int mode = (int)params[3];

    if (mode == 0)  // equals
      if (sd::math::sd_abs<Z,Z>(zd1 - compare) <= eps)
        return zd2;
      else
        return zd1;
    else if (mode == 1)  // not equals eps
      if (sd::math::sd_abs<Z,Z>(zd1 - compare) > eps)
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
      if (sd::math::sd_abs<Z,Z>(zd1) < compare)
        return zd2;
      else
        return zd1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<Z,Z>(zd1) > compare)
        return zd2;
      else
        return zd1;
    else if (mode == 8 || mode == 15)  // is inf
      if (sd::math::sd_isinf<X>(d1))  // Use original d1, not cast zd1
        return zd2;
      else
        return zd1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan<X>(d1))  // Use original d1, not cast zd1
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
      if (sd::math::sd_abs<Z,Z>(zd1) >= compare)
        return zd2;
      else
        return zd1;
    else if (mode == 13) {  // abs_less_or_equals_than
      if (sd::math::sd_abs<Z,Z>(zd1) <= compare)
        return zd2;
      else
        return zd1;
    }
    else if (mode == 14) {  // is_finite (not inf)
      if (!sd::math::sd_isinf<X>(d1))  // Use original d1, not cast zd1
        return zd2;
      else
        return zd1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return zd1;
  }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }

 public:
  static SD_HOST_DEVICE SD_INLINE Z op(X d1, Y d2, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2, params);
    else
      return op_simd(d1, d2, params);
  }
};

template <typename X, typename Y, typename Z>
class CompareAndSet {
 private:
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X dX, Y dY, Z *params) {
    auto d1 = static_cast<Z>(dX);
    auto d2 = static_cast<Z>(dY);
    auto compare = params[0];
    auto eps = params[2];
    auto mode = static_cast<int>(params[3]);

    if (mode == 0)  // equals
      if (sd::math::sd_abs<Z,Z>(d2 - compare) <= eps)
        return d2;
      else
        return d1;
    else if (mode == 1)  // not equals
      if (sd::math::sd_abs<Z,Z>(d2 - compare) > eps)
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
      if (sd::math::sd_abs<Z,Z>(d2) < compare)
        return d2;
      else
        return d1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<Z,Z>(d2) > compare)
        return d2;
      else
        return d1;
    else if (mode == 8 || mode == 15)  // is inf
      if (sd::math::sd_isinf<Y>(dY))  // Use original dY, not cast d2
        return d2;
      else
        return d1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan<Y>(dY))  // Use original dY, not cast d2
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
      if (sd::math::sd_abs<Z,Z>(d1) >= compare)
        return d2;
      else
        return d1;
    else if (mode == 13)  // abs_less_or_equals_than
      if (sd::math::sd_abs<Z,Z>(d1) <= compare)
        return d2;
      else
        return d1;
    else if (mode == 14) {  // is_finite (not inf)
      if (!sd::math::sd_isinf<X>(dX))  // Use original dX, not cast d1
        return d2;
      else
        return d1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return d1;
  }
  static SD_HOST_DEVICE SD_INLINE Z op_simd(X dX, Y dY, Z *params) { return op_logic(dX, dY, params); }

 public:
  static SD_HOST_DEVICE SD_INLINE Z op(X dX, Y dY, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(dX, dY, params);
    else
      return op_simd(dX, dY, params);
  }
};

template <typename X>
class CompareAndSetTransform {
 private:
  static SD_HOST_DEVICE SD_INLINE X op_logic(X d1, X *params) {
    auto compare = params[0];
    auto set = params[1];
    auto eps = params[2];
    int mode = (int)params[3];

    if (mode == 0)  // equals
      if (sd::math::sd_abs<X,X>(d1 - compare) <= eps)
        return set;
      else
        return d1;
    else if (mode == 1)  // not equals
      if (sd::math::sd_abs<X,X>(d1 - compare) > eps)
        return set;
      else
        return d1;
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
      if (sd::math::sd_abs<X,X>(d1) < compare)
        return set;
      else
        return d1;
    else if (mode == 7)  // abs_greater_than
      if (sd::math::sd_abs<X,X>(d1) > compare)
        return set;
      else
        return d1;
    else if (mode == 8)  // is inf
      if (sd::math::sd_isinf<X>(d1))
        return set;
      else
        return d1;
    else if (mode == 9)  // is nan
      if (sd::math::sd_isnan<X>(d1))
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
      if (sd::math::sd_abs<X,X>(d1) >= compare)
        return set;
      else
        return d1;
    else if (mode == 13)  // abs_less_or_equals_than
      if (sd::math::sd_abs<X,X>(d1) <= compare)
        return set;
      else
        return d1;
    else if (mode == 14) {  // is_finite (not inf)
      if (!sd::math::sd_isinf<X>(d1))
        return compare;  // Note: original code returns compare, not set
      else
        return d1;
    }
    else
      sd_printf("Undefined boolean operation: [%i]\n", mode);
    return d1;
  }
  static SD_HOST_DEVICE SD_INLINE X op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  no_op_exec_special_same no_op_exec_special_same_cuda;
  static SD_HOST_DEVICE SD_INLINE X op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)
      return op_logic(d1, params);
    else
      return op_simd(d1, params);
  }
};













DECLARE_UNARY_SIMD_SAFE_OP(SELUDerivative,
                           return d1 > static_cast<X>(0.f)
                                  ? static_cast<X>(SELU_LAMBDA)
                                  : static_cast<X>(SELU_ALPHA) * static_cast<X>(SELU_LAMBDA) * sd::math::sd_exp<X COMMA X>(d1);
)


DECLARE_UNARY_SIMD_SAFE_OP(HardSigmoid,
                           return sd::math::sd_min<X>(
                               static_cast<X>(1) COMMA sd::math::sd_max<X>(static_cast<X>(0) COMMA (static_cast<X>(0.2f)) * d1 + static_cast<X>(0.5f)));
)

DECLARE_UNARY_SIMD_SAFE_OP(SELU,
                           return d1 > static_cast<X>(0.0f)
                                  ? static_cast<X>(SELU_LAMBDA) * static_cast<X>(d1)
                                  : static_cast<X>(SELU_LAMBDA) * (static_cast<X>(SELU_ALPHA) * sd::math::sd_exp<X COMMA X>(d1) - static_cast<X>(SELU_ALPHA));
)





DECLARE_UNARY_SIMD_SAFE_OP(Swish,
                           return d1 * sd::math::sd_sigmoid<X COMMA X>(d1);
)


DECLARE_UNARY_SIMD_SAFE_OP(SwishDerivative,
                           X ex = sd::math::sd_pow<X COMMA X COMMA X>(static_cast<X>(M_E) COMMA d1);
                               return (ex * (d1 + ex + static_cast<X>(1.f))) / sd::math::sd_pow<X COMMA X COMMA X>((ex + static_cast<X>(1.f)) COMMA static_cast<X>(2.f));
)


DECLARE_UNARY_SIMD_SAFE_OP(Mish,
                           return d1 * sd::math::sd_tanh<X COMMA X>(sd::math::sd_softplus<X COMMA X>(d1));
)

DECLARE_UNARY_SIMD_SAFE_OP(MishDerivative,
                           auto ex = sd::math::sd_exp<X COMMA X>(d1);
                               auto e2x = ex * ex;
                               auto e3x = ex * ex * ex;
                               return (ex * (4 * (d1 + 1) + 4 * e2x + e3x + ex * (4 * d1 + 6))) / sd::math::sd_pow<X COMMA X COMMA X>((2 * ex + e2x + 2) COMMA (X)2.f);
)

DECLARE_UNARY_SIMD_SAFE_OP(GELU,
                           return d1 * sd::math::sd_sigmoid<X COMMA X>(static_cast<X>(1.702f) * d1);
)


DECLARE_UNARY_SIMD_SAFE_OP(PreciseGELU,
                           auto sp = sd::math::sd_sqrt<X COMMA X>(static_cast<X>(2) / static_cast<X>(M_PI));
                               auto xp = d1 + sd::math::sd_pow<X COMMA X COMMA X>(static_cast<X>(0.044715) * d1 COMMA static_cast<X>(3));
                               return (d1 / static_cast<X>(2)) * (static_cast<X>(1) + sd::math::sd_tanh<X COMMA X>(sp * xp));
)
DECLARE_UNARY_SIMD_SAFE_OP(GELUDerivative,
                           auto x17 = static_cast<X>(1.702f) * d1;
                               auto ep = sd::math::sd_exp<X COMMA X>(x17);
                               auto one_plus_ep = static_cast<X>(1.f) + ep;
                               return (ep * (one_plus_ep + x17)) / (one_plus_ep * one_plus_ep);

)
DECLARE_UNARY_SIMD_SAFE_OP(PreciseGELUDerivative,
                           auto x79 = static_cast<X>(0.797885) * d1;
                               auto temp1 = static_cast<X>(0.0356774) * d1;
                               auto x03 = temp1 * temp1 * temp1;  // cube without sd_pow
                               auto x39 = static_cast<X>(0.398942) * d1;
                               auto temp2 = static_cast<X>(0.0535161) * d1;
                               auto x05 = temp2 * temp2 * temp2;  // cube without sd_pow
                               auto scz = sd::math::sd_sech<X COMMA X>(x79 + x03);
                               return static_cast<X>(0.5) + (x39 + x05) * (scz * scz) + static_cast<X>(0.5) * sd::math::sd_tanh<X COMMA X>(x79 + x03);
)

DECLARE_UNARY_SIMD_SAFE_OP(LogSigmoid,
                           return sd::math::sd_log<X COMMA X>(sd::math::sd_sigmoid<X COMMA X>(d1));
)

DECLARE_UNARY_SIMD_SAFE_OP(LogSigmoidDerivative,
                           X ex = sd::math::sd_exp<X COMMA X>(d1);
                               return static_cast<X>(1.f) / (ex + static_cast<X>(1.f));
)
DECLARE_UNARY_MATH_OP(SigmoidDerivative, sd_sigmoidderivative)
DECLARE_UNARY_MATH_OP(TanhDerivative, sd_tanhderivative)
DECLARE_UNARY_MATH_OP(SinhDerivative, sd_cosh)  // sinh derivative is cosh
DECLARE_UNARY_MATH_OP(SoftSignDerivative, sd_softsignderivative)

DECLARE_MULTI_OP_SIMD_SAFE(And,
                           return d2 + d1;,
                           if (params != nullptr) {
                             auto comp = params[0];
                             return d1 != comp && d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
                           } else {
                             auto b1 = static_cast<bool>(d1);
                             auto b2 = static_cast<bool>(d2);
                             return (b1 && b2) ? static_cast<Z>(1) : static_cast<Z>(0);
                           },
                           return d1;,
                           return static_cast<Z>(119);
)

DECLARE_UNARY_SIMD_SAFE_OP(RationalTanh,
                           auto dis = (static_cast<X>(2) / static_cast<X>(3)) * d1;
                               auto tanh = sd::math::sd_sgn<X COMMA X>(dis) *
                                           (static_cast<X>(1) -
                                            (static_cast<X>(1) / (static_cast<X>(1) + static_cast<X>(sd::math::sd_abs<X COMMA X>(dis)) +
                               sd::math::sd_pow<X COMMA X COMMA X>(dis COMMA static_cast<X>(2)) +
                               static_cast<X>(1.41645f) * sd::math::sd_pow<X COMMA X COMMA X>(dis COMMA static_cast<X>(4)))));
                               return static_cast<X>(1.7159f) * tanh;
)


DECLARE_UNARY_SIMD_SAFE_OP(RationalTanhDerivative,
                           auto dis = (static_cast<X>(2.f) / static_cast<X>(3.f)) * d1;
                               auto a = static_cast<X>(1.f) + sd::math::sd_abs<X COMMA X>(dis) + sd::math::sd_pow<X COMMA X COMMA X>(dis COMMA static_cast<X>(2.f)) +
                               static_cast<X>(1.41645f) * sd::math::sd_pow<X COMMA X COMMA X>(dis COMMA static_cast<X>(4));
                               auto tDeriv =
                           (static_cast<X>(1.f) + sd::math::sd_sign<X COMMA X>(dis) * (static_cast<X>(2.f) * dis +
                               static_cast<X>(4.f) * static_cast<X>(1.41645f) *
                               sd::math::sd_pow<X COMMA X COMMA X>(dis COMMA static_cast<X>(3)))) /
                           (a * a);
                               return static_cast<X>(1.7159f) * (static_cast<X>(2.f) / static_cast<X>(3.f)) * tDeriv;
)

DECLARE_UNARY_SIMD_SAFE_OP(ScaledTanh,
                           return params[0] * sd::math::sd_tanh<X COMMA X>(params[1] * d1);
)

// RectifiedTanh operation
DECLARE_UNARY_SIMD_SAFE_OP(RectifiedTanh,
                           return sd::math::sd_max<X>(static_cast<X>(0) COMMA sd::math::sd_tanh<X COMMA X>(d1));
)

// ELU operation
DECLARE_BINARY_SIMD_SAFE_OP(ELU,
                            return sd::math::sd_elu<X COMMA Z>(d1 COMMA static_cast<X>(d2));
)

// ELUDerivative operation
DECLARE_BINARY_SIMD_SAFE_OP(ELUDerivative,
                            return sd::math::sd_eluderivative<X COMMA Z>(d1 COMMA static_cast<X>(d2));
)

// RELU operation
DECLARE_BINARY_SIMD_SAFE_OP(RELU,
                            auto xt = static_cast<Z>(d1);
                                auto xf = static_cast<Z>(d2);
                                return xt < xf ? xf : xt;
)

// RELUDerivative operation
DECLARE_BINARY_SIMD_SAFE_OP(RELUDerivative,
                            auto xt = static_cast<Z>(d1);
                                auto xf = static_cast<Z>(d2);
                                return xt > xf ? static_cast<Z>(1.f) : static_cast<Z>(0.f);
)


DECLARE_BINARY_SIMD_SAFE_OP(RELU6,
                            auto relu = RELU<X COMMA Y COMMA Z>::op(d1 COMMA d2 COMMA params);
                                return relu < static_cast<Z>(6) ? relu : static_cast<Z>(6);
)

DECLARE_BINARY_SIMD_SAFE_OP(LeakyRELU,
                            auto val = static_cast<Z>(d1);
                                auto alpha = static_cast<Z>(d2);
                                return val < 0.0f ? alpha * val : val;
)

DECLARE_BINARY_SIMD_SAFE_OP(LeakyRELUDerivative,
                            if (d1 >= static_cast<X>(0))
                              return static_cast<Z>(1);
                            else
                              return static_cast<Z>(d2);
)

DECLARE_REDUCE_SIMD_SAFE_OP(IsNan,
                            return sd::math::sd_isnan<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
)
DECLARE_REDUCE_SIMD_SAFE_OP(IsPositive,
                            return d1 > (X)0.f;
)

DECLARE_REDUCE_SIMD_SAFE_OP(IsNegative,
                            return d1 < (X)0.f;
)

DECLARE_REDUCE_SIMD_SAFE_OP(IsInf,
                            return sd::math::sd_isinf<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
)

// IsInfOrNan operation
DECLARE_REDUCE_SIMD_SAFE_OP(IsFinite,
                            return sd::math::sd_isfin<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
)

DECLARE_REDUCE_SIMD_SAFE_OP(IsInfOrNan,
                            return sd::math::sd_isfin<X>(d1) ? static_cast<Z>(0) : static_cast<Z>(1);
)


DECLARE_UNARY_SIMPLE_OP(TimesOneMinus, d1 * (static_cast<X>(1) - d1))
DECLARE_UNARY_SIMPLE_OP(CubeDerivative, static_cast<X>(3) * d1 * d1)
DECLARE_UNARY_SIMPLE_OP(SpecialDerivative, d1 * (static_cast<X>(1.f) - d1))


DECLARE_COMPLEX_ACCUMULATION_SIMD_SAFE_OP(ASum,
                                          return sd::math::sd_abs<X COMMA X>(d1);,
                                          return sd::math::sd_abs<X COMMA X>(d1) + sd::math::sd_abs<X COMMA X>(d2);,
                                          return sd::math::sd_abs<X COMMA X>(d1) + sd::math::sd_abs<X COMMA X>(d2);,
                                          ASUM, static_cast<X>(0),
                                          sd::math::sd_abs<X COMMA X>(opOutput) + sd::math::sd_abs<X COMMA X>(old),
                                          sd::math::sd_abs<X COMMA X>(opOutput) + sd::math::sd_abs<X COMMA X>(old),
                                          sd::math::sd_abs<X COMMA X>(reduction)
)

DECLARE_SIMPLE_REDUCTION_OP(
    CountNonZero,
    ASUM,
    static_cast<Z>(0),
    (d1 == static_cast<X>(0.0f) ? static_cast<InterType>(0.0f) : static_cast<InterType>(1.0f)),
    (opOutput + old),
    (opOutput + old),
    static_cast<Z>(reduction)
)

DECLARE_SIMPLE_REDUCTION_OP(
    CountZero,
    SUM,
    static_cast<Z>(0.0f),
    (d1 == static_cast<X>(0) ? static_cast<InterType>(1) : static_cast<InterType>(0)),
    (opOutput + old),
    (opOutput + old),
    static_cast<Z>(reduction)
)

DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP(Any,
                                        return d1;,
                                        SUM, static_cast<X>(0.0f),
                                        opOutput + old,
                                        opOutput + old,
                                        reduction > static_cast<Z>(0) ? static_cast<Z>(1) : static_cast<Z>(0)
)

DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP(All,
                                        return d1;,
                                        SUM, static_cast<X>(1),
                                        static_cast<Z>(static_cast<bool>(opOutput) && static_cast<bool>(old) ? 1 : 0),
                                        static_cast<Z>(static_cast<bool>(opOutput) && static_cast<bool>(old) ? 1 : 0),
                                        reduction > static_cast<Z>(0) ? static_cast<Z>(1) : static_cast<Z>(0)
)

// AMean operation
DECLARE_ACCUMULATION_SIMD_SAFE_OP(AMean,
                                  return static_cast<InterType>(sd::math::sd_abs<X COMMA X>(d1));,
                                  ASUM, static_cast<X>(0),
                                  sd::math::sd_abs<InterType COMMA InterType>(opOutput) + sd::math::sd_abs<InterType COMMA InterType>(old),
                                  opOutput + old,
                                  static_cast<InterType>(reduction / static_cast<InterType>(n))
)

DECLARE_COMPLEX_ACCUMULATION_SIMD_SAFE_OP(AMax,
                                          return sd::math::sd_abs<X COMMA X>(d1);,
                                          return sd::math::sd_max<X>(sd::math::sd_abs<X COMMA X>(d1) COMMA sd::math::sd_abs<X COMMA X>(d2));,
                                          return sd::math::sd_abs<X COMMA X>(d1) > sd::math::sd_abs<X COMMA X>(d2) ? d1 : d2;,
                                          AMAX, input[0],
                                          sd::math::sd_max<X>(sd::math::sd_abs<X COMMA X>(old) COMMA sd::math::sd_abs<X COMMA X>(opOutput)),
                                          sd::math::sd_max<X>(sd::math::sd_abs<X COMMA X>(opOutput) COMMA sd::math::sd_abs<X COMMA X>(old)),
                                          sd::math::sd_abs<X COMMA X>(reduction)
)

DECLARE_COMPLEX_ACCUMULATION_SIMD_SAFE_OP(AMin,
                                          return sd::math::sd_abs<X COMMA X>(d1);,
                                          return sd::math::sd_min<X>(sd::math::sd_abs<X COMMA X>(d1) COMMA sd::math::sd_abs<X COMMA X>(d2));,
                                          return sd::math::sd_min<X>(sd::math::sd_abs<X COMMA X>(d1) COMMA sd::math::sd_abs<X COMMA X>(d2));,
                                          AMIN, input[0],
                                          sd::math::sd_min<X>(sd::math::sd_abs<X COMMA X>(old) COMMA sd::math::sd_abs<X COMMA X>(opOutput)),
                                          sd::math::sd_min<X>(sd::math::sd_abs<X COMMA X>(opOutput) COMMA sd::math::sd_abs<X COMMA X>(old)),
                                          sd::math::sd_abs<X COMMA X>(reduction)
)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(Norm1,
                                  return static_cast<InterType>(sd::math::sd_abs<X COMMA X>(d1));,
                                  SUM, static_cast<X>(0),
                                  opOutput + old,
                                  opOutput + old,
                                  reduction
)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(Norm2,
                                  auto v = static_cast<InterType>(d1);
                                  return v * v;,
                                               SUM, static_cast<X>(0),
                                               opOutput + old,
                                               opOutput + old,
                                               sd::math::sd_sqrt<InterType COMMA Z>(reduction)
)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(SquaredNorm,
                                  auto v = static_cast<InterType>(d1);
                                  return v * v;,
                                               SUM, static_cast<X>(0),
                                               opOutput + old,
                                               opOutput + old,
                                               reduction
)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(NormFrobenius,
                                  auto v = static_cast<InterType>(sd::math::sd_abs<X COMMA X>(d1));
                                  return v * v;,
                                               SUM, static_cast<X>(0),
                                               opOutput + old,
                                               opOutput + old,
                                               sd::math::sd_sqrt<InterType COMMA Z>(reduction)
)

DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP(NormP,
                                  return sd::math::sd_pow<InterType COMMA Z COMMA InterType>(static_cast<InterType>(sd::math::sd_abs<X COMMA X>(d1)) COMMA static_cast<InterType>(params[0]));,
                                  SUM, static_cast<X>(0),
                                  opOutput + old,
                                  opOutput + old,
                                  sd::math::sd_pow<InterType COMMA Z COMMA Z>(reduction COMMA static_cast<Z>(1.0f) / extraParams[0])
)




DECLARE_MIXED_ACCUMULATION_SIMD_SAFE_OP(NormMax,
                                        return static_cast<Z>(d1);,
                                        SUM, static_cast<X>(0),
                                        opOutput + old,
                                        sd::math::sd_max<Z>(sd::math::sd_abs<Z COMMA Z>(old) COMMA sd::math::sd_abs<Z COMMA Z>(opOutput)),
                                        sd::math::sd_max<Z>(sd::math::sd_abs<Z COMMA Z>(reduction) COMMA sd::math::sd_abs<Z COMMA Z>(reduction))
)

// --- Generic Assign Template ---
template <typename X, typename Z>
class Assign {
 private:
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, X *params) {
    if constexpr (std::is_same_v<X, Z>) {
      return d1; // No conversion needed
    } else if constexpr (std::is_convertible_v<X, Z>) {
      return static_cast<Z>(d1); // Use static_cast for direct convertibility
    } else {
      // This will trigger a compile error for unsupported types,
      // requiring a specialization like the ones below.
      return static_cast<Z>(d1);
    }
  }

  static SD_HOST_DEVICE SD_INLINE Z op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  static SD_HOST_DEVICE SD_INLINE Z op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value)
      return op_logic(d1, params);
    else
      return op_simd(d1, params);
  }
};

// --- Specialization: std::basic_string<char16_t> (UTF-16) -> std::basic_string<char> (UTF-8) ---
template <>
class Assign<std::basic_string<char16_t>, std::basic_string<char>> {
 public:
  static const bool requiresSpecial = false;
  static void execSpecial(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char16_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char16_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE static std::basic_string<char>
  op(const std::basic_string<char16_t>& d1, std::basic_string<char16_t> * /*params*/) {
    char temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES];
    const char16_t* input_data = d1.data();
    const uint32_t input_length_char16_units = static_cast<uint32_t>(d1.length());
    sd::LongType required_bytes = sd::unicode::offsetUtf16StringInUtf8(input_data, input_length_char16_units);

    if (required_bytes > 0 && static_cast<size_t>(required_bytes) <= SD_STRING_ASSIGN_TEMP_BUFFER_BYTES) {
      void* end_ptr = sd::unicode::utf16to8Ptr(input_data, input_data + input_length_char16_units, temp_output_buffer);
      size_t bytes_written = static_cast<char*>(end_ptr) - temp_output_buffer;
      if (bytes_written == static_cast<size_t>(required_bytes)) {
        return std::basic_string<char>(temp_output_buffer, bytes_written);
      }
    }
    return std::basic_string<char>();
  }
};

// --- Specialization: std::basic_string<char> (UTF-8) -> std::basic_string<char16_t> (UTF-16) ---
template <>
class Assign<std::basic_string<char>, std::basic_string<char16_t>> {
 public:
  static const bool requiresSpecial = false;
  static void execSpecial(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char16_t> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE static std::basic_string<char16_t>
  op(const std::basic_string<char>& d1, std::basic_string<char> * /*params*/) {
    char16_t temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES / sizeof(char16_t) + 1];
    const char* input_data = d1.data();
    const uint32_t input_length_bytes = static_cast<uint32_t>(d1.length());
    sd::LongType required_bytes_for_utf16 = sd::unicode::offsetUtf8StringInUtf16(input_data, input_length_bytes);

    if (required_bytes_for_utf16 > 0 && static_cast<size_t>(required_bytes_for_utf16) < sizeof(temp_output_buffer) ) {
      void* end_ptr = sd::unicode::utf8to16Ptr(input_data, input_data + input_length_bytes, temp_output_buffer);
      size_t char16_units_written = static_cast<char16_t*>(end_ptr) - temp_output_buffer;
      if (char16_units_written * sizeof(char16_t) == static_cast<size_t>(required_bytes_for_utf16)) {
        return std::basic_string<char16_t>(temp_output_buffer, char16_units_written);
      }
    }
    return std::basic_string<char16_t>();
  }
};

// --- Specialization: std::basic_string<char32_t> (UTF-32) -> std::basic_string<char> (UTF-8) ---
template <>
class Assign<std::basic_string<char32_t>, std::basic_string<char>> {
 public:
  static const bool requiresSpecial = false;
  static void execSpecial(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char32_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char32_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char> *reductionPointer, // Z is std::basic_string<char>
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE static std::basic_string<char>
  op(const std::basic_string<char32_t>& d1, std::basic_string<char32_t> * /*params*/) {
    char temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES];
    const char32_t* input_data = d1.data();
    const uint32_t input_length_char32_units = static_cast<uint32_t>(d1.length());
    sd::LongType required_bytes = sd::unicode::offsetUtf32StringInUtf8(input_data, input_length_char32_units);

    if (required_bytes > 0 && static_cast<size_t>(required_bytes) <= SD_STRING_ASSIGN_TEMP_BUFFER_BYTES) {
      void* end_ptr = sd::unicode::utf32to8Ptr(input_data, input_data + input_length_char32_units, temp_output_buffer);
      size_t bytes_written = static_cast<char*>(end_ptr) - temp_output_buffer;
      if (bytes_written == static_cast<size_t>(required_bytes)) {
        return std::basic_string<char>(temp_output_buffer, bytes_written);
      }
    }
    return std::basic_string<char>();
  }
};

// --- Specialization: std::basic_string<char> (UTF-8) -> std::basic_string<char32_t> (UTF-32) ---
template <>
class Assign<std::basic_string<char>, std::basic_string<char32_t>> {
 public:
  static const bool requiresSpecial = false;
  static void execSpecial(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char32_t> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE static std::basic_string<char32_t>
  op(const std::basic_string<char>& d1, std::basic_string<char> * /*params*/) {
    char32_t temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES / sizeof(char32_t) + 1];
    const char* input_data = d1.data();
    const uint32_t input_length_bytes = static_cast<uint32_t>(d1.length());
    sd::LongType required_bytes_for_utf32_data = sd::unicode::offsetUtf8StringInUtf32(input_data, input_length_bytes);

    if (required_bytes_for_utf32_data > 0 && static_cast<size_t>(required_bytes_for_utf32_data) < sizeof(temp_output_buffer) ) {
      void* end_ptr = sd::unicode::utf8to32Ptr(input_data, input_data + input_length_bytes, temp_output_buffer);
      size_t char32_units_written = static_cast<char32_t*>(end_ptr) - temp_output_buffer;
      if (char32_units_written * sizeof(char32_t) == static_cast<size_t>(required_bytes_for_utf32_data)) {
        return std::basic_string<char32_t>(temp_output_buffer, char32_units_written);
      }
    }
    return std::basic_string<char32_t>();
  }
};

// --- Identity Specializations ---
template <>
class Assign<std::basic_string<char>, std::basic_string<char>> {
 public:
  static const bool requiresSpecial = false;
  static void execSpecial(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif
  SD_HOST_DEVICE static std::basic_string<char>
  op(const std::basic_string<char>& d1, std::basic_string<char> * /*params*/) {
    return d1;
  }
};

template <typename X, typename Y, typename Z>
class LogPoissonLossFull {
 private:
  static SD_INLINE Z op_logic(X z, Y c) {
    auto zz = static_cast<Z>(z);
    auto zc = static_cast<Z>(c);
    return (sd::math::sd_exp<Y, Z>(c) - zz * zc +
            (zz * sd::math::sd_log<X, Z>(z) - zz +
             static_cast<Z>(0.5f) * sd::math::sd_log<Z, Z>(static_cast<Z>(SD_DOUBLE_PI_X) * zz)));
  }
  static SD_INLINE Z op_logic(X z, Y c, Z *params) { return op_logic(z, c); }
  static SD_INLINE Z op_logic(X z) {
    auto zz = static_cast<Z>(z);
    return (zz * sd::math::sd_log<Y, Z>(z) - zz +
            static_cast<Z>(0.5f) * sd::math::sd_log<Z, Z>(static_cast<Z>(SD_DOUBLE_PI_X) * zz));
  }
  static SD_INLINE X op_logic(X z, Y *params) {
    return (sd::math::sd_exp<X, X>(params[0]) - z * params[0] +
            (z * sd::math::sd_log<X, Z>(z) - z + static_cast<X>(0.5f) * sd::math::sd_log<X, Z>(SD_DOUBLE_PI_X * z)));
  }
  static Z op_simd(X z, Y c) { return op_logic(z, c); }
  static Z op_simd(X z, Y c, Z *params) { return op_logic(z, c, params); }
  static Z op_simd(X z) { return op_logic(z); }
  static X op_simd(X z, Y *params) { return op_logic(z, params); }

 public:
  static Z op(X z, Y c) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(z, c);
    else return op_simd(z, c);
  }
  static Z op(X z, Y c, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(z, c, params);
    else return op_simd(z, c, params);
  }
  static Z op(X z) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(z);
    else return op_simd(z);
  }
  static X op(X z, Y *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(z, params);
    else return op_simd(z, params);
  }
};



template <typename X>
class Celu {
 private:
  static SD_INLINE X op_logic(X d1, X *params) {
    X alpha = params[0];
    return sd::math::sd_max<X>(static_cast<X>(0), d1) +
           sd::math::sd_min<X>(static_cast<X>(0), alpha * (sd::math::sd_exp<X, X>(d1/alpha) - static_cast<X>(1)));
  }
  static X op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  no_op_exec_special_same no_op_exec_special_same_cuda;
  static X op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<X>::value)
      return op_logic(d1, params);
    else
      return op_simd(d1, params);
  }
};




// ELUAlpha - simple conditional with parameter
DECLARE_UNARY_CLIPPING_OP(ELUAlpha,
                          X alpha = params[0];
                              return d1 > static_cast<X>(0) ? d1 : alpha * (sd::math::sd_exp<X COMMA X>(d1) - static_cast<X>(1));
)

// PReLU - simple conditional with parameter
DECLARE_UNARY_CLIPPING_OP(PReLU,
                          X alpha = params[0];
                              return d1 > static_cast<X>(0) ? d1 : alpha * d1;
)

// ThresholdedReLU - simple threshold operation
DECLARE_UNARY_CLIPPING_OP(ThresholdedReLU,
                          X theta = params[0];
                              return d1 > theta ? d1 : static_cast<X>(0);
)




// Use the new macro for bitwise operations
DECLARE_SIMPLE_BINARY_OP(IntOr, d2 | d1)
DECLARE_SIMPLE_BINARY_OP(IntAnd, d2 & d1)
DECLARE_SIMPLE_BINARY_OP(IntXor, d2 ^ d1)
DECLARE_SIMPLE_BINARY_OP(ShiftLeft, d1 << d2)
DECLARE_SIMPLE_BINARY_OP(ShiftRight, d1 >> d2)


DECLARE_SIMPLE_BINARY_TEMPLATE_OP(CyclicShiftLeft, sd::math::sd_rotl<X>(d1 COMMA d2))
DECLARE_SIMPLE_BINARY_TEMPLATE_OP(CyclicShiftRight, sd::math::sd_rotr<X>(d1 COMMA d2))



template <typename X, typename Y, typename Z>
class Mod {
 private:
  static SD_INLINE Z op_logic(X d1, Y d2) {
    auto dx = static_cast<X>(d2);
    auto f = sd::math::sd_floor<X, X>(d1 / dx);
    auto r = f * dx;
    return static_cast<Z>(d1 - r);
  }
  static SD_INLINE Z op_logic(X d1, Y d2, Z *params) { return op_logic(d1, d2); }
  static SD_INLINE Z op_logic(X d1, Y *params) { return op_logic(d1, params[0]); }
  static Z op_simd(X d1, Y d2) { return op_logic(d1, d2); }
  static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }
  static Z op_simd(X d1, Y *params) { return op_logic(d1, params); }

 public:
  static Z op(X d1, Y d2) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2);
    else return op_simd(d1, d2);
  }
  static Z op(X d1, Y d2, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2, params);
    else return op_simd(d1, d2, params);
  }
  static Z op(X d1, Y *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, params);
    else return op_simd(d1, params);
  }
};


DECLARE_BINARY_COPY_OP(ReverseMod,
                       static_cast<Z>(static_cast<int>(d2) % static_cast<int>(d1)),
                       static_cast<Z>(static_cast<int>(d2) % static_cast<int>(d1)),
                       static_cast<Z>(d1),
                       static_cast<Z>(static_cast<int>(params[0]) % static_cast<int>(d1))
)

template <typename X, typename Z>
class Epsilon {
 private:
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, X d2) {
    X diff = d1 - d2;
    X absDiff = sd::math::sd_abs<X,X>(diff);
    if (absDiff <= static_cast<X>(SD_MIN_V)) return static_cast<Z>(1);
    return static_cast<Z>(0);
  }
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, X d2, X *params) {
    X diff = d1 - d2;
    X absDiff = sd::math::sd_abs<X,X>(diff);
    if(params != nullptr && absDiff <= static_cast<X>(params[0])) {
      return static_cast<Z>(1);
    } else if(absDiff <= static_cast<X>(1e-5)) {
      return static_cast<Z>(1);
    }
    return static_cast<Z>(0);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, X *params) { return static_cast<Z>(d1); }
  static SD_HOST_DEVICE SD_INLINE  Z op_simd(X d1, X d2) { return op_logic(d1, d2); }
  static SD_HOST_DEVICE SD_INLINE  Z op_simd(X d1, X d2, X *params) { return op_logic(d1, d2, params); }
  static SD_HOST_DEVICE SD_INLINE  Z op_simd(X d1, X *params) { return op_logic(d1, params); }

 public:
  static SD_HOST_DEVICE SD_INLINE  Z op(X d1, X d2) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1, d2);
    else return op_simd(d1, d2);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op(X d1, X d2, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1, d2, params);
    else return op_simd(d1, d2, params);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1, params);
    else return op_simd(d1, params);
  }
};

template <typename X, typename Z>
class MatchConditionBool {
 private:
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, X *extraParams) {
    X compare = extraParams[0];
    X eps = extraParams[1];
    auto mode = static_cast<int>(extraParams[2]);
    sd_debug("value: %f; comp: %f; eps: %f; mode: %i;\n", d1, compare, eps, mode);

    switch (mode) {
      case 0:  // equals
        return sd::math::sd_abs<X,X>(d1 - compare) <= eps ? static_cast<Z>(1) : static_cast<Z>(0);
      case 1:  // not equals
        return sd::math::sd_abs<X,X>(d1 - compare) > eps ? static_cast<Z>(1) : static_cast<Z>(0);
      case 2:  // less_than
        return d1 < compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 3:  // greater_than
        return d1 > compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 4:  // less_or_equals_than
        return d1 <= compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 5:  // greater_or_equals_than
        return d1 >= compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 6:  // abs_less_than
        return sd::math::sd_abs<X,X>(d1) < compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 7:  // abs_greater_than
        return sd::math::sd_abs<X,X>(d1) > compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 8:  // is inf
        return sd::math::sd_isinf<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
      case 9:  // is nan
        return sd::math::sd_isnan<X>(d1) ? static_cast<Z>(1) : static_cast<Z>(0);
      case 10:
        return (d1 == compare) ? static_cast<Z>(1) : static_cast<Z>(0);
      case 11:
        return (d1 != compare) ? static_cast<Z>(1) : static_cast<Z>(0);
      case 12:  // abs_greater_or_equals_than
        return sd::math::sd_abs<X,X>(d1) >= compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 13:  // abs_less_or_equals_than
        return sd::math::sd_abs<X,X>(d1) <= compare ? static_cast<Z>(1) : static_cast<Z>(0);
      case 14:
        // isFinite
        return !(sd::math::sd_isinf<X>(d1) || sd::math::sd_isnan<X>(d1)) ? static_cast<Z>(1) : static_cast<Z>(0);
      case 15:
        // isInfinite
        return (sd::math::sd_isinf<X>(d1) || sd::math::sd_isnan<X>(d1)) ? static_cast<Z>(1) : static_cast<Z>(0);
      default:
        sd_debug("Undefined match condition: [%i]\n", mode);
    }
    return static_cast<Z>(d1);
  }

  // Remove SD_OP_DEF to avoid SIMD issues with float16/bfloat16
  static Z op_simd(X d1, X *extraParams) { return op_logic(d1, extraParams); }

 public:
  // Fix: Use explicit declarations instead of problematic macros
  no_op_exec_special no_op_exec_special_cuda;

  // Special handling for bool operations with type compatibility
  static const bool requiresSpecialAccumulation = false;

  // Primary execSpecial function with Z_TYPE* extraParams (for boolean case)
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, Z *extraParams, Z *result,
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength,
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}

  // Template overload to handle type conversions (for cases where extraParams is sd::LongType*)
  template<typename ExtraParamsType>
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, ExtraParamsType *extraParams, Z *result,
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength,
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {
    // Handle type conversion if needed - this handles the sd::LongType* to Z* conversion
    // For most cases, this will be empty since we don't actually implement special accumulation
  }

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void execSpecialCuda(
      const X *dx, const sd::LongType *xShapeInfo, Z *extraParams, Z *result,
      const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
      Z *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {}

  template<typename ExtraParamsType>
  static SD_INLINE SD_DEVICE void execSpecialCuda(
      const X *dx, const sd::LongType *xShapeInfo, ExtraParamsType *extraParams, Z *result,
      const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
      Z *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  static SD_HOST_DEVICE SD_INLINE Z op(X d1, X *extraParams) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1, extraParams);
    else
      return op_simd(d1, extraParams);
  }
};


DECLARE_MULTI_OP_SIMD_SAFE(Or,
                           return d2 + d1;,
                           if (params != nullptr) {
                             auto comp = params[0];
                             return d1 != comp || d2 != comp ? static_cast<Z>(1) : static_cast<Z>(0);
                           } else {
                             auto b1 = static_cast<bool>(d1);
                             auto b2 = static_cast<bool>(d2);
                             return b1 || b2 ? static_cast<Z>(1) : static_cast<Z>(0);
                           },
                           return d1;,
                           return static_cast<Z>(119);
)


DECLARE_XOR_SIMD_SAFE(Xor,
                      return d2 + d1;,
                      if (params != nullptr) {
                        auto comp = params[0];
                        return ((d1 == comp && d2 != comp) || (d1 != comp && d2 == comp)) ? static_cast<Z>(1) : static_cast<Z>(0);
                      } else {
                        auto b1 = static_cast<bool>(d1);
                        auto b2 = static_cast<bool>(d2);
                        return (!b1 && b2) || (b1 && !b2) ? static_cast<Z>(1) : static_cast<Z>(0);
                      },
                      return d1;
)




DECLARE_NOT_SIMD_SAFE(Not,
                      return static_cast<Z>(0);,
                      return d1 != d2 ? static_cast<Z>(1) : static_cast<Z>(0);,
                      auto b1 = static_cast<bool>(d1);
                          return !b1;
)


DECLARE_ACCUMULATION_SIMD_SAFE_OP(Variance,
                                  X mean = static_cast<InterType>(params[0]);
                                      X ret = d1 - mean;
                                      return ret * ret;,
                                  SUM, static_cast<X>(0.0f),
                                  old + opOutput,
                                  old + opOutput,
                                  static_cast<Z>(reduction / static_cast<InterType>(n - 1))
)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(StandardDeviation,
                                  InterType mean = static_cast<InterType>(params[0]);
                                      InterType ret = d1 - mean;
                                      return ret * ret;,
                                  SUM, static_cast<X>(0.0f),
                                  old + opOutput,
                                  old + opOutput,
                                  sd::math::sd_sqrt<InterType COMMA Z>(static_cast<InterType>(reduction / static_cast<InterType>(n - 1)))
)


DECLARE_ACCUMULATION_SIMD_SAFE_OP(ShannonEntropy,
                                  auto p = d1;
                                      return static_cast<Z>(p) * sd::math::sd_log2<X COMMA Z>(p);,
                                  SUM, static_cast<X>(0),
                                  opOutput + old,
                                  opOutput + old,
                                  -reduction
)


DECLARE_ACCUMULATION_SIMD_SAFE_OP(LogEntropy,
                                  return static_cast<InterType>(d1) * sd::math::sd_log<X COMMA InterType>(d1);,
                                  SUM, static_cast<X>(0),
                                  opOutput + old,
                                  opOutput + old,
                                  sd::math::sd_log<InterType COMMA Z>(-reduction)
)



template <typename X, typename Z>
class IndexAbsoluteMax {
 public:
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return sd::math::sd_abs<X,X>(val);
  }
  static SD_HOST_DEVICE SD_INLINE   functions::indexreduce::IndexValue<X> update(
      functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
    opOutput.value = sd::math::sd_abs<X,X>(opOutput.value);
    old.value = sd::math::sd_abs<X,X>(old.value);
    if (opOutput.value > old.value) return opOutput;
#ifdef __CUDACC__
    else if (opOutput.value == old.value && opOutput.index < old.index)
     return opOutput;
#endif
    return old;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (sd::math::sd_abs<X,X>(f1.value) > sd::math::sd_abs<X,X>(f2.value)) return f2;
    return f1;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
  static SD_HOST_DEVICE SD_INLINE  X startingValue(const X *input) { return static_cast<X>(0); }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class IndexAbsoluteMin {
 public:
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }
  static SD_HOST_DEVICE SD_INLINE  X startingValue(const X *input) { return sd::DataTypeUtils::infOrMax<X>(); }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = 0;
    return local;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> update(
      functions::indexreduce::IndexValue<X> &old, functions::indexreduce::IndexValue<X> &opOutput, X *extraParams) {
    opOutput.value = sd::math::sd_abs<X,X>(opOutput.value);
    old.value = sd::math::sd_abs<X,X>(old.value);
    if (opOutput.value < old.value) return opOutput;
#ifdef __CUDACC__
    else if (opOutput.value == old.value && opOutput.index < old.index)
     return opOutput;
#endif
    return old;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (sd::math::sd_abs<X,X>(f1.value) < sd::math::sd_abs<X,X>(f2.value)) return f2;
    return f1;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
  static  SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class FirstIndex {
 public:
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old,
                                                                     functions::indexreduce::IndexValue<X> &opOutput,
                                                                     X *extraParams) {
#ifdef __CUDACC__
    if (opOutput.index < 0) return old;
#endif
    auto res = MatchConditionBool<X, X>::op(opOutput.value, extraParams);
    if (res == static_cast<X>(0)) return old;
    if (old.index < 0) return opOutput;
    if (old.index > opOutput.index) return opOutput;
    return old;
  }
  static SD_HOST_DEVICE SD_INLINE  X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = -1;
    return local;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.index > f2.index) return f2;
    return f1;
  }
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
};


template <typename X, typename Z>
class LastIndex {
 public:
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val,
                                                                        X *extraParams) {
    return val;
  }
  static SD_HOST_DEVICE SD_INLINE  functions::indexreduce::IndexValue<X> update(functions::indexreduce::IndexValue<X> &old,
                                                                     functions::indexreduce::IndexValue<X> &opOutput,
                                                                     X *extraParams) {
#ifdef __CUDACC__
    if (opOutput.index < 0) return old;
#endif
    auto res = MatchConditionBool<X, X>::op(opOutput.value, extraParams);
    if (res == static_cast<X>(0)) return old;
    if (old.index < 0) return opOutput;
    if (old.index < opOutput.index) return opOutput;
    return old;
  }
  static SD_HOST_DEVICE SD_INLINE X startingValue(const X *input) { return -sd::DataTypeUtils::infOrMax<X>(); }
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> startingIndexValue(const X *input) {
    functions::indexreduce::IndexValue<X> local;
    local.value = startingValue(input);
    local.index = -1;
    return local;
  }
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> d1,
                                                                        functions::indexreduce::IndexValue<X> d2,
                                                                        X *extraParams) {
    return d1;
  }
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> merge(functions::indexreduce::IndexValue<X> f1,
                                                                           functions::indexreduce::IndexValue<X> f2,
                                                                           X *extraParams) {
    if (f1.index < f2.index) return f2;
    return f1;
  }
  static SD_HOST_DEVICE SD_INLINE functions::indexreduce::IndexValue<X> postProcess(
      functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X *dx, int incx, X *extraParams, X *result) {
    return reduction;
  }
};

DECLARE_ACCUMULATION_SIMD_SAFE_OP(Entropy,
                                  return static_cast<InterType>(d1) * sd::math::sd_log<X COMMA InterType>(d1);,
                                  SUM, static_cast<X>(0),
                                  opOutput + old,
                                  opOutput + old,
                                  static_cast<Z>(-reduction)
)


template <typename X, typename Z>
class SummaryStatsVariance {
 public:
  static SD_HOST_DEVICE SD_INLINE Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
    if (biasCorrected) {
      Z ret = static_cast<Z>(val.varianceBiasCorrected());
      if (ret < static_cast<Z>(0.0f)) return static_cast<Z>(val.variance());
      return ret;
    }
    return static_cast<Z>(val.variance());
  }
  static SD_HOST_DEVICE SD_INLINE functions::summarystats::SummaryStatsData<X> op(
      functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
    return d1;
  }
};

template <typename X, typename Z>
class SummaryStatsStandardDeviation {
 public:
  static SD_HOST_DEVICE SD_INLINE Z getValue(const bool biasCorrected, functions::summarystats::SummaryStatsData<X> val) {
    if (biasCorrected) {
      auto ret = static_cast<Z>(val.varianceBiasCorrected());
      if (ret < static_cast<Z>(0.0f))
        return sd::math::sd_sqrt<double, Z>(val.variance());
      else
        return sd::math::sd_sqrt<double, Z>(ret);
    }
    return sd::math::sd_sqrt<double, Z>(val.variance());
  }
  static SD_HOST_DEVICE SD_INLINE functions::summarystats::SummaryStatsData<X> op(
      functions::summarystats::SummaryStatsData<X> d1, Z *extraParams) {
    return d1;
  }
};

// =============================================================================
// MISSING ADDITIONAL UNARY OPERATIONS
// =============================================================================

DECLARE_UNARY_SIMD_SAFE_OP(Sqr,
                           return sd::math::sd_pow<X COMMA X COMMA X>(d1 COMMA static_cast<X>(2));
)

DECLARE_UNARY_MATH_OP_XZ(Sqrt, sd_sqrt)

// For RSqrt, use the complex math version:
DECLARE_UNARY_COMPLEX_MATH_OP_XZ(RSqrt,
                                 static_cast<Z>(1.0) / static_cast<Z>(sd::math::sd_sqrt<X, Z>(d1)))



DECLARE_BINARY_COPY_OP(RelativeError,
                       static_cast<Z>(sd::math::sd_re<X>(d1, static_cast<X>(d2))),
                       static_cast<Z>(sd::math::sd_re<X>(d1, static_cast<X>(d2))),
                       static_cast<Z>(0),
                       static_cast<Z>(0)
)

// BinaryRelativeError - Custom conditional logic, manual implementation needed
template <typename X, typename Y, typename Z>
class BinaryRelativeError {
 private:
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, Y d2, Z *params) {
    X threshold = static_cast<X>(params[0]);
    return sd::math::sd_re<X>(d1, static_cast<X>(d2)) > threshold ? static_cast<Z>(1) : static_cast<Z>(0);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1) { return static_cast<Z>(0); }
  SD_HOST_DEVICE SD_INLINE  static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }
  SD_HOST_DEVICE SD_INLINE  static Z op_simd(X d1) { return op_logic(d1); }

 public:
  no_op_exec_special no_op_exec_special_cuda
  static Z op(X d1, Y d2, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2, params);
    else return op_simd(d1, d2, params);
  }
  static Z op(X d1) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1);
    else return op_simd(d1);
  }
};

template <typename X, typename Y, typename Z>
class BinaryMinimumAbsoluteRelativeError {
 private:
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, X *params) {
    X d2 = params[0];
    X thresholdRelative = params[1];
    X thresholdAbsolute = params[2];
    return sd::math::sd_re<X>(d1, d2) > thresholdRelative
           ? (sd::math::sd_abs<X,X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0)
                                                                                 : static_cast<Z>(1))
           : static_cast<Z>(0);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1, Y d2, Z *params) {
    X thresholdRelative = static_cast<X>(params[0]);
    X thresholdAbsolute = static_cast<X>(params[1]);
    return sd::math::sd_re<X>(d1, static_cast<X>(d2)) > thresholdRelative
           ? (sd::math::sd_abs<X,X>(d1 - static_cast<X>(d2)) < thresholdAbsolute ? static_cast<Z>(0)
                                                                                 : static_cast<Z>(1))
           : static_cast<Z>(0);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op_logic(X d1) { return static_cast<Z>(0); }
  SD_HOST_DEVICE SD_INLINE  static Z op_simd(X d1, X *params) { return op_logic(d1, params); }
  SD_HOST_DEVICE SD_INLINE  static Z op_simd(X d1, Y d2, Z *params) { return op_logic(d1, d2, params); }
  SD_HOST_DEVICE SD_INLINE  static Z op_simd(X d1) { return op_logic(d1); }

 public:
  no_op_exec_special no_op_exec_special_cuda
  static SD_HOST_DEVICE SD_INLINE  Z op(X d1, X *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1, params);
    else return op_simd(d1, params);
  }
  static SD_HOST_DEVICE SD_INLINE  Z op(X d1, Y d2, Z *params) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value ||
                  simdOps::is_simd_unsupported_argument_type<Y>::value)
      return op_logic(d1, d2, params);
    else return op_simd(d1, d2, params);
  }
  static  SD_HOST_DEVICE SD_INLINE  Z op(X d1) {
    if constexpr (simdOps::is_simd_unsupported_return_type<Z>::value ||
                  simdOps::is_simd_unsupported_argument_type<X>::value)
      return op_logic(d1);
    else return op_simd(d1);
  }
};



// =============================================================================
// REDUCE OPERATIONS
// =============================================================================

DECLARE_REDUCE_OP(Sum, SUM, static_cast<X>(0.0f), opOutput + old, opOutput + old, reduction)
DECLARE_REDUCE_OP(Prod, PRODUCT, static_cast<X>(1), opOutput * old, opOutput * old, reduction)
DECLARE_REDUCE_OP(Max, MAX, -sd::DataTypeUtils::infOrMax<X>(),
                  sd::math::sd_max<X>(old, opOutput), sd::math::sd_max<X>(opOutput, old), reduction)
DECLARE_REDUCE_OP(Min, MIN, sd::DataTypeUtils::infOrMax<X>(),
                  sd::math::sd_min<X>(old, opOutput), sd::math::sd_min<X>(opOutput, old), reduction)

DECLARE_ACCUMULATION_SIMD_SAFE_OP(Mean,
                                  return static_cast<InterType>(d1);,
                                  SUM,
                                  static_cast<X>(0),
                                  old + opOutput,
                                  old + opOutput,
                                  reduction / static_cast<InterType>(n)
)

// =============================================================================
// INDEX REDUCE OPERATIONS
// =============================================================================

DECLARE_INDEX_REDUCE_OP(IndexMax, -sd::DataTypeUtils::infOrMax<X>(),
                        opOutput.value > old.value, f1.value > f2.value)
DECLARE_INDEX_REDUCE_OP(IndexMin, sd::DataTypeUtils::infOrMax<X>(),
                        opOutput.value < old.value, f1.value < f2.value)


DECLARE_BINARY_COPY_OP(LogPoissonLoss,
                       sd::math::sd_exp<Y COMMA Z>(d2) - static_cast<Z>(d1) * static_cast<Z>(d2),
                       sd::math::sd_exp<Y COMMA Z>(d2) - static_cast<Z>(d1) * static_cast<Z>(d2),
                       static_cast<Z>(d1),
                       sd::math::sd_exp<Y COMMA Z>(params[0]) - static_cast<Z>(d1) * static_cast<Z>(params[0])
)



// LogicalNot - using existing binary copy pattern
DECLARE_BINARY_COPY_OP(LogicalNot,
                       static_cast<Z>(!((int)d1 && (int)d2)),
                       static_cast<Z>(!(static_cast<int>(d1) && static_cast<int>(d2))),
                       static_cast<Z>(d1),
                       static_cast<Z>(119)
)

// LogicalXor - bitwise XOR logic
DECLARE_BINARY_COPY_OP(LogicalXor,
                       static_cast<Z>((static_cast<int>(d1) | static_cast<int>(d2)) & ~(static_cast<int>(d1) & static_cast<int>(d2))),
                       static_cast<Z>((static_cast<int>(d1) | static_cast<int>(d2)) & ~(static_cast<int>(d1) & static_cast<int>(d2))),
                       static_cast<Z>(d1),
                       static_cast<Z>(119)
)

// LogicalAnd - bitwise AND logic
DECLARE_BINARY_COPY_OP(LogicalAnd,
                       static_cast<Z>(static_cast<int>(d1) & static_cast<int>(d2)),
                       static_cast<Z>(static_cast<int>(d1) & static_cast<int>(d2)),
                       static_cast<Z>(d1),
                       static_cast<Z>(119)
)

// LogicalOr - bitwise OR logic
DECLARE_BINARY_COPY_OP(LogicalOr,
                       static_cast<Z>(static_cast<int>(d1) | static_cast<int>(d2)),
                       static_cast<Z>(static_cast<int>(d1) | static_cast<int>(d2)),
                       static_cast<Z>(d1),
                       static_cast<Z>(119)
)
template <typename X, typename Z>
class MatchCondition {
 private:
  static SD_HOST_DEVICE SD_INLINE Z op_logic(X d1, X compare, X eps, int mode) {
    switch (mode) {
      case 0: return static_cast<Z>(sd::math::sd_abs<X,X>(d1 - compare) <= eps ? 1 : 0);
      case 1: return static_cast<Z>(sd::math::sd_abs<X,X>(d1 - compare) > eps ? 1 : 0);
      case 2: return static_cast<Z>(d1 < compare ? 1 : 0);
      case 3: return static_cast<Z>(d1 > compare ? 1 : 0);
      case 4: return static_cast<Z>(d1 <= compare ? 1 : 0);
      case 5: return static_cast<Z>(d1 >= compare ? 1 : 0);
      case 6: return static_cast<Z>(sd::math::sd_abs<X,X>(d1) < compare ? 1 : 0);
      case 7: return static_cast<Z>(sd::math::sd_abs<X,X>(d1) > compare ? 1 : 0);
      case 8: return static_cast<Z>(sd::math::sd_isinf(d1) ? 1 : 0);
      case 9: return static_cast<Z>(sd::math::sd_isnan(d1) ? 1 : 0);
      case 10: return static_cast<Z>((d1 == compare) ? 1 : 0);
      case 11: return static_cast<Z>((d1 != compare) ? 1 : 0);
      case 12: return static_cast<Z>(sd::math::sd_abs<X,X>(d1) >= compare ? 1 : 0);
      case 13: return static_cast<Z>(sd::math::sd_abs<X,X>(d1) <= compare ? 1 : 0);
      case 14: return static_cast<Z>(!(sd::math::sd_isinf<X>(d1) || sd::math::sd_isnan<X>(d1)) ? 1 : 0);
      case 15: return static_cast<Z>(sd::math::sd_isinf<X>(d1) || sd::math::sd_isnan<X>(d1) ? 1 : 0);
      default: sd_printf("Undefined match condition: [%i]\n", mode);
    }
    return static_cast<Z>(d1);
  }

 public:
  static const bool requiresSpecialAccumulation = false;

  // CRITICAL: Add the missing InterType typedef
  using InterType = typename AggregateType<Z>::type;

  // execSpecial signatures - matches what reduce_long.hpp expects
  static void execSpecial(const X *x, const sd::LongType *xShapeInfo, sd::LongType *extraParams, Z *result,
                          const sd::LongType *resultShapeInfoBuffer, sd::LongType *dimension, sd::LongType dimensionLength,
                          const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {}

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void execSpecialCuda(
      const X *dx, const sd::LongType *xShapeInfo, sd::LongType *extraParams, Z *result,
      const sd::LongType *resultShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
      Z *reductionBuffer, const sd::LongType *tadOnlyShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  // Core reduction operation methods - these use Z* parameters
  SD_HOST_DEVICE SD_INLINE static Z startingValue(const X* input) { return static_cast<Z>(0); }
  SD_HOST_DEVICE SD_INLINE static InterType merge(InterType old, InterType opOutput, Z* extraParams) { return old + opOutput; }
  SD_HOST_DEVICE SD_INLINE static InterType update(InterType old, InterType opOutput, Z* extraParams) { return old + opOutput; }
  SD_HOST_DEVICE SD_INLINE static Z postProcess(InterType reduction, sd::LongType n, Z* extraParams) { return static_cast<Z>(reduction); }

  // Core op methods - these use Z* parameters
  static SD_HOST_DEVICE SD_INLINE InterType op(X d1, Z* extraParams) {
    if (extraParams == nullptr) return static_cast<InterType>(0);
    X compare = static_cast<X>(extraParams[0]);
    X eps = static_cast<X>(extraParams[1]);
    auto mode = static_cast<int>(extraParams[2]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  static SD_HOST_DEVICE SD_INLINE InterType op(X d1, X d2, Z* extraParams) {
    if (extraParams == nullptr) {
      // If no extraParams, use d2 as compare value, default eps=0, mode=0 (equals)
      return static_cast<InterType>(op_logic(d1, d2, static_cast<X>(0), 0));
    }

    // Use d2 as comparison value, extraParams for eps and mode
    X compare = d2;
    X eps = static_cast<X>(extraParams[0]);
    auto mode = static_cast<int>(extraParams[1]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  static SD_HOST_DEVICE SD_INLINE InterType op(X d1, X d2) {
    // Default: compare d1 to d2 with eps=0 and mode=0 (equals)
    return static_cast<InterType>(op_logic(d1, d2, static_cast<X>(0), 0));
  }

  // *** TEMPLATE OVERLOADS FOR DIFFERENT PARAMETER TYPES ***

  // Template overloads for X* parameters - only when X != Z
  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, X> && !std::is_same_v<X, Z>, InterType>::type
      op(X d1, ParamType* extraParams) {
    if (extraParams == nullptr) return static_cast<InterType>(0);
    X compare = extraParams[0];
    X eps = extraParams[1];
    auto mode = static_cast<int>(extraParams[2]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      op(X d1, X d2, ParamType* extraParams) {
    if (extraParams == nullptr) {
      return static_cast<InterType>(op_logic(d1, d2, static_cast<X>(0), 0));
    }
    X compare = d2;
    X eps = extraParams[0];
    auto mode = static_cast<int>(extraParams[1]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      merge(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      update(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, Z>::type
      postProcess(InterType reduction, sd::LongType n, ParamType* extraParams) {
    return static_cast<Z>(reduction);
  }

  // Template overloads for sd::LongType* parameters
  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, sd::LongType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      op(X d1, ParamType* extraParams) {
    if (extraParams == nullptr) return static_cast<InterType>(0);
    X compare = static_cast<X>(extraParams[0]);
    X eps = static_cast<X>(extraParams[1]);
    auto mode = static_cast<int>(extraParams[2]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, sd::LongType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      op(X d1, X d2, ParamType* extraParams) {
    if (extraParams == nullptr) {
      return static_cast<InterType>(op_logic(d1, d2, static_cast<X>(0), 0));
    }
    X compare = d2;
    X eps = static_cast<X>(extraParams[0]);
    auto mode = static_cast<int>(extraParams[1]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, sd::LongType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      merge(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, sd::LongType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      update(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_same_v<ParamType, sd::LongType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, Z>::type
      postProcess(InterType reduction, sd::LongType n, ParamType* extraParams) {
    return static_cast<Z>(reduction);
  }

  // Template overloads for float* parameters (for cases like bfloat16/bfloat16 with float* extraParams)
  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_floating_point_v<ParamType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      op(X d1, ParamType* extraParams) {
    if (extraParams == nullptr) return static_cast<InterType>(0);
    X compare = static_cast<X>(extraParams[0]);
    X eps = static_cast<X>(extraParams[1]);
    auto mode = static_cast<int>(extraParams[2]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_floating_point_v<ParamType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      op(X d1, X d2, ParamType* extraParams) {
    if (extraParams == nullptr) {
      return static_cast<InterType>(op_logic(d1, d2, static_cast<X>(0), 0));
    }
    X compare = d2;
    X eps = static_cast<X>(extraParams[0]);
    auto mode = static_cast<int>(extraParams[1]);
    return static_cast<InterType>(op_logic(d1, compare, eps, mode));
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_floating_point_v<ParamType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      merge(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_floating_point_v<ParamType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, InterType>::type
      update(InterType old, InterType opOutput, ParamType* extraParams) {
    return old + opOutput;
  }

  template<typename ParamType>
  static SD_HOST_DEVICE SD_INLINE
      typename std::enable_if<std::is_floating_point_v<ParamType> && !std::is_same_v<ParamType, X> && !std::is_same_v<ParamType, Z>, Z>::type
      postProcess(InterType reduction, sd::LongType n, ParamType* extraParams) {
    return static_cast<Z>(reduction);
  }
};


// --- Specialization: std::basic_string<char32_t> (UTF-32) -> std::basic_string<char16_t> (UTF-16) ---
template <>
class Assign<std::basic_string<char32_t>, std::basic_string<char16_t>> {
 public:
  static const bool requiresSpecial = false;
  static SD_HOST_DEVICE void execSpecial(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char32_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char32_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char16_t> *reductionPointer, // Z is std::basic_string<char16_t>
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE SD_INLINE  static std::basic_string<char16_t>
  op(const std::basic_string<char32_t>& d1, std::basic_string<char32_t> * /*params*/) {
    char16_t temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES / sizeof(char16_t) + 1];
    const char32_t* input_data = d1.data();
    const uint32_t input_length_char32_units = static_cast<uint32_t>(d1.length());

    sd::LongType required_bytes_for_utf16 = sd::unicode::offsetUtf32StringInUtf16(input_data, input_length_char32_units);

    if (required_bytes_for_utf16 > 0 && static_cast<size_t>(required_bytes_for_utf16) < sizeof(temp_output_buffer)) {
      void* end_ptr = sd::unicode::utf32to16Ptr(input_data, input_data + input_length_char32_units, temp_output_buffer);
      size_t char16_units_written = static_cast<char16_t*>(end_ptr) - temp_output_buffer;
      if (char16_units_written * sizeof(char16_t) == static_cast<size_t>(required_bytes_for_utf16)) {
        return std::basic_string<char16_t>(temp_output_buffer, char16_units_written);
      }
    }
    return std::basic_string<char16_t>();
  }
};

// --- Specialization: std::basic_string<char16_t> (UTF-16) -> std::basic_string<char32_t> (UTF-32) ---
template <>
class Assign<std::basic_string<char16_t>, std::basic_string<char32_t>> {
 public:
  static const bool requiresSpecial = false;
  static SD_HOST_DEVICE SD_INLINE  void execSpecial(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char16_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char16_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char32_t> *reductionPointer, // Z is std::basic_string<char32_t>
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif

  SD_HOST_DEVICE SD_INLINE  static std::basic_string<char32_t>
  op(const std::basic_string<char16_t>& d1, std::basic_string<char16_t> * /*params*/) {
    char32_t temp_output_buffer[SD_STRING_ASSIGN_TEMP_BUFFER_BYTES / sizeof(char32_t) + 1];
    const char16_t* input_data = d1.data();
    const uint32_t input_length_char16_units = static_cast<uint32_t>(d1.length());

    sd::LongType required_bytes_for_utf32_data = sd::unicode::offsetUtf16StringInUtf32(input_data, input_length_char16_units);

    if (required_bytes_for_utf32_data > 0 && static_cast<size_t>(required_bytes_for_utf32_data) < sizeof(temp_output_buffer)) {
      void* end_ptr = sd::unicode::utf16to32Ptr(input_data, input_data + input_length_char16_units, temp_output_buffer);
      size_t char32_units_written = static_cast<char32_t*>(end_ptr) - temp_output_buffer;
      if (char32_units_written * sizeof(char32_t) == static_cast<size_t>(required_bytes_for_utf32_data)) {
        return std::basic_string<char32_t>(temp_output_buffer, char32_units_written);
      }
    }
    return std::basic_string<char32_t>();
  }
};


// --- Identity Specializations (Redundant in modernized version but included for completeness) ---
template <>
class Assign<std::basic_string<char16_t>, std::basic_string<char16_t>> {
 public:
  static const bool requiresSpecial = false;
  static SD_HOST_DEVICE SD_INLINE  void execSpecial(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char16_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char16_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char16_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char16_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char16_t> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif
  SD_HOST_DEVICE SD_INLINE   static std::basic_string<char16_t>
  op(const std::basic_string<char16_t>& d1, std::basic_string<char16_t> * /*params*/) {
    return d1;
  }
};

template <>
class Assign<std::basic_string<char32_t>, std::basic_string<char32_t>> {
 public:
  static const bool requiresSpecial = false;
  static  SD_HOST_DEVICE SD_INLINE  void execSpecial(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                          std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                          std::basic_string<char32_t> *extraParams, const sd::LongType *tadShapeInfo,
                          const sd::LongType *tadOffsets) {}
#ifdef __CUDACC__
  static SD_DEVICE void execSpecialCuda(const std::basic_string<char32_t> *dx, const sd::LongType *xShapeBuffer,
                                        std::basic_string<char32_t> *result, const sd::LongType *resultShapeBuffer,
                                        std::basic_string<char32_t> *extraParams,
                                        sd::LongType *allocationPointer,
                                        std::basic_string<char32_t> *reductionPointer,
                                        const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets) {}
#endif
  SD_HOST_DEVICE SD_INLINE  static std::basic_string<char32_t>
  op(const std::basic_string<char32_t>& d1, std::basic_string<char32_t> * /*params*/) {
    return d1;
  }
};

}  // namespace simdOps

#endif