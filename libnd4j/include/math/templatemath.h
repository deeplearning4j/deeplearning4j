/*
*  ******************************************************************************
*  *
*  *
*  * This program and the accompanying materials are made available under the
*  * terms of the Apache License, Version 2.0 which is available at
*  * https://www.apache.org/licenses/LICENSE-2.0.
*  *
*  * See the NOTICE file distributed with this work for additional
*  * information regarding copyright ownership.
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*  * License for the specific language governing permissions and limitations
*  * under the License.
*  *
*  * SPDX-License-Identifier: Apache-2.0
*  *****************************************************************************
*/

/*
* templatemath.h
*
*  Created on: Jan 1, 2016
*      Author: agibsonccc
*/

#ifndef TEMPLATEMATH_H_
#define TEMPLATEMATH_H_

#include <array/DataTypeUtils.h>
#include <math/platformmath.h>
#include <system/common.h>

#include "platformmath.h"

#define BFLOAT16_MAX_VALUE 32737.
#define HALF_MAX_VALUE 65504.
#define FLOAT_MAX_VALUE 3.4028235E38
#define DOUBLE_MAX_VALUE 1.7976931348623157E308
#define SD_FLOAT_MIN_NORMAL 1.17549435e-38

#ifndef M_E
#define M_E 2.718281828459
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sd {

namespace math {

#include <type_traits>
#include <types/type_promote.h>

/*
*
*
* SD_PROMOTE_FUNC(FUNC_NAME, BODY): This macro takes two parameters: FUNC_NAME (the name of the function to be defined) and BODY (the body of the function).
Template Function:
The macro defines a template function with three template parameters: T, U (defaulting to T), and Z (defaulting to T).
The function returns a value of type Z and takes two parameters: val1 of type T and val2 of type U.
Type Promotion:
Inside the function, a type alias calc_type is defined using promote_type3<T, U, Z>::type, which determines the promoted type among T, U, and Z.
The input values val1 and val2 are cast to calc_type.
Function Body:
The BODY parameter is evaluated to compute the result, which is then cast to type Z before being returned.
* */
// Macro to define functions with advanced type promotion and debugging
// Updated SD_PROMOTE_FUNC macro
#define SD_PROMOTE_FUNC(FUNC_NAME, BODY)                                \
template<typename T, typename U = T, typename Z = T>                    \
SD_HOST_DEVICE SD_INLINE Z FUNC_NAME(T val1, U val2) {                  \
   using calc_type = typename promote_type3<T, U, Z>::type;            \
   calc_type promoted_val1 = static_cast<calc_type>(val1);             \
   calc_type promoted_val2 = static_cast<calc_type>(val2);             \
   calc_type result = BODY;                                            \
   SD_PRINT_MATH_FUNC2(#FUNC_NAME, promoted_val1, promoted_val2, result); \
   return static_cast<Z>(result);                                      \
}

#define SD_PROMOTE_FUNC3(FUNC_NAME, BODY)                                \
template<typename T, typename U = T, typename V = T, typename Z = T>     \
SD_HOST_DEVICE SD_INLINE Z FUNC_NAME(T val1, U val2, V eps) {            \
   using calc_type = typename promote_type3<T, U, Z>::type;             \
   calc_type promoted_val1 = static_cast<calc_type>(val1);              \
   calc_type promoted_val2 = static_cast<calc_type>(val2);              \
   calc_type promoted_eps = static_cast<calc_type>(eps);                \
   calc_type result = BODY;                                             \
   SD_PRINT_MATH_FUNC2(#FUNC_NAME, promoted_val1, promoted_val2, result); \
   return static_cast<Z>(result);                                       \
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_abs(T value);

SD_PROMOTE_FUNC3(sd_eq, (sd_abs<calc_type, calc_type>(promoted_val1 - promoted_val2) <= promoted_eps))

template <typename T>
SD_HOST_DEVICE SD_INLINE  void  sd_swap(T& val1, T& val2);

SD_PROMOTE_FUNC(sd_max, (promoted_val1 > promoted_val2 ? promoted_val1 : promoted_val2))

SD_PROMOTE_FUNC(sd_min, (promoted_val1 < promoted_val2 ? promoted_val1 : promoted_val2))

SD_PROMOTE_FUNC(sd_add, (promoted_val1 + promoted_val2))
SD_PROMOTE_FUNC(sd_subtract, (promoted_val1 - promoted_val2))
SD_PROMOTE_FUNC(sd_multiply, (promoted_val1 * promoted_val2))
SD_PROMOTE_FUNC(sd_divide, (promoted_val1 / promoted_val2))

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_re(T val1, T val2);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_rint(T val1);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_copysign(T val1, T val2);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_softplus(T val);

template <typename T>
SD_HOST_DEVICE SD_INLINE T sd_rotl(T val, T shift);

template <typename T>
SD_HOST_DEVICE SD_INLINE T sd_rotr(T val, T shift);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_dot(X* x, Y* y, int length);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_ceil(T val1);

template <typename T>
SD_HOST_DEVICE SD_INLINE bool sd_isnan(T val1);

template <typename T>
SD_HOST_DEVICE SD_INLINE bool sd_isinf(T val1);

template <typename T>
SD_HOST_DEVICE SD_INLINE bool sd_isfin(T val1);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_cos(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_cosh(T val);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_exp(X val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_floor(T val);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_log(X val);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_pow(X val, Y val2);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_floordiv(X val, Y val2);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_round(T val);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_remainder(X num, Y denom);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_fmod(X num, Y denom);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_erf(T num);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_erfc(T num);

SD_HOST_DEVICE SD_INLINE int32_t floatToRawIntBits(float d) {
 union {
   float f;
   int32_t i;
 } tmp;
 tmp.f = d;
 return tmp.i;
}

SD_HOST_DEVICE SD_INLINE float intBitsToFloat(int32_t i) {
 union {
   float f;
   int32_t i;
 } tmp;
 tmp.i = i;
 return tmp.f;
}

SD_HOST_DEVICE SD_INLINE float mulsignf(float x, float y) {
 return intBitsToFloat(floatToRawIntBits(x) ^ (floatToRawIntBits(y) & (1 << 31)));
}

SD_HOST_DEVICE SD_INLINE float copysignfk(float x, float y) {
 return intBitsToFloat((floatToRawIntBits(x) & ~(1 << 31)) ^ (floatToRawIntBits(y) & (1 << 31)));
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sigmoid(T val) {
 Z result = (Z)1.0f / ((Z)1.0f + sd_exp<T, Z>(-val));
 SD_PRINT_MATH_FUNC("sd_sigmoid", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_elu(T val, T alpha) {
 Z result;
 if (val >= (T)0.f)
   result = val;
 else
   result = static_cast<Z>(alpha) * (sd_exp<T, Z>(val) - static_cast<Z>(1.0f));
 SD_PRINT_MATH_FUNC2("sd_elu", val, alpha, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_leakyrelu(T val, T alpha) {
 Z result;
 if (val < (T)0.0f)
   result = alpha * val;
 else
   result = val;
 SD_PRINT_MATH_FUNC2("sd_leakyrelu", val, alpha, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_eluderivative(T val, T alpha) {
 Z result;
 if (val >= static_cast<T>(0.0f))
   result = static_cast<Z>(1.0f);
 else
   result = static_cast<Z>(alpha) * sd_exp<T, Z>(val);
 SD_PRINT_MATH_FUNC2("sd_eluderivative", val, alpha, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sin(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sinh(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_softplus(T val) {
 Z result = sd_log<T, Z>((Z)1.0f + sd_exp<T, Z>(val));
 SD_PRINT_MATH_FUNC("sd_softplus", val, result);
 return result;
}


template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_log(X val) {
  Z result = static_cast<Z>(p_log<X>(val));
  SD_PRINT_MATH_FUNC("sd_log", val, result);
  return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_log2(X val) {
  Z result = static_cast<Z>(p_log2<X>(val));
  SD_PRINT_MATH_FUNC("sd_log2", val, result);
  return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_softsign(T val) {
 Z result = val / ((T)1.0f + sd::math::sd_abs<T, T>(val));
 SD_PRINT_MATH_FUNC("sd_softsign", val, result);
 return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sqrt(X val);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_tanh(X val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_tan(T val);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_atan2(X val1, X val2);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_atan2(X val1, X val2) {
 Z result = p_atan2<Z>(static_cast<Z>(val1), static_cast<Z>(val2));
 SD_PRINT_MATH_FUNC2("sd_atan2", val1, val2, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_tan(T value) {
 Z result = p_tan<Z>(static_cast<Z>(value));
 SD_PRINT_MATH_FUNC("sd_tan", value, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_tanhderivative(T val) {
 Z tanh_val = sd_tanh<T, Z>(val);
 Z result = (Z)1.0f - tanh_val * tanh_val;
 SD_PRINT_MATH_FUNC("sd_tanhderivative", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE T sd_sigmoidderivative(T val) {
 Z sigmoid = sd_sigmoid<T, Z>(val);
 T result = sigmoid * ((Z)1.0f - sigmoid);
 SD_PRINT_MATH_FUNC("sd_sigmoidderivative", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE T sd_softsignderivative(T val) {
 T y = (T)1.0f + sd_abs<T, T>(val);
 T result = (Z)1.0f / (y * y);
 SD_PRINT_MATH_FUNC("sd_softsignderivative", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE T sd_sgn(T val) {
 Z result = val < (T)0.0f ? (Z)-1.0f : val > (T)0.0f ? (Z)1.0f : (Z)0.0f;
 SD_PRINT_MATH_FUNC("sd_sgn", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sign(T val) {
 Z result = sd_sgn<T, Z>(val);
 SD_PRINT_MATH_FUNC("sd_sign", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_signum(T val) {
 Z result = sd_sgn<T, Z>(val);
 SD_PRINT_MATH_FUNC("sd_signum", val, result);
 return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_gamma(X a);

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_lgamma(X x);

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_dot(X* x, Y* y, int length) {
 Z dot = (Z)0.0f;
 for (int e = 0; e < length; e++) {
   dot += static_cast<Z>(x[e]) * static_cast<Z>(y[e]);
 }
 SD_PRINT_MATH_FUNC("sd_dot", length, dot);
 return dot;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_acos(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sech(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_acosh(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_asin(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_asinh(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_asinh(T val) {
 Z result = sd_log<Z, Z>(sd_sqrt<Z, Z>(sd_pow<T, T, Z>(val, (T)2) + (Z)1.f) + (Z)val);
 SD_PRINT_MATH_FUNC("sd_asinh", val, result);
 return result;
}

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_atan(T val);

template <typename T, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_atanh(T val);

template <>
SD_HOST_DEVICE SD_INLINE float16 sd_abs<float16, float16>(float16 value) {
#ifdef SD_NATIVE_HALFS
 float16 result;
 if (value < (float16)0.f) {
   result = float16(__hneg(value.data));
 } else {
   result = value;
 }
#else
 float16 result = (float16)fabsf((float)value);
#endif
 SD_PRINT_MATH_FUNC("sd_abs<float16>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE bfloat16 sd_abs<bfloat16, bfloat16>(bfloat16 value) {
 bfloat16 result = (bfloat16)fabsf((float)value);
 SD_PRINT_MATH_FUNC("sd_abs<bfloat16>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE float sd_abs<float, float>(float value) {
 float result = fabsf(value);
 SD_PRINT_MATH_FUNC("sd_abs<float>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE double sd_abs<double, double>(double value) {
 double result = fabs(value);
 SD_PRINT_MATH_FUNC("sd_abs<double>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE int sd_abs<int, int>(int value) {
 int result = abs(value);
 SD_PRINT_MATH_FUNC("sd_abs<int>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE sd::LongType sd_abs<sd::LongType, sd::LongType>(sd::LongType value) {
 sd::LongType result = llabs(value);
 SD_PRINT_MATH_FUNC("sd_abs<sd::LongType>", value, result);
 return result;
}

// ... Continue adding print statements to other specializations and functions ...

template <>
SD_HOST_DEVICE SD_INLINE bool sd_abs<bool>(bool value) {
 SD_PRINT_MATH_FUNC("sd_abs<bool>", value, value);
 return value;
}

template <>
SD_HOST_DEVICE SD_INLINE uint8_t sd_abs<uint8_t>(uint8_t value) {
 SD_PRINT_MATH_FUNC("sd_abs<uint8_t>", value, value);
 return value;
}

template <>
SD_HOST_DEVICE SD_INLINE uint16_t sd_abs<uint16_t>(uint16_t value) {
 SD_PRINT_MATH_FUNC("sd_abs<uint16_t>", value, value);
 return value;
}

template <>
SD_HOST_DEVICE SD_INLINE uint32_t sd_abs<uint32_t>(uint32_t value) {
 SD_PRINT_MATH_FUNC("sd_abs<uint32_t>", value, value);
 return value;
}

template <>
SD_HOST_DEVICE SD_INLINE sd::UnsignedLong sd_abs<sd::UnsignedLong>(sd::UnsignedLong value) {
 SD_PRINT_MATH_FUNC("sd_abs<sd::UnsignedLong>", value, value);
 return value;
}

template <>
SD_HOST_DEVICE SD_INLINE int8_t sd_abs<int8_t>(int8_t value) {
 int8_t result = value < 0 ? -value : value;
 SD_PRINT_MATH_FUNC("sd_abs<int8_t>", value, result);
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE int16_t sd_abs<int16_t>(int16_t value) {
 int16_t result = value < 0 ? -value : value;
 SD_PRINT_MATH_FUNC("sd_abs<int16_t>", value, result);
 return result;
}

// Similarly, add print statements to the rest of the functions and specializations...

// For example, sd_isnan specializations:

template <>
SD_HOST_DEVICE SD_INLINE bool sd_isnan<float16>(float16 value) {
 bool result = (value) == 0x7fffU;
 SD_PRINT_MATH_FUNC("sd_isnan<float16>", value, static_cast<float16>(result));
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE bool sd_isnan<bfloat16>(bfloat16 value) {
 bool result = value == bfloat16::nan();  // 0x7fffU;
 SD_PRINT_MATH_FUNC("sd_isnan<bfloat16>", value, static_cast<bfloat16>(result));
 return result;
}

template <>
SD_HOST_DEVICE SD_INLINE bool sd_isnan<float>(float value) {
 bool result = value != value;
 SD_PRINT_MATH_FUNC("sd_isnan<float>", value, static_cast<float>(result));
 return result;
}

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_igamma(X a, Y x) {
  Z result;
  if (a <= X(0.000001)) {
    result = Z(0);
  } else {
    Z aim = sd_pow<X, X, Z>(x, a) / (sd_exp<X, Z>(x) * sd_gamma<Y, Z>(a));
    Z sum = Z(0.);
    Z denom = Z(1.);
    for (int i = 0; Z(1. / denom) > Z(1.0e-12); i++) {
      denom *= (a + i);
      sum += sd_pow<X, int, Z>(x, i) / denom;
    }
    result = aim * sum;
  }
  SD_PRINT_MATH_FUNC2("sd_igamma", a, x, result);
  return result;
}

// Implementing sd_igammac and adding print statements
template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_igammac(X a, Y x) {
  Z result = Z(1.) - sd_igamma<X, Y, Z>(a, x);
  SD_PRINT_MATH_FUNC2("sd_igammac", a, x, result);
  return result;
}


/**
* This func is special case - it must return floating point value, and optionally Y arg can be floating point argument
* @tparam X
* @tparam Y
* @tparam Z
* @param val
* @param val2
* @return
 */
template <>
SD_HOST_DEVICE SD_INLINE float sd_pow(float val, float val2) {
  float result = p_pow<float>(val, val2);
  SD_PRINT_MATH_FUNC2("sd_pow float", val, val2, result);
  return result;
}

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_pow(X val, Y val2) {
  Z result = p_pow<Z>(static_cast<Z>(val), static_cast<Z>(val2));
  SD_PRINT_MATH_FUNC2("sd_pow", val, val2, result);
  return result;
}

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_floordiv(X val, Y val2) {
  Z result = static_cast<Z>(std::floor(static_cast<double>(val) / static_cast<double>(val2)));
  SD_PRINT_MATH_FUNC2("sd_floordiv", val, val2, result);
  return result;
}

// Implement sd_lgamma with print statements
template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_lgamma(X x) {
  Z result;
  if (x < X(12.0)) {
    result = sd_log<Z, Z>(sd_gamma<X, Z>(x));
  } else {
    static const double c[8] = {1.0 / 12.0,   -1.0 / 360.0,      1.0 / 1260.0, -1.0 / 1680.0,
                                1.0 / 1188.0, -691.0 / 360360.0, 1.0 / 156.0,  -3617.0 / 122400.0};
    double z = Z(1.0 / Z(x * x));
    double sum = c[7];

    for (int i = 6; i >= 0; i--) {
      sum *= z;
      sum += c[i];
    }

    double series = sum / Z(x);
    static const double halfLogTwoPi = 0.91893853320467274178032973640562;

    result = Z((double(x) - 0.5) * sd_log<X, double>(x) - double(x) + halfLogTwoPi + series);
  }
  SD_PRINT_MATH_FUNC("sd_lgamma", x, result);
  return result;
}

template <typename T>
SD_HOST_DEVICE SD_INLINE T sd_re(T val1, T val2) {
  T result;
  if (val1 == (T)0.0f && val2 == (T)0.0f)
    result = (T)0.0f;
  else
    result = sd_abs<T,T>(val1 - val2) / (sd_abs<T,T>(val1) + sd_abs<T,T>(val2));
  SD_PRINT_MATH_FUNC2("sd_re", val1, val2, result);
  return result;
}

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_remainder(X val, Y val2) {
  Z result = p_remainder<Z>(static_cast<Z>(val), static_cast<Z>(val2));
  SD_PRINT_MATH_FUNC2("sd_remainder", val, val2, result);
  return result;
}

template <typename X, typename Y, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_fmod(X val, Y val2) {
  Z result = p_fmod<Z>(static_cast<Z>(val), static_cast<Z>(val2));
  SD_PRINT_MATH_FUNC2("sd_fmod", val, val2, result);
  return result;
}



template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sin(X val) {
  Z result = p_sin<Z>(static_cast<Z>(val));
  SD_PRINT_MATH_FUNC("sd_sin", val, result);
  return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_sqrt(X val) {
  Z result = p_sqrt<Z>(static_cast<Z>(val));
  SD_PRINT_MATH_FUNC("sd_sqrt", val, result);
  return result;
}

template <typename X>
SD_HOST_DEVICE SD_INLINE X neg_tanh(X val) {
  X o = static_cast<X>(1.0f);
  X t = static_cast<X>(2.0f);
  X e = static_cast<X>(M_E);

  auto p = sd::math::sd_pow<X, X, X>(e, val * t);
  X result = (p - o) / (p + o);
  SD_PRINT_MATH_FUNC("neg_tanh", val, result);
  return result;
}

template <typename X>
SD_HOST_DEVICE SD_INLINE X pos_tanh(X val) {
  X o = static_cast<X>(1.0f);
  X t = static_cast<X>(-2.0f);
  X e = static_cast<X>(M_E);

  auto p = sd::math::sd_pow<X, X, X>(e, val * t);
  X result = (o - p) / (o + p);
  SD_PRINT_MATH_FUNC("pos_tanh", val, result);
  return result;
}

SD_HOST_DEVICE SD_INLINE float neu_tanh(float val, float sign) {
  float e(M_E);
  float av = sign * val;
  auto p = sd::math::sd_pow<float, float, float>(e, -av * 2.f);
  float result = (1 - p) / (1 + p);
  SD_PRINT_MATH_FUNC2("neu_tanh", val, sign, result);
  return result;
}

template <>
SD_HOST_DEVICE SD_INLINE float sd_tanh(float val) {
  float sign = copysignfk(1.0f, val);
  float result = sign * neu_tanh(val, sign);
  SD_PRINT_MATH_FUNC("sd_tanh<float>", val, result);
  return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_tanh(X val) {
  Z result = val <= 0 ? neg_tanh(val) : pos_tanh(val);
  SD_PRINT_MATH_FUNC("sd_tanh", val, result);
  return result;
}

template <typename T>
SD_HOST_DEVICE SD_INLINE T sd_rotl(T val, T shift) {
  T result = p_rotl<T>(val, shift);
  SD_PRINT_MATH_FUNC2("sd_rotl", val, shift, result);
  return result;
}

template <typename T>
SD_HOST_DEVICE SD_INLINE T sd_rotr(T val, T shift) {
  T result = p_rotr<T>(val, shift);
  SD_PRINT_MATH_FUNC2("sd_rotr", val, shift, result);
  return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_erf(X val) {
  Z result = p_erf<Z>(static_cast<Z>(val));
  SD_PRINT_MATH_FUNC("sd_erf", val, result);
  return result;
}

template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_erfc(X val) {
  Z result = p_erfc<Z>(static_cast<Z>(val));
  SD_PRINT_MATH_FUNC("sd_erfc", val, result);
  return result;
}




template <typename T>
SD_HOST_DEVICE SD_INLINE  void  sd_swap(T& val1, T& val2) {
  T temp = val1;
  val1 = val2;
  val2 = temp;
};


// Implement sd_gamma with print statements
template <typename X, typename Z>
SD_HOST_DEVICE SD_INLINE Z sd_gamma(X a) {
  Z result;
  if (a < X(0.001)) {
    const double eulerGamma = 0.577215664901532860606512090;
    result = Z(1.0 / ((double)a * (1.0 + eulerGamma * (double)a)));
  } else if (a < X(12.0)) {
    double y = (double)a;
    int n = 0;
    bool argWasLessThanOne = y < 1.0;

    if (argWasLessThanOne) {
      y += 1.0;
    } else {
      n = static_cast<int>(floor(y)) - 1;
      y -= n;
    }

    static const double p[] = {-1.71618513886549492533811E+0, 2.47656508055759199108314E+1,
                               -3.79804256470945635097577E+2, 6.29331155312818442661052E+2,
                               8.66966202790413211295064E+2,  -3.14512729688483675254357E+4,
                               -3.61444134186911729807069E+4, 6.64561438202405440627855E+4};

    static const double q[] = {-3.08402300119738975254353E+1, 3.15350626979604161529144E+2,
                               -1.01515636749021914166146E+3, -3.10777167157231109440444E+3,
                               2.25381184209801510330112E+4,  4.75584627752788110767815E+3,
                               -1.34659959864969306392456E+5, -1.15132259675553483497211E+5};

    double num = 0.0;
    double den = 1.0;

    double z = y - 1;
    for (auto i = 0; i < 8; i++) {
      num = (num + p[i]) * z;
      den = den * z + q[i];
    }
    double result_temp = num / den + 1.0;

    if (argWasLessThanOne) {
      result_temp /= (y - 1.0);
    } else {
      for (auto i = 0; i < n; i++) result_temp *= y++;
    }
    result = Z(result_temp);
  } else {
    if (a > 171.624) {
      result = Z(DOUBLE_MAX_VALUE);
    } else {
      result = sd::math::sd_exp<Z, Z>(sd::math::sd_lgamma<X, Z>(a));
    }
  }
  SD_PRINT_MATH_FUNC("sd_gamma", a, result);
  return result;
}

}  // namespace math
}  // namespace sd

#endif /* TEMPLATEMATH_H_ */
