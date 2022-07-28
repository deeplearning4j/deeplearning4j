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

#ifndef LIBND4J_PLATFORM_MATH_H
#define LIBND4J_PLATFORM_MATH_H
#include <math.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include <cmath>

#ifdef __CUDACC__
#include <types/bfloat16.h>
#include <types/float16.h>

union BPAIR {
  struct {
    bfloat16 H;
    bfloat16 L;
  } B;
  int W;

  SD_HOST_DEVICE
  BPAIR(){};

  SD_HOST_DEVICE
  ~BPAIR(){};
};

#if CUDA_VERSION_MAJOR == 8
typedef union {
  struct {
    half H;
    half L;
  } B;
  int W;
} PAIR;
#else
struct HALFS {
  half H;
  half L;

  SD_HOST_DEVICE
  HALFS(){};

  SD_HOST_DEVICE
  ~HALFS(){};
};
union PAIR {
  HALFS B;
  int W;

  SD_HOST_DEVICE
  PAIR(){};

  SD_HOST_DEVICE
  ~PAIR() {}
};
#endif  // cuda_9

#else
#include <types/float16.h>

#endif

namespace sd {
namespace math {
template <typename T>
SD_INLINE SD_HOST_DEVICE T p_exp(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log2(T value);


template <typename T>
SD_INLINE SD_HOST_DEVICE T p_floor(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_ceil(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_cos(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_cosh(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_acos(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_acosh(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sin(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sinh(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_asin(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sqrt(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_tanh(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_erf(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_erfc(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atan(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_tan(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atanh(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rint(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotl(T value, T shift);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotr(T value, T shift);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_remainder(T val1, T val2);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_fmod(T val1, T val2);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_pow(T value, T power);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atan2(T val1, T val2);

//////

template <>
SD_INLINE SD_HOST_DEVICE float p_exp(float value) {
  return expf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_exp(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hexp(val.data);
#else
  return static_cast<float16>(expf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_exp(bfloat16 val) {
  return static_cast<bfloat16>(expf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_exp(double value) {
  return exp(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_exp(T value) {
  return static_cast<T>(expf(static_cast<float>(value)));
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float16 p_pow(float16 value, float16 power) {
  return static_cast<float16>(powf(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_pow(bfloat16 value, bfloat16 power) {
  return static_cast<bfloat16>(powf(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE float p_pow(float value, float power) {
  return powf(value, power);
}

template <>
SD_INLINE SD_HOST_DEVICE double p_pow(double value, double power) {
  return pow(value, power);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_pow(T value, T power) {
  return static_cast<T>(powf(static_cast<float>(value), static_cast<float>(power)));
}
/////////

template <>
SD_INLINE SD_HOST_DEVICE float16 p_fmod(float16 value, float16 power) {
  return static_cast<float16>(fmodf(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_fmod(bfloat16 value, bfloat16 power) {
  return static_cast<bfloat16>(fmodf(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE float p_fmod(float value, float power) {
  return fmodf(value, power);
}

template <>
SD_INLINE SD_HOST_DEVICE double p_fmod(double value, double power) {
  return fmod(value, power);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_fmod(T value, T power) {
  return static_cast<T>(fmodf(static_cast<float>(value), static_cast<float>(power)));
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float16 p_atan2(float16 value, float16 power) {
  return static_cast<float16>(atan2f(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE float p_atan2(float value, float power) {
  return atan2f(value, power);
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atan2(double value, double power) {
  return atan2(value, power);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atan2(T value, T power) {
  return static_cast<T>(atan2f(static_cast<float>(value), static_cast<float>(power)));
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float16 p_remainder(float16 value, float16 power) {
  return static_cast<float16>(remainderf(static_cast<float>(value), static_cast<float>(power)));
}

template <>
SD_INLINE SD_HOST_DEVICE float p_remainder(float value, float power) {
  return remainderf(value, power);
}

template <>
SD_INLINE SD_HOST_DEVICE double p_remainder(double value, double power) {
  return remainder(value, power);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_remainder(T value, T power) {
  return static_cast<T>(remainderf(static_cast<float>(value), static_cast<float>(power)));
}
/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_log(float value) {
  return logf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_log(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hlog(val.data);
#else
  return static_cast<float16>(logf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE double p_log(double value) {
  return log(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log(T value) {
  return static_cast<T>(logf(static_cast<float>(value)));
}


template <>
SD_INLINE SD_HOST_DEVICE float p_log2(float value) {
  return log2f(value);
}

template <>
SD_INLINE SD_HOST_DEVICE double p_log2(double value) {
  return log2(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log2(T value) {
  return static_cast<T>(log2f(static_cast<float>(value)));
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_floor(float value) {
  return floorf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_floor(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hfloor(val.data);
#else
  return static_cast<float16>(floorf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_floor(bfloat16 value) {
  return static_cast<bfloat16>(floorf((float)value));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_floor(double value) {
  return floor(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_floor(T value) {
  return value;
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_ceil(float value) {
  return ceilf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_ceil(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hceil(val.data);
#else
  return static_cast<float16>(ceilf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_ceil(bfloat16 value) {
  return static_cast<bfloat16>(ceilf((float)value));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_ceil(double value) {
  return ceil(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_ceil(T value) {
  return value;
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_round(float value) {
  return roundf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round(float16 val) {
  return static_cast<float16>(roundf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round(bfloat16 value) {
  return static_cast<bfloat16>(roundf((float)value));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round(double value) {
  return round(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round(T value) {
  return value;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round_prefer_ceil(T value) {
  return value;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_round_prefer_ceil(float value) {
  return roundf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round_prefer_ceil(float16 val) {
  return static_cast<float16>(roundf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round_prefer_ceil(bfloat16 value) {
  return static_cast<bfloat16>(roundf((float)value));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round_prefer_ceil(double value) {
  return round(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round_prefer_floor(T value) {
  return value;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_round_prefer_floor(float value) {
  if (value == static_cast<int64_t>(value) + 0.5f) {
    return floorf(value);
  }
  return roundf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round_prefer_floor(float16 val) {
  float float_val = static_cast<float>(val);
  if (float_val == static_cast<int64_t>(float_val) + 0.5f) {
    return static_cast<float16>(floorf(float_val));
  }
  return static_cast<float16>(roundf(float_val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round_prefer_floor(bfloat16 val) {
  float float_val = static_cast<float>(val);
  if (float_val == static_cast<int64_t>(float_val) + 0.5f) {
    return static_cast<bfloat16>(floorf(float_val));
  }
  return static_cast<bfloat16>(roundf(float_val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round_prefer_floor(double value) {
  if (value == static_cast<int64_t>(value) + 0.5) {
    return floor(value);
  }
  return round(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_rint(float value) {
  return rintf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_rint(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hrint(val.data);
#else
  return static_cast<float16>(rintf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_rint(bfloat16 val) {
  return static_cast<bfloat16>(rintf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_rint(double value) {
  return rint(value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rint(T value) {
  return value;
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_cos(float value) {
  return cosf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_cos(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hcos(val.data);
#else
  return static_cast<float16>(cosf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_cos(bfloat16 val) {
  return static_cast<bfloat16>(cosf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_cos(double value) {
  return cos(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_sin(float value) {
  return sinf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sin(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hsin(val.data);
#else
  return static_cast<float16>(sinf((float)val));
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sin(bfloat16 val) {
  return static_cast<bfloat16>(sinf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sin(double value) {
  return sin(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_sqrt(float value) {
  return sqrtf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sqrt(float16 val) {
#ifdef SD_NATIVE_HALFS
  return hsqrt(val.data);
#else
  return static_cast<float16>(sqrtf((float)val));
#endif
}
template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sqrt(bfloat16 val) {
  return static_cast<float16>(sqrtf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sqrt(double value) {
  return sqrt(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_tanh(float value) {
  return tanhf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_tanh(float16 val) {
  return static_cast<float16>(tanhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_tanh(bfloat16 val) {
  return static_cast<bfloat16>(tanhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_tanh(double value) {
  return tanh(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_erf(float value) {
  return erff(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_erf(float16 val) {
  return static_cast<float16>(erff((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_erf(bfloat16 val) {
  return static_cast<bfloat16>(erff((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_erf(double value) {
  return erf(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_erfc(float value) {
  return erfcf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_erfc(float16 val) {
  return static_cast<float16>(erfcf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_erfc(bfloat16 val) {
  return static_cast<bfloat16>(erfcf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_erfc(double value) {
  return erfc(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_acos(float value) {
  return acosf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_acos(float16 val) {
  return static_cast<float16>(acosf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_acos(bfloat16 val) {
  return static_cast<bfloat16>(acosf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_acos(double value) {
  return acos(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_sinh(float value) {
  return sinhf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sinh(float16 val) {
  return static_cast<float16>(sinhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sinh(bfloat16 val) {
  return static_cast<bfloat16>(sinhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sinh(double value) {
  return sinh(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_acosh(float value) {
  return acoshf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_acosh(float16 val) {
  return static_cast<float16>(acoshf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_acosh(bfloat16 val) {
  return static_cast<bfloat16>(acoshf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_acosh(double value) {
  return acosh(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_cosh(float value) {
  return coshf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_cosh(float16 val) {
  return static_cast<float16>(coshf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_cosh(bfloat16 val) {
  return static_cast<bfloat16>(coshf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_cosh(double value) {
  return cosh(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_asin(float value) {
  return asinf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_asin(float16 val) {
  return static_cast<float16>(asinf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_asin(bfloat16 val) {
  return static_cast<bfloat16>(asinf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_asin(double value) {
  return asin(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_atan(float value) {
  return atanf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_atan(float16 val) {
  return static_cast<float16>(atanf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_atan(bfloat16 val) {
  return static_cast<bfloat16>(atanf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atan(double value) {
  return atan(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_tan(float value) {
  return tanf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_tan(float16 val) {
  return static_cast<float16>(tanf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_tan(bfloat16 val) {
  return static_cast<bfloat16>(tanf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_tan(double value) {
  return tan(value);
}

/////////

template <>
SD_INLINE SD_HOST_DEVICE float p_atanh(float value) {
  return atanhf(value);
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_atanh(float16 val) {
  return static_cast<float16>(atanhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_atanh(bfloat16 val) {
  return static_cast<bfloat16>(atanhf((float)val));
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atanh(double value) {
  return atanh(value);
}

/////////
template <typename T>
SD_INLINE SD_HOST_DEVICE T _rotate_left(T value, T shift);

template <typename T>
SD_INLINE SD_HOST_DEVICE T _rotate_right(T value, T shift);

template <>
SD_INLINE SD_HOST_DEVICE int8_t _rotate_left(int8_t value, int8_t shift) {
  return value << shift | value >> (8 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t _rotate_right(int8_t value, int8_t shift) {
  return value >> shift | value << (8 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t _rotate_left(uint8_t value, uint8_t shift) {
  return value << shift | value >> (8 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t _rotate_right(uint8_t value, uint8_t shift) {
  return value >> shift | value << (8 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t _rotate_left(int16_t value, int16_t shift) {
  return value << shift | value >> (16 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t _rotate_right(int16_t value, int16_t shift) {
  return value >> shift | value << (16 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t _rotate_left(uint16_t value, uint16_t shift) {
  return value << shift | value >> (16 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t _rotate_right(uint16_t value, uint16_t shift) {
  return value >> shift | value << (16 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE int _rotate_left(int value, int shift) {
  return value << shift | value >> (32 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE int _rotate_right(int value, int shift) {
  return value >> shift | value << (32 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t _rotate_left(uint32_t value, uint32_t shift) {
  return value << shift | value >> (32 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t _rotate_right(uint32_t value, uint32_t shift) {
  return value >> shift | value << (32 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType _rotate_left(sd::LongType value, sd::LongType shift) {
  return value << shift | value >> (64 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType _rotate_right(sd::LongType value, sd::LongType shift) {
  return value >> shift | value << (64 - shift);
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t _rotate_left(uint64_t value, uint64_t shift) {
#ifdef SD_ARM_BUILD
  // TODO: eventually remove this once gcc fixes the bug
  sd::LongType val =
      _rotate_left<sd::LongType>(*reinterpret_cast<sd::LongType *>(&value), *reinterpret_cast<sd::LongType *>(&shift));
  return *reinterpret_cast<uint64_t *>(&val);
#else
  return value << shift | value >> (64 - shift);
#endif
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t _rotate_right(uint64_t value, uint64_t shift) {
#ifdef SD_ARM_BUILD
  // TODO: eventually remove this once gcc fixes the bug
  sd::LongType val =
      _rotate_right<sd::LongType>(*reinterpret_cast<sd::LongType *>(&value), *reinterpret_cast<sd::LongType *>(&shift));
  return *reinterpret_cast<uint64_t *>(&val);
#else
  return value >> shift | value << (64 - shift);
#endif
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotl(T value, T shift) {
  return _rotate_left<T>(value, shift);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotr(T value, T shift) {
  return _rotate_right<T>(value, shift);
}
}  // namespace math
}  // namespace sd

#endif  // DEV_TESTS_PLATFORM_MATH_H
