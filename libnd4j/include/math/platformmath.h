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


// Include SD_PRINT_MATH_FUNC and SD_PRINT_MATH_FUNC2 macros
#ifdef SD_PRINT_MATH
#include <cstdio>
#include <cstdint>  // Include for fixed-width integer types
// New sd_print_math2 functions for functions with two inputs

#define PRINT_IF_NECESSARY(funcName) \
  const char* envFuncName = std::getenv("PRINT_MATH_FUNCTION_NAME"); \
  if (envFuncName != nullptr && std::string(envFuncName) != "" && (funcName != nullptr && std::string(funcName) != "") && std::string(envFuncName) == funcName) { \
    StackTrace st; \
    st.load_here(); \
    Printer p; \
    p.print(st); \
  }

template <typename T>
SD_INLINE SD_HOST void sd_print_math2(char* func_name, T input1, T input2, T output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %f, input2 = %f, output = %f\n",
         func_name, static_cast<double>(input1), static_cast<double>(input2), static_cast<double>(output));
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2(char* func_name, float input1, float input2, float output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %f, input2 = %f, output = %f\n",
         func_name, static_cast<double>(input1), static_cast<double>(input2), static_cast<double>(output));
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<uint16_t>(char* func_name, uint16_t input1, uint16_t input2, uint16_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
  fflush(stdout);
}

// Specializations for integer types
template <>
SD_INLINE SD_HOST void sd_print_math2<int>(char* func_name, int input1, int input2, int output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<uint32_t>(char* func_name, uint32_t input1, uint32_t input2, uint32_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <typename T>
SD_INLINE SD_HOST void sd_print_math(char* func_name, T input, T output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %f, output = %f\n", func_name, static_cast<double>(input), static_cast<double>(output));
  fflush(stdout);
}

// Specializations for integer types
template <>
SD_INLINE SD_HOST void sd_print_math<int>(char* func_name, int input, int output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %d, output = %d\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<long>(char* func_name, long input, long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %ld, output = %ld\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<unsigned long>(char* func_name, unsigned long input, unsigned long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %lu, output = %lu\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<long long>(char* func_name, long long input, long long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %lld, output = %lld\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<unsigned long long>(char* func_name, unsigned long long input, unsigned long long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %llu, output = %llu\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<int16_t>(char* func_name, int16_t input, int16_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %d, output = %d\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<uint16_t>(char* func_name, uint16_t input, uint16_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %u, output = %u\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<int8_t>(char* func_name, int8_t input, int8_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %d, output = %d\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<uint8_t>(char* func_name, uint8_t input, uint8_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %u, output = %u\n", func_name, input, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math<uint32_t>(char* func_name, uint32_t input, uint32_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %u, output = %u\n", func_name, input, output);
  fflush(stdout);
}

// Specializations for float16
template <>
SD_INLINE SD_HOST void sd_print_math(char* func_name, float16 input, float16 output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %f, output = %f\n", func_name, static_cast<float>(input), static_cast<float>(output));
  fflush(stdout);
}

// Specializations for bfloat16
template <>
SD_INLINE SD_HOST void sd_print_math<bfloat16>(char* func_name, bfloat16 input, bfloat16 output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %f, output = %f\n", func_name, static_cast<float>(input), static_cast<float>(output));
  fflush(stdout);
}

// Specialization for bool
template <>
SD_INLINE SD_HOST void sd_print_math<bool>(char* func_name, bool input, bool output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input = %s, output = %s\n", func_name, input ? "true" : "false", output ? "true" : "false");
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<uint64_t>(char* func_name, uint64_t input1, uint64_t input2, uint64_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %ld, input2 = %ld, output = %ld\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<long>(char* func_name, long input1, long input2, long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %ld, input2 = %ld, output = %ld\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<long long>(char* func_name, long long input1, long long input2, long long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %lld, input2 = %lld, output = %lld\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<unsigned long long>(char* func_name, unsigned long long input1, unsigned long long input2, unsigned long long output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %llu, input2 = %llu, output = %llu\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<int16_t>(char* func_name, int16_t input1, int16_t input2, int16_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<int8_t>(char* func_name, int8_t input1, int8_t input2, int8_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %d, input2 = %d, output = %d\n", func_name, input1, input2, output);
  fflush(stdout);
}

template <>
SD_INLINE SD_HOST void sd_print_math2<uint8_t>(char* func_name, uint8_t input1, uint8_t input2, uint8_t output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %u, input2 = %u, output = %u\n", func_name, input1, input2, output);
  fflush(stdout);
}

// Specializations for float16
template <>
SD_INLINE SD_HOST void sd_print_math2<float16>(char* func_name, float16 input1, float16 input2, float16 output) {
#if defined(SD_GCC_FUNCTRACE)
  PRINT_IF_NECESSARY(func_name);
#endif
  printf("%s: input1 = %f, input2 = %f, output = %f\n",
         func_name, static_cast<float>(input1), static_cast<float>(input2), static_cast<float>(output));
  fflush(stdout);
}



#define SD_PRINT_MATH_FUNC(func_name, input, output) \
 BUILD_SINGLE_TEMPLATE(sd_print_math, (func_name, input, output), SD_NUMERIC_TYPES)

#define SD_PRINT_MATH_FUNC2(func_name, input1, input2, output) \
 BUILD_SINGLE_TEMPLATE(sd_print_math2,(func_name, input1, input2, output),SD_NUMERIC_TYPES)                                                              \

#else
#define SD_PRINT_MATH_FUNC(func_name, input, output)
#define SD_PRINT_MATH_FUNC2(func_name, input1, input2, output)
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
SD_INLINE SD_HOST_DEVICE T p_round_prefer_ceil(T value);

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round_prefer_floor(T value);

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

// Function implementations with SD_PRINT_MATH_FUNC added

// p_exp
template <>
SD_INLINE SD_HOST_DEVICE float p_exp(float value) {
 float result = expf(value);
 SD_PRINT_MATH_FUNC("p_exp<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_exp(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hexp(val.data);
#else
 float16 result = static_cast<float16>(expf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_exp<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_exp(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(expf((float)val));
 SD_PRINT_MATH_FUNC("p_exp<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_exp(double value) {
 double result = exp(value);
 SD_PRINT_MATH_FUNC("p_exp<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_exp(T value) {
 T result = static_cast<T>(expf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_exp<T>", value, result);
 return result;
}

// p_pow
template <>
SD_INLINE SD_HOST_DEVICE float16 p_pow(float16 value, float16 power) {
 float16 result = static_cast<float16>(powf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_pow<float16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_pow(bfloat16 value, bfloat16 power) {
 bfloat16 result = static_cast<bfloat16>(powf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_pow<bfloat16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_pow(float value, float power) {
 float result = powf(value, power);
 SD_PRINT_MATH_FUNC("p_pow<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_pow(double value, double power) {
 double result = pow(value, power);
 SD_PRINT_MATH_FUNC("p_pow<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_pow(T value, T power) {
 T result = static_cast<T>(powf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_pow<T>", value, result);
 return result;
}

// p_fmod
template <>
SD_INLINE SD_HOST_DEVICE float16 p_fmod(float16 value, float16 power) {
 float16 result = static_cast<float16>(fmodf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_fmod<float16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_fmod(bfloat16 value, bfloat16 power) {
 bfloat16 result = static_cast<bfloat16>(fmodf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_fmod<bfloat16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_fmod(float value, float power) {
 float result = fmodf(value, power);
 SD_PRINT_MATH_FUNC("p_fmod<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_fmod(double value, double power) {
 double result = fmod(value, power);
 SD_PRINT_MATH_FUNC("p_fmod<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_fmod(T value, T power) {
 T result = static_cast<T>(fmodf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_fmod<T>", value, result);
 return result;
}

// p_atan2
template <>
SD_INLINE SD_HOST_DEVICE float16 p_atan2(float16 value, float16 power) {
 float16 result = static_cast<float16>(atan2f(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_atan2<float16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_atan2(float value, float power) {
 float result = atan2f(value, power);
 SD_PRINT_MATH_FUNC("p_atan2<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atan2(double value, double power) {
 double result = atan2(value, power);
 SD_PRINT_MATH_FUNC("p_atan2<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atan2(T value, T power) {
 T result = static_cast<T>(atan2f(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_atan2<T>", value, result);
 return result;
}

// p_remainder
template <>
SD_INLINE SD_HOST_DEVICE float16 p_remainder(float16 value, float16 power) {
 float16 result = static_cast<float16>(remainderf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_remainder<float16>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float p_remainder(float value, float power) {
 float result = remainderf(value, power);
 SD_PRINT_MATH_FUNC("p_remainder<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_remainder(double value, double power) {
 double result = remainder(value, power);
 SD_PRINT_MATH_FUNC("p_remainder<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_remainder(T value, T power) {
 T result = static_cast<T>(remainderf(static_cast<float>(value), static_cast<float>(power)));
 SD_PRINT_MATH_FUNC("p_remainder<T>", value, result);
 return result;
}

// p_log
template <>
SD_INLINE SD_HOST_DEVICE float p_log(float value) {
 if (value == 0.0f)
   value = SD_EPSILON;
 float result = logf(value);
 SD_PRINT_MATH_FUNC("p_log<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_log(float16 val) {
#ifdef SD_NATIVE_HALFS
 if ((float)val == 0.0f)
   val = static_cast<float16>(SD_EPSILON);
 float16 result = hlog(val.data);
#else
 if (val == 0.0f)
   val = static_cast<float16>(SD_EPSILON);
 float16 result = static_cast<float16>(logf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_log<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_log(bfloat16 val) {
 if (val == 0.0f)
   val = static_cast<bfloat16>(SD_EPSILON);
 bfloat16 result = static_cast<bfloat16>(logf((float)val));
 SD_PRINT_MATH_FUNC("p_log<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_log(double value) {
 if (value == 0.0)
   value = SD_EPSILON;
 double result = log(value);
 SD_PRINT_MATH_FUNC("p_log<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log(T value) {
 if (value == static_cast<T>(0.0f))
   value = static_cast<T>(SD_EPSILON);
 T result = static_cast<T>(logf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_log<T>", value, result);
 return result;
}

// p_log2
template <>
SD_INLINE SD_HOST_DEVICE float p_log2(float value) {
 if (value == 0.0f)
   value = SD_EPSILON;
 float result = log2f(value);
 SD_PRINT_MATH_FUNC("p_log2<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_log2(double value) {
 if (value == 0.0)
   value = SD_EPSILON;
 double result = log2(value);
 SD_PRINT_MATH_FUNC("p_log2<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_log2(T value) {
 if (value == static_cast<T>(0.0f))
   value = static_cast<T>(SD_EPSILON);
 T result = static_cast<T>(log2f(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_log2<T>", value, result);
 return result;
}

// p_floor
template <>
SD_INLINE SD_HOST_DEVICE float p_floor(float value) {
 float result = floorf(value);
 SD_PRINT_MATH_FUNC("p_floor<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_floor(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hfloor(val.data);
#else
 float16 result = static_cast<float16>(floorf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_floor<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_floor(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(floorf((float)val));
 SD_PRINT_MATH_FUNC("p_floor<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_floor(double value) {
 double result = floor(value);
 SD_PRINT_MATH_FUNC("p_floor<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_floor(T value) {
 SD_PRINT_MATH_FUNC("p_floor<T>", value, value);
 return value;
}

// p_ceil
template <>
SD_INLINE SD_HOST_DEVICE float p_ceil(float value) {
 float result = ceilf(value);
 SD_PRINT_MATH_FUNC("p_ceil<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_ceil(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hceil(val.data);
#else
 float16 result = static_cast<float16>(ceilf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_ceil<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_ceil(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(ceilf((float)val));
 SD_PRINT_MATH_FUNC("p_ceil<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_ceil(double value) {
 double result = ceil(value);
 SD_PRINT_MATH_FUNC("p_ceil<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_ceil(T value) {
 SD_PRINT_MATH_FUNC("p_ceil<T>", value, value);
 return value;
}

// p_round
template <>
SD_INLINE SD_HOST_DEVICE float p_round(float value) {
 float result = roundf(value);
 SD_PRINT_MATH_FUNC("p_round<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round(float16 val) {
 float16 result = static_cast<float16>(roundf((float)val));
 SD_PRINT_MATH_FUNC("p_round<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(roundf((float)val));
 SD_PRINT_MATH_FUNC("p_round<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round(double value) {
 double result = round(value);
 SD_PRINT_MATH_FUNC("p_round<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round(T value) {
 SD_PRINT_MATH_FUNC("p_round<T>", value, value);
 return value;
}

// p_round_prefer_ceil
template <>
SD_INLINE SD_HOST_DEVICE float p_round_prefer_ceil(float value) {
 float result = roundf(value);
 SD_PRINT_MATH_FUNC("p_round_prefer_ceil<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round_prefer_ceil(float16 val) {
 float16 result = static_cast<float16>(roundf((float)val));
 SD_PRINT_MATH_FUNC("p_round_prefer_ceil<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round_prefer_ceil(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(roundf((float)val));
 SD_PRINT_MATH_FUNC("p_round_prefer_ceil<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round_prefer_ceil(double value) {
 double result = round(value);
 SD_PRINT_MATH_FUNC("p_round_prefer_ceil<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round_prefer_ceil(T value) {
 SD_PRINT_MATH_FUNC("p_round_prefer_ceil<T>", value, value);
 return value;
}

// p_round_prefer_floor
template <>
SD_INLINE SD_HOST_DEVICE float p_round_prefer_floor(float value) {
 float result;
 if (value == static_cast<int64_t>(value) + 0.5f) {
   result = floorf(value);
 } else {
   result = roundf(value);
 }
 SD_PRINT_MATH_FUNC("p_round_prefer_floor<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_round_prefer_floor(float16 val) {
 float float_val = static_cast<float>(val);
 float16 result;
 if (float_val == static_cast<int64_t>(float_val) + 0.5f) {
   result = static_cast<float16>(floorf(float_val));
 } else {
   result = static_cast<float16>(roundf(float_val));
 }
 SD_PRINT_MATH_FUNC("p_round_prefer_floor<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_round_prefer_floor(bfloat16 val) {
 float float_val = static_cast<float>(val);
 bfloat16 result;
 if (float_val == static_cast<int64_t>(float_val) + 0.5f) {
   result = static_cast<bfloat16>(floorf(float_val));
 } else {
   result = static_cast<bfloat16>(roundf(float_val));
 }
 SD_PRINT_MATH_FUNC("p_round_prefer_floor<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_round_prefer_floor(double value) {
 double result;
 if (value == static_cast<int64_t>(value) + 0.5) {
   result = floor(value);
 } else {
   result = round(value);
 }
 SD_PRINT_MATH_FUNC("p_round_prefer_floor<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_round_prefer_floor(T value) {
 SD_PRINT_MATH_FUNC("p_round_prefer_floor<T>", value, value);
 return value;
}

// p_rint
template <>
SD_INLINE SD_HOST_DEVICE float p_rint(float value) {
 float result = rintf(value);
 SD_PRINT_MATH_FUNC("p_rint<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_rint(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hrint(val.data);
#else
 float16 result = static_cast<float16>(rintf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_rint<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_rint(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(rintf((float)val));
 SD_PRINT_MATH_FUNC("p_rint<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_rint(double value) {
 double result = rint(value);
 SD_PRINT_MATH_FUNC("p_rint<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rint(T value) {
 SD_PRINT_MATH_FUNC("p_rint<T>", value, value);
 return value;
}

// p_cos
template <>
SD_INLINE SD_HOST_DEVICE float p_cos(float value) {
 float result = cosf(value);
 SD_PRINT_MATH_FUNC("p_cos<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_cos(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hcos(val.data);
#else
 float16 result = static_cast<float16>(cosf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_cos<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_cos(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(cosf((float)val));
 SD_PRINT_MATH_FUNC("p_cos<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_cos(double value) {
 double result = cos(value);
 SD_PRINT_MATH_FUNC("p_cos<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_cos(T value) {
 T result = static_cast<T>(cosf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_cos<T>", value, result);
 return result;
}

// p_sin
template <>
SD_INLINE SD_HOST_DEVICE float p_sin(float value) {
 float result = sinf(value);
 SD_PRINT_MATH_FUNC("p_sin<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sin(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hsin(val.data);
#else
 float16 result = static_cast<float16>(sinf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_sin<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sin(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(sinf((float)val));
 SD_PRINT_MATH_FUNC("p_sin<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sin(double value) {
 double result = sin(value);
 SD_PRINT_MATH_FUNC("p_sin<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sin(T value) {
 T result = static_cast<T>(sinf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_sin<T>", value, result);
 return result;
}

// p_sqrt
template <>
SD_INLINE SD_HOST_DEVICE float p_sqrt(float value) {
 float result = sqrtf(value);
 SD_PRINT_MATH_FUNC("p_sqrt<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sqrt(float16 val) {
#ifdef SD_NATIVE_HALFS
 float16 result = hsqrt(val.data);
#else
 float16 result = static_cast<float16>(sqrtf((float)val));
#endif
 SD_PRINT_MATH_FUNC("p_sqrt<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sqrt(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(sqrtf((float)val));
 SD_PRINT_MATH_FUNC("p_sqrt<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sqrt(double value) {
 double result = sqrt(value);
 SD_PRINT_MATH_FUNC("p_sqrt<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sqrt(T value) {
 T result = static_cast<T>(sqrtf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_sqrt<T>", value, result);
 return result;
}

// p_tanh
template <>
SD_INLINE SD_HOST_DEVICE float p_tanh(float value) {
 float result = tanhf(value);
 SD_PRINT_MATH_FUNC("p_tanh<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_tanh(float16 val) {
 float16 result = static_cast<float16>(tanhf((float)val));
 SD_PRINT_MATH_FUNC("p_tanh<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_tanh(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(tanhf((float)val));
 SD_PRINT_MATH_FUNC("p_tanh<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_tanh(double value) {
 double result = tanh(value);
 SD_PRINT_MATH_FUNC("p_tanh<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_tanh(T value) {
 T result = static_cast<T>(tanhf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_tanh<T>", value, result);
 return result;
}

// p_erf
template <>
SD_INLINE SD_HOST_DEVICE float p_erf(float value) {
 float result = erff(value);
 SD_PRINT_MATH_FUNC("p_erf<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_erf(float16 val) {
 float16 result = static_cast<float16>(erff((float)val));
 SD_PRINT_MATH_FUNC("p_erf<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_erf(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(erff((float)val));
 SD_PRINT_MATH_FUNC("p_erf<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_erf(double value) {
 double result = erf(value);
 SD_PRINT_MATH_FUNC("p_erf<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_erf(T value) {
 T result = static_cast<T>(erff(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_erf<T>", value, result);
 return result;
}

// p_erfc
template <>
SD_INLINE SD_HOST_DEVICE float p_erfc(float value) {
 float result = erfcf(value);
 SD_PRINT_MATH_FUNC("p_erfc<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_erfc(float16 val) {
 float16 result = static_cast<float16>(erfcf((float)val));
 SD_PRINT_MATH_FUNC("p_erfc<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_erfc(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(erfcf((float)val));
 SD_PRINT_MATH_FUNC("p_erfc<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_erfc(double value) {
 double result = erfc(value);
 SD_PRINT_MATH_FUNC("p_erfc<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_erfc(T value) {
 T result = static_cast<T>(erfcf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_erfc<T>", value, result);
 return result;
}

// p_acos
template <>
SD_INLINE SD_HOST_DEVICE float p_acos(float value) {
 float result = acosf(value);
 SD_PRINT_MATH_FUNC("p_acos<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_acos(float16 val) {
 float16 result = static_cast<float16>(acosf((float)val));
 SD_PRINT_MATH_FUNC("p_acos<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_acos(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(acosf((float)val));
 SD_PRINT_MATH_FUNC("p_acos<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_acos(double value) {
 double result = acos(value);
 SD_PRINT_MATH_FUNC("p_acos<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_acos(T value) {
 T result = static_cast<T>(acosf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_acos<T>", value, result);
 return result;
}

// p_cosh
template <>
SD_INLINE SD_HOST_DEVICE float p_cosh(float value) {
 float result = coshf(value);
 SD_PRINT_MATH_FUNC("p_cosh<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_cosh(float16 val) {
 float16 result = static_cast<float16>(coshf((float)val));
 SD_PRINT_MATH_FUNC("p_cosh<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_cosh(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(coshf((float)val));
 SD_PRINT_MATH_FUNC("p_cosh<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_cosh(double value) {
 double result = cosh(value);
 SD_PRINT_MATH_FUNC("p_cosh<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_cosh(T value) {
 T result = static_cast<T>(coshf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_cosh<T>", value, result);
 return result;
}

// p_acosh
template <>
SD_INLINE SD_HOST_DEVICE float p_acosh(float value) {
 float result = acoshf(value);
 SD_PRINT_MATH_FUNC("p_acosh<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_acosh(float16 val) {
 float16 result = static_cast<float16>(acoshf((float)val));
 SD_PRINT_MATH_FUNC("p_acosh<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_acosh(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(acoshf((float)val));
 SD_PRINT_MATH_FUNC("p_acosh<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_acosh(double value) {
 double result = acosh(value);
 SD_PRINT_MATH_FUNC("p_acosh<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_acosh(T value) {
 T result = static_cast<T>(acoshf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_acosh<T>", value, result);
 return result;
}

// p_sinh
template <>
SD_INLINE SD_HOST_DEVICE float p_sinh(float value) {
 float result = sinhf(value);
 SD_PRINT_MATH_FUNC("p_sinh<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_sinh(float16 val) {
 float16 result = static_cast<float16>(sinhf((float)val));
 SD_PRINT_MATH_FUNC("p_sinh<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_sinh(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(sinhf((float)val));
 SD_PRINT_MATH_FUNC("p_sinh<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_sinh(double value) {
 double result = sinh(value);
 SD_PRINT_MATH_FUNC("p_sinh<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_sinh(T value) {
 T result = static_cast<T>(sinhf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_sinh<T>", value, result);
 return result;
}

// p_asin
template <>
SD_INLINE SD_HOST_DEVICE float p_asin(float value) {
 float result = asinf(value);
 SD_PRINT_MATH_FUNC("p_asin<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_asin(float16 val) {
 float16 result = static_cast<float16>(asinf((float)val));
 SD_PRINT_MATH_FUNC("p_asin<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_asin(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(asinf((float)val));
 SD_PRINT_MATH_FUNC("p_asin<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_asin(double value) {
 double result = asin(value);
 SD_PRINT_MATH_FUNC("p_asin<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_asin(T value) {
 T result = static_cast<T>(asinf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_asin<T>", value, result);
 return result;
}

// p_atan
template <>
SD_INLINE SD_HOST_DEVICE float p_atan(float value) {
 float result = atanf(value);
 SD_PRINT_MATH_FUNC("p_atan<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_atan(float16 val) {
 float16 result = static_cast<float16>(atanf((float)val));
 SD_PRINT_MATH_FUNC("p_atan<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_atan(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(atanf((float)val));
 SD_PRINT_MATH_FUNC("p_atan<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atan(double value) {
 double result = atan(value);
 SD_PRINT_MATH_FUNC("p_atan<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atan(T value) {
 T result = static_cast<T>(atanf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_atan<T>", value, result);
 return result;
}

// p_tan
template <>
SD_INLINE SD_HOST_DEVICE float p_tan(float value) {
 float result = tanf(value);
 SD_PRINT_MATH_FUNC("p_tan<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_tan(float16 val) {
 float16 result = static_cast<float16>(tanf((float)val));
 SD_PRINT_MATH_FUNC("p_tan<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_tan(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(tanf((float)val));
 SD_PRINT_MATH_FUNC("p_tan<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_tan(double value) {
 double result = tan(value);
 SD_PRINT_MATH_FUNC("p_tan<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_tan(T value) {
 T result = static_cast<T>(tanf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_tan<T>", value, result);
 return result;
}

// p_atanh
template <>
SD_INLINE SD_HOST_DEVICE float p_atanh(float value) {
 float result = atanhf(value);
 SD_PRINT_MATH_FUNC("p_atanh<float>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE float16 p_atanh(float16 val) {
 float16 result = static_cast<float16>(atanhf((float)val));
 SD_PRINT_MATH_FUNC("p_atanh<float16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE bfloat16 p_atanh(bfloat16 val) {
 bfloat16 result = static_cast<bfloat16>(atanhf((float)val));
 SD_PRINT_MATH_FUNC("p_atanh<bfloat16>", val, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE double p_atanh(double value) {
 double result = atanh(value);
 SD_PRINT_MATH_FUNC("p_atanh<double>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_atanh(T value) {
 T result = static_cast<T>(atanhf(static_cast<float>(value)));
 SD_PRINT_MATH_FUNC("p_atanh<T>", value, result);
 return result;
}

// Rotational functions
template <typename T>
SD_INLINE SD_HOST_DEVICE T _rotate_left(T value, T shift);

template <typename T>
SD_INLINE SD_HOST_DEVICE T _rotate_right(T value, T shift);

template <>
SD_INLINE SD_HOST_DEVICE int8_t _rotate_left(int8_t value, int8_t shift) {
 int8_t result = value << shift | value >> (8 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<int8_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE int8_t _rotate_right(int8_t value, int8_t shift) {
 int8_t result = value >> shift | value << (8 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<int8_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t _rotate_left(uint8_t value, uint8_t shift) {
 uint8_t result = value << shift | value >> (8 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<uint8_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint8_t _rotate_right(uint8_t value, uint8_t shift) {
 uint8_t result = value >> shift | value << (8 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<uint8_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t _rotate_left(int16_t value, int16_t shift) {
 int16_t result = value << shift | value >> (16 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<int16_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE int16_t _rotate_right(int16_t value, int16_t shift) {
 int16_t result = value >> shift | value << (16 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<int16_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t _rotate_left(uint16_t value, uint16_t shift) {
 uint16_t result = value << shift | value >> (16 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<uint16_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint16_t _rotate_right(uint16_t value, uint16_t shift) {
 uint16_t result = value >> shift | value << (16 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<uint16_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE int _rotate_left(int value, int shift) {
 int result = value << shift | value >> (32 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<int>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE int _rotate_right(int value, int shift) {
 int result = value >> shift | value << (32 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<int>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t _rotate_left(uint32_t value, uint32_t shift) {
 uint32_t result = value << shift | value >> (32 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<uint32_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint32_t _rotate_right(uint32_t value, uint32_t shift) {
 uint32_t result = value >> shift | value << (32 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<uint32_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType _rotate_left(sd::LongType value, sd::LongType shift) {
 sd::LongType result = value << shift | value >> (64 - shift);
 SD_PRINT_MATH_FUNC("_rotate_left<sd::LongType>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType _rotate_right(sd::LongType value, sd::LongType shift) {
 sd::LongType result = value >> shift | value << (64 - shift);
 SD_PRINT_MATH_FUNC("_rotate_right<sd::LongType>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t _rotate_left(uint64_t value, uint64_t shift) {
#ifdef SD_ARM_BUILD
 // TODO: eventually remove this once gcc fixes the bug
 sd::LongType val =
     _rotate_left<sd::LongType>(*reinterpret_cast<sd::LongType *>(&value), *reinterpret_cast<sd::LongType *>(&shift));
 uint64_t result = *reinterpret_cast<uint64_t *>(&val);
#else
 uint64_t result = value << shift | value >> (64 - shift);
#endif
 SD_PRINT_MATH_FUNC("_rotate_left<uint64_t>", value, result);
 return result;
}

template <>
SD_INLINE SD_HOST_DEVICE uint64_t _rotate_right(uint64_t value, uint64_t shift) {
#ifdef SD_ARM_BUILD
 // TODO: eventually remove this once gcc fixes the bug
 sd::LongType val =
     _rotate_right<sd::LongType>(*reinterpret_cast<sd::LongType *>(&value), *reinterpret_cast<sd::LongType *>(&shift));
 uint64_t result = *reinterpret_cast<uint64_t *>(&val);
#else
 uint64_t result = value >> shift | value << (64 - shift);
#endif
 SD_PRINT_MATH_FUNC("_rotate_right<uint64_t>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotl(T value, T shift) {
 T result = _rotate_left<T>(value, shift);
 SD_PRINT_MATH_FUNC("p_rotl<T>", value, result);
 return result;
}

template <typename T>
SD_INLINE SD_HOST_DEVICE T p_rotr(T value, T shift) {
 T result = _rotate_right<T>(value, shift);
 SD_PRINT_MATH_FUNC("p_rotr<T>", value, result);
 return result;
}

}  // namespace math
}  // namespace sd

#endif  // LIBND4J_PLATFORM_MATH_H
