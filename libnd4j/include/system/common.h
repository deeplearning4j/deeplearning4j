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

#ifndef SD_SYSTEM_COMMON_H
#define SD_SYSTEM_COMMON_H

#include <cstdint>

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)
#define COMMA ,

#if defined(_MSC_VER)

#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4251)
#pragma warning(disable : 4101)
#pragma warning(disable : 4305)
#pragma warning(disable : 4309)
#pragma warning(disable : 4333)
#pragma warning(disable : 4146)
#pragma warning(disable : 4018)
// we're ignoring warning about non-exportable parent class, since std::runtime_error is a part of Standard C++ Library
#pragma warning(disable : 4275)
#pragma warning(disable : 4297)

#endif



#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define SD_LIB_EXPORT __attribute__((dllexport))
#else
#define SD_LIB_EXPORT __declspec(dllexport)
#endif
#define SD_LIB_HIDDEN
#else
#if __GNUC__ >= 4
#define SD_LIB_EXPORT __attribute__((visibility("default")))
#define SD_LIB_HIDDEN __attribute__((visibility("hidden")))
#else
#define SD_LIB_EXPORT
#define SD_LIB_HIDDEN
#endif
#endif

// Cross-platform compiler attributes
#if defined(__GNUC__)
#define SD_NO_INSTRUMENT __attribute__((no_instrument_function))
#elif defined(_MSC_VER)
#define SD_NO_INSTRUMENT __declspec(noinline)
#else
#define SD_NO_INSTRUMENT
#endif

#ifdef __clang__
#include <unordered_map>
#define SD_MAP_IMPL std::unordered_map
#define SD_LOOPS_INLINED
#define SD_INLINE inline
#elif _MSC_VER
#include <map>
#define SD_MAP_IMPL std::map
#define SD_INLINE __forceinline
#elif __GNUC__
#include <unordered_map>
#define SD_MAP_IMPL std::unordered_map
#define SD_LOOPS_INLINED
#define SD_INLINE  inline
#elif __CUDACC__
#include <unordered_map>
#define SD_MAP_IMPL std::unordered_map
#define SD_INLINE __forceinline__ inline
#else
#include <unordered_map>
#define SD_MAP_IMPL std::unordered_map
#define SD_INLINE inline
#endif

#ifdef __CUDACC__

#define SD_HOST __host__
#define SD_DEVICE __device__
#define SD_KERNEL __global__
#define SD_HOST_DEVICE __host__ __device__

#else

#define SD_HOST
#define SD_DEVICE
#define SD_KERNEL
#define SD_HOST_DEVICE

#endif  // CUDACC

#if defined(_ISOC11_SOURCE) && defined(__AVX2__)
#define SD_DESIRED_ALIGNMENT 32
#define SD_ALIGNED_ALLOC 1
#endif

#if defined(__GNUC__)
#define SD_ALIGN32 __attribute__((aligned(32)))
#elif defined(_MSC_VER)
#define SD_ALIGN32 __declspec(align(32))
#else
#define SD_ALIGN32
#endif

#ifdef __CUDACC__
// 610 is for tests only
// 600 is Tesla P100
// 530 is Tegra
#if __CUDA_ARCH__ == 600 || __CUDA_ARCH__ == 530 || __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 720 || __CUDA_ARCH__ == 750
#define NATIVE_HALFS
#endif

#endif

// Include openmp_pragmas.h AFTER SD_INLINE is defined
#include <system/openmp_pragmas.h>

#ifdef __CUDACC__

#define SD_META_DEF SD_INLINE SD_HOST
#define SD_OP_DEF SD_INLINE SD_HOST_DEVICE

#elif __JAVACPP_HACK__

#define SD_META_DEF
#define SD_OP_DEF

#else

// Proper fix: Use SIMD but handle the float16/bfloat16 issue with template specializations
#define SD_META_DEF PRAGMA_OMP_DECLARE_SIMD SD_INLINE
#define SD_OP_DEF PRAGMA_OMP_DECLARE_SIMD SD_INLINE

// Alternative macro for problematic template instantiations
#define SD_OP_DEF_NO_SIMD SD_INLINE

#endif

#define SD_MIN_V 1e-12
#define SD_MAX_FLOAT 1e37
#define SD_MIN_FLOAT 1e-37
#define SD_MAX_INT 2147483647
#define SD_MIN_CUTFOFF -3.79297773665f
#define SD_MAX_CUTFOFF 3.79297773665f
#define SD_FLOAT_MIN_NORMAL 1.17549435e-38
#define SD_EPSILON 1e-5
#define SD_DOUBLE_PI_T T(2.0 * 3.14159265358979323846)
#define SD_DOUBLE_PI_X X(2.0 * 3.14159265358979323846)

#include <cstdarg>
#include <cstdio>
#include <string>

namespace sd {

using Pointer = void*;
using LongType = long long;
using UnsignedLong = uint64_t;
using Unsigned = unsigned int;



enum class Status : int {
  OK = 0,
  BAD_INPUT = 1,
  BAD_SHAPE = 2,
  BAD_RANK = 3,
  BAD_PARAMS = 4,
  BAD_OUTPUT = 5,
  BAD_RNG = 6,
  BAD_EPSILON = 7,
  BAD_GRADIENTS = 8,
  BAD_BIAS = 9,
  VALIDATION = 20,
  BAD_GRAPH = 30,
  BAD_LENGTH = 31,
  BAD_DIMENSIONS = 32,
  BAD_ORDER = 33,
  BAD_ARGUMENTS = 34,
  DOUBLE_WRITE = 40,
  DOUBLE_READ = 45,
  KERNEL_FAILURE = 50,
  EQ_TRUE = 100,
  EQ_FALSE = 101,
  MAYBE = 119
};
struct ErrorResult {
  sd::Status status;
  std::string message;
};

}  // namespace sd

#define SD_MAX_DIMENSION 0x7fffffff
#define SD_MAX_NUM_THREADS 1024
#define SD_MAX_RANK 32
#define SD_MAX_SHAPEINFOLENGTH 2 * SD_MAX_RANK + 4
#define SD_MAX_COORD 3
#define SD_PREALLOC_SIZE 33554432

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#define SD_CUDA_BLOCK_SIZE 256

#if !defined(_OPENMP)
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#define omp_set_num_threads(threads)
#else
#include <omp.h>
#endif


#ifndef __JAVACPP_HACK__

#if defined(SD_GCC_FUNCTRACE) && !defined(OP_BOILER_PLATE_THROW_EXCEPTIONS)
#define OP_BOILER_PLATE_THROW_EXCEPTIONS
#include <exceptions/backward.hpp>
using namespace backward;
void throwException(const char* exceptionMessage);
#else
void throwException(const char* exceptionMessage);

#endif
#define THROW_EXCEPTION(exceptionMessage) throwException(exceptionMessage);
#endif

#define CONCAT2(A, B) A##B
#define CONCAT3_IMPL(a, b, c) a##b##c
#define CONCAT3(a, b, c) CONCAT3_IMPL(a, b, c)


#define MIX2(A, B) A##_##B
#define MIX3(A, B, C) A##_##B##_##C
#define MIX4(A, B, C, D) A##_##B##_##C##_##D

#define EMPTY()
#define DEFER(id) id EMPTY()

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT4(a, b, c, d) a##b##c##d
#define CONCAT5_IMPL(a, b, c, d, e) a##b##c##d##e
#define CONCAT5(a, b, c, d, e) CONCAT5_IMPL(a, b, c, d, e)
#define CONCAT6_IMPL(a, b, c, d, e, f) a##b##c##d##e##f
#define CONCAT6(a, b, c, d, e, f) CONCAT6_IMPL(a, b, c, d, e, f)
#define CONCAT7_IMPL(a, b, c, d, e, f, g) a##b##c##d##e##f##g
#define CONCAT7(a, b, c, d, e, f, g) CONCAT7_IMPL(a, b, c, d, e, f, g)
#define CONCAT8_IMPL(a, b, c, d, e, f, g, h) a##b##c##d##e##f##g##h
#define CONCAT8(a, b, c, d, e, f, g, h) CONCAT8_IMPL(a, b, c, d, e, f, g, h)

#define UNDERSCORE _
#define COMMA_MATH ,

#define EXPAND(...) __VA_ARGS__
#define EXPAND2(...) __VA_ARGS__
#define EXPAND3(...) __VA_ARGS__
#define EXTRACT(...) EXTRACT __VA_ARGS__
#define NOTHING_EXTRACT
#define PASTE(x, ...) x##__VA_ARGS__
#define PASTE2(x, ...) x##__VA_ARGS__
#define PASTE3(x, ...) x##__VA_ARGS__
#define EVALUATING_PASTE(x, ...) PASTE(x, __VA_ARGS__)
#define EVALUATING_PASTE2(x, ...) PASTE2(x, __VA_ARGS__)
#define EVALUATING_PASTE3(x, ...) PASTE3(x, __VA_ARGS__)
#define UNPAREN(x) EVALUATING_PASTE(NOTHING_, EXTRACT x)
#define UNPAREN2(x) EVALUATING_PASTE2(NOTHING_, EXTRACT x)
#define UNPAREN3(x) EVALUATING_PASTE3(NOTHING_, EXTRACT x)
#define EVAL(...) EVAL0(__VA_ARGS__)
#define EVALX(x) x
#define EVAL0(...) EVAL1(EVAL1(EVAL1(__VA_ARGS__)))
#define EVAL1(...) EVAL2(EVAL2(EVAL2(__VA_ARGS__)))
#define EVAL2(...) EVAL3(EVAL3(EVAL3(__VA_ARGS__)))
#define EVAL3(...) EVAL4(EVAL4(EVAL4(__VA_ARGS__)))
#define EVAL4(...) EVAL5(EVAL5(EVAL5(__VA_ARGS__)))
#define EVAL5(...) __VA_ARGS__


#endif