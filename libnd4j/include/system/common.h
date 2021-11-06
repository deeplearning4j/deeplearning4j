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

#include <system/openmp_pragmas.h>
#include <stdint.h>
#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

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
#define SD_INLINE __attribute__((always_inline)) inline
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

#ifdef __CUDACC__

#define SD_META_DEF SD_INLINE SD_HOST
#define SD_OP_DEF SD_INLINE SD_HOST_DEVICE

#elif __JAVACPP_HACK__

#define SD_META_DEF
#define SD_OP_DEF

#else

#define SD_META_DEF PRAGMA_OMP_DECLARE_SIMD SD_INLINE
#define SD_OP_DEF PRAGMA_OMP_DECLARE_SIMD SD_INLINE

#endif

#define SD_MIN_V 1e-12
#define SD_MAX_FLOAT 1e37
#define SD_MIN_FLOAT 1e-37
#define SD_MAX_INT 2147483647
#define SD_MIN_CUTFOFF -3.79297773665f
#define SD_FLOAT_MIN_NORMAL 1.17549435e-38
#define SD_EPSILON 1e-5
#define SD_DOUBLE_PI_T T(2.0 * 3.14159265358979323846)
#define SD_DOUBLE_PI_X X(2.0 * 3.14159265358979323846)

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
        TRUE = 100,
        FALSE = 101,
        MAYBE = 119
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

#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

static SD_INLINE sd::LongType microTime() {
#ifdef WIN32
    LARGE_INTEGER freq, count;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&count);
  return (sd::LongType)count.QuadPart / freq.QuadPart;
#else
    timeval tv;
    gettimeofday(&tv, NULL);
    return (sd::LongType)tv.tv_sec * 1000000 + tv.tv_usec;
#endif
}

#endif
