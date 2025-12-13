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

#ifndef DEV_TESTS_OPENMP_PRAGMAS_H
#define DEV_TESTS_OPENMP_PRAGMAS_H

#include <cfloat>
#include <climits>
#include <cstdint>
#include <limits>
#include <system/type_boilerplate.h>  // For sd::LongType and other types

// Add compiler-specific SIMD handling for problematic types
#if defined(__GNUC__) && !defined(__clang__)
// GCC-specific: Disable SIMD warnings for unsupported types
#define PRAGMA_OMP_DECLARE_SIMD_SAFE \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wattributes\"") \
    _Pragma("omp declare simd") \
    _Pragma("GCC diagnostic pop")
#elif defined(__clang__)
// Clang-specific: Similar approach
#define PRAGMA_OMP_DECLARE_SIMD_SAFE \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wignored-attributes\"") \
    _Pragma("omp declare simd") \
    _Pragma("clang diagnostic pop")
#elif defined(_MSC_VER)
// MSVC: No SIMD pragma, just inline
#define PRAGMA_OMP_DECLARE_SIMD_SAFE
#else
// Default fallback
#define PRAGMA_OMP_DECLARE_SIMD_SAFE _Pragma("omp declare simd")
#endif


#if defined(_MSC_VER)

#define OMP_STRINGIFY_HELPER(args) #args
#define OMP_STRINGIFY(args) OMP_STRINGIFY_HELPER(args)
#define OMP_IF(args)
#define OMP_SCHEDULE(args)
#define PRAGMA_OMP_ATOMIC
#define PRAGMA_OMP_ATOMIC_ARGS(args)
#define PRAGMA_OMP_CRITICAL
#define PRAGMA_OMP_DECLARE_SIMD
#define PRAGMA_OMP_SIMD __pragma(omp simd)
#define PRAGMA_OMP_SIMD_ARGS(args)
#define PRAGMA_OMP_DECLARE_REDUCTION(OP_NAME, TYPE, OPERATOR, INIT)
#define PRAGMA_OMP_PARALLEL
#define PRAGMA_OMP_PARALLEL_REDUCTION(args)
#define PRAGMA_OMP_PARALLEL_ARGS(args)
#define PRAGMA_OMP_PARALLEL_THREADS(args)
#define PRAGMA_OMP_PARALLEL_THREADS_IF(threads, condition)
#define PRAGMA_OMP_PARALLEL_FOR
#define PRAGMA_OMP_PARALLEL_FOR_ARGS(args)
#define PRAGMA_OMP_PARALLEL_FOR_IF(args)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(loops)
#define PRAGMA_OMP_PARALLEL_FOR_REDUCTION(args)
#define PRAGMA_OMP_PARALLEL_FOR_THREADS(args)
#define PRAGMA_OMP_PARALLEL_FOR_SIMD
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(args)
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(loops)
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(args)
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(args)
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(threads, loops)
#define PRAGMA_OMP_PARALLEL_SECTIONS
#define PRAGMA_OMP_SECTION
#define PRAGMA_OMP_SINGLE
#define PRAGMA_OMP_SINGLE_ARGS(args)
#define PRAGMA_OMP_TASK

#else

#define OMP_STRINGIFY_HELPER(args) #args
#define OMP_STRINGIFY(args) OMP_STRINGIFY_HELPER(args)
#define OMP_IF(args) if (args)
#define OMP_SCHEDULE(args) schedule(args)

#define PRAGMA_OMP_ATOMIC _Pragma(OMP_STRINGIFY(omp atomic))
#define PRAGMA_OMP_ATOMIC_ARGS(args) _Pragma(OMP_STRINGIFY(omp atomic args))
#define PRAGMA_OMP_CRITICAL _Pragma(OMP_STRINGIFY(omp critical))
#define PRAGMA_OMP_DECLARE_SIMD _Pragma("omp declare simd")
#define PRAGMA_OMP_SIMD _Pragma(OMP_STRINGIFY(omp simd))
#define PRAGMA_OMP_SIMD_ARGS(args) _Pragma(OMP_STRINGIFY(omp simd args))

#define PRAGMA_OMP_DECLARE_REDUCTION(OP_NAME, TYPE, OPERATOR, INIT) \
 _Pragma(OMP_STRINGIFY(omp declare reduction(OP_NAME : TYPE : omp_out OPERATOR omp_in) \
                       initializer(omp_priv = INIT)))

#define PRAGMA_OMP_PARALLEL _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_REDUCTION(...) \
 _Pragma(OMP_STRINGIFY(omp parallel reduction(__VA_ARGS__) default(shared)))
#define PRAGMA_OMP_PARALLEL_ARGS(...) \
 _Pragma(OMP_STRINGIFY(omp parallel __VA_ARGS__ default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS(args) \
 _Pragma(OMP_STRINGIFY(omp parallel num_threads(args) if (args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS_IF(threads, condition) \
 _Pragma(OMP_STRINGIFY(omp parallel num_threads(threads) if (condition) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR \
 _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_REDUCTION(...) \
 _Pragma(OMP_STRINGIFY(omp parallel for reduction(__VA_ARGS__) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_ARGS(...) \
 _Pragma(OMP_STRINGIFY(omp parallel for __VA_ARGS__ default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_IF(args) \
 _Pragma(OMP_STRINGIFY(omp parallel for if(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(loops) \
 _Pragma(OMP_STRINGIFY(omp parallel for default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_FOR_THREADS(args) \
 _Pragma(OMP_STRINGIFY(omp parallel for num_threads(args) if(args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD \
 _Pragma(OMP_STRINGIFY(omp parallel for simd default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(...) \
 _Pragma(OMP_STRINGIFY(omp parallel for simd __VA_ARGS__ default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(loops) \
 _Pragma(OMP_STRINGIFY(omp parallel for simd default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(...) \
 _Pragma(OMP_STRINGIFY(omp parallel for simd reduction(__VA_ARGS__) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(args) \
 _Pragma(OMP_STRINGIFY(omp parallel for simd num_threads(args) if(args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(threads, loops) \
 _Pragma(OMP_STRINGIFY(omp parallel for simd num_threads(threads) if(threads > 1) default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_SECTIONS \
 _Pragma(OMP_STRINGIFY(omp parallel sections))
#define PRAGMA_OMP_SECTION \
 _Pragma(OMP_STRINGIFY(omp section))
#define PRAGMA_OMP_SINGLE \
 _Pragma(OMP_STRINGIFY(omp single))
#define PRAGMA_OMP_SINGLE_ARGS(...) \
 _Pragma(OMP_STRINGIFY(omp single __VA_ARGS__))
#define PRAGMA_OMP_TASK \
 _Pragma(OMP_STRINGIFY(omp task))

#endif

// Reduction function templates
#define FUNC_RL std::function<int64_t(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_AL std::function<sd::LongType(sd::LongType, sd::LongType)>

// Aggregation functions
#define FUNC_RD std::function<double(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_AD std::function<double(double, double)>

// Parallel block
#define FUNC_DO std::function<void(sd::LongType, sd::LongType)>

// Parallel_for block
#define FUNC_1D std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_2D \
std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType, \
                   sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_3D \
std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType, \
                   sd::LongType, sd::LongType, sd::LongType, sd::LongType, \
                   sd::LongType, sd::LongType)>

// Aggregation lambda
#define LAMBDA_AL [&](sd::LongType _old, sd::LongType _new) -> sd::LongType
#define LAMBDA_AD [&](double _old, double _new) -> double

#define LAMBDA_SUML \
LAMBDA_AL { return _old + _new; }
#define LAMBDA_SUMD \
LAMBDA_AD { return _old + _new; }

// Reduction lambda
#define PRAGMA_REDUCE_LONG \
[&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) \
    mutable -> sd::LongType
#define PRAGMA_REDUCE_DOUBLE \
[&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) \
    mutable -> double

// Parallel block lambda
#define PRAGMA_THREADS_DO \
[&](sd::LongType thread_id, sd::LongType numThreads) -> void

// Parallel_for lambdas
#define PRAGMA_THREADS_FOR \
[&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) \
    -> void
#define PRAGMA_THREADS_FOR_2D \
[&](sd::LongType thread_id, sd::LongType start_x, sd::LongType stop_x, sd::LongType inc_x, \
    sd::LongType start_y, sd::LongType stop_y, sd::LongType inc_y) -> void
#define PRAGMA_THREADS_FOR_3D \
[&](sd::LongType thread_id, sd::LongType start_x, sd::LongType stop_x, sd::LongType inc_x, \
    sd::LongType start_y, sd::LongType stop_y, sd::LongType inc_y, \
    sd::LongType start_z, sd::LongType stop_z, sd::LongType inc_z) -> void

// OpenMP reduction declarations

#if !defined(_MSC_VER)

// Declare custom reductions using the '+' operator for custom types

#if defined(HAS_FLOAT16)
#pragma omp declare reduction(+: float16 : omp_out += omp_in) \
  initializer(omp_priv = float16(0.0f))
#endif

#if defined(HAS_BFLOAT16)
#pragma omp declare reduction(+: bfloat16 : omp_out += omp_in) \
  initializer(omp_priv = bfloat16(0.0f))
#endif

// For other custom types, declare similar reductions using the '+' operator

#endif  // !defined(_MSC_VER)

#endif  // DEV_TESTS_OPENMP_PRAGMAS_H
