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
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_OPENMP_PRAGMAS_H
#define DEV_TESTS_OPENMP_PRAGMAS_H

#if defined(__NEC__)

#define OMP_STRINGIFY(args) #args
#define OMP_IF(args)
#define OMP_NUM_THREADS(args)
#define OMP_SCHEDULE(args) schedule(args)
#define OMP_MAXT max
#define OMP_SUMT +
#define OMP_PRODT *
#define OMP_REDUCTION(args) reduction(args)
#define OMP_COLLAPSE(loops)
#define PRAGMA_OMP_ATOMIC _Pragma(OMP_STRINGIFY(omp atomic))
#define PRAGMA_OMP_ATOMIC_ARGS(args) _Pragma(OMP_STRINGIFY(omp atomic args))
#define PRAGMA_OMP_CRITICAL _Pragma(OMP_STRINGIFY(omp critical))
#define PRAGMA_OMP_DECLARE_SIMD
#define PRAGMA_OMP_SIMD
#define PRAGMA_OMP_SIMD_ARGS(args)
#define PRAGMA_OMP_SIMD_SUM(args)
#define PRAGMA_OMP_SIMD_MAX(args)
#define PRAGMA_OMP_SIMD_MAX_2(args)
#define PRAGMA_OMP_PARALLEL _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel args default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS_IF(threads, condition) _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel for reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel for args default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_IF(args) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(loops) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel for args default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(threads, loops) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(loops) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel for reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_SECTIONS _Pragma(OMP_STRINGIFY(omp parallel sections))
#define PRAGMA_OMP_SECTION _Pragma(OMP_STRINGIFY(omp section))
#define PRAGMA_OMP_SINGLE _Pragma(OMP_STRINGIFY(omp single))
#define PRAGMA_OMP_SINGLE_ARGS(args) _Pragma(OMP_STRINGIFY(omp single args))
#define PRAGMA_OMP_TASK _Pragma(OMP_STRINGIFY(omp task))
#define PRAGMA_SUM_ENV(length, sum)

#elif defined(_MSC_VER)

#define OMP_STRINGIFY(args) #args
#define OMP_IF(args)
#define OMP_SCHEDULE(args)
#define OMP_MAXT
#define OMP_SUMT
#define OMP_REDUCTION(args)
#define PRAGMA_OMP_ATOMIC
#define PRAGMA_OMP_ATOMIC_ARGS(args)
#define PRAGMA_OMP_CRITICAL
#define PRAGMA_OMP_DECLARE_SIMD
#define PRAGMA_OMP_SIMD __pragma(omp simd)
#define PRAGMA_OMP_SIMD_ARGS(args)
#define PRAGMA_OMP_SIMD_SUM(args)
#define PRAGMA_OMP_SIMD_MAX(args)
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

#define OMP_STRINGIFY(args) #args
#define OMP_IF(args) if (args)
#define OMP_SCHEDULE(args) schedule(args)
#define OMP_MAXT maxT
#define OMP_SUMT sumT
#define OMP_REDUCTION(args) reduction(args)
#define PRAGMA_OMP_ATOMIC _Pragma(OMP_STRINGIFY(omp atomic))
#define PRAGMA_OMP_ATOMIC_ARGS(args) _Pragma(OMP_STRINGIFY(omp atomic args))
#define PRAGMA_OMP_CRITICAL _Pragma(OMP_STRINGIFY(omp critical))
#define PRAGMA_OMP_DECLARE_SIMD _Pragma("omp declare simd")
#define PRAGMA_OMP_SIMD _Pragma(OMP_STRINGIFY(omp simd))
#define PRAGMA_OMP_SIMD_ARGS(args) _Pragma(OMP_STRINGIFY(omp simd args))
#define PRAGMA_OMP_SIMD_SUM(args) _Pragma(OMP_STRINGIFY(omp simd reduction(sumT : args)))
#define PRAGMA_OMP_SIMD_MAX(args) _Pragma(OMP_STRINGIFY(omp simd reduction(maxTF : args)))
#define PRAGMA_OMP_SIMD_MAX_2(args) _Pragma(OMP_STRINGIFY(omp simd reduction(maxT : args)))
#define PRAGMA_OMP_PARALLEL _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel args default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS(args) \
  _Pragma(OMP_STRINGIFY(omp parallel num_threads(args) if (args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS_IF(threads, condition) \
  _Pragma(OMP_STRINGIFY(omp parallel num_threads(threads) if (condition) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR _Pragma(OMP_STRINGIFY(omp parallel for default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel for reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel for args default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_IF(args) _Pragma(OMP_STRINGIFY(omp parallel for if(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(loops) _Pragma(OMP_STRINGIFY(omp parallel for default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_FOR_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel for num_threads(args) if(args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD _Pragma(OMP_STRINGIFY(omp parallel for simd default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel for simd args default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(threads, loops) _Pragma(OMP_STRINGIFY(omp parallel for simd num_threads(threads) if(threads > 1) default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(loops) _Pragma(OMP_STRINGIFY(omp parallel for simd default(shared) collapse(loops)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel for simd reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel for simd num_threads(args) if(args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_SECTIONS _Pragma(OMP_STRINGIFY(omp parallel sections))
#define PRAGMA_OMP_SECTION _Pragma(OMP_STRINGIFY(omp section))
#define PRAGMA_OMP_SINGLE _Pragma(OMP_STRINGIFY(omp single))
#define PRAGMA_OMP_SINGLE_ARGS(args) _Pragma(OMP_STRINGIFY(omp single args))
#define PRAGMA_OMP_TASK _Pragma(OMP_STRINGIFY(omp task))
#define PRAGMA_SUM_ENV(length, sum)                                                                                \
  PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(length > Environment::getInstance().elementwiseThreshold()) schedule(guided) \
                                   reduction(OMP_SUMT                                                              \
                                             : sum))
#define PRAGMA_OMP_SIMD_REDUCE(reduce)
#endif

// reductions
#define FUNC_RL std::function<int64_t(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_AL std::function<sd::LongType(sd::LongType, sd::LongType)>

// aggregation functions
#define FUNC_RD std::function<double(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_AD std::function<double(double, double)>

// parallel block
#define FUNC_DO std::function<void(sd::LongType, sd::LongType)>

// parallel_for block
#define FUNC_1D std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_2D std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType)>
#define FUNC_3D \
  std::function<void(sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType, sd::LongType)>

// aggregation lambda
#define LAMBDA_AL [&](int64_t _old, int64_t _new) -> int64_t
#define LAMBDA_AD [&](double _old, double _new) -> double

#define LAMBDA_SUML \
  LAMBDA_AL { return _old + _new; }
#define LAMBDA_SUMD \
  LAMBDA_AD { return _old + _new; }

// reduction lambda
#define PRAGMA_REDUCE_LONG [&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) mutable -> int64_t
#define PRAGMA_REDUCE_DOUBLE [&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) mutable -> double

// paralllel block lambda
#define PRAGMA_THREADS_DO [&](sd::LongType thread_id, sd::LongType numThreads) -> void

// paralllel_for lambdas
#define PRAGMA_THREADS_FOR [&](sd::LongType thread_id, sd::LongType start, sd::LongType stop, sd::LongType increment) -> void
#define PRAGMA_THREADS_FOR_2D                                                                              \
  [&](sd::LongType thread_id, sd::LongType start_x, sd::LongType stop_x, sd::LongType inc_x, sd::LongType start_y, sd::LongType stop_y, \
      sd::LongType inc_y) -> void
#define PRAGMA_THREADS_FOR_3D                                                                              \
  [&](sd::LongType thread_id, sd::LongType start_x, sd::LongType stop_x, sd::LongType inc_x, sd::LongType start_y, sd::LongType stop_y, \
      sd::LongType inc_y, sd::LongType start_z, sd::LongType stop_z, sd::LongType inc_z) -> void

#endif  // DEV_TESTS_OPENMP_PRAGMAS_H
