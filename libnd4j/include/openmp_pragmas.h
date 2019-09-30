/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#if defined(_MSC_VER)

#define OMP_STRINGIFY(args)
#define OMP_IF(args)
#define OMP_SCHEDULE(args)
#define OMP_MAXT
#define OMP_SUMT
#define OMP_REDUCTION(args)
#define PRAGMA_OMP_ATOMIC
#define PRAGMA_OMP_ATOMIC_ARGS(args)
#define PRAGMA_OMP_CRITICAL
#define PRAGMA_OMP_SIMD
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
#define OMP_IF(args) if(args)
#define OMP_SCHEDULE(args) schedule(args)
#define OMP_MAXT maxT
#define OMP_SUMT sumT
#define OMP_REDUCTION(args) reduction(args)
#define PRAGMA_OMP_ATOMIC _Pragma(OMP_STRINGIFY(omp atomic))
#define PRAGMA_OMP_ATOMIC_ARGS(args) _Pragma(OMP_STRINGIFY(omp atomic args))
#define PRAGMA_OMP_CRITICAL _Pragma(OMP_STRINGIFY(omp critical))
#define PRAGMA_OMP_SIMD _Pragma(OMP_STRINGIFY(omp simd))
#define PRAGMA_OMP_SIMD_ARGS(args) _Pragma(OMP_STRINGIFY(omp simd args))
#define PRAGMA_OMP_SIMD_SUM(args) _Pragma(OMP_STRINGIFY(omp simd reduction(sumT:args)))
#define PRAGMA_OMP_SIMD_MAX(args) _Pragma(OMP_STRINGIFY(omp simd reduction(maxTF:args)))
#define PRAGMA_OMP_PARALLEL _Pragma(OMP_STRINGIFY(omp parallel default(shared)))
#define PRAGMA_OMP_PARALLEL_REDUCTION(args) _Pragma(OMP_STRINGIFY(omp parallel reduction(args) default(shared)))
#define PRAGMA_OMP_PARALLEL_ARGS(args) _Pragma(OMP_STRINGIFY(omp parallel args default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS(args) _Pragma(OMP_STRINGIFY(omp parallel num_threads(args) if(args > 1) default(shared)))
#define PRAGMA_OMP_PARALLEL_THREADS_IF(threads, condition) _Pragma(OMP_STRINGIFY(omp parallel num_threads(threads) if(condition) default(shared)))
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

#endif

#define FUNC_DO std::function<void(uint64_t)>
#define FUNC_1D std::function<void(uint64_t, uint64_t, uint64_t, uint64_t)>
#define FUNC_2D std::function<void(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)>
#define FUNC_3D std::function<void(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)>

#define PRAGMA_THREADS_DO  [&](uint64_t thread_id) -> void
#define PRAGMA_THREADS_FOR  [&](uint64_t thread_id, uint64_t start, uint64_t stop, uint64_t increment) -> void
#define PRAGMA_THREADS_FOR_2D [&](uint64_t thread_id, uint64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y) -> void
#define PRAGMA_THREADS_FOR_3D [&](uint64_t thread_id, int64_t start_x, uint64_t stop_x, uint64_t inc_x, uint64_t start_y, uint64_t stop_y, uint64_t inc_y, uint64_t start_z, uint64_t stop_z, uint64_t inc_z) -> void

#endif //DEV_TESTS_OPENMP_PRAGMAS_H
