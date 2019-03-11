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


#define OMP_STRINGIFY(args) #args
#define OMP_IF(args) if(args)
#define OMP_SCHEDULE(args) schedule(args)
#define OMP_MAXT maxT
#define OMP_SUMT sumT
#define OMP_REDUCTION(args) reduction(args)
#define PRAGMA_OMP_CRITICAL _Pragma(OMP_STRINGIFY(omp critical))
#define PRAGMA_OMP_SIMD(args) _Pragma(OMP_STRINGIFY(omp simd args))
#define PRAGMA_OMP_PARALLEL(args) _Pragma(OMP_STRINGIFY(omp parallel args))
#define PRAGMA_OMP_PARALLEL_FOR(args) _Pragma(OMP_STRINGIFY(omp parallel for args))
#define PRAGMA_OMP_PARALLEL_FOR_SIMD(args) _Pragma(OMP_STRINGIFY(omp parallel for simd args))


#endif //DEV_TESTS_OPENMP_PRAGMAS_H
