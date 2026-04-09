/* ******************************************************************************
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

#ifndef LIBND4J_OMP_REDUCTIONS_H
#define LIBND4J_OMP_REDUCTIONS_H

// OpenMP custom reduction declarations for float16/bfloat16.
// This header MUST be included AFTER float16.h and bfloat16.h are fully defined.
// It is included from types.h at the end, after all type headers.

#if defined(_OPENMP) && !defined(_MSC_VER)

#if defined(HAS_FLOAT16)
#include <types/float16.h>
#pragma omp declare reduction(+: float16 : omp_out += omp_in) \
  initializer(omp_priv = float16(0.0f))
#endif

#if defined(HAS_BFLOAT16)
#include <types/bfloat16.h>
#pragma omp declare reduction(+: bfloat16 : omp_out += omp_in) \
  initializer(omp_priv = bfloat16(0.0f))
#endif

#endif  // defined(_OPENMP) && !defined(_MSC_VER)

#endif  // LIBND4J_OMP_REDUCTIONS_H
