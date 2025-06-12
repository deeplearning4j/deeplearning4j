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

#pragma once
#ifndef OP_MACROS_INDEX_REDUCE_H_
#define OP_MACROS_INDEX_REDUCE_H_

#include "op_types.h"

namespace simdOps {

// =============================================================================
// INDEX REDUCE OPERATION MACROS
// =============================================================================

/**
 * @brief Declares an index reduce operation with proper SIMD handling
 */
#define DECLARE_INDEX_REDUCE_OP(OP_NAME, STARTING_VAL, UPDATE_CONDITION, MERGE_CONDITION)       \
  template <typename X, typename Z>                                                             \
  class OP_NAME {                                                                               \
   public:                                                                                      \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(functions::indexreduce::IndexValue<X> val, \
                                                                          X* extraParams) {                         \
      return val;                                                                               \
    }                                                                                           \
    static SD_HOST_DEVICE inline X startingValue(const X* input) { return STARTING_VAL; }      \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> startingIndexValue(const X* input) {        \
      functions::indexreduce::IndexValue<X> local;                                             \
      local.value = startingValue(input);                                                      \
      local.index = 0;                                                                         \
      return local;                                                                             \
    }                                                                                           \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> update(                \
        functions::indexreduce::IndexValue<X>& old, functions::indexreduce::IndexValue<X>& opOutput, \
        X* extraParams) {                                                                      \
      if (UPDATE_CONDITION) return opOutput;                                                   \
      return old;                                                                               \
    }                                                                                           \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> merge(                 \
        functions::indexreduce::IndexValue<X> f1, functions::indexreduce::IndexValue<X> f2, X* extraParams) { \
      if (MERGE_CONDITION) return f2;                                                          \
      return f1;                                                                               \
    }                                                                                           \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> postProcess(           \
        functions::indexreduce::IndexValue<X> reduction, int n, int xOffset, X* dx, int incx, X* extraParams, \
        X* result) {                                                                           \
      return reduction;                                                                         \
    }                                                                                           \
    static SD_HOST_DEVICE inline functions::indexreduce::IndexValue<X> op(                    \
        functions::indexreduce::IndexValue<X> d1, functions::indexreduce::IndexValue<X> d2, X* extraParams) { \
      return d1;                                                                               \
    }                                                                                           \
  };

} // namespace simdOps

#endif // OP_MACROS_INDEX_REDUCE_H_