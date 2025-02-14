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
// Created by raver119 on 24.04.17.
//

#ifndef LIBND4J_SPECIALS_H
#define LIBND4J_SPECIALS_H

#ifdef __CUDACC__
#define ELEMENT_THRESHOLD 8192
#define TAD_THRESHOLD 2
#endif

#include <vector>


#include "legacy/NativeOps.h"


namespace sd {
class NDArray;

// FIXME: get rid of this redefinition
typedef union {
  float f_;
  int i_;
} FloatBits2;

class SD_LIB_EXPORT SpecialTypeConverter {
 public:
  template <typename S, typename T>
  static void convertGeneric(sd::Pointer *extras, void *dx, sd::LongType N, void *dz);
};

template <typename T>
class SD_LIB_EXPORT SpecialMethods {
 public:
  static void concatCpuGeneric(const std::vector<NDArray *> &inArrs, NDArray &output, const LongType axis);
  static void concatCpuGeneric(LongType dimension, int numArrays, NDArray **inArrs,
                               NDArray *result);
  static void splitCpuGeneric(NDArray&input, const std::vector<NDArray *> &outArrs, const LongType axis);

  static void quickSort_parallel_internal(NDArray *x, int left, int right, int cutoff,
                                          bool descending);
  static void quickSort_parallel(NDArray *x, int numThreads,
                                 bool descending);

  static void sortGeneric(NDArray *input, bool descending);
  static void sortTadGeneric(NDArray *input, sd::LongType  *dimension, int dimensionLength,bool descending);
};

template <typename X, typename Y>
class SD_LIB_EXPORT DoubleMethods {
 public:
  static void sortByKey(NDArray *x,NDArray *y,
                        bool descending);
  static void sortByValue(NDArray *x,NDArray *y,
                          bool descending);

  static void sortTadByKey(NDArray *x,NDArray *y,
                           NDArray *dimension, bool descending);
  static void sortTadByValue(NDArray *x, NDArray *y,
                             NDArray *dimension, bool descending);
};
}  // namespace sd

#endif  // LIBND4J_SPECIALS_H
