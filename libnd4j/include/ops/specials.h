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
  static void concatCpuGeneric(const std::vector<const NDArray *> &inArrs, NDArray &output, const LongType axis);
  static void concatCpuGeneric(int dimension, int numArrays, sd::Pointer *data, sd::Pointer *inputShapeInfo,
                               void *result, sd::LongType const *resultShapeInfo);
  static void splitCpuGeneric(const NDArray &input, const std::vector<NDArray *> &outArrs, int axis);
  static void accumulateGeneric(void **x, void *z, const sd::LongType *zShapeInfo, int n, sd::LongType length);
  static void averageGeneric(void **x, void *z, const sd::LongType *zShapeInfo, int n, sd::LongType length,
                             bool propagate);

  static sd::LongType getPosition(const sd::LongType *xShapeInfo, sd::LongType index);
  static void quickSort_parallel_internal(T *array, const sd::LongType *xShapeInfo, int left, int right, int cutoff,
                                          bool descending);
  static void quickSort_parallel(void *array, const sd::LongType *xShapeInfo, sd::LongType lenArray, int numThreads,
                                 bool descending);

  static int nextPowerOf2(int number);
  static int lastPowerOf2(int number);

  static void sortGeneric(void *x, const sd::LongType *xShapeInfo, bool descending);
  static void sortTadGeneric(void *x, const sd::LongType *xShapeInfo, sd::LongType  *dimension, LongType dimensionLength,
                             const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets, bool descending);

  static void decodeBitmapGeneric(const void *dx, sd::LongType N, void *dz, const sd::LongType *zShapeInfo);
  static sd::LongType encodeBitmapGeneric(void *dx, const sd::LongType *zShapeInfo, sd::LongType N, sd::LongType  *dz,
                                          float threshold);
};

template <typename X, typename Y>
class SD_LIB_EXPORT DoubleMethods {
 public:
  static void sortByKey(void *vx, sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                        bool descending);
  static void sortByValue(void *vx, sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                          bool descending);

  static void sortTadByKey(void *vx, sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                           sd::LongType  *dimension, LongType dimensionLength, bool descending);
  static void sortTadByValue(void *vx, sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                             sd::LongType  *dimension, LongType dimensionLength, bool descending);
};
}  // namespace sd

#endif  // LIBND4J_SPECIALS_H
