/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// Created by agibsonccc on 11/6/24.
//
#include <legacy/NativeOpExecutioner.h>

inline static void execSort(sd::NDArray *x, bool descending) {
  auto xType = x->dataType();

  BUILD_SINGLE_SELECTOR(xType, sd::SpecialMethods, ::sortGeneric(x, descending), SD_COMMON_TYPES);
}

static void execSort(sd::NDArray *x, sd::LongType *dimension,  sd::LongType dimensionLength,
                     bool descending) {
  auto xType = x->dataType();

  BUILD_SINGLE_SELECTOR(
      xType, sd::SpecialMethods,
      ::sortTadGeneric(x, dimension, dimensionLength, descending),
      SD_COMMON_TYPES);
}

inline static void execSortCooIndices(sd::LongType *indices, void *x, sd::LongType length,
                                      const sd::LongType *xShapeInfo) {
  auto xType = sd::ArrayOptions::dataType(xShapeInfo);
  int rank = shape::rank(xShapeInfo);

  BUILD_SINGLE_SELECTOR(xType, sd::sparse::SparseUtils, ::sortCooIndicesGeneric(indices, x, length, rank),
                        SD_COMMON_TYPES);
}

inline static void execRavelMultiIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                                       sd::LongType *shapeInfo, int mode) {
  sd::sparse::IndexUtils::ravelMultiIndex(indices, flatIndices, length, shapeInfo, mode);
}

inline static void execUnravelIndex(sd::LongType *indices, sd::LongType *flatIndices, sd::LongType length,
                                    sd::LongType *shapeInfo) {
  sd::sparse::IndexUtils::unravelIndex(indices, flatIndices, length, shapeInfo);
}

inline static sd::LongType encodeBitmap(sd::NDArray *x, sd::LongType N, long long int *dz,
                                        float threshold) {
  auto xType = x->dataType();

  BUILD_SINGLE_SELECTOR(xType, return sd::SpecialMethods, ::encodeBitmapGeneric(x, N, dz, threshold),
                        SD_FLOAT_TYPES);
}

inline static void decodeBitmap(sd::NDArray *dx, sd::LongType N, sd::NDArray *z) {
  auto zType = z->dataType();

  BUILD_SINGLE_SELECTOR(zType, sd::SpecialMethods, ::decodeBitmapGeneric(dx,z,N), SD_FLOAT_TYPES);
}