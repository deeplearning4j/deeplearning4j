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
// Created by raver119 on 08.10.2017.
//
#include "../scalar_bool.h"

#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace scalar {

template <typename X, typename Z>
template <typename OpType>
void ScalarBoolTransform<X, Z>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                          const sd::LongType *zShapeInfo, const void *vscalars,
                                          sd::LongType *dimension,
                                          sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                          const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                          const sd::LongType *zTadOffsets, const sd::LongType start, const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalars = reinterpret_cast<const X *>(vscalars);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = xTadShapeInfo;
    zTadOffsets = xTadOffsets;
  }

  // Cache shape-related values for TAD operations
  sd::LongType xTadRank = shape::rank(xTadShapeInfo);
  sd::LongType zTadRank = shape::rank(zTadShapeInfo);
  sd::LongType *xTadShape = shape::shapeOf(xTadShapeInfo);
  sd::LongType *zTadShape = shape::shapeOf(zTadShapeInfo);
  sd::LongType *xTadStride = shape::stride(xTadShapeInfo);
  sd::LongType *zTadStride = shape::stride(zTadShapeInfo);

  const int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
  const int numTads = shape::length(xShapeInfo) / tadLength;
  int num_threads = sd::math::sd_min<int>(numTads, sd::Environment::getInstance().maxThreads());

  for (auto r = start; r < stop; r++) {
    auto oZ = z + zTadOffsets[r];
    auto oX = x + xTadOffsets[r];

    PRAGMA_OMP_SIMD
    for (int f = 0; f < tadLength; f++) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType xOffset, zOffset;
      INDEX2COORDS(f, xTadRank, xTadShape, coords);
      COORDS2INDEX(xTadRank, xTadStride, coords, xOffset);
      COORDS2INDEX(zTadRank, zTadStride, coords, zOffset);
      oZ[zOffset] = OpType::op(oX[xOffset], scalars[r], extraParams);
    }
  }
}

template <typename X, typename Y>
void ScalarBoolTransform<X, Y>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                          void *z, const sd::LongType *zShapeInfo, const void *scalars,
                                          sd::LongType *dimension,
                                          sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                          const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                          const sd::LongType *zTadOffsets, const sd::LongType start, const sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(transform,
                       PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength,
                              xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, start, stop),
                       SCALAR_BOOL_OPS);
}

template <typename X, typename Y>
void ScalarBoolTransform<X, Y>::transform(const int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                                          const sd::LongType *zShapeInfo, const void *scalar, void *extraParams,
                                          const sd::LongType start, const sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams, start, stop),
                       SCALAR_BOOL_OPS);
}

template <typename X, typename Z>
template <typename OpType>
void ScalarBoolTransform<X, Z>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                          const sd::LongType *zShapeInfo, const void *vscalar, void *vextraParams,
                                          const sd::LongType start, const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalar = reinterpret_cast<const X *>(vscalar)[0];
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  // Cache shape-related values
  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType zRank = shape::rank(zShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *zShape = shape::shapeOf(zShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *zStride = shape::stride(zShapeInfo);

  if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
    PRAGMA_OMP_SIMD
    for (auto i2 = start; i2 < stop; i2++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(i2, xRank, xShape, coords);
      sd::LongType offset;
      COORDS2INDEX(xRank, xStride, coords, offset);
      z[offset] = OpType::op(x[offset], scalar, extraParams);
    };
  } else {
    PRAGMA_OMP_SIMD
    for (auto i2 = start; i2 < stop; i2++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(i2, xRank, xShape, coords);
      sd::LongType xOffset, zOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      COORDS2INDEX(zRank, zStride, coords, zOffset);
      z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
    };
  }
}

BUILD_DOUBLE_TEMPLATE( class ScalarBoolTransform, , SD_COMMON_TYPES, SD_BOOL_TYPES);

}  // namespace scalar
}  // namespace functions