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
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include <cstdint>

#include "../legacy_ops.h"
#include "../scalar.h"

using namespace simdOps;

namespace functions {
namespace scalar {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
template <typename OpType>
void ScalarTransform<X, Y, Z>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                         const sd::LongType *zShapeInfo, const void *vscalars, sd::LongType *dimension,
                                         sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                         const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                         const sd::LongType *zTadOffsets, sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalars = reinterpret_cast<const Y *>(vscalars);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

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
  for (auto r = start; r < stop; r++) {
    auto oZ = z + zTadOffsets[r];
    auto oX = x + xTadOffsets[r];
    PRAGMA_OMP_SIMD
    for (int f = 0; f < tadLength; f++) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType xOffset, zOffset;
      INDEX2COORDS(f, xTadRank, xTadShape, coords);
      INDEX2COORDS(f, zTadRank, zTadShape, coords);
      COORDS2INDEX(xTadRank, xTadStride, coords, xOffset);
      COORDS2INDEX(zTadRank, zTadStride, coords, zOffset);
      oZ[zOffset] = OpType::op(oX[xOffset], scalars[0], extraParams);
    }
  }
}

////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                         void *z, const sd::LongType *zShapeInfo, const void *scalars,
                                         sd::LongType *dimension,
                                         sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                         const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                         const sd::LongType *zTadOffsets,
                                         sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(transform,
                        PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength,
                               xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, start, stop),
                        SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                                         const sd::LongType *zShapeInfo, const void *scalar, void *extraParams,
                                         const sd::LongType start, const sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams, start, stop), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
template <typename OpType>
void ScalarTransform<X, Y, Z>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                         const sd::LongType *zShapeInfo, const void *vscalar,
                                         void *vextraParams,
                                         const sd::LongType start, const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalar = reinterpret_cast<const Y *>(vscalar);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);
  //need special handling for scalars as strides may not be set for scalars
  if(shape::length(xShapeInfo) <= 1 && shape::length(zShapeInfo) <= 1) {
    z[0] = OpType::op(x[0], scalar[0], extraParams);
    return;

  }
  // Cache shape-related values
  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType zRank = shape::rank(zShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *zShape = shape::shapeOf(zShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *zStride = shape::stride(zShapeInfo);

  PRAGMA_OMP_SIMD
  for (auto i = start; i < stop; i++) {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset, zOffset;
    INDEX2COORDS(i, xRank, xShape, coords);
    INDEX2COORDS(i, zRank, zShape, zCoords);
    COORDS2INDEX(xRank, xStride, coords, xOffset);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);
    z[zOffset] = OpType::op(x[xOffset], scalar[0], extraParams);
  };
}
////////////////////////////////////////////////////////////////////////

}  // namespace scalar
}  // namespace functions