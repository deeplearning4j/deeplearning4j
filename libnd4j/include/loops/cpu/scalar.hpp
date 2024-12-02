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

  const int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
  for (auto r = start; r < stop; r++) {
    auto oZ = z + zTadOffsets[r];
    auto oX = x + xTadOffsets[r];
    PRAGMA_OMP_SIMD
    for (int f = 0; f < tadLength; f++) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType xOffset, zOffset;
      INDEX2COORDS(f, shape::rank(xTadShapeInfo), shape::shapeOf(xTadShapeInfo), coords);
      INDEX2COORDS(f, shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), coords);
      COORDS2INDEX(shape::rank(xTadShapeInfo), shape::stride(xTadShapeInfo), coords, xOffset);
      COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), coords, zOffset);
      oZ[zOffset] = OpType::op(oX[xOffset], scalars[r], extraParams);
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
  auto scalar = reinterpret_cast<const Y *>(vscalar)[0];
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  PRAGMA_OMP_SIMD
  for (auto i = start; i < stop; i++) {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];
    sd::LongType xOffset, zOffset;
    INDEX2COORDS(i, shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), coords);
    INDEX2COORDS(i, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), zCoords);
    COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
    COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), zCoords, zOffset);
    z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
  };

}
////////////////////////////////////////////////////////////////////////



}  // namespace scalar
}  // namespace functions