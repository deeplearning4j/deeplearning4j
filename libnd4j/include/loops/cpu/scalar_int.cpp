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
// @author raver119@gmail.com
//
#include "../scalar_int.h"

#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace scalar {

template <typename X>
template <typename OpType>
void ScalarIntTransform<X>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                      const sd::LongType *zShapeInfo, const void *vscalars, sd::LongType *dimension,
                                      sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                      const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                      const sd::LongType *zTadOffsets, const sd::LongType start, const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
  auto scalars = reinterpret_cast<const X *>(vscalars);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = xTadShapeInfo;
    zTadOffsets = xTadOffsets;
  }

  // Cache shape-related values
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
      COORDS2INDEX(xTadRank, xTadStride, coords, xOffset);
      COORDS2INDEX(zTadRank, zTadStride, coords, zOffset);
      oZ[zOffset] = OpType::op(oX[xOffset], scalars[r], extraParams);
    }
  }
}

template <typename X>
void ScalarIntTransform<X>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                      void *z, const sd::LongType *zShapeInfo, const void *scalars,
                                      sd::LongType *dimension,
                                      sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                      const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                      const sd::LongType *zTadOffsets, const sd::LongType start, const sd::LongType stop) {
  DISPATCH_BY_OPNUM_T(transform,
                      PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength,
                             xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, start, stop),
                      SCALAR_INT_OPS);
}

template <typename X>
void ScalarIntTransform<X>::transform(const int opNum, const void *x, sd::LongType xEws, void *z, sd::LongType zEws,
                                      const void *scalar, void *extraParams, const sd::LongType n, const sd::LongType start,
                                      const sd::LongType stop) {
  DISPATCH_BY_OPNUM_T(transform, PARAMS(x, xEws, z, zEws, scalar, extraParams, n, start, stop), SCALAR_INT_OPS);
}

template <typename X>
void ScalarIntTransform<X>::transform(const int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                                      const sd::LongType *zShapeInfo, const void *scalar, void *extraParams,
                                      const sd::LongType start, const sd::LongType stop) {
  DISPATCH_BY_OPNUM_T(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams, start, stop),
                      SCALAR_INT_OPS);
}

template <typename X>
template <typename OpType>
void ScalarIntTransform<X>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                      const sd::LongType *zShapeInfo, const void *vscalar, void *vextraParams,
                                      const sd::LongType start, const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
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
    for (auto i3 = start; i3 < stop; i3++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(i3, xRank, xShape, coords);
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

template <typename X>
template <typename OpType>
void ScalarIntTransform<X>::transform(const void *vx, sd::LongType xEws, void *vz, sd::LongType zEws,
                                      const void *vscalar, void *vextraParams, const sd::LongType len, const sd::LongType start,
                                      const sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<X *>(vz);
  auto scalar = reinterpret_cast<const X *>(vscalar)[0];
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  if (scalar < static_cast<X>((sizeof(X) * 8))) {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) {
      auto xi = i * xEws;
      auto zi = i * zEws;
      z[zi] = OpType::op(x[xi], scalar, extraParams);
    }
  }
}

BUILD_SINGLE_TEMPLATE(template class ScalarIntTransform, , SD_INTEGER_TYPES);

}  // namespace scalar
}  // namespace functions