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
//  @author raver119@gmail.com
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/LoopKind.h>
#include <helpers/ShapeUtils.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <cstdio>

using namespace simdOps;

namespace functions {
namespace broadcast {

template <typename X, typename Y, typename Z>
void Broadcast<X, Y, Z>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                     const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                     sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                     const sd::LongType *xTadOffset, const sd::LongType *zTadShapeInfo,
                                     const sd::LongType *zTadOffset, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(execInverse,
                        PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, xTadShapeInfo,
                               xTadOffset, zTadShapeInfo, zTadOffset, start, stop),
                        BROADCAST_OPS);
}

template <typename X, typename Y, typename Z>
void Broadcast<X, Y, Z>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                              const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                              sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                              const sd::LongType *xTadOffset,
                              const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset,
                              sd::LoopKind::Kind loopKind, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(exec,
                        PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, xTadShapeInfo,
                               xTadOffset, zTadShapeInfo, zTadOffset, loopKind, start, stop),
                        BROADCAST_OPS);
}

template <typename X, typename Y, typename Z>
template <typename OpType>
void Broadcast<X, Y, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                              const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                              sd::LongType *dimension,
                              sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                              const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset,
                              sd::LoopKind::Kind loopKind, sd::LongType start,
                              sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Z *>(vz);

  auto xTadShapeShapeInfo = xTadShapeInfo;
  auto tadOffsets = xTadOffset;

  if (xTadShapeInfo == nullptr || tadOffsets == nullptr) {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);
    xTadShapeShapeInfo = tadPack->primaryShapeInfo();
    tadOffsets = tadPack->primaryOffsets();
  }

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = xTadShapeShapeInfo;
    zTadOffset = tadOffsets;
  }

  // Cache all shape-related values
  sd::LongType xTadRank = shape::rank(xTadShapeShapeInfo);
  sd::LongType yRank = shape::rank(yShapeInfo);
  sd::LongType zTadRank = shape::rank(zTadShapeInfo);
  sd::LongType *xTadShape = shape::shapeOf(xTadShapeShapeInfo);
  sd::LongType *yShape = shape::shapeOf(yShapeInfo);
  sd::LongType *zTadShape = shape::shapeOf(zTadShapeInfo);
  sd::LongType *xTadStride = shape::stride(xTadShapeShapeInfo);
  sd::LongType *yStride = shape::stride(yShapeInfo);
  sd::LongType *zTadStride = shape::stride(zTadShapeInfo);

  sd::LongType tadLength = shape::length(xTadShapeShapeInfo);

  // Case 5
  for (auto i = start; i < stop; i++) {
    auto oZ = z + zTadOffset[i];
    auto oX = x + tadOffsets[i];
    PRAGMA_OMP_SIMD
    for (sd::LongType f = 0; f < tadLength; f++) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType zCoords[SD_MAX_RANK];

      INDEX2COORDS(f, xTadRank, xTadShape, coords);
      INDEX2COORDS(f, yRank, yShape, yCoords);
      INDEX2COORDS(f, zTadRank, zTadShape, zCoords);

      sd::LongType xOffset, yOffset, zOffset;
      COORDS2INDEX(xTadRank, xTadStride, coords, xOffset);
      COORDS2INDEX(yRank, yStride, yCoords, yOffset);
      COORDS2INDEX(zTadRank, zTadStride, zCoords, zOffset);

      oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
    }
  }
}

template <typename X, typename Y, typename Z>
template <typename OpType>
void Broadcast<X, Y, Z>::execInverse(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                     const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                     sd::LongType *dimension, sd::LongType dimensionLength,
                                     const sd::LongType *yTadShapeInfo, const sd::LongType *yTadOffset,
                                     const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset,
                                     sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Z *>(vz);

  auto yTadShapeShapeInfo = yTadShapeInfo;
  auto tadOffsets = yTadOffset;

  if (yTadShapeInfo == nullptr || tadOffsets == nullptr) {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(yShapeInfo, dimension, dimensionLength);
    yTadShapeShapeInfo = tadPack->primaryShapeInfo();
    tadOffsets = tadPack->primaryOffsets();
  }

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = yTadShapeShapeInfo;
    zTadOffset = tadOffsets;
  }

  // Cache all shape-related values
  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType yTadRank = shape::rank(yTadShapeShapeInfo);
  sd::LongType zTadRank = shape::rank(zTadShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *yTadStride = shape::stride(yTadShapeShapeInfo);
  sd::LongType *zTadStride = shape::stride(zTadShapeInfo);
  sd::LongType *zTadShape = shape::shapeOf(zTadShapeInfo);

  sd::LongType tadLength = shape::length(yTadShapeShapeInfo);

  for (auto i = start; i < stop; i++) {
    auto oZ = z + zTadOffset[i];
    auto oY = y + tadOffsets[i];

    PRAGMA_OMP_SIMD
    for (sd::LongType f = 0; f < tadLength; f++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(f, zTadRank, zTadShape, coords);

      sd::LongType xOffset, yOffset, zOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      COORDS2INDEX(yTadRank, yTadStride, coords, yOffset);
      COORDS2INDEX(zTadRank, zTadStride, coords, zOffset);

      oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
    }
  }
}

template <typename X, typename Y, typename Z>
void Broadcast<X, Y, Z>::exec(const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                              const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo) {
  DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo), BROADCAST_OPS);
}

template <typename X, typename Y, typename Z, typename OpType>
static void execDefault(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                        const sd::LongType *zShapeInfo) {
  // Cache shape-related values
  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType yRank = shape::rank(yShapeInfo);
  sd::LongType zRank = shape::rank(zShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *yShape = shape::shapeOf(yShapeInfo);
  sd::LongType *zShape = shape::shapeOf(zShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *yStride = shape::stride(yShapeInfo);
  sd::LongType *zStride = shape::stride(zShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; ++i) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType zCoords[SD_MAX_RANK];

      INDEX2COORDS(i, xRank, xShape, coords);
      INDEX2COORDS(i, yRank, yShape, yCoords);
      INDEX2COORDS(i, zRank, zShape, zCoords);

      sd::LongType xOffset, yOffset, zOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      COORDS2INDEX(yRank, yStride, yCoords, yOffset);
      COORDS2INDEX(zRank, zStride, zCoords, zOffset);

      z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), shape::length(zShapeInfo));
}

template <typename X, typename Y, typename Z>
template <typename OpType>
void Broadcast<X, Y, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                              const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const X *x = reinterpret_cast<const X *>(vx);
  const Y *y = reinterpret_cast<const Y *>(vy);
  Z *z = reinterpret_cast<Z *>(vz);

  const int rank = shape::rank(zShapeInfo);  // xRank = yRank = zRank

  switch (rank) {
    default:
      execDefault<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  }
}

}  // namespace broadcast
}  // namespace functions