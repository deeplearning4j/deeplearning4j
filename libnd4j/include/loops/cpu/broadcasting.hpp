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
void Broadcast<X, Y, Z>::exec(const void* vx, const sd::LongType* xShapeInfo,
                              const void* vy, const sd::LongType* yShapeInfo,
                              void* vz, const sd::LongType* zShapeInfo,
                              sd::LongType* dimension, sd::LongType dimensionLength,
                              const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffset,
                              const sd::LongType* zTadShapeInfo, const sd::LongType* zTadOffset,
                              sd::LoopKind::Kind loopKind, sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X*>(vx);
  auto y = reinterpret_cast<const Y*>(vy);
  auto z = reinterpret_cast<Z*>(vz);

  if (loopKind == sd::LoopKind::BROADCAST_SCALAR_X) {
    sd::LongType tadLength = shape::length(xTadShapeInfo);
    for (auto i = start; i < stop; i++) {
      auto oY = y + (i * tadLength);
      auto oZ = z + (i * tadLength);
      const auto oX = x[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++)
        oZ[f] = OpType::op(oX, oY[f]);
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_SCALAR_Y) {
    sd::LongType tadLength = shape::length(xTadShapeInfo);
    for (auto i = start; i < stop; i++) {
      auto oX = x + (i * tadLength);
      auto oZ = z + (i * tadLength);
      const auto oY = y[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++)
        oZ[f] = OpType::op(oX[f], oY);
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_2D) {
    const sd::LongType nSize1 = shape::sizeAt(zShapeInfo, 1);
    const sd::LongType* xStrides = shape::stride(xTadShapeInfo);
    const sd::LongType* yStrides = shape::stride(yShapeInfo);
    const sd::LongType* zStrides = shape::stride(zTadShapeInfo);

    for (auto i0 = start; i0 < stop; i0++) {
      auto baseX = x + xTadOffset[i0];
      auto baseZ = z + zTadOffset[i0];

      PRAGMA_OMP_SIMD
      for (sd::LongType i1 = 0; i1 < nSize1; i1++) {
        auto rX = baseX + xStrides[1] * i1;
        auto rY = y + yStrides[1] * i1;
        auto rZ = baseZ + zStrides[1] * i1;

        *rZ = OpType::op(*rX, *rY);
      }
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_3D) {
    const sd::LongType nSize1 = shape::sizeAt(zShapeInfo, 1);
    const sd::LongType nSize2 = shape::sizeAt(zShapeInfo, 2);

    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* yStrides = shape::stride(yShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    for (auto i0 = start; i0 < stop; i0++) {
      PRAGMA_OMP_SIMD
      for (sd::LongType i1 = 0; i1 < nSize1; i1++) {
        for (sd::LongType i2 = 0; i2 < nSize2; i2++) {
          auto rX = x + (xStrides[0] * i0 + xStrides[1] * i1 + xStrides[2] * i2);
          auto rY = y + (yStrides[0] * i0 + yStrides[1] * i1 + yStrides[2] * i2);
          auto rZ = z + (zStrides[0] * i0 + zStrides[1] * i1 + zStrides[2] * i2);

          *rZ = OpType::op(*rX, *rY);
        }
      }
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_4D) {
    const sd::LongType nSize1 = shape::sizeAt(zShapeInfo, 1);
    const sd::LongType nSize2 = shape::sizeAt(zShapeInfo, 2);
    const sd::LongType nSize3 = shape::sizeAt(zShapeInfo, 3);

    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* yStrides = shape::stride(yShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    for (auto i = start; i < stop; i++) {
      uint64_t i0 = i / nSize1;
      uint64_t i1 = i % nSize1;

      PRAGMA_OMP_SIMD
      for (sd::LongType i2 = 0; i2 < nSize2; i2++) {
        for (sd::LongType i3 = 0; i3 < nSize3; i3++) {
          auto rX = x + (xStrides[0] * i0 + xStrides[1] * i1 +
                         xStrides[2] * i2 + xStrides[3] * i3);
          auto rY = y + (yStrides[0] * i0 + yStrides[1] * i1 +
                         yStrides[2] * i2 + yStrides[3] * i3);
          auto rZ = z + (zStrides[0] * i0 + zStrides[1] * i1 +
                         zStrides[2] * i2 + zStrides[3] * i3);

          *rZ = OpType::op(*rX, *rY);
        }
      }
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_5D) {
    const sd::LongType nSize1 = shape::sizeAt(zShapeInfo, 1);
    const sd::LongType nSize2 = shape::sizeAt(zShapeInfo, 2);
    const sd::LongType nSize3 = shape::sizeAt(zShapeInfo, 3);
    const sd::LongType nSize4 = shape::sizeAt(zShapeInfo, 4);

    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* yStrides = shape::stride(yShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    for (auto i = start; i < stop; i++) {
      uint32_t i0 = i / nSize1;
      uint32_t i1 = i % nSize1;

      PRAGMA_OMP_SIMD
      for (sd::LongType i2 = 0; i2 < nSize2; i2++) {
        for (sd::LongType i3 = 0; i3 < nSize3; i3++) {
          for (sd::LongType i4 = 0; i4 < nSize4; i4++) {
            auto rX = x + (xStrides[0] * i0 + xStrides[1] * i1 +
                           xStrides[2] * i2 + xStrides[3] * i3 + xStrides[4] * i4);
            auto rY = y + (yStrides[0] * i0 + yStrides[1] * i1 +
                           yStrides[2] * i2 + yStrides[3] * i3 + yStrides[4] * i4);
            auto rZ = z + (zStrides[0] * i0 + zStrides[1] * i1 +
                           zStrides[2] * i2 + zStrides[3] * i3 + zStrides[4] * i4);

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
    }
  }
  else {
    // Default case for other ranks
    const int xRank = shape::rank(xShapeInfo);
    const int yRank = shape::rank(yShapeInfo);
    const int zRank = shape::rank(zShapeInfo);

    const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
    const sd::LongType* yShape = shape::shapeOf(yShapeInfo);
    const sd::LongType* zShape = shape::shapeOf(zShapeInfo);
    const sd::LongType* xStrides = shape::stride(xShapeInfo);
    const sd::LongType* yStrides = shape::stride(yShapeInfo);
    const sd::LongType* zStrides = shape::stride(zShapeInfo);

    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType yCoords[SD_MAX_RANK];
    sd::LongType zCoords[SD_MAX_RANK];

    for (auto i = start; i < stop; i++) {
      // Calculate independent coordinates for each array
      INDEX2COORDS(i, xRank, xShape, xCoords);
      INDEX2COORDS(i, yRank, yShape, yCoords);
      INDEX2COORDS(i, zRank, zShape, zCoords);

      // Calculate offsets based on each array's coordinates and strides
      sd::LongType xOffset, yOffset, zOffset;
      COORDS2INDEX(xRank, xStrides, xCoords, xOffset);
      COORDS2INDEX(yRank, yStrides, yCoords, yOffset);
      COORDS2INDEX(zRank, zStrides, zCoords, zOffset);

      z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
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

  // Handle TAD setup
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

  // Get shape information
  const auto xRank = shape::rank(xShapeInfo);
  const auto yTadRank = shape::rank(yTadShapeShapeInfo);
  const auto zTadRank = shape::rank(zTadShapeInfo);

  const auto xStrides = shape::stride(xShapeInfo);
  const auto yTadStrides = shape::stride(yTadShapeShapeInfo);
  const auto zTadStrides = shape::stride(zTadShapeInfo);

  const auto xShape = shape::shapeOf(xShapeInfo);
  const auto yTadShape = shape::shapeOf(yTadShapeShapeInfo);
  const auto zTadShape = shape::shapeOf(zTadShapeInfo);

  const sd::LongType tadLength = shape::length(yTadShapeShapeInfo);

  if (yTadRank <= 3) {
    // Optimized path for lower ranks
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      if (yTadRank == 1) {
        PRAGMA_OMP_SIMD
        for (sd::LongType j = 0; j < tadLength; j++) {
          oZ[j * zTadStrides[0]] = OpType::op(x[j * xStrides[0]], oY[j * yTadStrides[0]]);
        }
      }
      else if (yTadRank == 2) {
        const sd::LongType dim0 = yTadShape[0];
        const sd::LongType dim1 = yTadShape[1];

        for (sd::LongType j0 = 0; j0 < dim0; j0++) {
          PRAGMA_OMP_SIMD
          for (sd::LongType j1 = 0; j1 < dim1; j1++) {
            const auto xOffset = j0 * xStrides[0] + j1 * xStrides[1];
            const auto yOffset = j0 * yTadStrides[0] + j1 * yTadStrides[1];
            const auto zOffset = j0 * zTadStrides[0] + j1 * zTadStrides[1];
            oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
          }
        }
      }
      else { // rank 3
        const sd::LongType dim0 = yTadShape[0];
        const sd::LongType dim1 = yTadShape[1];
        const sd::LongType dim2 = yTadShape[2];

        for (sd::LongType j0 = 0; j0 < dim0; j0++) {
          for (sd::LongType j1 = 0; j1 < dim1; j1++) {
            PRAGMA_OMP_SIMD
            for (sd::LongType j2 = 0; j2 < dim2; j2++) {
              const auto xOffset = j0 * xStrides[0] + j1 * xStrides[1] + j2 * xStrides[2];
              const auto yOffset = j0 * yTadStrides[0] + j1 * yTadStrides[1] + j2 * yTadStrides[2];
              const auto zOffset = j0 * zTadStrides[0] + j1 * zTadStrides[1] + j2 * zTadStrides[2];
              oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
            }
          }
        }
      }
    }
  }
  else {
    // Use macros for higher ranks
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, yTadRank, yTadShape, coords);

        sd::LongType xOffset, yOffset, zOffset;
        COORDS2INDEX(xRank, xStrides, coords, xOffset);
        COORDS2INDEX(yTadRank, yTadStrides, coords, yOffset);
        COORDS2INDEX(zTadRank, zTadStrides, coords, zOffset);

        oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
      }
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