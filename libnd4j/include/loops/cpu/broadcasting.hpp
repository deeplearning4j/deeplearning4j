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

  // Get rank information
  const int xRank = shape::rank(xShapeInfo);
  const int yRank = shape::rank(yShapeInfo);
  const int zRank = shape::rank(zShapeInfo);
  const int xTadRank = xTadShapeInfo ? shape::rank(xTadShapeInfo) : xRank;
  const int zTadRank = zTadShapeInfo ? shape::rank(zTadShapeInfo) : zRank;

  // Get shape information
  const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  const sd::LongType* yShape = shape::shapeOf(yShapeInfo);
  const sd::LongType* zShape = shape::shapeOf(zShapeInfo);
  const sd::LongType* xTadShape = shape::shapeOf(xTadShapeInfo);
  const sd::LongType* zTadShape = shape::shapeOf(zTadShapeInfo);

  // Get stride information
  const sd::LongType* xStrides = shape::stride(xShapeInfo);
  const sd::LongType* yStrides = shape::stride(yShapeInfo);
  const sd::LongType* zStrides = shape::stride(zShapeInfo);
  const sd::LongType* xTadStrides = shape::stride(xTadShapeInfo);
  const sd::LongType* zTadStrides = shape::stride(zTadShapeInfo);

  // Classify array types
  // For X array or X TAD
  bool isXScalar = xTadRank == 0 || (xTadRank == 1 && xTadShape[0] == 1);
  bool isXVector = (xTadRank == 1 && xTadShape[0] > 1) ||
                   (xTadRank == 2 && (xTadShape[0] == 1 || xTadShape[1] == 1));
  bool isXRowVector = (xTadRank == 1 && xTadShape[0] > 1) ||
                      (xTadRank == 2 && xTadShape[0] == 1 && xTadShape[1] > 1);
  bool isXColumnVector = (xTadRank == 2 && xTadShape[0] > 1 && xTadShape[1] == 1);

  // For Y array
  bool isYScalar = yRank == 0 || (yRank == 1 && yShape[0] == 1);
  bool isYVector = (yRank == 1 && yShape[0] > 1) ||
                   (yRank == 2 && (yShape[0] == 1 || yShape[1] == 1));
  bool isYRowVector = (yRank == 1 && yShape[0] > 1) ||
                      (yRank == 2 && yShape[0] == 1 && yShape[1] > 1);
  bool isYColumnVector = (yRank == 2 && yShape[0] > 1 && yShape[1] == 1);

  // For Z array or Z TAD
  bool isZScalar = zTadRank == 0 || (zTadRank == 1 && zTadShape[0] == 1);
  bool isZVector = (zTadRank == 1 && zTadShape[0] > 1) ||
                   (zTadRank == 2 && (zTadShape[0] == 1 || zTadShape[1] == 1));
  bool isZRowVector = (zTadRank == 1 && zTadShape[0] > 1) ||
                      (zTadRank == 2 && zTadShape[0] == 1 && zTadShape[1] > 1);
  bool isZColumnVector = (zTadRank == 2 && zTadShape[0] > 1 && zTadShape[1] == 1);

  // Handle scalar broadcasting as a special case first
  if (isYScalar) {
    // Scalar broadcast - apply same value to every element
    const Y scalarY = y[0];
    sd::LongType length = shape::length(xTadShapeInfo ? xTadShapeInfo : xShapeInfo);

    if (xTadShapeInfo && zTadShapeInfo) {
      // TAD case
      for (auto i = start; i < stop; i++) {
        auto oX = x + xTadOffset[i];
        auto oZ = z + zTadOffset[i];

        // Handle different X and Z shapes
        if (isXVector && isZVector) {
          sd::LongType len = shape::length(xTadShapeInfo);
          PRAGMA_OMP_SIMD
          for (sd::LongType f = 0; f < len; f++) {
            sd::LongType xOffset = f * xTadStrides[xTadRank-1];
            sd::LongType zOffset = f * zTadStrides[zTadRank-1];
            oZ[zOffset] = OpType::op(oX[xOffset], scalarY);
          }
        } else {
          // General case
          for (sd::LongType f = 0; f < length; f++) {
            // Calculate proper offsets for current position
            sd::LongType xCoord[SD_MAX_RANK], zCoord[SD_MAX_RANK];
            sd::LongType xOffset, zOffset;

            INDEX2COORDS(f, xTadRank, xTadShape, xCoord);
            INDEX2COORDS(f, zTadRank, zTadShape, zCoord);

            COORDS2INDEX(xTadRank, xTadStrides, xCoord, xOffset);
            COORDS2INDEX(zTadRank, zTadStrides, zCoord, zOffset);

            oZ[zOffset] = OpType::op(oX[xOffset], scalarY);
          }
        }
      }
    } else {
      // Non-TAD case
      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < length; f++)
        z[f] = OpType::op(x[f], scalarY);
    }
  }
    // Handle 2D broadcasting
  else if (loopKind == sd::LoopKind::BROADCAST_2D) {
    // Determine shapes for broadcasting
    sd::LongType nRows = zTadShape[0];
    sd::LongType nCols = zTadRank > 1 ? zTadShape[1] : shape::length(zTadShapeInfo);

    // Special vector broadcasting cases
    if (isYVector && (isXRowVector || isXColumnVector || isXVector)) {
      // Vector to vector broadcasting
      if (isYRowVector && (isXRowVector || isXVector)) {
        // Row vector to row vector
        for (auto i = start; i < stop; i++) {
          auto baseX = x + xTadOffset[i];
          auto baseZ = z + zTadOffset[i];

          sd::LongType xStride = xTadRank > 1 ? xTadStrides[xTadRank - 1] : xTadStrides[0];
          sd::LongType yStride = yRank == 1 ? yStrides[0] : yStrides[1];
          sd::LongType zStride = zTadRank ? zTadStrides[zTadRank - 1] : zTadStrides[0];


          PRAGMA_OMP_SIMD
          for (sd::LongType i1 = 0; i1 < nCols; i1++) {
            auto rX = baseX + i1 * xStride;
            auto rY = y + i1 * yStride;
            auto rZ = baseZ + i1 * zStride;

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
      else if (isYColumnVector && (isXColumnVector || isXVector)) {
        // Column vector to column vector
        // Row vector to row vector
        for (auto i = start; i < stop; i++) {
          auto baseX = x + (xTadOffset ? xTadOffset[i] : 0);
          auto baseZ = z + (zTadOffset ? zTadOffset[i] : 0);

          // Calculate correct strides based on shape and rank
          sd::LongType xStride;
          if (xTadRank == 1) {
            xStride = xTadStrides[0];
          } else { // xTadRank == 2
            xStride = xTadStrides[0]; // For 2D column vector, use row stride
          }

          sd::LongType yStride;
          if (yRank == 1) {
            yStride = yStrides[0];
          } else { // yRank == 2
            yStride = yStrides[0]; // For 2D column vector, use row stride
          }

          sd::LongType zStride;
          if (zTadRank == 1) {
            zStride = zTadStrides[0];
          } else { // zTadRank == 2
            zStride = zTadStrides[0]; // For 2D column vector, use row stride
          }

          // Verify dimensions match
          sd::LongType xLen = isXColumnVector ? (xTadRank == 2 ? xTadShape[0] : xTadShape[0]) : xTadShape[0];
          sd::LongType yLen = yRank == 2 ? yShape[0] : yShape[0];
          printf("xLen: %lld; yLen: %lld nRows %lld,xStride %lld,yStride %lld, zStride %lld\n", xLen, yLen,nRows,xStride,yStride,zStride);
          PRAGMA_OMP_SIMD
          for (sd::LongType i1 = 0; i1 < nRows; i1++) {
            auto rX = baseX + i1 * xStride;
            auto rY = y + i1 * yStride;
            auto rZ = baseZ + i1 * zStride;

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
      else if (isYColumnVector && isXRowVector) {
        // Column vector to row vector (outer product)
        for (auto i = start; i < stop; i++) {
          auto baseX = x + (xTadOffset ? xTadOffset[i] : 0);
          auto baseZ = z + (zTadOffset ? zTadOffset[i] : 0);

          for (sd::LongType i0 = 0; i0 < nRows; i0++) {
            auto colValue = y[i0 * yStrides[0]];

            PRAGMA_OMP_SIMD
            for (sd::LongType i1 = 0; i1 < nCols; i1++) {
              auto rX = baseX + i1 * xTadStrides[xTadRank-1];
              auto rZ = baseZ + i0 * zTadStrides[0] + i1 * zTadStrides[1];

              *rZ = OpType::op(*rX, colValue);
            }
          }
        }
      }
      else if (isYRowVector && isXColumnVector) {
        // Row vector to column vector (outer product)
        for (auto i = start; i < stop; i++) {
          printf("4 2d tad: %lld\n", i);
          fflush(stdout);
          auto baseX = x + (xTadOffset ? xTadOffset[i] : 0);
          auto baseZ = z + (zTadOffset ? zTadOffset[i] : 0);

          for (sd::LongType i0 = 0; i0 < nRows; i0++) {
            auto xValue = baseX[i0 * xTadStrides[0]];

            PRAGMA_OMP_SIMD
            for (sd::LongType i1 = 0; i1 < nCols; i1++) {
              auto rY = y + i1 * (yRank == 1 ? yStrides[0] : yStrides[1]);
              auto rZ = baseZ + i0 * zTadStrides[0] + i1 * zTadStrides[1];

              *rZ = OpType::op(xValue, *rY);
            }
          }
        }
      }
    }
      // Matrix with vector broadcasting
    else if ((isXRowVector && isYRowVector) || (isXColumnVector && isYColumnVector)) {
      // Matching vectors - element-wise operation
      for (auto i = start; i < stop; i++) {
        auto baseX = x + (xTadOffset ? xTadOffset[i] : 0);
        auto baseZ = z + (zTadOffset ? zTadOffset[i] : 0);

        sd::LongType vecLength = isXRowVector ? nCols : nRows;
        sd::LongType xStride = isXRowVector ? xTadStrides[xTadRank-1] : xTadStrides[0];
        sd::LongType yStride = isYRowVector ? (yRank == 1 ? yStrides[0] : yStrides[1]) : yStrides[0];
        sd::LongType zStride = isZRowVector ? zTadStrides[zTadRank-1] : zTadStrides[0];

        PRAGMA_OMP_SIMD
        for (sd::LongType i1 = 0; i1 < vecLength; i1++) {
          auto rX = baseX + i1 * xStride;
          auto rY = y + i1 * yStride;
          auto rZ = baseZ + i1 * zStride;

          *rZ = OpType::op(*rX, *rY);
        }
      }
    }
      // Matrix with row vector broadcasting
    else if (isYRowVector && xTadRank == 2 && zTadRank == 2 &&
             xTadShape[1] == (yRank == 1 ? yShape[0] : yShape[1])) {
      // Broadcasting row vector (each element applied to a column)
      for (auto i0 = start; i0 < stop; i0++) {
        auto baseX = x + (xTadOffset ? xTadOffset[i0] : 0);
        auto baseZ = z + (zTadOffset ? zTadOffset[i0] : 0);

        for (sd::LongType i1 = 0; i1 < nRows; i1++) {
          for (sd::LongType i2 = 0; i2 < nCols; i2++) {
            // Get element from X at current position
            auto xOffset = i1 * xTadStrides[0] + i2 * xTadStrides[1];
            // Get element from Y row vector based on column index only
            auto yOffset = i2 * (yRank == 1 ? yStrides[0] : yStrides[1]);
            // Get destination element in Z at current position
            auto zOffset = i1 * zTadStrides[0] + i2 * zTadStrides[1];

            // Apply operation
            baseZ[zOffset] = OpType::op(baseX[xOffset], y[yOffset]);
          }
        }
      }
    }
      // Matrix with column vector broadcasting
    else if (isYColumnVector) {
      // Broadcasting column vector (each element applied to a row)
      for (auto i0 = start; i0 < stop; i0++) {
        auto baseX = x + (xTadOffset ? xTadOffset[i0] : 0);
        auto baseZ = z + (zTadOffset ? zTadOffset[i0] : 0);

        for (sd::LongType i1 = 0; i1 < nRows; i1++) {
          // Get element from column vector based on row index
          auto rY = y + i1 * yStrides[0];

          PRAGMA_OMP_SIMD
          for (sd::LongType i2 = 0; i2 < nCols; i2++) {
            auto rX = baseX + i1 * xTadStrides[0] + i2 * xTadStrides[1];
            auto rZ = baseZ + i1 * zTadStrides[0] + i2 * zTadStrides[1];

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
    }
      // Standard 2D broadcasting
    else {
      for (auto i0 = start; i0 < stop; i0++) {
        auto baseX = x + (xTadOffset ? xTadOffset[i0] : 0);
        auto baseZ = z + (zTadOffset ? zTadOffset[i0] : 0);

        for (sd::LongType i1 = 0; i1 < nRows; i1++) {
          PRAGMA_OMP_SIMD
          for (sd::LongType i2 = 0; i2 < nCols; i2++) {
            auto rX = baseX + i1 * xTadStrides[0] + i2 * xTadStrides[1];
            auto rY = y + i1 * yStrides[0] + i2 * yStrides[1];
            auto rZ = baseZ + i1 * zTadStrides[0] + i2 * zTadStrides[1];

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
    }
  }
    // Handle remaining loop kinds
  else if (loopKind == sd::LoopKind::BROADCAST_SCALAR_X) {
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
    // Handle 3D broadcasting (generalized like 2D case)
  else if (loopKind == sd::LoopKind::BROADCAST_3D) {
    // Get TAD info
    const sd::LongType tadRank = xTadShapeInfo ? shape::rank(xTadShapeInfo) : 3;
    const sd::LongType* tadShape = xTadShapeInfo ? shape::shapeOf(xTadShapeInfo) : xShape;
    const sd::LongType* tadStride = xTadShapeInfo ? shape::stride(xTadShapeInfo) : xStrides;
    const sd::LongType tadLength = xTadShapeInfo ? shape::length(xTadShapeInfo) : shape::length(xShapeInfo);
    if (isYVector) {
      // Vector broadcasting
      const sd::LongType yLength = yRank == 1 ? yShape[0] : (yShape[0] == 1 ? yShape[1] : yShape[0]);
      const sd::LongType yStride = yRank == 1 ? yStrides[0] : (yShape[0] == 1 ? yStrides[1] : yStrides[0]);

      // Determine which dimension this vector should be broadcast along
      // For a 3D TAD, check if vector length matches any dimension
      if (tadRank == 3) {
        if (yLength == tadShape[2]) {
          // Broadcast along last dimension
          for (auto i = start; i < stop; i++) {
            auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
            auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

            for (sd::LongType j = 0; j < tadLength; j++) {
              // Calculate TAD coords
              sd::LongType coords[SD_MAX_RANK];
              INDEX2COORDS(j, tadRank, tadShape, coords);

              // Get offsets
              sd::LongType xOffset, zOffset;
              COORDS2INDEX(tadRank, tadStride, coords, xOffset);
              COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

              // Get Y index - use the last dimension (coords[2])
              sd::LongType yOffset = coords[2] * yStride;

              // Apply operation
              oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
            }
          }
        }
        else if (yLength == tadShape[1]) {
          // Broadcast along middle dimension
          PRAGMA_OMP_SIMD
          for (auto i = start; i < stop; i++) {
            auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
            auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

            for (sd::LongType j = 0; j < tadLength; j++) {
              // Calculate TAD coords
              sd::LongType coords[SD_MAX_RANK];
              INDEX2COORDS(j, tadRank, tadShape, coords);

              // Get offsets
              sd::LongType xOffset, zOffset;
              COORDS2INDEX(tadRank, tadStride, coords, xOffset);
              COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

              // Get Y index - use the middle dimension (coords[1])
              sd::LongType yOffset = coords[1] * yStride;

              // Apply operation
              oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
            }
          }
        }
        else if (yLength == tadShape[0]) {
          // Broadcast along first dimension
          PRAGMA_OMP_SIMD
          for (auto i = start; i < stop; i++) {
            auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
            auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

            for (sd::LongType j = 0; j < tadLength; j++) {
              // Calculate TAD coords
              sd::LongType coords[SD_MAX_RANK];
              INDEX2COORDS(j, tadRank, tadShape, coords);

              // Get offsets
              sd::LongType xOffset, zOffset;
              COORDS2INDEX(tadRank, tadStride, coords, xOffset);
              COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

              // Get Y index - use the first dimension (coords[0])
              sd::LongType yOffset = coords[0] * yStride;

              // Apply operation
              oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
            }
          }
        }
        else {
          // Default broadcasting behavior - broadcast along the last dimension
          PRAGMA_OMP_SIMD
          for (auto i = start; i < stop; i++) {
            auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
            auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

            for (sd::LongType j = 0; j < tadLength; j++) {
              // Calculate TAD coords
              sd::LongType coords[SD_MAX_RANK];
              INDEX2COORDS(j, tadRank, tadShape, coords);

              // Get offsets
              sd::LongType xOffset, zOffset;
              COORDS2INDEX(tadRank, tadStride, coords, xOffset);
              COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

              // Get Y index with wrapping/broadcasting
              sd::LongType yOffset = (coords[2] % yLength) * yStride;

              // Apply operation
              oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
            }
          }
        }
      }
      else {
        // Handle lower rank TADs (1D or 2D)
        PRAGMA_OMP_SIMD
        for (auto i = start; i < stop; i++) {
          auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
          auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

          for (sd::LongType j = 0; j < tadLength; j++) {
            // Calculate TAD coords
            sd::LongType coords[SD_MAX_RANK];
            INDEX2COORDS(j, tadRank, tadShape, coords);

            // Get offsets
            sd::LongType xOffset, zOffset;
            COORDS2INDEX(tadRank, tadStride, coords, xOffset);
            COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

            // Get Y index - for lower ranks, broadcast along the last available dimension
            sd::LongType lastCoord = tadRank > 0 ? coords[tadRank - 1] : 0;
            sd::LongType yOffset = (lastCoord % yLength) * yStride;

            // Apply operation
            oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
          }
        }
      }
    }
    else if (yRank == 2) {
      // Y is a 2D matrix - determine which dimensions it aligns with
      for (auto i = start; i < stop; i++) {
        auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
        auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);
        PRAGMA_OMP_SIMD
        for (sd::LongType j = 0; j < tadLength; j++) {
          // Calculate TAD coords
          sd::LongType coords[SD_MAX_RANK];
          INDEX2COORDS(j, tadRank, tadShape, coords);

          // Get offsets
          sd::LongType xOffset, zOffset;
          COORDS2INDEX(tadRank, tadStride, coords, xOffset);
          COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

          // Calculate Y offset based on dimension matching
          sd::LongType yOffset;

          // Default behavior for different 2D matrix broadcasting patterns
          if (tadRank == 3) {
            if (yShape[0] == tadShape[0] && yShape[1] == tadShape[2]) {
              // Y is aligned with dimensions 0 and 2
              yOffset = coords[0] * yStrides[0] + coords[2] * yStrides[1];
            }
            else if (yShape[0] == tadShape[0] && yShape[1] == tadShape[1]) {
              // Y is aligned with dimensions 0 and 1
              yOffset = coords[0] * yStrides[0] + coords[1] * yStrides[1];
            }
            else if (yShape[0] == tadShape[1] && yShape[1] == tadShape[2]) {
              // Y is aligned with dimensions 1 and 2
              yOffset = coords[1] * yStrides[0] + coords[2] * yStrides[1];
            }
            else {
              // Default: broadcast Y to match the last two dimensions with modulo
              yOffset = (coords[1] % yShape[0]) * yStrides[0] + (coords[2] % yShape[1]) * yStrides[1];
            }
          }
          else if (tadRank == 2) {
            // Direct mapping for 2D TAD and 2D Y
            yOffset = (coords[0] % yShape[0]) * yStrides[0] + (coords[1] % yShape[1]) * yStrides[1];
          }
          else {
            // For 1D TAD, map to the first dimension of Y
            yOffset = (coords[0] % yShape[0]) * yStrides[0];
          }

          // Apply operation
          oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
        }
      }
    }
    else if (yRank == 3) {
      // Y is a 3D tensor
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
        auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

        for (sd::LongType j = 0; j < tadLength; j++) {
          // Calculate TAD coords
          sd::LongType coords[SD_MAX_RANK];
          INDEX2COORDS(j, tadRank, tadShape, coords);

          // Get offsets
          sd::LongType xOffset, zOffset;
          COORDS2INDEX(tadRank, tadStride, coords, xOffset);
          COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

          // Calculate Y offset with modulo for broadcasting if needed
          sd::LongType yCoords[3] = {0, 0, 0};

          // Map coordinates appropriately based on ranks
          if (tadRank == 3) {
            yCoords[0] = coords[0] % yShape[0];
            yCoords[1] = coords[1] % yShape[1];
            yCoords[2] = coords[2] % yShape[2];
          }
          else if (tadRank == 2) {
            // Map 2D to last 2 dimensions of 3D
            yCoords[0] = 0;  // First dimension is broadcasted
            yCoords[1] = coords[0] % yShape[1];
            yCoords[2] = coords[1] % yShape[2];
          }
          else {
            // Map 1D to last dimension of 3D
            yCoords[0] = 0;
            yCoords[1] = 0;
            yCoords[2] = coords[0] % yShape[2];
          }

          sd::LongType yOffset = yCoords[0] * yStrides[0] + yCoords[1] * yStrides[1] + yCoords[2] * yStrides[2];

          // Apply operation
          oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
        }
      }
    }
    else {
      // General case for other ranks of Y
      for (auto i = start; i < stop; i++) {
        auto oX = x + (xTadOffset ? xTadOffset[i] : 0);
        auto oZ = z + (zTadOffset ? zTadOffset[i] : 0);

        for (sd::LongType j = 0; j < tadLength; j++) {
          // Calculate TAD coords
          sd::LongType coords[SD_MAX_RANK];
          INDEX2COORDS(j, tadRank, tadShape, coords);

          // Get offsets
          sd::LongType xOffset, zOffset;
          COORDS2INDEX(tadRank, tadStride, coords, xOffset);
          COORDS2INDEX(tadRank, zTadStrides, coords, zOffset);

          // Calculate Y offset based on rank
          sd::LongType yOffset = 0;

          // Map coordinates to Y based on rank
          for (int d = 0; d < tadRank && d < yRank; d++) {
            yOffset += (coords[d] % yShape[d]) * yStrides[d];
          }

          // Apply operation
          oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
        }
      }
    }
  }
  else if (loopKind == sd::LoopKind::BROADCAST_4D) {
    const sd::LongType nSize1 = shape::sizeAt(zShapeInfo, 1);
    const sd::LongType nSize2 = shape::sizeAt(zShapeInfo, 2);
    const sd::LongType nSize3 = shape::sizeAt(zShapeInfo, 3);

    for (auto i = start; i < stop; i++) {
      uint64_t i0 = i / nSize1;
      uint64_t i1 = i % nSize1;

      for (sd::LongType i2 = 0; i2 < nSize2; i2++) {
        PRAGMA_OMP_SIMD
        for (sd::LongType i3 = 0; i3 < nSize3; i3++) {
          auto rX = x + (xStrides[0] * i0 + xStrides[1] * i1 + xStrides[2] * i2 + xStrides[3] * i3);
          auto rY = y + (yStrides[0] * i0 + yStrides[1] * i1 + yStrides[2] * i2 + yStrides[3] * i3);
          auto rZ = z + (zStrides[0] * i0 + zStrides[1] * i1 + zStrides[2] * i2 + zStrides[3] * i3);

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

    for (auto i = start; i < stop; i++) {
      uint32_t i0 = i / nSize1;
      uint32_t i1 = i % nSize1;

      for (sd::LongType i2 = 0; i2 < nSize2; i2++) {
        for (sd::LongType i3 = 0; i3 < nSize3; i3++) {
          PRAGMA_OMP_SIMD
          for (sd::LongType i4 = 0; i4 < nSize4; i4++) {
            auto rX = x + (xStrides[0] * i0 + xStrides[1] * i1 + xStrides[2] * i2 + xStrides[3] * i3 + xStrides[4] * i4);
            auto rY = y + (yStrides[0] * i0 + yStrides[1] * i1 + yStrides[2] * i2 + yStrides[3] * i3 + yStrides[4] * i4);
            auto rZ = z + (zStrides[0] * i0 + zStrides[1] * i1 + zStrides[2] * i2 + zStrides[3] * i3 + zStrides[4] * i4);

            *rZ = OpType::op(*rX, *rY);
          }
        }
      }
    }
  }
  else {
    // Default case for other ranks - general purpose implementation
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
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(yShapeInfo), dimension,
                                                                         dimensionLength);
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