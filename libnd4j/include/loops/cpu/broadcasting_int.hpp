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
#include <loops/broadcasting_int.h>
#include <loops/legacy_ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace broadcast {

template <typename X>
void BroadcastInt<X>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                          const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                          sd::LongType*dimension,
                          sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                          const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, sd::LongType start,
                          sd::LongType stop) {
 DISPATCH_BY_OPNUM_T(exec,
                     PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, xTadShapeInfo,
                            xTadOffset, zTadShapeInfo, zTadOffset, start, stop),
                     BROADCAST_INT_OPS);
}

template <typename X>
void BroadcastInt<X>::exec(const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                          const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo) {
 DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo), BROADCAST_INT_OPS);
}

template <typename X>
void BroadcastInt<X>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                 const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                 sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                                 const sd::LongType *xTadOffset, const sd::LongType *zTadShapeInfo,
                                 const sd::LongType *zTadOffset, sd::LongType start, sd::LongType stop) {
 DISPATCH_BY_OPNUM_T(execInverse,
                     PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, xTadShapeInfo,
                            xTadOffset, zTadShapeInfo, zTadOffset, start, stop),
                     BROADCAST_INT_OPS);
}

template <typename X>
template <typename OpType>
void BroadcastInt<X>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                           const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                           sd::LongType *dimension,
                           sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                           const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, sd::LongType start,
                           sd::LongType stop) {
 auto x = reinterpret_cast<const X*>(vx);
 auto y = reinterpret_cast<const X*>(vy);
 auto z = reinterpret_cast<X*>(vz);
 sd::LoopKind::Kind loopKind= sd::LoopKind::deduceKindOfLoopXYZ(xTadShapeInfo, yShapeInfo, zTadShapeInfo);

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

template <typename X>
template <typename OpType>
void BroadcastInt<X>::execInverse(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                  const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                  sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *yTadShapeInfo,
                                  const sd::LongType *yTadOffset, const sd::LongType *zTadShapeInfo,
                                  const sd::LongType *zTadOffset, sd::LongType start, sd::LongType stop) {
 auto x = reinterpret_cast<const X *>(vx);
 auto y = reinterpret_cast<const X *>(vy);
 auto z = reinterpret_cast<X *>(vz);

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
////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
static void execRank1(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo,  static_cast<sd::LongType>(0));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo,  static_cast<sd::LongType>(0));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo,  static_cast<sd::LongType>(0));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo,  static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR {
    if (zStrd0 == 1 && xStrd0 == 1 && yStrd0 == 0) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(x[i0], *y);
    } else if (zStrd0 == 1 && xStrd0 == 0 && yStrd0 == 1) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(*x, y[i0]);
    } else if (zStrd0 == 1 && xStrd0 == 1 && yStrd0 == 1) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(x[i0], y[i0]);
    } else {
      for (auto i0 = start; i0 < stop; ++i0) z[i0 * zStrd0] = OpType::op(x[i0 * xStrd0], y[i0 * yStrd0]);
    }
  };
  samediff::Threads::parallel_tad(func,  static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
static void execRank2(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0)   : static_cast<sd::LongType>(1) );
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0)   : static_cast<sd::LongType>(1) );
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0)   : static_cast<sd::LongType>(1) );
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0)   : static_cast<sd::LongType>(1) );

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1)   : static_cast<sd::LongType>(0) );
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1)   : static_cast<sd::LongType>(0) );
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1)   : static_cast<sd::LongType>(0) );
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1)   : static_cast<sd::LongType>(0) );

  auto func = PRAGMA_THREADS_FOR {
    for (auto i0 = start; i0 < stop; ++i0) {
      auto x0 = x + i0 * xStrd0;
      auto y0 = y + i0 * yStrd0;
      auto z0 = z + i0 * zStrd0;

      if (zStrd1 == 1 && xStrd1 == 1 && yStrd1 == 0)
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(x0[i1], *y0);
      else if (zStrd1 == 1 && xStrd1 == 0 && yStrd1 == 1)
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(*x0, y0[i1]);
      else if (zStrd1 == 1 && xStrd1 == 1 && yStrd1 == 1)
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(x0[i1], y0[i1]);
      else
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1 * zStrd1] = OpType::op(x0[i1 * xStrd1], y0[i1 * yStrd1]);
    }
  };

  samediff::Threads::parallel_tad(func,  static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
static void execRank3(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                      const sd::LongType *zShapeInfo) {
  // Cache all shape-related values
  const sd::LongType xRank = shape::rank(xShapeInfo);
  const sd::LongType yRank = shape::rank(yShapeInfo);
  const sd::LongType zRank = shape::rank(zShapeInfo);

  auto xStride = shape::stride(xShapeInfo);
  auto yStride = shape::stride(yShapeInfo);
  auto zStride = shape::stride(zShapeInfo);

  const sd::LongType zLen0 = shape::sizeAt(zShapeInfo, 0);
  const sd::LongType zLen1 = shape::sizeAt(zShapeInfo, 1);
  const sd::LongType zLen2 = shape::sizeAt(zShapeInfo, 2);

  auto func = PRAGMA_THREADS_FOR_2D {
    sd::LongType coords[3];
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (sd::LongType i2 = 0; i2 < zLen2; ++i2) {
          coords[0] = i0;
          coords[1] = i1;
          coords[2] = i2;

          sd::LongType xOffset, yOffset, zOffset;
          COORDS2INDEX(xRank, xStride, coords, xOffset);
          COORDS2INDEX(yRank, yStride, coords, yOffset);
          COORDS2INDEX(zRank, zStride, coords, zOffset);

          z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, zLen0, 1, 0, zLen1, 1);
}

template <typename X, typename OpType>
static void execRank4(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(2));
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(2));
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(2));
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(2));

  sd::LongType zAxis2 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(1));
  sd::LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(1));
  sd::LongType yStrd2 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(1));
  sd::LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(1));

  sd::LongType zAxis3 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(0));
  sd::LongType xStrd3 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(0));
  sd::LongType yStrd3 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(0));
  sd::LongType zStrd3 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (auto i2 = start_z; i2 < stop_z; ++i2) {
          auto x2 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2;
          auto y2 = y + i0 * yStrd0 + i1 * yStrd1 + i2 * yStrd2;
          auto z2 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2;

          if (zStrd3 == 1 && xStrd3 == 1 && yStrd3 == 0)
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(x2[i3], *y2);
          else if (zStrd3 == 1 && xStrd3 == 0 && yStrd3 == 1)
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(*x2, y2[i3]);
          else if (zStrd3 == 1 && xStrd3 == 1 && yStrd3 == 1)
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(x2[i3], y2[i3]);
          else
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3 * zStrd3] = OpType::op(x2[i3 * xStrd3], y2[i3 * yStrd3]);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis2, static_cast<sd::LongType>(1));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
static void execRank5(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(4));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(4));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(4));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(4));

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(3));
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(3));
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(3));
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(3));

  sd::LongType zAxis2 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(2));
  sd::LongType xStrd2 = shape::strideAt(xShapeInfo, static_cast<sd::LongType>(2));
  sd::LongType yStrd2 = shape::strideAt(yShapeInfo, static_cast<sd::LongType>(2));
  sd::LongType zStrd2 = shape::strideAt(zShapeInfo, static_cast<sd::LongType>(2));

  sd::LongType zAxis3 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(1));
  sd::LongType xStrd3 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(1));
  sd::LongType yStrd3 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(1));
  sd::LongType zStrd3 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3) : static_cast<sd::LongType>(1));

  sd::LongType zAxis4 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(4) : static_cast<sd::LongType>(0));
  sd::LongType xStrd4 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(4) : static_cast<sd::LongType>(0));
  sd::LongType yStrd4 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(4) : static_cast<sd::LongType>(0));
  sd::LongType zStrd4 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(4) : static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR_3D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        for (auto i2 = start_z; i2 < stop_z; ++i2) {
          for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) {
            auto x3 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3;
            auto y3 = y + i0 * yStrd0 + i1 * yStrd1 + i2 * yStrd2 + i3 * yStrd3;
            auto z3 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2 + i3 * zStrd3;

            if (zStrd4 == 1 && xStrd4 == 1 && yStrd4 == 0)
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(x3[i4], *y3);
            else if (zStrd4 == 1 && xStrd4 == 0 && yStrd4 == 1)
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(*x3, y3[i4]);
            else if (zStrd4 == 1 && xStrd4 == 1 && yStrd4 == 1)
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(x3[i4], y3[i4]);
            else
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4)
                z3[i4 * zStrd4] = OpType::op(x3[i4 * xStrd4], y3[i4 * yStrd4]);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis2, static_cast<sd::LongType>(1));
}



template <typename X, typename OpType>
static void execDefault(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, X *z,
                        const sd::LongType *zShapeInfo) {
  // Cache all shape-related values
  const sd::LongType xRank = shape::rank(xShapeInfo);
  const sd::LongType yRank = shape::rank(yShapeInfo);
  const sd::LongType zRank = shape::rank(zShapeInfo);

  auto xStride = shape::stride(xShapeInfo);
  auto yStride = shape::stride(yShapeInfo);
  auto zStride = shape::stride(zShapeInfo);

  auto zShape = shape::shapeOf(zShapeInfo);
  const sd::LongType zLen = shape::length(zShapeInfo);

  const bool xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
  const bool yzSameOffsets = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType xOffset, yOffset, zOffset;

    for (auto i = start; i < stop; ++i) {
      INDEX2COORDS(i, zRank, zShape, coords);
      COORDS2INDEX(zRank, zStride, coords, zOffset);

      if (xzSameOffsets) {
        xOffset = zOffset;
      } else {
        COORDS2INDEX(xRank, xStride, coords, xOffset);
      }

      if (yzSameOffsets) {
        yOffset = zOffset;
      } else {
        COORDS2INDEX(yRank, yStride, coords, yOffset);
      }
      z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zLen);
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
void BroadcastInt<X>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                       const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
const X *x = reinterpret_cast<const X *>(vx);
const X *y = reinterpret_cast<const X *>(vy);
X *z = reinterpret_cast<X *>(vz);

const int rank = shape::rank(zShapeInfo);  // xRank = yRank = zRank

switch (rank) {
case 1:
  execRank1<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  break;
case 2:
  execRank2<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  break;
case 3:
  execRank3<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  break;
case 4:
  execRank4<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  break;
case 5:
  execRank5<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  break;
default:
  execDefault<X, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
}
}

}  // namespace broadcast
}  // namespace functions