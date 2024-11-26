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
                             sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
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

 sd::LongType tadLength = shape::length(xTadShapeShapeInfo);
 if (zTadShapeInfo == nullptr) {
   zTadShapeInfo = xTadShapeShapeInfo;
   zTadOffset = tadOffsets;
 }

 auto xEws = shape::elementWiseStride(xTadShapeShapeInfo);
 auto yEws = shape::elementWiseStride(yShapeInfo);
 auto zEws = shape::elementWiseStride(zTadShapeInfo);

 const sd::LoopKind::Kind kindOfLoop =
     (loopKind == sd::LoopKind::BROADCAST_SCALAR_X || loopKind == sd::LoopKind::BROADCAST_SCALAR_Y ||
      loopKind == sd::LoopKind::BROADCAST_3D || loopKind == sd::LoopKind::BROADCAST_4D ||
      loopKind == sd::LoopKind::BROADCAST_5D)
         ? loopKind
         : sd::LoopKind::deduceKindOfLoopXYZ(xTadShapeShapeInfo, yShapeInfo, zTadShapeInfo);

 if (kindOfLoop == sd::LoopKind::BROADCAST_SCALAR_X) {
   auto loopLength = yShapeInfo[shape::rank(yShapeInfo)];

   for (auto i = start; i < stop; i++) {
     auto oY = y + (i * loopLength);
     auto oZ = z + (i * loopLength);

     const auto oX = x[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < loopLength; f++) oZ[f] = OpType::op(oX, oY[f]);
   }
 } else if (kindOfLoop == sd::LoopKind::BROADCAST_SCALAR_Y) {
   auto loopLength = xShapeInfo[shape::rank(xShapeInfo)];

   for (auto i = start; i < stop; i++) {
     auto oX = x + (i * loopLength);
     auto oZ = z + (i * loopLength);

     const auto oY = y[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < loopLength; f++) oZ[f] = OpType::op(oX[f], oY);
   }
 } else if (kindOfLoop == sd::LoopKind::BROADCAST_3D) {
   int xRank = shape::rank(xShapeInfo);
   int yRank = shape::rank(yShapeInfo);

   auto xStrides = shape::stride(xShapeInfo);
   auto zStrides = shape::stride(zShapeInfo);

   sd::LongType yStrides[3] = {static_cast<sd::LongType>(0), static_cast<sd::LongType>(0), static_cast<sd::LongType>(0)};
   sd::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

   sd::LongType nSize1 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(1));
   sd::LongType nSize2 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(2));

   for (auto index0 = start; index0 < stop; index0++) {
     PRAGMA_OMP_SIMD
     for (sd::LongType index1 = 0; index1 < nSize1; index1++) {
       for (sd::LongType index2 = 0; index2 < nSize2; index2++) {
         auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2);
         auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2);
         auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2);
         *rZ = OpType::op(*rX, *rY);
       }
     }
   }

 } else if (kindOfLoop == sd::LoopKind::BROADCAST_4D) {
   sd::LongType xRank = shape::rank(xShapeInfo);
   sd::LongType yRank = shape::rank(yShapeInfo);

   auto xStrides = shape::stride(xShapeInfo);
   auto zStrides = shape::stride(zShapeInfo);

   sd::LongType yStrides[4] = {static_cast<sd::LongType>(0), static_cast<sd::LongType>(0), static_cast<sd::LongType>(0), static_cast<sd::LongType>(0)};
   sd::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

   uint64_t nSize1 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(1));
   uint64_t nSize2 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(2));
   uint64_t nSize3 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(3));

   for (auto i = start; i < stop; i++) {
     uint64_t index0 = i / nSize1;
     uint64_t index1 = i % nSize1;

     PRAGMA_OMP_SIMD
     for (sd::LongType index2 = 0; index2 < nSize2; index2++) {
       for (sd::LongType index3 = 0; index3 < nSize3; index3++) {
         auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2 + xStrides[3] * index3);
         auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2 + yStrides[3] * index3);
         auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2 + zStrides[3] * index3);
         *rZ = OpType::op(*rX, *rY);
       }
     }
   }

 } else if (kindOfLoop == sd::LoopKind::BROADCAST_5D) {
   sd::LongType xRank = shape::rank(xShapeInfo);
   sd::LongType yRank = shape::rank(yShapeInfo);

   auto xStrides = shape::stride(xShapeInfo);
   auto zStrides = shape::stride(zShapeInfo);

   sd::LongType yStrides[5] = {0, static_cast<sd::LongType>(0), static_cast<sd::LongType>(0), static_cast<sd::LongType>(0), static_cast<sd::LongType>(0)};
   sd::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

   sd::LongType nSize1 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(1));
   sd::LongType nSize2 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(2));
   sd::LongType nSize3 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(3));
   sd::LongType nSize4 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(4));

   for (auto i = start; i < stop; i++) {
     uint32_t index0 = i / nSize1;
     uint32_t index1 = i % nSize1;

     PRAGMA_OMP_SIMD
     for (sd::LongType index2 = 0; index2 < nSize2; index2++) {
       for (sd::LongType index3 = 0; index3 < nSize3; index3++) {
         for (sd::LongType index4 = 0; index4 < nSize4; index4++) {
           auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2 + xStrides[3] * index3 +
                          xStrides[4] * index4);
           auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2 + yStrides[3] * index3 +
                          yStrides[4] * index4);
           auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2 + zStrides[3] * index3 +
                          zStrides[4] * index4);

           *rZ = OpType::op(*rX, *rY);
         }
       }
     }
   }

 } else if (shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo) &&
            shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
   for (auto i = start; i < stop; i++) {
     auto oX = x + tadOffsets[i];
     auto oZ = z + zTadOffset[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < tadLength; f++) {
       sd::LongType coords[SD_MAX_RANK];
       INDEX2COORDS(f, shape::rank(xTadShapeShapeInfo), shape::shapeOf(xTadShapeShapeInfo), coords);
       sd::LongType offset;
       COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, offset);
       oZ[offset] = OpType::op(oX[offset], y[offset]);
     }
   }
 } else if (shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo)) {
   for (auto i = start; i < stop; i++) {
     auto oZ = z + zTadOffset[i];
     auto oX = x + tadOffsets[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < tadLength; f++) {
       sd::LongType coords[SD_MAX_RANK];
       INDEX2COORDS(f, shape::rank(xTadShapeShapeInfo), shape::shapeOf(xTadShapeShapeInfo), coords);
       sd::LongType offset;
       COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, offset);
       sd::LongType zOffset;
       COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), coords, zOffset);
       oZ[zOffset] = OpType::op(oX[offset], y[offset]);
     }
   }
 } else if (shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
   for (auto i = start; i < stop; i++) {
     auto oZ = z + zTadOffset[i];
     auto oX = x + tadOffsets[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < tadLength; f++) {
       sd::LongType coords[SD_MAX_RANK];
       INDEX2COORDS(f, shape::rank(xTadShapeShapeInfo), shape::shapeOf(xTadShapeShapeInfo), coords);
       sd::LongType offset;
       COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, offset);
       sd::LongType yOffset;
       COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, yOffset);
       oZ[offset] = OpType::op(oX[offset], y[yOffset]);
     }
   }
 } else if (shape::haveSameShapeAndStrides(yShapeInfo, zTadShapeInfo)) {
   for (auto i = start; i < stop; i++) {
     auto oZ = z + zTadOffset[i];
     auto oX = x + tadOffsets[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < tadLength; f++) {
       sd::LongType coords[SD_MAX_RANK];
       INDEX2COORDS(f, shape::rank(yShapeInfo), shape::shapeOf(yShapeInfo), coords);
       sd::LongType xOffset;
       COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, xOffset);
       sd::LongType offset;
       COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, offset);
       oZ[offset] = OpType::op(oX[xOffset], y[offset]);
     }
   }
 } else {
   for (auto i = start; i < stop; i++) {
     auto oZ = z + zTadOffset[i];
     auto oX = x + tadOffsets[i];

     PRAGMA_OMP_SIMD
     for (sd::LongType f = 0; f < tadLength; f++) {
       sd::LongType coords[SD_MAX_RANK];
       INDEX2COORDS(f, shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), coords);
       sd::LongType xOffset;
       COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, xOffset);
       sd::LongType yOffset;
       COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, yOffset);
       sd::LongType zOffset;
       COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), coords, zOffset);
       oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
     }
   }
 }
}

template <typename X, typename Y, typename Z>
template <typename OpType>
void Broadcast<X, Y, Z>::execInverse(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                     const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                     sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *yTadShapeInfo,
                                     const sd::LongType *yTadOffset, const sd::LongType *zTadShapeInfo,
                                     const sd::LongType *zTadOffset,
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

  sd::LongType tadLength = shape::length(yTadShapeShapeInfo);
  sd::LongType tads = shape::length(yShapeInfo) / tadLength;

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = yTadShapeShapeInfo;
    zTadOffset = tadOffsets;
  }

  auto lenZ = shape::length(zTadShapeInfo);
  auto lenX = shape::length(xShapeInfo);

  sd::LongType tadsPerThread = tads / TAD_THRESHOLD;
  sd::LongType threads = sd::math::sd_max<sd::LongType>(1, tadsPerThread);
  threads = sd::math::sd_min<sd::LongType>(threads, sd::Environment::getInstance().maxThreads());

  auto yEws = shape::elementWiseStride(yTadShapeShapeInfo);
  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto zEws = shape::elementWiseStride(zTadShapeInfo);

  const sd::LoopKind::Kind kindOfLoop =
      sd::LoopKind::deduceKindOfLoopXYZ(yTadShapeShapeInfo, xShapeInfo, zTadShapeInfo);

  if (kindOfLoop == sd::LoopKind::EWS1) {
    for (auto i = start; i < stop; i++) {
      auto oY = y + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f] = OpType::op(x[f], oY[f]);
    }
  } else if (kindOfLoop == sd::LoopKind::EWSNONZERO) {
    for (auto i = start; i < stop; i++) {
      auto oY = y + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f * zEws] = OpType::op(x[f * xEws], oY[f * yEws]);
    };
  } else if (shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo) &&
             shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oY = y + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(yTadShapeShapeInfo), shape::shapeOf(yTadShapeShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, offset);
        oZ[offset] = OpType::op(x[offset], oY[offset]);
      }
    };
  } else if (shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(yTadShapeShapeInfo), shape::shapeOf(yTadShapeShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, offset);
        sd::LongType zOffset;
        COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), coords, zOffset);
        oZ[zOffset] = OpType::op(x[offset], oY[offset]);
      }
    };
  } else if (shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(yTadShapeShapeInfo), shape::shapeOf(yTadShapeShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, offset);
        sd::LongType xOffset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
        oZ[offset] = OpType::op(x[xOffset], oY[offset]);
      }
    };
  } else if (shape::haveSameShapeAndStrides(xShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), coords);
        sd::LongType yOffset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, yOffset);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, offset);
        oZ[offset] = OpType::op(x[offset], oY[yOffset]);
      }
    };
  } else {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(zTadShapeInfo), shape::shapeOf(zTadShapeInfo), coords);
        sd::LongType xOffset;
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
        sd::LongType yOffset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, yOffset);
        sd::LongType zOffset;
        COORDS2INDEX(shape::rank(zTadShapeInfo), shape::stride(zTadShapeInfo), coords, zOffset);
        oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
      }
    };
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void Broadcast<X, Y, Z>::exec(const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                              const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo) {
  DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo), BROADCAST_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
static void execRank1(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<sd::LongType>(0));

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
  samediff::Threads::parallel_tad(func, static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
static void execRank2(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(1));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(1));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(1));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(1));

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(0));
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(0));
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(0));
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : static_cast<sd::LongType>(0));

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

  samediff::Threads::parallel_tad(func, static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
static void execRank3(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : 2);
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : 2);
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : 2);
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : 2);

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, static_cast<sd::LongType>(1));

  sd::LongType zAxis2 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(0));
  sd::LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(0));
  sd::LongType yStrd2 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(0));
  sd::LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
        auto y1 = y + i0 * yStrd0 + i1 * yStrd1;
        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

        if (zStrd2 == 1 && xStrd2 == 1 && yStrd2 == 0)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(x1[i2], *y1);
        else if (zStrd2 == 1 && xStrd2 == 0 && yStrd2 == 1)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(*x1, y1[i2]);
        else if (zStrd2 == 1 && xStrd2 == 1 && yStrd2 == 1)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(x1[i2], y1[i2]);
        else
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2 * zStrd2] = OpType::op(x1[i2 * xStrd2], y1[i2 * yStrd2]);
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
static void execRank4(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(3));

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : 2);
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : 2);
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : 2);
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(1) : 2);

  sd::LongType zAxis2 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(1));
  sd::LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(1));
  sd::LongType yStrd2 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(1));
  sd::LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? 2 : static_cast<sd::LongType>(1));

  sd::LongType zAxis3 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3): static_cast<sd::LongType>(0));
  sd::LongType xStrd3 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3): static_cast<sd::LongType>(0));
  sd::LongType yStrd3 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3): static_cast<sd::LongType>(0));
  sd::LongType zStrd3 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(3): static_cast<sd::LongType>(0));

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
template <typename X, typename Y, typename Z, typename OpType>
static void execRank5(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
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

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
static void execDefault(const X *x, const sd::LongType *xShapeInfo, const Y *y, const sd::LongType *yShapeInfo, Z *z,
                        const sd::LongType *zShapeInfo) {
  const bool xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
  const bool yzSameOffsets = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType xOffset, yOffset, zOffset;

    for (auto i = start; i < stop; ++i) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(i, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), coords);

      COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), coords, zOffset);
      if (xzSameOffsets) {
        xOffset = zOffset;
      } else {
        COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);
      }
      if (yzSameOffsets) {
        yOffset = zOffset;
      } else {
        COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, yOffset);
      }

      z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), shape::length(zShapeInfo));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
template <typename OpType>
void Broadcast<X, Y, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                              const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  const X *x = reinterpret_cast<const X *>(vx);
  const Y *y = reinterpret_cast<const Y *>(vy);
  Z *z = reinterpret_cast<Z *>(vz);

  const int rank = shape::rank(zShapeInfo);  // xRank = yRank = zRank

  switch (rank) {
    case 1:
      execRank1<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
      break;
    case 2:
      execRank2<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
      break;
    case 3:
      execRank3<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
      break;
    case 4:
      execRank4<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
      break;
    case 5:
      execRank5<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
      break;
    default:
      execDefault<X, Y, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo);
  }
}

}  // namespace broadcast
}  // namespace functions