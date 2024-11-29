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
#include <loops/broadcasting_bool.h>
#include <loops/legacy_ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace broadcast {

template <typename X, typename Y>
void BroadcastBool<X, Y>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                               const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                               void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                               const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                               const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, sd::LongType start,
                               sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(exec,
                       PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength,
                              xTadShapeInfo, xTadOffset, zTadShapeInfo, zTadOffset, start, stop),
                       BROADCAST_BOOL_OPS);
}

template <typename X, typename Y>
void BroadcastBool<X, Y>::exec(const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                               const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                               void *extraParams) {
  DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams), BROADCAST_BOOL_OPS);
}

template <typename X, typename Y>
void BroadcastBool<X, Y>::execInverse(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                      const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                      void *extraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                                      const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                                      const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, sd::LongType start,
                                      sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(execInverse,
                       PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, dimension, dimensionLength,
                              xTadShapeInfo, xTadOffset, zTadShapeInfo, zTadOffset, start, stop),
                       BROADCAST_BOOL_OPS);
}

template <typename X, typename Z>
template <typename OpType>
void BroadcastBool<X, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                               const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                               void *vextraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                               const sd::LongType *xTadShapeInfo, const sd::LongType *xTadOffset,
                               const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, uint64_t start,
                               uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  auto xTadShapeShapeInfo = xTadShapeInfo;
  auto tadOffsets = xTadOffset;

  if (xTadShapeInfo == nullptr || tadOffsets == nullptr) {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);

    xTadShapeShapeInfo = const_cast<sd::LongType *>(tadPack->primaryShapeInfo());
    tadOffsets = const_cast<sd::LongType *>(tadPack->primaryOffsets());
  }

  sd::LongType tadLength = shape::length(xTadShapeShapeInfo);
  sd::LongType tads = shape::length(xShapeInfo) / tadLength;

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = xTadShapeShapeInfo;
    zTadOffset = tadOffsets;
  }

  auto lenZ = shape::length(zTadShapeInfo);
  auto lenY = shape::length(yShapeInfo);

  sd::LongType tadsPerThread = tads / TAD_THRESHOLD;
  sd::LongType threads = sd::math::sd_max<sd::LongType>(1, tadsPerThread);
  threads = sd::math::sd_min<int>(threads, sd::Environment::getInstance().maxThreads());

  auto xEws = shape::elementWiseStride(xTadShapeShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zTadShapeInfo);

  const sd::LoopKind::Kind kindOfLoop =
      sd::LoopKind::deduceKindOfLoopXYZ(xTadShapeShapeInfo, yShapeInfo, zTadShapeInfo);

  if (kindOfLoop == sd::LoopKind::EWS1) {
    for (auto i = start; i < stop; i++) {
      auto oX = x + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f] = OpType::op(oX[f], y[f], extraParams);
    }
  } else if (kindOfLoop == sd::LoopKind::EWSNONZERO) {
    for (auto i = start; i < stop; i++) {
      auto oX = x + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f * zEws] = OpType::op(oX[f * xEws], y[f * yEws], extraParams);
    };
  } else if (shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo) &&
             shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oX = x + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(xTadShapeShapeInfo), shape::shapeOf(xTadShapeShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, offset);
        oZ[offset] = OpType::op(oX[offset], y[offset], extraParams);
      }
    };
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
        oZ[zOffset] = OpType::op(oX[offset], y[offset], extraParams);
      }
    };
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
        oZ[offset] = OpType::op(oX[offset], y[yOffset], extraParams);
      }
    };

  } else if (shape::haveSameShapeAndStrides(yShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oX = x + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(yShapeInfo), shape::shapeOf(yShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, offset);
        sd::LongType xOffset;
        COORDS2INDEX(shape::rank(xTadShapeShapeInfo), shape::stride(xTadShapeShapeInfo), coords, xOffset);
        oZ[offset] = OpType::op(oX[xOffset], y[offset], extraParams);
      }
    };
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
        oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset], extraParams);
      }
    };
  }
}

template <typename X, typename Z>
template <typename OpType>
void BroadcastBool<X, Z>::execInverse(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                      void *vextraParams, sd::LongType *dimension, sd::LongType dimensionLength,
                                      const sd::LongType *yTadShapeInfo, const sd::LongType *yTadOffset,
                                      const sd::LongType *zTadShapeInfo, const sd::LongType *zTadOffset, uint64_t start,
                                      uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  auto yTadShapeShapeInfo = yTadShapeInfo;
  auto tadOffsets = yTadOffset;

  if (yTadShapeInfo == nullptr || tadOffsets == nullptr) {
    auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(yShapeInfo, dimension, dimensionLength);
    yTadShapeShapeInfo = const_cast<sd::LongType *>(tadPack->primaryShapeInfo());
    tadOffsets = const_cast<sd::LongType *>(tadPack->primaryOffsets());
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
  threads = sd::math::sd_min<int>(threads, sd::Environment::getInstance().maxThreads());

  const sd::LoopKind::Kind kindOfLoop =
      sd::LoopKind::deduceKindOfLoopXYZ(yTadShapeShapeInfo, xShapeInfo, zTadShapeInfo);

  if (kindOfLoop == sd::LoopKind::EWS1) {
    for (auto i = start; i < stop; i++) {
      auto oY = y + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f] = OpType::op(x[f], oY[f], extraParams);
    }
  } else if (kindOfLoop == sd::LoopKind::EWSNONZERO) {
    for (auto i = start; i < stop; i++) {
      auto oY = y + tadOffsets[i];
      auto oZ = z + zTadOffset[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) oZ[f * shape::stride(zTadShapeInfo)[0]] = OpType::op(x[f * shape::stride(xShapeInfo)[0]], oY[f * shape::stride(yTadShapeShapeInfo)[0]], extraParams);
    }
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
        oZ[offset] = OpType::op(x[offset], oY[offset], extraParams);
      }
    }
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
        oZ[zOffset] = OpType::op(x[offset], oY[offset], extraParams);
      }
    }
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
        oZ[offset] = OpType::op(x[xOffset], oY[offset], extraParams);
      }
    }
  } else if (shape::haveSameShapeAndStrides(xShapeInfo, zTadShapeInfo)) {
    for (auto i = start; i < stop; i++) {
      auto oZ = z + zTadOffset[i];
      auto oY = y + tadOffsets[i];

      PRAGMA_OMP_SIMD
      for (sd::LongType f = 0; f < tadLength; f++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(f, shape::rank(yTadShapeShapeInfo), shape::shapeOf(yTadShapeShapeInfo), coords);
        sd::LongType offset;
        COORDS2INDEX(shape::rank(yTadShapeShapeInfo), shape::stride(yTadShapeShapeInfo), coords, offset);
        sd::LongType yOffset;
        COORDS2INDEX(shape::rank(yShapeInfo), shape::stride(yShapeInfo), coords, yOffset);
        oZ[offset] = OpType::op(x[offset], oY[yOffset], extraParams);
      }
    }
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
        oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset], extraParams);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execRank1(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo, X *extraParams) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, static_cast<sd::LongType>(0));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR {
    if (zStrd0 == 1 && xStrd0 == 1 && yStrd0 == 0) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(x[i0], *y, extraParams);
    } else if (zStrd0 == 1 && xStrd0 == 0 && yStrd0 == 1) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(*x, y[i0], extraParams);
    } else if (zStrd0 == 1 && xStrd0 == 1 && yStrd0 == 1) {
      for (auto i0 = start; i0 < stop; ++i0) z[i0] = OpType::op(x[i0], y[i0], extraParams);
    } else {
      for (auto i0 = start; i0 < stop; ++i0) z[i0 * zStrd0] = OpType::op(x[i0 * xStrd0], y[i0 * yStrd0], extraParams);
    }
  };
  samediff::Threads::parallel_tad(func, static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execRank2(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo, X *extraParams) {
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
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(x0[i1], *y0, extraParams);
      else if (zStrd1 == 1 && xStrd1 == 0 && yStrd1 == 1)
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(*x0, y0[i1], extraParams);
      else if (zStrd1 == 1 && xStrd1 == 1 && yStrd1 == 1)
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1) z0[i1] = OpType::op(x0[i1], y0[i1], extraParams);
      else
        for (sd::LongType i1 = 0; i1 < zAxis1; ++i1)
          z0[i1 * zStrd1] = OpType::op(x0[i1 * xStrd1], y0[i1 * yStrd1], extraParams);
    }
  };

  samediff::Threads::parallel_tad(func, static_cast<sd::LongType>(0), zAxis0);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execRank3(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo, X *extraParams) {
  sd::LongType zAxis0 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(2));
  sd::LongType xStrd0 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(2));
  sd::LongType yStrd0 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(2));
  sd::LongType zStrd0 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(0) : static_cast<sd::LongType>(2));

  sd::LongType zAxis1 = shape::sizeAt(zShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType xStrd1 = shape::strideAt(xShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType yStrd1 = shape::strideAt(yShapeInfo, static_cast<sd::LongType>(1));
  sd::LongType zStrd1 = shape::strideAt(zShapeInfo, static_cast<sd::LongType>(1));

  sd::LongType zAxis2 = shape::sizeAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(0));
  sd::LongType xStrd2 = shape::strideAt(xShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(0));
  sd::LongType yStrd2 = shape::strideAt(yShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(0));
  sd::LongType zStrd2 = shape::strideAt(zShapeInfo, shape::order(zShapeInfo) == 'c' ? static_cast<sd::LongType>(2) : static_cast<sd::LongType>(0));

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto i0 = start_x; i0 < stop_x; ++i0) {
      for (auto i1 = start_y; i1 < stop_y; ++i1) {
        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
        auto y1 = y + i0 * yStrd0 + i1 * yStrd1;
        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

        if (zStrd2 == 1 && xStrd2 == 1 && yStrd2 == 0)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(x1[i2], *y1, extraParams);
        else if (zStrd2 == 1 && xStrd2 == 0 && yStrd2 == 1)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(*x1, y1[i2], extraParams);
        else if (zStrd2 == 1 && xStrd2 == 1 && yStrd2 == 1)
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2) z1[i2] = OpType::op(x1[i2], y1[i2], extraParams);
        else
          for (sd::LongType i2 = 0; i2 < zAxis2; ++i2)
            z1[i2 * zStrd2] = OpType::op(x1[i2 * xStrd2], y1[i2 * yStrd2], extraParams);
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execRank4(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo, X *extraParams) {
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
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(x2[i3], *y2, extraParams);
          else if (zStrd3 == 1 && xStrd3 == 0 && yStrd3 == 1)
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(*x2, y2[i3], extraParams);
          else if (zStrd3 == 1 && xStrd3 == 1 && yStrd3 == 1)
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3) z2[i3] = OpType::op(x2[i3], y2[i3], extraParams);
          else
            for (sd::LongType i3 = 0; i3 < zAxis3; ++i3)
              z2[i3 * zStrd3] = OpType::op(x2[i3 * xStrd3], y2[i3 * yStrd3], extraParams);
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis2, static_cast<sd::LongType>(1));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execRank5(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                      const sd::LongType *zShapeInfo, X *extraParams) {
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
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(x3[i4], *y3, extraParams);
            else if (zStrd4 == 1 && xStrd4 == 0 && yStrd4 == 1)
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(*x3, y3[i4], extraParams);
            else if (zStrd4 == 1 && xStrd4 == 1 && yStrd4 == 1)
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4) z3[i4] = OpType::op(x3[i4], y3[i4], extraParams);
            else
              for (sd::LongType i4 = 0; i4 < zAxis4; ++i4)
                z3[i4 * zStrd4] = OpType::op(x3[i4 * xStrd4], y3[i4 * yStrd4], extraParams);
          }
        }
      }
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), zAxis0, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis1, static_cast<sd::LongType>(1), static_cast<sd::LongType>(0), zAxis2, static_cast<sd::LongType>(1));
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename OpType>
static void execDefault(const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Z *z,
                        const sd::LongType *zShapeInfo, X *extraParams) {
  const bool xzSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
  const bool yzSameOffsets = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);

  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    sd::LongType xOffset, yOffset, zOffset;

    for (auto i = start; i < stop; ++i) {
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

      z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
    }
  };

  samediff::Threads::parallel_for(func, static_cast<sd::LongType>(0), shape::length(zShapeInfo));
}
////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void BroadcastBool<X, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                               const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                               void *vextraParams) {
  const X *x = reinterpret_cast<const X *>(vx);
  const X *y = reinterpret_cast<const X *>(vy);
  Z *z = reinterpret_cast<Z *>(vz);

  X *extraParams = reinterpret_cast<X *>(vextraParams);

  const int rank = shape::rank(zShapeInfo);  // xRank = yRank = zRank

  switch (rank) {
    case 1:
      execRank1<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
      break;
    case 2:
      execRank2<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
      break;
    case 3:
      execRank3<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
      break;
    case 4:
      execRank4<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
      break;
    case 5:
      execRank5<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
      break;
    default:
      execDefault<X, Z, OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams);
  }
}


}  // namespace broadcast
}  // namespace functions
