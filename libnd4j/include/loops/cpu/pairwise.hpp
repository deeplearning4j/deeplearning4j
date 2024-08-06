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
// Created by remote on 2018-09-20.
//
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/shape.h>
#include <loops/pairwise_transform.h>
#include <math/templatemath.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace pairwise_transforms {

template <typename X, typename Y, typename Z>
void PairWiseTransform<X, Y, Z>::exec(int opNum,
                                      const void *x,
                                      sd::LongType xEws,
                                      const void *y,
                                      sd::LongType yEws,
                                      void *z,
                                      sd::LongType zEws,
                                      void *extraParams,
                                      sd::LongType n,
                                      sd::LongType start,
                                      sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x, xEws, y, yEws, z, zEws, extraParams, n, start, stop), PAIRWISE_TRANSFORM_OPS);
};

template <typename X, typename Y, typename Z>
template <typename OpType>
void PairWiseTransform<X, Y, Z>::exec(const void *vx, sd::LongType xEws, const void *vy, sd::LongType yEws, void *vz,
                                      sd::LongType zEws, void *vextraParams, sd::LongType n, sd::LongType start,
                                      sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Z *>(vz);

  auto extraParams = reinterpret_cast<Z *>(vextraParams);


  if (xEws == 1 && yEws == 1 && zEws == 1) {
    for (sd::LongType i = start; i < stop; i++) {
      z[i] = OpType::op(x[i], y[i], extraParams);
    }

  } else {
    for (sd::LongType i = start; i < stop; i++) z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], extraParams);
  }


}

template <typename X, typename Y, typename Z>
void PairWiseTransform<X, Y, Z>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                      const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                      void *extraParams, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, start, stop),
                        PAIRWISE_TRANSFORM_OPS);
};

template <typename X, typename Y, typename Z>
template <typename OpType>
void PairWiseTransform<X, Y, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                      const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                      void *vextraParams, sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const Y *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  auto n = shape::length(xShapeInfo);
  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);
  bool allSameOrder = true;
  if(yShapeInfo != nullptr) {
    allSameOrder = shape::order(xShapeInfo) == shape::order(yShapeInfo) && shape::order(yShapeInfo) == shape::order(zShapeInfo);
  } else {
    allSameOrder = shape::order(xShapeInfo) == shape::order(zShapeInfo);
  }

  if (shape::isScalar(yShapeInfo)) {
    sd::LongType xShapeInfoCast[SD_MAX_RANK];
    const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
      for (sd::LongType i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        z[offset] = OpType::op(x[offset], y[0], extraParams);
      };
    } else {
      sd::LongType zShapeInfoCast[SD_MAX_RANK];
      const bool canCastZ = sd::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);
      sd::LongType zCoords[SD_MAX_RANK];

      PRAGMA_OMP_SIMD
      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coords(i, zShapeInfo, zCoords);
        auto xOffset = shape::getOffset(xShapeInfo, zCoords);
        auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        z[zOffset] = OpType::op(x[xOffset], y[0], extraParams);
      };
    }
    return;
  }

  const sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);
  const bool sameShapesXY = shape::shapeEquals(xShapeInfo, yShapeInfo);

  if ((kindOfLoop == sd::LoopKind::EWS1 ||
       kindOfLoop == sd::LoopKind::EWSNONZERO) && sameShapesXY
      && !shape::isViewConst(xShapeInfo) && !shape::isViewConst(yShapeInfo)
      && !shape::isViewConst(zShapeInfo)
      && allSameOrder
       && shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)
      && shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
    printf("pairwise broadcast case 1\n");
    fflush(stdout);
    exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, n, start, stop);
  } else if ((kindOfLoop == sd::LoopKind::EWS1 || kindOfLoop == sd::LoopKind::EWSNONZERO) &&
             !sameShapesXY
             && allSameOrder) {  // not same shape
    printf("pairwise broadcast case 2\n");
    fflush(stdout);
    exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, shape::length(yShapeInfo), start, stop);
  } else {

    if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)
        && shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)
        && !shape::isViewConst(xShapeInfo)
        &&  !shape::isViewConst(yShapeInfo) && !shape::isViewConst(zShapeInfo)
        && allSameOrder) {
      printf("pairwise broadcast case 3\n");
      fflush(stdout);
      sd::LongType xShapeInfoCast[SD_MAX_RANK];
      bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      sd::LongType zCoords[SD_MAX_RANK];

      PRAGMA_OMP_SIMD
      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coords(i, zShapeInfo, zCoords);
        auto xOffset = shape::getOffset(xShapeInfo, zCoords);
        auto yOffset = shape::getOffset(yShapeInfo, zCoords);
        auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
      }
    } else if ((shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)
                || shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo))
               &&  allSameOrder) {
      printf("pairwise broadcast case 4\n");
      fflush(stdout);
      //general case. note we use to do element wise stride here
      //but for some cases of views it doesn't work
      sd::LongType zCoords[SD_MAX_RANK];
      PRAGMA_OMP_SIMD
      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coords(i, zShapeInfo, zCoords);
        auto xOffset = shape::getOffset(xShapeInfo, zCoords);
        auto yOffset = shape::getOffset(yShapeInfo, zCoords);
        auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
      };
    } else if (shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo)
               && !shape::isViewConst(xShapeInfo)
               && !shape::isViewConst(yShapeInfo) && !shape::isViewConst(zShapeInfo)
               && allSameOrder) {
      printf("pairwise broadcast case 5\n");
      fflush(stdout);
      sd::LongType xShapeInfoCast[SD_MAX_RANK];
      sd::LongType yShapeInfoCast[SD_MAX_RANK];
      bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
      sd::LongType zCoords[SD_MAX_RANK];

      PRAGMA_OMP_SIMD
      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coords(i, zShapeInfo, zCoords);
        auto xOffset = shape::getOffset(xShapeInfo, zCoords);
        auto yOffset = shape::getOffset(yShapeInfo, zCoords);
        auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
      };
    } else {
      printf("pairwise broadcast case 6\n");
      fflush(stdout);
      sd::LongType zCoords[SD_MAX_RANK];

      for (sd::LongType i = start; i < stop; i++) {
        shape::index2coords(i, zShapeInfo, zCoords);

        auto xOffset = shape::getOffset(xShapeInfo, zCoords);
        auto yOffset = shape::getOffset(yShapeInfo, zCoords);
        auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
      };
    }
  }
}
}  // namespace pairwise_transforms
}  // namespace functions
