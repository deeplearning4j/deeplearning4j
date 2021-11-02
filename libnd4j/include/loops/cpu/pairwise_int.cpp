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
#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <helpers/OmpLaunchHelper.h>
#include <loops/pairwise_int.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace pairwise_transforms {

template <typename X>
void PairWiseIntTransform<X>::exec(const int opNum, const void *x, sd::LongType xEws, const void *y, sd::LongType yEws,
                                   void *z, sd::LongType zEws, void *extraParams, sd::LongType n, const uint64_t start,
                                   const uint64_t stop) {
  DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xEws, y, yEws, z, zEws, extraParams, n, start, stop), PAIRWISE_INT_OPS);
};

template <typename X>
template <typename OpType>
void PairWiseIntTransform<X>::exec(const void *vx, sd::LongType xEws, const void *vy, sd::LongType yEws, void *vz,
                                   sd::LongType zEws, void *vextraParams, const sd::LongType n, const uint64_t start,
                                   const uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<X *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  if (xEws == 1 && yEws == 1 && zEws == 1) {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) z[i] = OpType::op(x[i], y[i], extraParams);
  } else {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) z[i * zEws] = OpType::op(x[i * xEws], y[i * yEws], extraParams);
  }
}

template <typename X>
void PairWiseIntTransform<X>::exec(const int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                   const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                   void *extraParams, const uint64_t start, const uint64_t stop) {
  DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, extraParams, start, stop),
                      PAIRWISE_INT_OPS);
};

template <typename X>
template <typename OpType>
void PairWiseIntTransform<X>::exec(const void *vx, const sd::LongType *xShapeInfo, const void *vy,
                                   const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                                   void *vextraParams, const uint64_t start, const uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<X *>(vz);
  auto extraParams = reinterpret_cast<X *>(vextraParams);

  auto n = shape::length(xShapeInfo);
  auto xEws = shape::elementWiseStride(xShapeInfo);
  auto yEws = shape::elementWiseStride(yShapeInfo);
  auto zEws = shape::elementWiseStride(zShapeInfo);

  if (shape::isScalar(yShapeInfo)) {
    sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
    const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        z[offset] = OpType::op(x[offset], y[0], extraParams);
      };
    } else {
      sd::Unsigned zShapeInfoCast[SD_MAX_RANK];
      const bool canCastZ = sd::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
        z[zOffset] = OpType::op(x[xOffset], y[0], extraParams);
      };
    }
    return;
  }

  const sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);
  const bool sameShapesXY = shape::shapeEquals(xShapeInfo, yShapeInfo);

  if ((kindOfLoop == sd::LoopKind::EWS1 || kindOfLoop == sd::LoopKind::EWSNONZERO) && sameShapesXY) {
    exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, n, start, stop);
  } else if ((kindOfLoop == sd::LoopKind::EWS1 || kindOfLoop == sd::LoopKind::EWSNONZERO) &&
             !sameShapesXY) {  // not same shape
    exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, shape::length(yShapeInfo), start, stop);
  } else {
    if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo) &&
        shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
      sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
      const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        z[offset] = OpType::op(x[offset], y[offset], extraParams);
      };
    } else if (shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
      sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
      sd::Unsigned zShapeInfoCast[SD_MAX_RANK];
      const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      const bool canCastZ = sd::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
        z[zOffset] = OpType::op(x[offset], y[offset], extraParams);
      };
    } else if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
      sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
      sd::Unsigned yShapeInfoCast[SD_MAX_RANK];
      const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      const bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto yOffset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
        z[offset] = OpType::op(x[offset], y[yOffset], extraParams);
      };
    } else if (shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo)) {
      sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
      sd::Unsigned yShapeInfoCast[SD_MAX_RANK];
      const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      const bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto offset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
        z[offset] = OpType::op(x[xOffset], y[offset], extraParams);
      };
    } else {
      sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
      sd::Unsigned yShapeInfoCast[SD_MAX_RANK];
      sd::Unsigned zShapeInfoCast[SD_MAX_RANK];
      const bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
      const bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
      const bool canCastZ = sd::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto yOffset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
        z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
      };
    }
  }
}

BUILD_SINGLE_TEMPLATE(template class PairWiseIntTransform, , SD_INTEGER_TYPES);
}  // namespace pairwise_transforms
}  // namespace functions
