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
                                         const sd::LongType *zShapeInfo, const void *vscalars, long long int *dimension,
                                         int dimensionLength, const sd::LongType *xTadShapeInfo,
                                         const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                         const sd::LongType *zTadOffsets, uint64_t start, uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalars = reinterpret_cast<const Y *>(vscalars);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  if (zTadShapeInfo == nullptr) {
    zTadShapeInfo = xTadShapeInfo;
    zTadOffsets = xTadOffsets;
  }

  const int xTadEws = shape::elementWiseStride(xTadShapeInfo);
  const int zTadEws = shape::elementWiseStride(zTadShapeInfo);
  const int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
  const int numTads = shape::length(xShapeInfo) / tadLength;

  sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXZ(xTadShapeInfo, zTadShapeInfo);

  if (kindOfLoop != sd::LoopKind::EWS1 && kindOfLoop != sd::LoopKind::EWSNONZERO) {
    printf("ScalarTransform<X, Z>::transform: super-bad loop visited. Shouldn't ever happen\n");
    return;
  }

  int num_threads = sd::math::sd_min<int>(numTads, sd::Environment::getInstance().maxThreads());

  if (kindOfLoop == sd::LoopKind::EWS1) {
    for (auto r = start; r < stop; r++) {
      auto oZ = z + zTadOffsets[r];
      auto oX = x + xTadOffsets[r];

      PRAGMA_OMP_SIMD
      for (int f = 0; f < tadLength; f++) oZ[f] = OpType::op(oX[f], scalars[r], extraParams);
    };
  } else {
    for (auto r = start; r < stop; r++) {
      auto oZ = z + zTadOffsets[r];
      auto oX = x + xTadOffsets[r];

      PRAGMA_OMP_SIMD
      for (int f = 0; f < tadLength; f++) oZ[f * zTadEws] = OpType::op(oX[f * xTadEws], scalars[r], extraParams);
    };
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                         void *z, const sd::LongType *zShapeInfo, const void *scalars,
                                         long long int *dimension,
                                         int dimensionLength, const sd::LongType *xTadShapeInfo,
                                         const sd::LongType *xTadOffsets, const sd::LongType *zTadShapeInfo,
                                         const sd::LongType *zTadOffsets,
                                         uint64_t start, uint64_t stop) {
  DISPATCH_BY_OPNUM_TTT(transform,
                        PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength,
                               xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets, start, stop),
                        SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum, const void *x, sd::LongType xStride, void *z,
                                         sd::LongType zStride, const void *scalar, void *extraParams, const uint64_t n,
                                         const uint64_t start, const uint64_t stop) {
  DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xStride, z, zStride, scalar, extraParams, n, start, stop), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum, const void *x, const sd::LongType *xShapeInfo, void *z,
                                         const sd::LongType *zShapeInfo, const void *scalar, void *extraParams,
                                         const uint64_t start, const uint64_t stop) {
  DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams, start, stop), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
template <typename OpType>
void ScalarTransform<X, Y, Z>::transform(const void *vx, const sd::LongType *xShapeInfo, void *vz,
                                         const sd::LongType *zShapeInfo, const void *vscalar, void *vextraParams,
                                         const uint64_t start, const uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalar = reinterpret_cast<const Y *>(vscalar)[0];
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  const auto len = shape::length(xShapeInfo);
  const auto xEws = shape::elementWiseStride(xShapeInfo);
  const auto zEws = shape::elementWiseStride(zShapeInfo);

  sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

  if (kindOfLoop == sd::LoopKind::EWS1 || kindOfLoop == sd::LoopKind::EWSNONZERO) {
    transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len, start, stop);
  } else {
    sd::LongType xShapeInfoCast[SD_MAX_RANK];
    const bool canCastX = sd::DataTypeUtils::castShapeInfo<sd::LongType>(xShapeInfo, xShapeInfoCast);

    if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        z[offset] = OpType::op(x[offset], scalar, extraParams);
      };
    } else {
      sd::LongType zShapeInfoCast[SD_MAX_RANK];
      const bool canCastZ = sd::DataTypeUtils::castShapeInfo<sd::LongType>(zShapeInfo, zShapeInfoCast);

      PRAGMA_OMP_SIMD
      for (auto i = start; i < stop; i++) {
        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
        z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
      };
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
template <typename OpType>
void ScalarTransform<X, Y, Z>::transform(const void *vx, sd::LongType xEws, void *vz, sd::LongType zEws,
                                         const void *vscalar, void *vextraParams, const uint64_t len,
                                         const uint64_t start, const uint64_t stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto scalar = reinterpret_cast<const Y *>(vscalar)[0];
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  if (xEws == 1 && zEws == 1) {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) z[i] = OpType::op(x[i], scalar, extraParams);
  } else {
    PRAGMA_OMP_SIMD
    for (auto i = start; i < stop; i++) z[i * zEws] = OpType::op(x[i * xEws], scalar, extraParams);
  }
}

}  // namespace scalar
}  // namespace functions
