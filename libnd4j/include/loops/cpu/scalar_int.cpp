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

 // tad preparation
 const int xTadEws = shape::elementWiseStride(xTadShapeInfo);
 const int zTadEws = shape::elementWiseStride(zTadShapeInfo);
 const int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
 const int numTads = shape::length(xShapeInfo) / tadLength;

 sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXZ(xTadShapeInfo, zTadShapeInfo);

 if (kindOfLoop != sd::LoopKind::EWS1 && kindOfLoop != sd::LoopKind::EWSNONZERO) {
   printf("ScalarIntTransform<X>::transform: super-bad loop visited. Shouldn't ever happen\n");
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

 auto xEws = shape::elementWiseStride(xShapeInfo);
 auto zEws = shape::elementWiseStride(zShapeInfo);
 auto len = shape::length(xShapeInfo);

 sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

 if (kindOfLoop == sd::LoopKind::EWS1 || kindOfLoop == sd::LoopKind::EWSNONZERO) {
   transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len, start, stop);
   return;
 }

 if (shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
   PRAGMA_OMP_SIMD
   for (auto i = start; i < stop; i++) {
     sd::LongType coords[SD_MAX_RANK];
     shape::index2coords(i, xShapeInfo, coords);
     auto offset = shape::getOffset(xShapeInfo, coords);
     z[offset] = OpType::op(x[offset], scalar, extraParams);
   };
 } else {
   PRAGMA_OMP_SIMD
   for (auto i = start; i < stop; i++) {
     sd::LongType coords[SD_MAX_RANK];
     shape::index2coords(i, xShapeInfo, coords);
     auto xOffset = shape::getOffset(xShapeInfo, coords);
     auto zOffset = shape::getOffset(zShapeInfo, coords);
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

 if (scalar < (sizeof(X) * 8)) {
   if (xEws == 1 && zEws == 1) {
     for (auto i = start; i < stop; i++) z[i] = OpType::op(x[i], scalar, extraParams);
   } else {
     for (auto i = start; i < stop; i++) z[i * zEws] = OpType::op(x[i * xEws], scalar, extraParams);
   }
 }
}

BUILD_SINGLE_TEMPLATE(template class ScalarIntTransform, , SD_INTEGER_TYPES);

}  // namespace scalar
}  // namespace functions