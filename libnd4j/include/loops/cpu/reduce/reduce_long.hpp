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
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/Loops.h>
#include <helpers/OmpLaunchHelper.h>
#include <loops/legacy_ops.h>
#include <loops/reduce_long.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace reduce {
template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceLongFunction<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams,
                                                 void *vz, const sd::LongType *zShapeInfo) {
 auto x = reinterpret_cast<const X *>(vx);
 auto z = reinterpret_cast<Z *>(vz);
 auto extraParams = reinterpret_cast<X *>(vextraParams);

 const sd::LongType length = shape::length(xShapeInfo);

 if (shape::isEmptyConst(xShapeInfo)) {
   z[0] = OpType::startingValue(x);
   return;
 }

 if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
   if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
   const auto startingVal = OpType::startingValue(x);

   for (sd::LongType i = 0; i < length; i++) z[i] = startingVal;
   return;
 }

 sd::LongType xShapeInfoCast[SD_MAX_RANK];
 int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
 Z intermediate[64];

 PRAGMA_OMP_SIMD
 for (auto e = 0; e < maxThreads; e++) intermediate[e] = OpType::startingValue(x);

 sd::LongType xRank = shape::rank(xShapeInfo);
 sd::LongType* xShape = shape::shapeOf(xShapeInfo);
 sd::LongType* xStride = shape::stride(xShapeInfo);


 auto func = PRAGMA_THREADS_FOR {
   for (auto i = start; i < stop; i++) {
     sd::LongType coords[SD_MAX_RANK];
     INDEX2COORDS(i, xRank, xShape, coords);
     sd::LongType indexOffset;
     COORDS2INDEX(xRank, xStride, coords, indexOffset);
     intermediate[thread_id] = OpType::update(
         intermediate[thread_id],
         OpType::op(x[indexOffset], extraParams),
         extraParams);
   }
 };
 maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

 for (int e = 1; e < maxThreads; e++)
   intermediate[0] = OpType::update(intermediate[0], intermediate[e], extraParams);

 z[0] = OpType::postProcess(intermediate[0], length, extraParams);
}

template <typename X, typename Z>
template <typename OpType>
Z SD_HOST ReduceLongFunction<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams) {
 auto x = reinterpret_cast<const X *>(vx);
 auto extraParams = reinterpret_cast<X *>(vextraParams);

 const sd::LongType length = shape::length(xShapeInfo);
 auto xEws = shape::elementWiseStride(xShapeInfo);

 if (xEws >= 1) {
   return execScalar<OpType>(x, xEws, length, extraParams);
 } else {
   auto startingValue = OpType::startingValue(x);
   sd::LongType xShapeInfoCast[SD_MAX_RANK];

   sd::LongType xRank = shape::rank(xShapeInfo);
   sd::LongType* xShape = shape::shapeOf(xShapeInfo);
   sd::LongType* xStride = shape::stride(xShapeInfo);

   for (sd::LongType i = 0; i < length; i++) {
     sd::LongType coords[SD_MAX_RANK];
     INDEX2COORDS(i, xRank, xShape, coords);
     sd::LongType indexOffset;
     COORDS2INDEX(xRank, xStride, coords, indexOffset);
     startingValue = OpType::update(startingValue, OpType::op(x[indexOffset], extraParams), extraParams);
   }
   return OpType::postProcess(startingValue, length, extraParams);
 }
}

template <typename X, typename Y>
Y ReduceLongFunction<X, Y>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                      void *extraParams) {
 RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_LONG_OPS);
}

template <typename X, typename Y>
void ReduceLongFunction<X, Y>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                         void *extraParams, void *z, const sd::LongType *zShapeInfo) {
 DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_LONG_OPS);
}

template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceLongFunction<X, Z>::exec(const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                           void *vresult, const sd::LongType *resultShapeInfo) {
 auto z = reinterpret_cast<Z *>(vresult);
 z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
}

template <typename X, typename Z>
template <typename OpType>
Z SD_HOST ReduceLongFunction<X, Z>::execScalar(const void *vx, sd::LongType xEws, sd::LongType length,
                                              void *vextraParams) {
 auto x = reinterpret_cast<const X *>(vx);
 auto extraParams = reinterpret_cast<X *>(vextraParams);
 int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
 Z intermediate[64];

 PRAGMA_OMP_SIMD
 for (auto e = 0; e < maxThreads; e++) intermediate[e] = OpType::startingValue(x);

 auto func = PRAGMA_THREADS_FOR {
   if (xEws == 1) {
     for (auto i = start; i < stop; i++)
       intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i], extraParams), extraParams);
   } else {
     for (auto i = start; i < stop; i++)
       intermediate[thread_id] =
           OpType::update(intermediate[thread_id], OpType::op(x[i * xEws], extraParams), extraParams);
   }
 };

 maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

 for (int e = 1; e < maxThreads; e++)
   intermediate[0] = OpType::update(intermediate[0], intermediate[e], extraParams);

 return OpType::postProcess(intermediate[0], length, extraParams);
}

template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceLongFunction<X, Z>::exec(sd::memory::Workspace *workspace, const void *vx,
                                           const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                           const sd::LongType *zShapeInfo, const sd::LongType *dims) {
 const X *x = reinterpret_cast<const X *>(vx);
 Z *z = reinterpret_cast<Z *>(vz);
 sd::LongType *extraParams = reinterpret_cast<sd::LongType *>(vextraParams);

 const int xRank = shape::rank(xShapeInfo);
 const int zRank = shape::rank(zShapeInfo);

 if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
   const auto startingVal = OpType::startingValue(x);
   const auto zLen = shape::length(zShapeInfo);

   for (sd::LongType i = 0; i < zLen; i++) z[i] = startingVal;
   return;
 }

 if (shape::length(zShapeInfo) == 1) {
   // Convert sd::LongType* to X* for the scalar operation
   X convertedExtraParams[3];  // Most ops need at most 3 extra params
   if (extraParams != nullptr) {
     // Convert the first few long parameters to X type
     // This handles common cases where we need type conversion
     for (int i = 0; i < 3; i++) {
       convertedExtraParams[i] = static_cast<X>(extraParams[i]);
     }
   }
   z[0] = execScalar<OpType>(x, xShapeInfo, extraParams != nullptr ? convertedExtraParams : nullptr);
   return;
 }

 if (OpType::requiresSpecialAccumulation) {
   OpType::execSpecial(x, xShapeInfo, extraParams, z, zShapeInfo, const_cast<sd::LongType *>(dims) + zRank,
                       xRank - zRank, nullptr, nullptr);
   return;
 }

 // Convert sd::LongType* to X* for ReductionLongLoops which expects X* extraParams
 X convertedExtraParams[3];  // Most ops need at most 3 extra params
 X* convertedExtraParamsPtr = nullptr;
 if (extraParams != nullptr) {
   // Convert the first few long parameters to X type
   for (int i = 0; i < 3; i++) {
     convertedExtraParams[i] = static_cast<X>(extraParams[i]);
   }
   convertedExtraParamsPtr = convertedExtraParams;
 }

 // Fix: Use ReductionLongLoops directly without #ifdef guards - the implementation will handle inlining
 sd::ReductionLongLoops<X, Z>::template innerloopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, convertedExtraParamsPtr);
}

template <typename X, typename Y>
void ReduceLongFunction<X, Y>::exec(const int opNum, sd::memory::Workspace *workspace, const void *vx,
                                   const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                   const sd::LongType *zShapeInfo,  sd::LongType *dims) {
 DISPATCH_BY_OPNUM_TT(exec, PARAMS(workspace, vx, xShapeInfo, vextraParams, vz, zShapeInfo, dims), REDUCE_LONG_OPS);
}
}  // namespace reduce
}  // namespace functions