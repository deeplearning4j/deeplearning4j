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
// Created by raver on 4/9/2018.
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/Loops.h>
#include <loops/indexreduce.h>
#include <loops/legacy_ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace indexreduce {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
sd::LongType IndexReduce<X, Y>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                          void *extraParams) {
 RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void IndexReduce<X, Y>::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, void *extraParams, void *z,
                            const sd::LongType *zShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {
 DISPATCH_BY_OPNUM_TT(
     exec, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset),
     INDEX_REDUCE_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
template <typename OpType>
sd::LongType IndexReduce<X, Y>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams) {
 auto x = reinterpret_cast<const X *>(vx);
 auto extraParams = reinterpret_cast<X *>(vextraParams);

 auto startingIndex = OpType::startingIndexValue(x);
 auto len = shape::length(xShapeInfo);
 sd::OmpLaunchHelper info(len);

 // Cache shape-related values
 sd::LongType xRank = shape::rank(xShapeInfo);
 sd::LongType *xShape = shape::shapeOf(xShapeInfo);
 sd::LongType *xStride = shape::stride(xShapeInfo);

 sd::LongType xShapeInfoCast[SD_MAX_RANK];
 bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
 int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
 IndexValue<X> intermediatery[64];
 for (int e = 0; e < maxThreads; e++) intermediatery[e].index = -1;

 auto func = PRAGMA_THREADS_FOR {
   intermediatery[thread_id] = OpType::startingIndexValue(x);

   for (auto i = start; i < stop; i++) {
     sd::LongType coords[SD_MAX_RANK];
     INDEX2COORDS(i, xRank, xShape, coords);
     sd::LongType offset;
     COORDS2INDEX(xRank, xStride, coords, offset);
     IndexValue<X> curr(x[offset], i);
     intermediatery[thread_id] = OpType::update(intermediatery[thread_id], curr, extraParams);
   }
 };

 maxThreads = samediff::Threads::parallel_for(func, 0, len, 1, maxThreads);

 for (int e = 0; e < maxThreads; e++) startingIndex = OpType::update(startingIndex, intermediatery[e], extraParams);

 return startingIndex.index;
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void IndexReduce<X, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                            const sd::LongType *zShapeInfo, sd::LongType *dimension, sd::LongType dimensionLength,
                            const sd::LongType *tadShapeInfo, const sd::LongType *tadOffset) {
 auto x = reinterpret_cast<X *>(const_cast<void *>(vx));
 auto z = reinterpret_cast<Z *>(vz);
 auto extraParams = reinterpret_cast<X *>(vextraParams);

 const sd::LongType zLen = shape::length(zShapeInfo);

 if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
   if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
   const auto indexValue = OpType::startingIndexValue(x);

   for (sd::LongType i = 0; i < zLen; i++) z[i] = (Z)indexValue.index;

   return;
 }

 if (shape::isScalar(zShapeInfo)) {
   z[0] = (Z)execScalar<OpType>(x, xShapeInfo, extraParams);
   return;
 }

 auto tadOnlyShapeInfo = tadShapeInfo;
 auto tadOffsets = tadOffset;

 if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
   if (dimensionLength < 1) return;

   auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(const_cast<sd::LongType*>(xShapeInfo), dimension,
                                                                        dimensionLength);
   tadOnlyShapeInfo = tadPack->primaryShapeInfo();
   tadOffsets = tadPack->primaryOffsets();
 }

 // Let IndexReductionLoops handle the shape caching internally since it's a separate component
 sd::IndexReductionLoops<X, Z>::template loopIndexReduce<OpType>(x, xShapeInfo, z, zShapeInfo, tadOnlyShapeInfo,
                                                                 tadOffsets, vextraParams);
}

}  // namespace indexreduce
}  // namespace functions