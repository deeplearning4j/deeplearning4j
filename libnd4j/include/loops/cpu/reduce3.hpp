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

// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.11.2018

#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/Loops.h>
#include <loops/legacy_ops.h>
#include <loops/reduce3.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace reduce3 {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void Reduce3<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                               const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  auto length = shape::length(xShapeInfo);

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY ||
      sd::ArrayOptions::arrayType(yShapeInfo) == sd::ArrayType::EMPTY) {
    if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
    const auto startingVal = OpType::startingValue(x);

    for (sd::LongType i = 0; i < length; i++) z[i] = startingVal;
    return;
  }

  // Cache shape-related values
  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType yRank = shape::rank(yShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *yShape = shape::shapeOf(yShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);
  sd::LongType *yStride = shape::stride(yShapeInfo);

  Z extraParamsVals[3] = {(Z)0.0f, (Z)0.0f, (Z)0.0f};
  Z startingVal = OpType::startingValue(x);
  int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
  Z intermediate[64];
  Z extraParamsLocal[3 * 64];

  PRAGMA_OMP_SIMD
  for (int e = 0; e < maxThreads; e++) intermediate[e] = startingVal;

  memset(extraParamsLocal, 0, 3 * 64 * sizeof(Z));
  if (extraParams != nullptr) {
    PRAGMA_OMP_SIMD
    for (int e = 0; e < maxThreads; e++) {
      extraParamsLocal[3 * e] = extraParams[0];
      extraParamsLocal[3 * e + 1] = extraParams[1];
      extraParamsLocal[3 * e + 2] = extraParams[2];
    }
  }

  auto func = PRAGMA_THREADS_FOR {
    for (auto i2 = start; i2 < stop; i2++) {
      sd::LongType coords[SD_MAX_RANK];
      sd::LongType yCoords[SD_MAX_RANK];

      INDEX2COORDS(i2, xRank, xShape, coords);
      INDEX2COORDS(i2, yRank, yShape, yCoords);
      sd::LongType xOffset = 0, yOffset = 0;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      COORDS2INDEX(yRank, yStride, yCoords, yOffset);

      intermediate[thread_id] = OpType::update(intermediate[thread_id],
                                               OpType::op(x[xOffset], y[yOffset], extraParamsLocal + 3 * thread_id),
                                               extraParamsLocal + 3 * thread_id);
    }
  };

  maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

  // merge step
  for (int e = 0; e < maxThreads; e++) OpType::aggregateExtraParams(extraParamsVals, extraParamsLocal + 3 * e);

  for (int e = 0; e < maxThreads; e++) startingVal = OpType::update(startingVal, intermediate[e], extraParamsVals);

  // writing out result
  z[0] = OpType::postProcess(startingVal, length, extraParamsVals);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X, Y>::execScalar(const int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals,
                               const void *vy, const sd::LongType *yShapeInfo, void *vz,
                               const sd::LongType *zShapeInfo) {
  DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo),
                       REDUCE3_OPS);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void Reduce3<X, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                         const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                         sd::LongType *dimension,
                         sd::LongType dimensionLength, sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  if (shape::isScalar(zShapeInfo)) {
    execScalar<OpType>(vx, xShapeInfo, vextraParams, vy, yShapeInfo, vz, zShapeInfo);
    return;
  }

#ifdef SD_LOOPS_INLINED
  sd::Reduction3Loops<X, Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                                                          dimensionLength, extraParams, start, stop);
#else
  sd::Reduction3Loops<X, Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                                                               dimensionLength, extraParams, start, stop);
#endif
}

//////////////////////////////////////////////////////////////////////////
// Rest of the functions remain the same as they primarily dispatch to other functions
// or use the Reduction3Loops class which handles its own shape caching internally

template <typename X, typename Z>
template <typename OpType>
void Reduce3<X, Z>::exec(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                         const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                         sd::LongType *dimension,
                         sd::LongType dimensionLength, const sd::LongType *tadShapeInfo, const sd::LongType *tadOffsets,
                         sd::LongType start, sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);
#ifdef SD_LOOPS_INLINED
  sd::Reduction3Loops<X, Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                                                          dimensionLength, extraParams, start, stop);
#else
  sd::Reduction3Loops<X, Z>::template innerloopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension,
                                                               dimensionLength, extraParams, start, stop);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
void Reduce3<X, Z>::execAll(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams, const void *vy,
                            const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                            sd::LongType *dimension,
                            sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo, const sd::LongType *xOffsets,
                            const sd::LongType *yTadShapeInfo, const sd::LongType *yOffsets, sd::LongType start,
                            sd::LongType stop) {
  auto x = reinterpret_cast<const X *>(vx);
  auto y = reinterpret_cast<const X *>(vy);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

#ifdef SD_LOOPS_INLINED
  sd::Reduction3Loops<X, Z>::template loopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo,
                                                             xOffsets, yTadShapeInfo, yOffsets, extraParams, start,
                                                             stop);
#else
  sd::Reduction3Loops<X, Z>::template innerloopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,
                                                                  xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets,
                                                                  extraParams, start, stop);
#endif
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X, Y>::exec(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals,
                         const void *vy, const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                         sd::LongType *dimension, sd::LongType dimensionLength, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(
      exec,
      PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength, start, stop),
      REDUCE3_OPS);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X, Y>::exec(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals,
                         const void *vy, const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                         sd::LongType *dimension, sd::LongType dimensionLength, const sd::LongType *tadShapeInfo,
                         const sd::LongType *tadOffsets, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(exec,
                       PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension,
                              dimensionLength, tadShapeInfo, tadOffsets, start, stop),
                       REDUCE3_OPS);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X, Y>::execAll(int opNum, const void *vx, const sd::LongType *xShapeInfo, void *extraParamsVals,
                            const void *vy, const sd::LongType *yShapeInfo, void *vz, const sd::LongType *zShapeInfo,
                            long long int *dimension, sd::LongType dimensionLength, const sd::LongType *xTadShapeInfo,
                            const sd::LongType *xOffsets, const sd::LongType *yTadShapeInfo,
                            const sd::LongType *yOffsets, sd::LongType start, sd::LongType stop) {
  DISPATCH_BY_OPNUM_TT(execAll,
                       PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension,
                              dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets, start, stop),
                       REDUCE3_OPS);
}

}  // namespace reduce3
}  // namespace functions