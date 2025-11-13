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

  // For reduce_long operations like MatchCondition, extraParams contain comparison values
  // (e.g., compare value, epsilon) that must remain in input type X, not be converted to Z.
  // Converting double→LongType would truncate 0.5→0, breaking comparisons.
  auto compatibleExtraParams = reinterpret_cast<X*>(vextraParams);

  const sd::LongType length = shape::length(xShapeInfo);

  if (shape::isEmptyConst(xShapeInfo)) {
    z[0] = static_cast<Z>(OpType::startingValue(x));
    return;
  }

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
    const auto startingVal = static_cast<Z>(OpType::startingValue(x));

    for (sd::LongType i = 0; i < length; i++) z[i] = startingVal;
    return;
  }

  int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
  typename OpType::InterType intermediate[64];

  PRAGMA_OMP_SIMD
  for (auto e = 0; e < maxThreads; e++) {
    intermediate[e] = static_cast<typename OpType::InterType>(OpType::startingValue(x));
  }

  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  sd::LongType* xStride = shape::stride(xShapeInfo);


  auto func = PRAGMA_THREADS_FOR {
    sd::LongType coords[SD_MAX_RANK];
    for (auto i = start; i < stop; i++) {
      INDEX2COORDS(i, xRank, xShape, coords);
      sd::LongType indexOffset;
      COORDS2INDEX(xRank, xStride, coords, indexOffset);
      
      intermediate[thread_id] = OpType::update(
          intermediate[thread_id],
          OpType::op(x[indexOffset], compatibleExtraParams),
          compatibleExtraParams);
    }
  };
  maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

  for (int e = 1; e < maxThreads; e++)
    intermediate[0] = OpType::update(intermediate[0], intermediate[e], compatibleExtraParams);

  z[0] = OpType::postProcess(intermediate[0], length, compatibleExtraParams);
}

template <typename X, typename Z>
template <typename OpType>
Z SD_HOST ReduceLongFunction<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams) {
  auto x = reinterpret_cast<const X *>(vx);


  // For reduce_long operations like MatchCondition, extraParams contain comparison values
  // (e.g., compare value, epsilon) that must remain in input type X, not be converted to Z.
  // Converting double→LongType would truncate 0.5→0, breaking comparisons.
  auto compatibleExtraParams = reinterpret_cast<X*>(vextraParams);

  const sd::LongType length = shape::length(xShapeInfo);

  auto startingValue = static_cast<typename OpType::InterType>(OpType::startingValue(x));

  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  sd::LongType* xStride = shape::stride(xShapeInfo);


  sd::LongType coords[SD_MAX_RANK];
  for (sd::LongType i = 0; i < length; i++) {
    INDEX2COORDS(i, xRank, xShape, coords);
    sd::LongType indexOffset;
    COORDS2INDEX(xRank, xStride, coords, indexOffset);

    
    startingValue = OpType::update(startingValue, OpType::op(x[indexOffset], compatibleExtraParams), compatibleExtraParams);
  }
  
  Z result = OpType::postProcess(startingValue, length, compatibleExtraParams);
  return result;
}

template <typename X, typename Z>
Z ReduceLongFunction<X, Z>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                       void *extraParams) {
  RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_LONG_OPS);
}

template <typename X, typename Z>
void ReduceLongFunction<X, Z>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
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


  // For reduce_long operations like MatchCondition, extraParams contain comparison values
  // (e.g., compare value, epsilon) that must remain in input type X, not be converted to Z.
  // Converting double→LongType would truncate 0.5→0, breaking comparisons.
  auto compatibleExtraParams = reinterpret_cast<X*>(vextraParams);

  int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
  typename OpType::InterType intermediate[64];

  PRAGMA_OMP_SIMD
  for (auto e = 0; e < maxThreads; e++) {
    intermediate[e] = static_cast<typename OpType::InterType>(OpType::startingValue(x));
  }


  auto func = PRAGMA_THREADS_FOR {
    if (xEws == 1) {
      for (auto i = start; i < stop; i++) {
        intermediate[thread_id] = OpType::update(intermediate[thread_id], OpType::op(x[i], compatibleExtraParams), compatibleExtraParams);
      }
    } else {
      for (auto i = start; i < stop; i++) {
        intermediate[thread_id] =
            OpType::update(intermediate[thread_id], OpType::op(x[i * xEws], compatibleExtraParams), compatibleExtraParams);
      }
    }
  };

  maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

  for (int e = 1; e < maxThreads; e++)
    intermediate[0] = OpType::update(intermediate[0], intermediate[e], compatibleExtraParams);

  Z result = OpType::postProcess(intermediate[0], length, compatibleExtraParams);
  return result;
}

template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceLongFunction<X, Z>::exec(sd::memory::Workspace *workspace, const void *vx,
                                            const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                            const sd::LongType *zShapeInfo, const sd::LongType *dims) {
  const X *x = reinterpret_cast<const X *>(vx);
  Z *z = reinterpret_cast<Z *>(vz);

  // For reduce_long operations like MatchCondition, extraParams contain comparison values
  // (e.g., compare value, epsilon) that must remain in input type X, not be converted to Z.
  // Converting double→LongType would corrupt the comparison values.
  auto compatibleExtraParams = reinterpret_cast<X*>(vextraParams);

  const int xRank = shape::rank(xShapeInfo);
  const int zRank = shape::rank(zShapeInfo);


  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    const auto startingVal = static_cast<Z>(OpType::startingValue(x));
    const auto zLen = shape::length(zShapeInfo);

    for (sd::LongType i = 0; i < zLen; i++) z[i] = startingVal;
    return;
  }

  if (shape::length(zShapeInfo) == 1) {
    z[0] = execScalar<OpType>(x, xShapeInfo, vextraParams);
    return;
  }

  if (OpType::requiresSpecialAccumulation) {
    // For execSpecial, use original LongType* extraParams as expected by that API
    auto originalParams = reinterpret_cast<sd::LongType *>(vextraParams);
    OpType::execSpecial(x, xShapeInfo, originalParams, z, zShapeInfo, const_cast<sd::LongType *>(dims) + zRank,
                        xRank - zRank, nullptr, nullptr);
    return;
  }
  // Call ReductionLongLoops with properly typed X* extraParams
  sd::ReductionLongLoops<X, Z>::template innerloopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, compatibleExtraParams);

}

template <typename X, typename Z>
void ReduceLongFunction<X, Z>::exec(const int opNum, sd::memory::Workspace *workspace, const void *vx,
                                    const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                    const sd::LongType *zShapeInfo,  sd::LongType *dims) {

  DISPATCH_BY_OPNUM_TT(exec, PARAMS(workspace, vx, xShapeInfo, vextraParams, vz, zShapeInfo, dims), REDUCE_LONG_OPS);
}

}  // namespace reduce
}  // namespace functions
