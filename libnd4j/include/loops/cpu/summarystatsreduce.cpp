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
// Created by raver119 on 18.12.17.
//
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/TAD.h>
#include <helpers/shape.h>
#include <loops/summarystatsreduce.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
namespace summarystats {

template <typename X, typename Y>
Y SummaryStatsReduce<X, Y>::execScalar(const int opNum, const bool biasCorrected, const void *x,
                                       const sd::LongType *xShapeInfo, void *extraParams) {
  RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams), SUMMARY_STATS_OPS);
}

template <typename X, typename Y>
void SummaryStatsReduce<X, Y>::execScalar(const int opNum, const bool biasCorrected, const void *x,
                                          const sd::LongType *xShapeInfo, void *extraParams, void *z,
                                          const sd::LongType *zShapeInfo) {
  DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams, z, zShapeInfo), SUMMARY_STATS_OPS);
}

template <typename X, typename Y>
void SummaryStatsReduce<X, Y>::exec(const int opNum, const bool biasCorrected, const void *x,
                                    const sd::LongType *xShapeInfo, void *extraParams, void *z,
                                    const sd::LongType *zShapeInfo, int *dimension, int dimensionLength) {
  DISPATCH_BY_OPNUM_TT(exec,
                       PARAMS(biasCorrected, x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength),
                       SUMMARY_STATS_OPS);
}

template <typename X, typename Z>
template <typename OpType>
void SummaryStatsReduce<X, Z>::execScalar(const bool biasCorrected, const void *vx, const sd::LongType *xShapeInfo,
                                          void *vextraParams, void *vz, const sd::LongType *zShapeInfo) {
  auto z = reinterpret_cast<Z *>(vz);
  z[0] = execScalar<OpType>(biasCorrected, vx, xShapeInfo, vextraParams);
}

template <typename X, typename Z>
template <typename OpType>
Z SummaryStatsReduce<X, Z>::execScalar(const bool biasCorrected, const void *vx, const sd::LongType *xShapeInfo,
                                       void *vextraParams) {
  auto x = reinterpret_cast<const X *>(vx);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);

  SummaryStatsData<X> startingIndex;
  startingIndex.initialize();
  auto length = shape::length(xShapeInfo);

  sd::Unsigned xShapeInfoCast[SD_MAX_RANK];
  const bool canCast = sd::DataTypeUtils::castShapeInfo<sd::Unsigned>(xShapeInfo, xShapeInfoCast);

  for (sd::LongType i = 0; i < length; i++) {
    auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCast);

    SummaryStatsData<X> curr;
    curr.initWithValue(x[xOffset]);
    startingIndex = update(startingIndex, curr, extraParams);
  }

  return OpType::getValue(biasCorrected, startingIndex);
}

template <typename X, typename Z>
template <typename OpType>
void SummaryStatsReduce<X, Z>::exec(const bool biasCorrected, const void *vx, const sd::LongType *xShapeInfo,
                                    void *vextraParams, void *vz, const sd::LongType *zShapeInfo, int *dimension,
                                    int dimensionLength) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);
  auto extraParams = reinterpret_cast<Z *>(vextraParams);
  auto resultLength = shape::length(zShapeInfo);

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
    SummaryStatsData<X> comp;
    comp.initWithValue(x[0]);

    for (sd::LongType i = 0; i < resultLength; i++) z[i] = OpType::getValue(biasCorrected, comp);
    return;
  }

  if (shape::isScalar(zShapeInfo)) {
    z[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
    return;
  }

  // no-op
  if (dimensionLength < 1) return;

  auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(xShapeInfo, dimension, dimensionLength);

  // pre squeezed: this is for keeping the pointer to the original
  // shape information for tad offset
  // the squeezed information doesn't render the right strides for
  // tad offset
  if (resultLength == 1 || dimensionLength == shape::rank(xShapeInfo) || tadPack.numberOfTads() == 1) {
    z[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
    return;
  }

  auto tadShapeShapeInfo = tadPack.primaryShapeInfo();
  auto tadLength = shape::length(tadPack.primaryShapeInfo());
  auto tadEWS = shape::elementWiseStride(tadPack.primaryShapeInfo());
  auto tadOrder = shape::order(tadPack.primaryShapeInfo());

  sd::Unsigned tadShapeShapeInfoCast[SD_MAX_RANK];
  const bool canCast = tadEWS == 1 && tadOrder == 'c'
                           ? false
                           : sd::DataTypeUtils::castShapeInfo<sd::Unsigned>(tadShapeShapeInfo, tadShapeShapeInfoCast);

  auto func = PRAGMA_THREADS_FOR {
    for (auto r = start; r < stop; r++) {
      auto tadOffsetForBlock = tadPack.primaryOffsets()[r];
      auto tx = x + tadOffsetForBlock;
      SummaryStatsData<X> comp;
      comp.initWithValue(tx[0]);

      if (tadEWS == 1 && tadOrder == 'c') {
        for (sd::LongType i = 1; i < tadLength; i++) {
          SummaryStatsData<X> indexVal2;
          indexVal2.initWithValue(tx[i]);

          comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
        }
      } else {
        for (sd::LongType i = 1; i < tadLength; i++) {
          auto xOffset = shape::indexOffset(i, tadShapeShapeInfo, tadShapeShapeInfoCast, canCast);

          SummaryStatsData<X> indexVal2;
          indexVal2.initWithValue(tx[xOffset]);

          comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
        }
      }

      z[r] = OpType::getValue(biasCorrected, comp);
    }
  };

  samediff::Threads::parallel_tad(func, 0, resultLength, 1);
}

BUILD_DOUBLE_TEMPLATE(template class SummaryStatsReduce, , SD_COMMON_TYPES, SD_FLOAT_TYPES);
}  // namespace summarystats
}  // namespace functions
