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
// Created by raver119 on 17.10.2017.
//
#include <array/ArrayOptions.h>
#include <array/DataTypeUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/Environment.h>

#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
Status LegacyStatsOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x});

  // we assume that opNuk is either stored in block, or was provided via op constructor
  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  // bias goes as first argument, unlike all other reductions
  bool biasCorrected = false;
  if (block.getIArguments()->size() > 0) biasCorrected = INT_ARG(0) > 0;

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyStatsOp");

  if (block.getIArguments()->size() == 1 ||
      (block.getIArguments()->size() == 2 && INT_ARG(1) == DataTypeUtils::max<int>())) {
    // scalar
    NativeOpExecutioner::execSummaryStatsScalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(),
                                                x->specialBuffer(), x->specialShapeInfo(),
                                                extras.argumentsAsT(z->dataType()), z->buffer(), z->shapeInfo(),
                                                z->specialBuffer(), z->specialShapeInfo(), biasCorrected);
  } else {
    // dimensions for TAD
    // skip iArgs[0] because it's biasCorrected, not a dim
    std::vector<LongType> dims(block.getIArguments()->begin() + 1, block.getIArguments()->end());
    for (size_t e = 0; e < dims.size(); e++)
      if (dims[e] < 0) dims[e] += x->rankOf();

    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

    auto packX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);

    auto pTadShape = Environment::getInstance().isCPU()
                         ? packX->primaryShapeInfo()
                         : packX->specialShapeInfo();

    auto pTadOffsets = Environment::getInstance().isCPU()
                           ? packX->primaryOffsets()
                           : packX->specialOffsets();

    NativeOpExecutioner::execSummaryStats(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                          x->specialShapeInfo(), extras.argumentsAsT(z->dataType()), z->buffer(),
                                          z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), dims.data(),
                                          (int)dims.size(), pTadShape, pTadOffsets, biasCorrected);
  }

  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}

LegacyStatsOp::LegacyStatsOp() : LegacyOp(1) {
  //
}

LegacyStatsOp::LegacyStatsOp(int opNum) : LegacyOp(1, opNum) {
  //
}

LegacyOp *LegacyStatsOp::clone() { return new LegacyStatsOp(this->_opNum); }

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
ShapeList *LegacyStatsOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  // Summary-stats (variance, stdev) always produce a floating-point result.
  // Preserve input FP dtype; promote non-FP inputs to the default FP type.
  DataType inDtype = ArrayOptions::dataType(inShape);
  DataType outDtype = DataTypeUtils::isR(inDtype) ? inDtype : Environment::getInstance().defaultFloatDataType();

  // iArgs[0] = biasCorrected, iArgs[1..] = reduction dimensions (matches validateAndExecute).
  // Scalar reduction: no iArgs, just biasCorrected, or biasCorrected + MAX sentinel.
  const auto nIArgs = block.getIArguments()->size();
  if (nIArgs == 0 || nIArgs == 1 ||
      (nIArgs == 2 && INT_ARG(1) == DataTypeUtils::max<int>())) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(outDtype));
  }

  // TAD reduction: skip iArgs[0] (biasCorrected) when gathering dimensions.
  std::vector<sd::LongType> dims(block.getIArguments()->begin() + 1, block.getIArguments()->end());
  auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
  sd::LongType *xShape2 = ShapeUtils::evalReduceShapeInfo('c', &dims, inShape, outDtype, keepDims, false);
  return SHAPELIST(xShape2);
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
