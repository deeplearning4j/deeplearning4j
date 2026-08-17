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
#include <array/DataTypeUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <system/env_functions.h>

#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>
namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
Status LegacyReduce3Op::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x, y});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  sd_debug("Executing LegacyReduce3Op: [%i]\n", opNum);

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyReduce3Op");

  // Detect "all dims" reduction: either no iArgs (empty reduce = reduce all),
  // sentinel iArg[0]=INT_MAX (explicit reduce-all), OR iArgs covering every rank
  // of x (DSP-packed dims [0..rank-1]). In any of these cases we must take the
  // scalar fast path — the TAD path below will try to TAD the output with dims
  // that exceed the output's rank (scalar is rank 2 [1,1]).
  bool allDimsReduction = (block.getIArguments()->size() == 0) ||
                          (block.getIArguments()->size() == 1 &&
                           INT_ARG(0) == DataTypeUtils::max<int>()) ||
                          (static_cast<LongType>(block.getIArguments()->size()) ==
                           x->rankOf());
  if (x->isSameShape(y) && allDimsReduction) {
    // reduce3 to scalar
    NativeOpExecutioner::execReduce3Scalar(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        extras.argumentsAsT(z->dataType()), y->buffer(), y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
  } else {
    std::vector<LongType> dims(*block.getAxis());
    for (size_t e = 0; e < dims.size(); e++)
      if (dims[e] < 0) dims[e] += x->rankOf();

    auto packX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);
    auto packY = ConstantTadHelper::getInstance().tadForDimensions(y->shapeInfo(), &dims);

    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions requuired for reduction!");

    auto xTadShape = sd::env_isCPU()
                         ? packX->primaryShapeInfo()
                         : packX->specialShapeInfo();
    auto xTadOffsets = sd::env_isCPU()
                           ? packX->primaryOffsets()
                           : packX->specialOffsets();

    auto yTadShape = sd::env_isCPU()
                         ? packY->primaryShapeInfo()
                         : packY->specialShapeInfo();
    auto yTadOffsets = sd::env_isCPU()
                           ? packY->primaryOffsets()
                           : packY->specialOffsets();

    NativeOpExecutioner::execReduce3(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                     x->specialShapeInfo(), extras.argumentsAsT(z->dataType()), y->buffer(),
                                     y->shapeInfo(), y->specialBuffer(), y->specialShapeInfo(), z->buffer(),
                                     z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), dims.data(),
                                     dims.size(), xTadShape, xTadOffsets, yTadShape, yTadOffsets);
  }

  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);
  return Status::OK;
}


LegacyReduce3Op::LegacyReduce3Op() : LegacyOp(2) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}

LegacyReduce3Op::LegacyReduce3Op(int opNum) : LegacyOp(2, opNum) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}

LegacyOp *LegacyReduce3Op::clone() { return new LegacyReduce3Op(this->_opNum); }

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
ShapeList *LegacyReduce3Op::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto xShape = inputShape->at(0);

  // evalReduceShapeInfo handles all cases correctly: empty iArgs (reduce all),
  // INT_MAX sentinel, full-rank dims list, and partial reductions. It also
  // preserves dtype from xShape rather than leaving it UNKNOWN, avoiding the
  // "Shape info created with invalid data type" failure that happened when the
  // manual scalar-shape buffer below was built with flags=0.
  auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
  sd::LongType *xShape2 =
      ShapeUtils::evalReduceShapeInfo('c', block.getIArguments(), xShape, keepDims, false);
  return SHAPELIST(xShape2);
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
