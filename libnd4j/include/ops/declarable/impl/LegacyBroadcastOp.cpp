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
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
Status LegacyBroadcastOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x, y});

  std::vector<LongType> dims(*block.getAxis());
  if (dims.size() == 0 && block.width() > 2) {
    auto axis = INPUT_VARIABLE(2);
    helpers::adjustAxis(x->rankOf(), axis, dims);
  }
  if (dims.size() > 0) std::sort(dims.begin(), dims.end());

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);

  auto tadLen = shape::length(packX->primaryShapeInfo());
  REQUIRE_TRUE(tadLen == y->lengthOf(), 0,
               "Length of broadcast TAD should be equal to length of Y operand, but got [%i] vs [%i]", tadLen,
               (int)y->lengthOf());

  PointersManager manager(block.launchContext(), "LegacyBroadcastOp");
  auto pTadShape = Environment::getInstance().isCPU()
                       ? packX->primaryShapeInfo()
                       : packX->specialShapeInfo();
  auto pTadOffsets = Environment::getInstance().isCPU()
                         ? packX->primaryOffsets()
                         : packX->specialOffsets();

  if (x == z)
    NativeOpExecutioner::execBroadcast(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                       x->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                       y->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                       z->specialShapeInfo(), dims.data(), dims.size(), pTadShape, pTadOffsets,
                                       pTadShape, pTadOffsets);
  else {
    // this is rare, but possible use case - X and Z might have different shapes/strides/orders. In this case we prepare
    // and pass separate TAD info
    auto packZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), &dims);

    auto zTadShape = Environment::getInstance().isCPU()
                         ? packZ->primaryShapeInfo()
                         : packZ->specialShapeInfo();
    auto zTadOffsets = Environment::getInstance().isCPU()
                           ? packZ->primaryOffsets()
                           : packZ->specialOffsets();

    NativeOpExecutioner::execBroadcast(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                       x->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                       y->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                       z->specialShapeInfo(), dims.data(), dims.size(), pTadShape, pTadOffsets,
                                       zTadShape, zTadOffsets);
  }

  manager.synchronize();

  traceExecIfNeeded(block);

  STORE_RESULT(*z);

  return Status::OK;
}

LegacyBroadcastOp::LegacyBroadcastOp() : LegacyOp(2) {
  //
}

LegacyBroadcastOp::LegacyBroadcastOp(int opNum) : LegacyOp(2, opNum) {
  //
}

LegacyOp *LegacyBroadcastOp::clone() { return new LegacyBroadcastOp(this->_opNum); }

/**
 *   If external NDArray wasn't specified - the same shape is returned by all broadcast ops.
 */
ShapeList *LegacyBroadcastOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  // FIXME: remove memcpy
  LongType *newShape;
  ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), sd::LongType);
  memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

  return SHAPELIST(CONSTANT(newShape));
}
}  // namespace ops
}  // namespace sd
