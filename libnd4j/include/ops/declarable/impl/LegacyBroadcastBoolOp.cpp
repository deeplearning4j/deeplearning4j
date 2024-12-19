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
#include <helpers/PointersManager.h>
#include <helpers/TAD.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
Status LegacyBroadcastBoolOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> dims(*block.getIArguments());
  if (dims.size() > 0) std::sort(dims.begin(), dims.end());

  NDArray::prepareSpecialUse({z}, {x, y});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);

  PointersManager manager(block.launchContext(), "LegacyBroadcastBoolOp");
  auto pTadShape = Environment::getInstance().isCPU()
                   ? packX->primaryShapeInfo()
                   : packX->specialShapeInfo();
  auto pTadOffsets = Environment::getInstance().isCPU()
                     ? packX->primaryOffsets()
                     : packX->specialOffsets();

  REQUIRE_TRUE(shape::length(packX->primaryShapeInfo()) == y->lengthOf(), 0,
               "Length of broadcast TAD should be equal to length of Y operand, but got [%i] vs [%i]",
               (int)shape::length(packX->primaryShapeInfo()), (int)y->lengthOf());

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
                       : packZ->specialOffsets();  //(sd::LongType *) manager.replicatePointer(tadZ.tadOffsets,

    NativeOpExecutioner::execBroadcast(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                       x->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                       y->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                       z->specialShapeInfo(), dims.data(), dims.size(), pTadShape, pTadOffsets,
                                       zTadShape, zTadOffsets);
  }

  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}

LegacyBroadcastBoolOp::LegacyBroadcastBoolOp() : LegacyOp(2) {
  //
}

LegacyBroadcastBoolOp::LegacyBroadcastBoolOp(int opNum) : LegacyOp(2, opNum) {
  //
}

LegacyOp *LegacyBroadcastBoolOp::clone() { return new LegacyBroadcastBoolOp(this->_opNum); }

/**
 *   If external NDArray wasn't specified - the same shape is returned by all broadcast ops.
 */
ShapeList *LegacyBroadcastBoolOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().castToDataType(inShape, BOOL));
  return ret;
}
}  // namespace ops
}  // namespace sd
