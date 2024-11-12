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
// Created by raver119 on 16.10.2017.
//
#include <legacy/NativeOpExecutioner.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace ops {
LegacyTransformStrictOp::LegacyTransformStrictOp() : LegacyOp(1) {
  this->getOpDescriptor()->allowInplace(true);
}

LegacyTransformStrictOp::LegacyTransformStrictOp(int opNum) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->allowInplace(true);
}

LegacyOp *LegacyTransformStrictOp::clone() { return new LegacyTransformStrictOp(this->_opNum); }

Status LegacyTransformStrictOp::validateAndExecute(Context &block) {
  auto input = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {input});

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyTransformStrictOp");

  NativeOpExecutioner::execTransformStrict(block.launchContext(), opNum, input->buffer(), input->shapeInfo(),
                                           input->specialBuffer(), input->specialShapeInfo(), z->buffer(),
                                           z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                                           extras.argumentsAsT(z->dataType()));

  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}

/**
 * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and
 * col2im. But these ops already have CustomOp implementations.
 *
 */
ShapeList *LegacyTransformStrictOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  LongType *newShape;
  COPY_SHAPE(inShape, newShape);

  return SHAPELIST(CONSTANT(newShape));
}
}  // namespace ops
}  // namespace sd
