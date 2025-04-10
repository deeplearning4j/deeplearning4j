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
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/OpRegistrator.h>

#ifdef ONLY_SAME_TRANSFORM
namespace sd {
namespace ops {
LegacyTransformOp::LegacyTransformOp() : LegacyOp::LegacyOp(1) {
  // just a no-op
}

LegacyTransformOp::LegacyTransformOp(int opType) : LegacyOp::LegacyOp(1, opType) {
  // just a no-op
}

LegacyOp *LegacyTransformOp::clone() { return new LegacyTransformOp(this->_opNum); }

sd::Status LegacyTransformOp::validateAndExecute(Context &block) {
  auto input = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opType = block.opType() < 0 ? this->_opNum : block.opType();

  NativeOpExcutioner::execTransformSame(opType, input->buffer(), input->shapeInfo(), z->buffer(), z->shapeInfo(),
                                        block.getTArguments()->data(), nullptr, nullptr);

  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return sd::Status::OK;
}

/**
 * For transform operations, output shape always equals to input shape. With just a few exclusions, like im2col and
 * col2im. But these ops already have CustomOp implementations.
 *
 */
ShapeList *LegacyTransformOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
  auto inShape = inputShape->at(0);
  return SHAPELIST(CONSTANT(inShape));
}
}  // namespace ops
}  // namespace sd
#endif
