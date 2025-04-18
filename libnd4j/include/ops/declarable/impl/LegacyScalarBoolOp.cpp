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
#include <array/NDArrayFactory.h>
#include <ops/declarable/LegacyScalarBoolOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
LegacyScalarBoolOp::LegacyScalarBoolOp() : LegacyOp(1) {
  // no-op
}

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum) : LegacyOp(1, opNum) {
  // no-op
}

LegacyOp *LegacyScalarBoolOp::clone() { return new LegacyScalarBoolOp(this->_opNum, *this->_scalar); }

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum, NDArray &scalar) : LegacyOp(1, opNum) {
  _scalar = new NDArray(scalar.dup(scalar.ordering(), false));
}

ShapeList *LegacyScalarBoolOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);
  return SHAPELIST(CONSTANT(inShape));
}

Status LegacyScalarBoolOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyScalarBoolOp");

  if (block.width() > 1) {
    auto y = INPUT_VARIABLE(1);

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                        y->specialShapeInfo(), extras.argumentsAsT(x->dataType()));
  } else if (block.getTArguments()->size() > 0) {
    auto y = NDArrayFactory::create(T_ARG(0), block.launchContext());

    NDArray::prepareSpecialUse({z}, {x, &y});

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(),
                                        y.specialShapeInfo(), extras.argumentsAsT(x->dataType(), 1));

    manager.synchronize();
  } else {
    NDArray::prepareSpecialUse({z}, {x, _scalar});

    NativeOpExecutioner::execScalarBool(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), _scalar->buffer(), _scalar->shapeInfo(),
        _scalar->specialBuffer(), _scalar->specialShapeInfo(), extras.argumentsAsT(x->dataType()));
  }
  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}
}  // namespace ops
}  // namespace sd
