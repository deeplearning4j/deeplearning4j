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
#include <helpers/ConstantShapeHelper.h>
#include <ops/declarable/LegacyScalarBoolOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
LegacyScalarBoolOp::LegacyScalarBoolOp() : LegacyOp(1) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
}

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
}

LegacyOp *LegacyScalarBoolOp::clone() { return new LegacyScalarBoolOp(this->_opNum, *this->_scalar); }

LegacyScalarBoolOp::LegacyScalarBoolOp(int opNum, NDArray &scalar) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->addTraits(
      OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_COMPARISON |
      OP_TRAIT_FULLY_WRITING);
  _scalar = scalar.dup(scalar.ordering(), false);
}

void LegacyScalarBoolOp::registerTypes() {
  // Bool ops produce BOOL output regardless of input type — NOT same mode.
  this->getOpDescriptor()->setSameMode(false);
  this->getOpDescriptor()->setAllowedOutputTypes({BOOL});
  this->getOpDescriptor()->setAllowedInputTypes(ANY);
}

ShapeList *LegacyScalarBoolOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);
  // Bool ops always produce BOOL output regardless of input type
  return SHAPELIST(ConstantShapeHelper::getInstance().castToDataType(inShape, BOOL));
}

Status LegacyScalarBoolOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());

  if (block.width() > 1) {
    auto y = INPUT_VARIABLE(1);

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                        y->specialShapeInfo(), extras.argumentsAsT(x->dataType()));

    NDArray::registerSpecialUse({z}, {x, y});
  } else if (block.getTArguments()->size() > 0) {
    // Cache the scalar NDArray in _scalar to avoid creating and destroying a
    // temporary on every call.  During CUDA graph capture the kernel records
    // the device address of the scalar buffer; if that buffer is freed before
    // replay the graph reads stale/garbage memory for the scalar value.
    // Caching keeps the buffer alive for the entire op lifetime.
    double scalarVal = T_ARG(0);
    auto xDt = x->dataType();
    if (_scalar == nullptr || !_cachedScalarValid ||
        _cachedScalarType != xDt || _cachedScalarValue != scalarVal) {
      delete _scalar;
      _scalar = NDArrayFactory::create(xDt, scalarVal, block.launchContext());
      _cachedScalarValid = true;
      _cachedScalarValue = scalarVal;
      _cachedScalarType = xDt;
    }

    NDArray::prepareSpecialUse({z}, {x, _scalar});

    NativeOpExecutioner::execScalarBool(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                        x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                        z->specialShapeInfo(), _scalar->buffer(), _scalar->shapeInfo(), _scalar->specialBuffer(),
                                        _scalar->specialShapeInfo(),
                                        extras.length() > 1 ? extras.argumentsAsT(x->dataType(), 1) : nullptr);

    NDArray::registerSpecialUse({z}, {x, _scalar});
  } else {
    REQUIRE_TRUE(_scalar != nullptr, 0,
                 "LegacyScalarBoolOp: no scalar value provided (neither via tArgs, input[1], nor pre-set _scalar). "
                 "OpNum=%d. This typically means the DSP plan compiler did not extract the scalar value.", opNum);
    NDArray::prepareSpecialUse({z}, {x, _scalar});

    NativeOpExecutioner::execScalarBool(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), _scalar->buffer(), _scalar->shapeInfo(),
        _scalar->specialBuffer(), _scalar->specialShapeInfo(), extras.argumentsAsT(x->dataType()));

    NDArray::registerSpecialUse({z}, {x, _scalar});
  }
  STORE_RESULT(*z);
  traceExecIfNeeded(block);

  return Status::OK;
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
