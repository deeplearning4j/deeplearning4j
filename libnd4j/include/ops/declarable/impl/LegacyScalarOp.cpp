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
#include <ops/declarable/LegacyScalarOp.h>

#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
LegacyScalarOp::LegacyScalarOp() : LegacyOp(1) { this->getOpDescriptor()->allowInplace(true); }

LegacyScalarOp::LegacyScalarOp(int opNum) : LegacyOp(1, opNum) {
  this->getOpDescriptor()->allowInplace(true);
}

LegacyOp *LegacyScalarOp::clone() { return new LegacyScalarOp(this->_opNum, *this->_scalar); }

LegacyScalarOp::LegacyScalarOp(int opNum, NDArray &scalar) : LegacyOp(1, opNum) {
  _scalar = scalar.dup(scalar.ordering(), false);
}

ShapeList *LegacyScalarOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  LongType *newShape;
  COPY_SHAPE(inShape, newShape);

  return SHAPELIST(CONSTANT(newShape));
}

Status LegacyScalarOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  ExtraArguments extras(*block.getTArguments());

  if (block.width() > 1) {
    auto y = INPUT_VARIABLE(1);

    NDArray::prepareSpecialUse({z}, {x, y});

    NativeOpExecutioner::execScalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(),
                                    x->specialShapeInfo(), z->buffer(), z->shapeInfo(), z->specialBuffer(),
                                    z->specialShapeInfo(), y->buffer(), y->shapeInfo(), y->specialBuffer(),
                                    y->specialShapeInfo(), extras.argumentsAsT(z->dataType()));

    NDArray::registerSpecialUse({z}, {x, y});
  } else if (block.getTArguments()->size() > 0) {
    // Cache the scalar NDArray in _scalar to avoid creating and destroying a
    // temporary on every call.  During CUDA graph capture the kernel records
    // the device address of the scalar buffer; if that buffer is freed before
    // replay the graph reads stale/garbage memory for the scalar value.
    // Caching keeps the buffer alive for the entire op lifetime, which covers
    // both capture and all subsequent replays.
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

    NativeOpExecutioner::execScalar(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), _scalar->buffer(), _scalar->shapeInfo(),
        _scalar->specialBuffer(), _scalar->specialShapeInfo(),
        extras.length() > 1 ? extras.argumentsAsT(z->dataType(), 1) : nullptr);

    NDArray::registerSpecialUse({z}, {x, _scalar});
  } else {
    NDArray::prepareSpecialUse({z}, {x, _scalar});

    NativeOpExecutioner::execScalar(
        block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
        z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), _scalar->buffer(), _scalar->shapeInfo(),
        _scalar->specialBuffer(), _scalar->specialShapeInfo(), extras.argumentsAsT(z->dataType()));

    NDArray::registerSpecialUse({z}, {x, _scalar});
  }


  traceExecIfNeeded(block);

  return Status::OK;
}
}  // namespace ops
}  // namespace sd
