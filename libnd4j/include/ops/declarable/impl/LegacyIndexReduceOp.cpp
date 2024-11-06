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
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
LegacyIndexReduceOp::LegacyIndexReduceOp() : LegacyOp(1) {
  //
}

LegacyIndexReduceOp::LegacyIndexReduceOp(int opNum) : LegacyOp(1, opNum) {
  //
}

LegacyOp *LegacyIndexReduceOp::clone() { return new LegacyIndexReduceOp(this->_opNum); }

ShapeList *LegacyIndexReduceOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto inShape = inputShape->at(0);

  if (block.getAxis()->size() == 0 && block.width() == 1) {
    LongType *newShape;
    // in this case we just return scalar
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
    newShape[0] = 2;
    newShape[1] = 1;
    newShape[2] = 1;
    newShape[3] = 1;
    newShape[4] = 1;
    newShape[6] = 1;
    newShape[7] = 99;

    auto desc = new ShapeDescriptor(newShape, INT64, false);
    auto result = ConstantShapeHelper::getInstance().createShapeInfo(desc);
    RELEASE(newShape, block.getWorkspace());
  if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
    return SHAPELIST(result);
  } else if (block.getAxis()->size()) {
    // in this case we're building proper shape for reduction
    auto array = INPUT_VARIABLE(0);

    auto newShape =
        ShapeUtils::evalReduceShapeInfo('c', block.getAxis(), *array, INT64, false, true, block.workspace());
    return SHAPELIST(newShape);
  } else {
    bool allAxes = false;
    auto indices = INPUT_VARIABLE(1);
    LongType rank = shape::rank(inShape);
    if (indices->lengthOf() == rank) allAxes = true;

    std::vector<LongType> axis(indices->lengthOf());
    for (int e = 0; e < indices->lengthOf(); e++) {
      // lol otherwise we segfault on macOS
      int f = indices->e<int>(e);
      axis[e] = f >= 0 ? f : f += rank;
    }
    if (allAxes) {
      LongType *newShape;
      // in this case we just return scalar
      ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), sd::LongType);
      newShape[0] = 2;
      newShape[1] = 1;
      newShape[2] = 1;
      newShape[3] = 1;
      newShape[4] = 1;
      newShape[6] = 1;
      newShape[7] = 99;

      auto desc = new ShapeDescriptor(newShape, INT64, false);
      auto result = ConstantShapeHelper::getInstance().createShapeInfo(desc);
      RELEASE(newShape, block.getWorkspace());
  if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
      return SHAPELIST(result);
    } else {
      // in this case we're building proper shape for reduction
      auto array = INPUT_VARIABLE(0);
      return SHAPELIST(
          ShapeUtils::evalReduceShapeInfo('c', &axis, *array, DataType::INT64, false, true, block.workspace()));
    }
  }
}

/**
 *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
 *   It solely depends on input shape, and requested dimensions
 */
Status LegacyIndexReduceOp::validateAndExecute(Context &block) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  NDArray::prepareSpecialUse({z}, {x});

  if (z->dataType() != INT64) {
    THROW_EXCEPTION("IndexReduce operations require output to be INT64");
  }

  int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

  bool allAxes = false;

  ExtraArguments extras(*block.getTArguments());
  PointersManager manager(block.launchContext(), "LegacyIndexReduceOp");

  if (block.width() == 1) {
    if (block.getAxis()->size() == 0) {
      // scalar
      NativeOpExecutioner::execIndexReduceScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
    } else {
      // TAD
      std::vector<LongType> dims(block.getAxis()->size());
      for (size_t e = 0; e < dims.size(); e++) {
        auto axe = block.getAxis()->at(e);
        dims[e] = axe < 0 ? axe + x->rankOf() : axe;
      }
      if (dims.size() > 1) std::sort(dims.begin(), dims.end());

      auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &dims);

      NativeOpExecutioner::execIndexReduce(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), reinterpret_cast<LongType *>(z->buffer()), z->shapeInfo(),
          z->specialBuffer(), z->specialShapeInfo(), nullptr, (int)dims.size(),
          Environment::getInstance().isCPU() ? tadPack->primaryShapeInfo() : tadPack->specialShapeInfo(),
          Environment::getInstance().isCPU() ? tadPack->primaryOffsets() : tadPack->specialOffsets());
    }
  } else {
    // TF mode
    auto indices = INPUT_VARIABLE(1);
    if (indices->lengthOf() == x->rankOf()) allAxes = true;

    std::vector<LongType> axis(indices->lengthOf());
    for (LongType e = 0; e < indices->lengthOf(); e++) {
      LongType f = indices->e<LongType>(e);
      axis[e] = f >= 0 ? f : f += x->rankOf();
    }

    if (allAxes) {
      NativeOpExecutioner::execIndexReduceScalar(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());

    } else {
      if (indices->lengthOf() > 1) std::sort(axis.begin(), axis.end());

      REQUIRE_TRUE(axis.size() > 0, 0, "Some dimensions required for reduction!");

      auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), &axis);

      NativeOpExecutioner::execIndexReduce(
          block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
          extras.argumentsAsT(x->dataType()), reinterpret_cast<LongType *>(z->buffer()), z->shapeInfo(),
          z->specialBuffer(), z->specialShapeInfo(), nullptr, (int)axis.size(),
          Environment::getInstance().isCPU() ? tadPack->primaryShapeInfo() : tadPack->specialShapeInfo(),
          Environment::getInstance().isCPU() ? tadPack->primaryOffsets() : tadPack->specialOffsets());
    }
  }

  manager.synchronize();
  STORE_RESULT(*z);
  traceExecIfNeeded(block);


  return Status::OK;
}
}  // namespace ops
}  // namespace sd
