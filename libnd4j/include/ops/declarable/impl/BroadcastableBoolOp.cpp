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
// Created by raver on 6/6/2018.
//
#include <helpers/ShapeUtils.h>
#include <ops/declarable/BroadcastableBoolOp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
BroadcastableBoolOp::BroadcastableBoolOp(const char *name, int numTArgs, int numIArgs)
    : DeclarableCustomOp::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
  //
}
ShapeList *BroadcastableBoolOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
  auto shapeList = SHAPELIST();
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  sd::DataType dtype = sd::DataType::BOOL;

  if (shape::isEmptyConst(x) || shape::isEmptyConst(y)) {
    // this is edge case, [3, 4] + [] = []
    if ((shape::isEmptyConst(x) && shape::rank(x) == 0) || (shape::isEmptyConst(y) && shape::rank(y) == 0)) {
      shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor::emptyDescriptor(dtype)));
      return shapeList;
    }

    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    // Cast to boolean and let ConstantShapeHelper manage the memory
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(newshape, dtype);
    shapeList->push_back(castedShape);
  } else if (shape::isScalar(x) && shape::isScalar(y)) {
    if (shape::rank(x) >= shape::rank(y)) {
      auto castedShape = ConstantShapeHelper::getInstance().castToDataType(x, dtype);
      shapeList->push_back(castedShape);
    } else {
      auto castedShape = ConstantShapeHelper::getInstance().castToDataType(y, dtype);
      shapeList->push_back(castedShape);
    }
  } else if (shape::equalsSoft(x, y)) {
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(x, dtype);
    shapeList->push_back(castedShape);
  } else if (shape::isScalar(x) && !shape::isScalar(y)) {
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(y, dtype);
    shapeList->push_back(castedShape);
  } else if (!shape::isScalar(x) && shape::isScalar(y)) {
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(x, dtype);
    shapeList->push_back(castedShape);
  } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    // Cast to boolean and let ConstantShapeHelper manage the memory
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(newshape, dtype);
    shapeList->push_back(castedShape);
  } else {
    auto castedShape = ConstantShapeHelper::getInstance().castToDataType(x, dtype);
    shapeList->push_back(castedShape);
  }

  return shapeList;
}
}  // namespace ops
}  // namespace sd
