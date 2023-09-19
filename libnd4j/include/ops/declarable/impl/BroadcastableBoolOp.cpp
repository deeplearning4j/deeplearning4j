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

  if (shape::isEmpty(x) || shape::isEmpty(y)) {
    // this is edge case, [3, 4] + [] = []
    if ((shape::isEmpty(x) && shape::rank(x) == 0) || (shape::isEmpty(y) && shape::rank(y) == 0)) {
      std::vector<sd::LongType> vecShape;
      auto xShape = shape::shapeOf(x);
      for(int i = 0; i < shape::rank(x); i++)
        vecShape.emplace_back(xShape[i]);
      shapeList->push_back(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(dtype,vecShape));
      return shapeList;
    }

    const sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    auto desc = new ShapeDescriptor(newshape, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  } else if (shape::isScalar(x) && shape::isScalar(y)) {
    if (shape::rank(x) >= shape::rank(y)) {
      auto desc = new ShapeDescriptor(x, dtype);
      shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
      delete desc;
    } else {
      auto desc = new ShapeDescriptor(y, dtype);
      shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
      delete desc;
    }
  } else if (shape::equalsSoft(x, y)) {
    auto desc = new ShapeDescriptor(x, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  } else if (shape::isScalar(x) && !shape::isScalar(y)) {
    auto desc = new ShapeDescriptor(y, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  } else if (!shape::isScalar(x) && shape::isScalar(y)) {
    auto desc = new ShapeDescriptor(x, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
    const sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    auto desc = new ShapeDescriptor(newshape, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  } else {
    // in this case we'll throw exception later
    auto desc = new ShapeDescriptor(x, dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;

  }

  return shapeList;
}
}  // namespace ops
}  // namespace sd
