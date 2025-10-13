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
#include <ops/declarable/BroadcastableOp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
BroadcastableOp::BroadcastableOp(const char *name, int numTArgs, int numIArgs)
    : DeclarableCustomOp::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
  //
}

ShapeList *BroadcastableOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
  auto shapeList = SHAPELIST();
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto outputs = _descriptor->getOutputTypesForOutput(0);
  sd::DataType dtype = block.dataType(0);
  if (block.dataType(0) != sd::DataType::BOOL && !(outputs.size() == 1 && outputs[0] == sd::DataType::BOOL)) {
    if (Environment::getInstance().isExperimentalBuild()) {
      if (shape::length(y) > shape::length(x)) {
        dtype = DataTypeUtils::pickPairwiseResultType(y, x);
      } else {
        dtype = DataTypeUtils::pickPairwiseResultType(x, y);
      }
    } else {
      dtype = ArrayOptions::dataType(x);
    }
  } else
    dtype = sd::DataType::BOOL;

  if (shape::isEmptyConst(x) || shape::isEmptyConst(y)) {
    // this is edge case, [3, 4] + [] = []
    if ((shape::isEmptyConst(x) && shape::rank(x) == 0) || (shape::isEmptyConst(y) && shape::rank(y) == 0)) {
      auto desc = ShapeBuilders::emptyShapeInfo(dtype);
      shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
      return shapeList;
    }

    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    auto constShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(newshape)->primary();
    shapeList->push_back(constShape);
  } else if (shape::isScalar(x) && shape::isScalar(y)) {
    if (shape::rank(x) >= shape::rank(y)) {
      shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(x)->primary());
    } else {
      shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(y)->primary());
    }
  } else if (shape::equalsSoft(x, y)) {
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(x)->primary());
  } else if (shape::isScalar(x) && !shape::isScalar(y)) {
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(y)->primary());
  } else if (!shape::isScalar(x) && shape::isScalar(y)) {
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(x)->primary());
  } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    auto constShape = ConstantShapeHelper::getInstance().bufferForShapeInfo(newshape)->primary();
    shapeList->push_back(constShape);
  } else {
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(x)->primary());

  }

  return shapeList;
}
}  // namespace ops
}  // namespace sd
