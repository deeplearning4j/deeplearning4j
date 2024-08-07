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
    : DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
  //
}

ShapeList *BroadcastableOp::calculateOutputShape(ShapeList *inputShape, Context &block) {
  auto shapeList = SHAPELIST();
  auto x = inputShape->at(0);
  auto y = inputShape->size() > 1  ? inputShape->at(1) : x;
  auto outputs = _descriptor->getOutputTypesForOutput(0);
  DataType dtype = block.dataType(0);
  if (block.dataType(0) != BOOL && !(outputs.size() == 1 && outputs[0] == BOOL)) {
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
    dtype = BOOL;

  if (shape::isEmptyConst(x) || shape::isEmptyConst(y)) {
    // this is edge case, [3, 4] + [] = []
    if ((shape::isEmptyConst(x) && shape::rank(x) == 0)
        || (shape::isEmptyConst(y) && shape::rank(y) == 0)
        || (shape::isEmptyConst(x) && shape::rank(x) == 1 && shape::shapeOf(x)[0] == 0)
        ||  (shape::isEmptyConst(y) && shape::rank(y) == 1 && shape::shapeOf(y)[0] == 0)) {
      std::vector<LongType> vecShape;
      auto xShape = shape::shapeOf(x);
      for(int i = 0; i < shape::rank(x); i++)
        vecShape.emplace_back(xShape[i]);
      shapeList->push_back(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(dtype,vecShape));
      return shapeList;
    }

    if(dtype == ANY) {
      THROW_EXCEPTION("No data type found!");
    }


    const LongType *newshape = nullptr;
    if(!ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace())) {
      std::string errorMessage;
      errorMessage += "Unable to evaluate broadcast shape info:";
      errorMessage += shape::shapeToString(x,"");
      errorMessage += " vs ";
      errorMessage += shape::shapeToString(y,"");
      errorMessage += "\n";
      THROW_EXCEPTION(errorMessage.c_str());

    }

    auto newShape = ConstantShapeHelper::getInstance().createFromExisting(newshape, dtype);
    shapeList->push_back(newShape);
  } else if (shape::isScalar(x) && shape::isScalar(y)) {
    if (shape::rank(x) >= shape::rank(y)) {
      auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,x);
      shapeList->push_back(newShape);
    } else {
      auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,y);
      shapeList->push_back(newShape);
    }
  } else if (shape::equalsSoft(x, y)) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,x);
    shapeList->push_back(newShape);
  } else if (shape::isScalar(x) && !shape::isScalar(y)) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,y);
    shapeList->push_back(newShape);
  } else if (!shape::isScalar(x) && shape::isScalar(y)) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,x);
    shapeList->push_back(newShape);
  } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
    const LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,newshape);
    shapeList->push_back(newShape);
  } else {
    // in this case we'll throw exception later
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype,x);
    shapeList->push_back(newShape);
  }

  return shapeList;
}
}  // namespace ops
}  // namespace sd
