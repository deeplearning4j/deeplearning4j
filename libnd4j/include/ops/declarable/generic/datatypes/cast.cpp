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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_cast)

#include <array/DataTypeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(cast, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  if(input->dataType() != ArrayOptions::dataType(input->shapeInfo())) {
    std::string errorMessage;
    errorMessage += "Input data type is not equal to data type reflected in shape info: ";
    errorMessage += DataTypeUtils::asString(input->dataType());
    errorMessage += " != ";
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(input->shapeInfo()));
    errorMessage += " for input shape info: ";
    errorMessage += ShapeUtils::shapeAsString(input->shapeInfo());
    errorMessage += " and output shape info: ";
    errorMessage += ShapeUtils::shapeAsString(OUTPUT_VARIABLE(0)->shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());

  }
  auto output = OUTPUT_VARIABLE(0);
  if(output->dataType() != ArrayOptions::dataType(output->shapeInfo())) {
    std::string errorMessage;
    errorMessage += "Input data type is not equal to data type reflected in shape info: ";
    errorMessage += DataTypeUtils::asString(input->dataType());
    errorMessage += " != ";
    errorMessage += DataTypeUtils::asString(ArrayOptions::dataType(input->shapeInfo()));
    errorMessage += " for input shape info: ";
    errorMessage += ShapeUtils::shapeAsString(input->shapeInfo());
    errorMessage += " and output shape info: ";
    errorMessage += ShapeUtils::shapeAsString(OUTPUT_VARIABLE(0)->shapeInfo());
    THROW_EXCEPTION(errorMessage.c_str());

  }
  if (input->isEmpty()) {
    REQUIRE_TRUE(output->isEmpty(), 0, "If input is empty, output array must also be empty");
    return Status::OK;
  }


  if (!block.isInplace()) output->assign(input);

  STORE_RESULT(output);
  return Status::OK;
}
DECLARE_SYN(Cast, cast);

DECLARE_SHAPE_FN(cast) {
  auto inShape = inputShape->at(0);
  if(!block.getDArguments()->empty()) {
    DataType newType = D_ARG(0);
    auto desc = new ShapeDescriptor(inShape, newType);
    if(desc->dataType() != newType) {
      std::string errorMessage;
      errorMessage += "New data type is not reflected in the created descriptor: ";
      errorMessage += DataTypeUtils::asString(desc->dataType());
      errorMessage += " != ";
      errorMessage += DataTypeUtils::asString(newType);
      errorMessage += " for input shape info: ";
      errorMessage += ShapeUtils::shapeAsString(inShape);
      THROW_EXCEPTION(errorMessage.c_str());
    }
    auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    if(desc->order() != shape::order(inShape)) {
      THROW_EXCEPTION("Order of the new shape descriptor is not equal to the order of the input shape descriptor!");
    }
    REQUIRE_TRUE(desc->dataType() == ArrayOptions::dataType(ret->at(0)),0,"Data types for cast did not equal!");
     if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
    return ret;

  } else {
    auto it = INT_ARG(0);
    DataType newType = DataTypeUtils::fromInt(it);
    auto desc = new ShapeDescriptor(inShape, newType);
    auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
     if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
    return ret;
  }
}

DECLARE_TYPES(cast) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
}
}  // namespace ops
}  // namespace sd

#endif
