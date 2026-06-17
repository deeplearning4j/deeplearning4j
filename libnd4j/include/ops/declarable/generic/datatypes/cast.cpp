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
#include <ops/declarable/headers/datatypes.h>
#include <ops/declarable/helpers/assign.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(cast, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (input->isEmpty()) {
    REQUIRE_TRUE(output->isEmpty(), 0, "If input is empty, output array must also be empty");
    return sd::Status::OK;
  }

  // Fast path: same data type - no conversion needed
  if (input->dataType() == output->dataType()) {
    if (!block.isInplace()) {
      output->assign(input);
    }
  } else if (!block.isInplace()) {
    helpers::assign(block.launchContext(), output, input);
  }

  STORE_RESULT(output);
  return sd::Status::OK;
}
DECLARE_SYN(Cast, cast);

DECLARE_SHAPE_FN(cast) {
  auto inShape = inputShape->at(0);

  // Check empty from both native shape info AND from the NDArray object.
  // Java-created empty singletons (Nd4j.empty()) may lack the ARRAY_EMPTY bit
  // in the native C++ shape pointer even though isEmpty() returns true on the Java side.
  // NDArray::isEmpty() has a fallback: rank==0 && _buffer==nullptr catches these.
  bool wasEmptyFromShape = ArrayOptions::hasPropertyBitSet(inShape, ARRAY_EMPTY);
  bool wasEmptyFromArray = false;
  if (block.isFastPath()) {
    const auto& fp = block.fastpath_in();
    if (fp.size() > 0 && fp[0] != nullptr) wasEmptyFromArray = fp[0]->isEmpty();
  }
  bool wasEmpty = wasEmptyFromShape || wasEmptyFromArray;

  if(!block.getDArguments()->empty()) {
    DataType newType = block.dataType(0);

    if (wasEmpty) {
      // Return rank-0 empty shape with the new dtype
      auto desc = ShapeBuilders::emptyShapeInfo(newType);
      auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
      delete[] desc;
      return ret;
    }

    auto newShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(newType, inShape);

    auto compDataType = ArrayOptions::dataType(newShapeInfo);
    if(compDataType != newType) {
      std::string errorMessage;
      errorMessage += "cast: new data type is ";
      errorMessage += DataTypeUtils::asString(newType);
      errorMessage += " data type from new constant created data type ";
      errorMessage += DataTypeUtils::asString(compDataType);
      errorMessage += "\n";
      THROW_EXCEPTION(errorMessage.c_str());
    }
    auto ret =  SHAPELIST(newShapeInfo);
    return ret;

  } else {
    auto it = INT_ARG(0);
    DataType newType = DataTypeUtils::fromInt(it);

    if (wasEmpty) {
      // Return rank-0 empty shape with the new dtype
      auto desc = ShapeBuilders::emptyShapeInfo(newType);
      auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
      delete[] desc;
      return ret;
    }

    auto resultShapeInfo = ConstantShapeHelper::getInstance().castToDataType(inShape, newType);
    auto ret = SHAPELIST(resultShapeInfo);
    return ret;
  }
}

DECLARE_TYPES(cast) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes(sd::DataType::ANY);
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_CAST);
}
}  // namespace ops
}  // namespace sd

#endif
