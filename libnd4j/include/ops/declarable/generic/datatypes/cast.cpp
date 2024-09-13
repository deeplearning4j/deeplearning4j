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
  auto output = OUTPUT_VARIABLE(0);

  if (input->isEmpty()) {
    REQUIRE_TRUE(output->isEmpty(), 0, "If input is empty, output array must also be empty");
    return sd::Status::OK;
  }

  if (!block.isInplace()) {
    NDArray& deRef = *input;
    output->assign(deRef);
    output->printIndexedBuffer("OUTPUT INDEXED OUTPUT AFTER CAST:");
    fflush(stdout);
  }

  STORE_RESULT(output);
  return sd::Status::OK;
}
DECLARE_SYN(Cast, cast);

DECLARE_SHAPE_FN(cast) {
  auto inShape = inputShape->at(0);
  if(!block.getDArguments()->empty()) {
    DataType newType = block.dataType(0);
    DataType secondComp = block.getDArguments()->at(0);
    printf("new type is %s block d args size is %d other data type is %s\n",
           DataTypeUtils::asString(newType).c_str(),
           block.getDArguments()->size(),
           DataTypeUtils::asString(secondComp).c_str());
    fflush(stdout);
    auto desc = new ShapeDescriptor(inShape, newType, true);
    auto newShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(desc);
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
    printf("cast new data type is : %s data type from new constant created data type %s\n",
           DataTypeUtils::asString(newType).c_str(),
           DataTypeUtils::asString(compDataType).c_str());
    fflush(stdout);
    return ret;

  } else {
    auto it = INT_ARG(0);
    DataType newType = DataTypeUtils::fromInt(it);
    auto desc = new ShapeDescriptor(inShape, newType, false);
    auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
    return ret;
  }
}

DECLARE_TYPES(cast) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes(sd::DataType::ANY);
}
}  // namespace ops
}  // namespace sd

#endif
