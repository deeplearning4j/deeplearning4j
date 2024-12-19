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
// @author Shyrma Yurii (iuriish@yahoo.com), created on 06.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_pad)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(pad, 2, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();

  // input validation
  std::vector<sd::LongType> expectedPaddingsShape = {rank, 2};
  std::vector<sd::LongType> currentPaddingsShape = paddings->getShapeAsVector();
  REQUIRE_TRUE(expectedPaddingsShape == currentPaddingsShape, 0,
               "PAD op: wrong shape of paddings array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedPaddingsShape).c_str(),
               ShapeUtils::shapeAsString(currentPaddingsShape).c_str());

  NDArray padValue(input->dataType(), block.launchContext());

  // in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements
  if (INT_ARG(0) == 0) {  // CONSTANT mode
    if (block.width() > 2) {
      REQUIRE_TRUE(
          input->dataType() == INPUT_VARIABLE(2)->dataType(), 0,
          "PAD op: data types of input and padValue arrays should be the same but got %i and %i correspondingly !",
          input->dataType(), INPUT_VARIABLE(2)->dataType());
      padValue.assign(INPUT_VARIABLE(2)->e(0));
    } else if (!block.getTArguments()->empty())
      padValue = T_ARG(0);
  } else if (INT_ARG(0) == 1) {  // REFLECT mode
    for (int dim = 0; dim < rank; ++dim)
    REQUIRE_TRUE(paddings->e<sd::LongType>(dim, 0) <= (input->shapeOf()[dim] - 1) &&
                 paddings->e<sd::LongType>(dim, 1) <= (input->shapeOf()[dim] - 1),
                 0, "PAD op: wrong content of paddings array for REFLECT mode !");
  }
  if (INT_ARG(0) == 2) {  // SYMMETRIC mode
    for (int dim = 0; dim < rank; ++dim)
    REQUIRE_TRUE(paddings->e<sd::LongType>(dim, 0) <= input->shapeOf()[dim] &&
                 paddings->e<sd::LongType>(dim, 1) <= input->shapeOf()[dim],
                 0, "PAD op: wrong content of paddings array for SYMMETRIC mode !");
  }

  // CONSTANT->0, REFLECT->1, SYMMETRIC->2
  REQUIRE_TRUE(
      INT_ARG(0) >= 0 && INT_ARG(0) <= 2, 0,
      "PAD op: unknown padding mode, there are only three possible legal values -> 0,1,2, but got %i instead !",
      INT_ARG(0));

  helpers::pad(block.launchContext(), INT_ARG(0), *input, *paddings, *output, padValue);

  return sd::Status::OK;
}

DECLARE_TYPES(pad) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64})  // INT32 with TF
      ->setSameMode(true);
}

DECLARE_SHAPE_FN(pad) {
  // check shape of paddings
  auto inputShapeInfo = inputShape->at(0);
  auto paddings = INPUT_VARIABLE(1);
  const int rank = inputShapeInfo[0];
  if(rank < 0 || rank > SD_MAX_RANK) {
    THROW_EXCEPTION("PAD op: Bad shape buffer. Likely corrupt. Please ensure buffer was not deallocated.");
  }
  // paddings validation
  const std::vector<sd::LongType> expectedPaddingsShape = {rank, 2};
  const std::vector<sd::LongType> currentPaddingsShape = paddings->getShapeAsVector();
  REQUIRE_TRUE(expectedPaddingsShape == currentPaddingsShape, 0,
               "PAD op: wrong shape of paddings array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedPaddingsShape).c_str(),
               ShapeUtils::shapeAsString(currentPaddingsShape).c_str());

  sd::LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  outShapeInfo[0] = rank;
  for (int i = 1; i <= rank; ++i)
    outShapeInfo[i] = inputShapeInfo[i] + paddings->e<sd::LongType>(i - 1, 0) + paddings->e<sd::LongType>(i - 1, 1);

  ShapeUtils::updateStridesAndType(outShapeInfo, inputShapeInfo, shape::order(inputShapeInfo));
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(outShapeInfo)->primary());
  return ret;
}

}  // namespace ops
}  // namespace sd

#endif
