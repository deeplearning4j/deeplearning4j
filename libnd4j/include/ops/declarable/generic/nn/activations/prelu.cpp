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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 24.07.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_prelu)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>

#include <numeric>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> sharedAxes = *block.getIArguments();

  const int inputRank = input->rankOf();
  const int numSharedAxes = sharedAxes.size();  // can be zero as well
  const LongType inputLen = input->lengthOf();
  const LongType alphaLen = alpha->lengthOf();
  const std::vector<LongType> inputShape = input->getShapeAsVector();
  const std::vector<LongType> alphaShape = alpha->getShapeAsVector();

  //***** input validation *****//
  std::vector<LongType> expectedAlphaShape(&inputShape[1], &inputShape[inputRank]);

  REQUIRE_TRUE(inputRank > 1, 0,
               "PRELU OP: wrong rank of input array, expected rank should be > 1, but got %i instead !", inputRank);

  for (int i = 0; i < numSharedAxes; ++i) {
    if (sharedAxes[i] <= 0) sharedAxes[i] += inputRank - 1;
    REQUIRE_TRUE(1 <= sharedAxes[i] && sharedAxes[i] <= inputRank - 1, 0,
                 "PRELU OP: wrong axis value %i in sharedAxes at position %i, axis value must be within range [1, "
                 "input_rank-1] !",
                 sharedAxes[i], i);
    expectedAlphaShape[sharedAxes[i] - 1] = 1;
  }


  NDArray alpha2 =  alphaShape != expectedAlphaShape ? alpha->reshape(alpha->ordering(), expectedAlphaShape) : *alpha;
  helpers::prelu(block.launchContext(), input,
                 &alpha2,
                 output);

  return Status::OK;
}

DECLARE_TYPES(prelu) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS});
}

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu_bp, 3, 2, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto dLdO = INPUT_VARIABLE(2);

  auto dLdI = OUTPUT_VARIABLE(0);
  auto dLdA = OUTPUT_VARIABLE(1);

  std::vector<LongType> sharedAxes = *block.getIArguments();

  const int inputRank = input->rankOf();
  const int numSharedAxes = sharedAxes.size();  // can be zero as well
  const LongType inputLen = input->lengthOf();
  const LongType alphaLen = alpha->lengthOf();
  const std::vector<LongType> inputShape = input->getShapeAsVector();
  const std::vector<LongType> alphaShape = alpha->getShapeAsVector();

  //***** input validation *****//

  // temporary limitation imposed by Yurii
  REQUIRE_TRUE(inputRank <= SD_MAX_RANK / 2, 0, "rank of input array should be <= SD_MAX_RANK/2, but got %i instead!",
               inputRank);
  REQUIRE_TRUE(input->lengthOf() / alpha->lengthOf() <= SD_MAX_RANK * 2, 0,
               "the length of input array should be no more than SD_MAX_RANK*2 times the alpha array length, but got "
               "%lld and %lld correspondingly!",
               input->lengthOf(), alpha->lengthOf());

  std::vector<LongType> expectedAlphaShape(&inputShape[1], &inputShape[inputRank]);

  REQUIRE_TRUE(inputRank > 1, 0,
               "PRELU_BP OP: wrong rank of input array, expected rank should be > 1, but got %i instead !", inputRank);

  for (int i = 0; i < numSharedAxes; ++i) {
    if (sharedAxes[i] <= 0) sharedAxes[i] += inputRank - 1;
    REQUIRE_TRUE(1 <= sharedAxes[i] && sharedAxes[i] <= inputRank - 1, 0,
                 "PRELU_BP OP: wrong axis value %i in sharedAxes at position %i, axis value must be within range [1, "
                 "input_rank-1] !",
                 sharedAxes[i], i);
    expectedAlphaShape[sharedAxes[i] - 1] = 1;
  }

  LongType product = 1;
  for (const auto& item : expectedAlphaShape) product *= item;

  REQUIRE_TRUE(product == alphaLen, 0, "PRELU_BP OP: wrong shape of alpha array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedAlphaShape).c_str(), ShapeUtils::shapeAsString(alphaShape).c_str());
  // ***** end of validation ***** //

  if (alphaShape != expectedAlphaShape) {
    alpha = new NDArray(alpha->reshape(alpha->ordering(), expectedAlphaShape));
    dLdA = new NDArray(dLdA->reshape(dLdA->ordering(), expectedAlphaShape));
  }

  helpers::preluBP(block.launchContext(), input, alpha, dLdO, dLdI, dLdA);

  if (alphaShape != expectedAlphaShape) {
    delete alpha;
    delete dLdA;
  }

  return Status::OK;
}

DECLARE_TYPES(prelu_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {FLOAT32, DOUBLE, HALF})
      ->setAllowedInputTypes(2, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(0, {FLOAT32, DOUBLE, HALF})
      ->setAllowedOutputTypes(1, {FLOAT32, DOUBLE, HALF});
}

}  // namespace ops
}  // namespace sd

#endif
