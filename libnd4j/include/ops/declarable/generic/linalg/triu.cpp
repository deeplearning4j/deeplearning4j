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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

#if NOT_EXCLUDED(OP_triu)
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() > 0, 0, "TRIU OP: the rank of input array must be > 0, but got %i instead !",
               input->rankOf());

  int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

  BUILD_SINGLE_SELECTOR(input->dataType(), input->fillAsTriangular, (0, diag, diag, *output, 'l'), SD_COMMON_TYPES);

  return sd::Status::OK;
}

DECLARE_TYPES(triu) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS})->setAllowedOutputTypes(0, {ALL_FLOATS});
}

DECLARE_SHAPE_FN(triu) {
  auto inShapeInfo = inputShape->at(0);

  REQUIRE_TRUE(inShapeInfo[0] > 0, 0, "TRIU OP: the rank of input array must be > 0, but got %i instead !",
               inShapeInfo[0]);

  int rank = (inShapeInfo[0] == 1) ? 2 : inShapeInfo[0];

  sd::LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  memcpy(outShapeInfo, inShapeInfo, (1 + rank) * sizeof(sd::LongType));  // copy rank and dimensions values only

  if (inShapeInfo[0] == 1) {
    outShapeInfo[0] = rank;
    outShapeInfo[1] = inShapeInfo[1];
    outShapeInfo[2] = inShapeInfo[1];
  }

  ShapeUtils::updateStridesAndType(outShapeInfo, inShapeInfo, shape::order(inShapeInfo));

  return SHAPELIST(CONSTANT(outShapeInfo));
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu_bp, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);  // dLoss/dO
  auto gradOPass = gradO;
  std::unique_ptr<NDArray> pass;
  pass.reset(gradOPass);

  auto gradI = OUTPUT_VARIABLE(0);  // dLoss/dI
  if(gradI->isScalar()) {
    gradI->p(0,0.0);
    return sd::Status::OK;
  }

  REQUIRE_TRUE(input->rankOf() > 0, 0, "TRIU_BP OP: the rank of input array must be > 0, but got %i instead !",
               input->rankOf());

  const int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

  helpers::triuBP(block.launchContext(), *input, *pass, *gradI, diag);

  return sd::Status::OK;
}

DECLARE_TYPES(triu_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedOutputTypes(0, {ALL_FLOATS});
}

DECLARE_SHAPE_FN(triu_bp) {
  auto gradOShapeInfo = inputShape->at(0);
  int rank = gradOShapeInfo[0];

  sd::LongType* outShapeInfo = nullptr;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), sd::LongType);
  memcpy(outShapeInfo, gradOShapeInfo, (1 + rank) * sizeof(sd::LongType));  // copy rank and dimensions values only

  auto in = inputShape->at(0);
  ShapeUtils::updateStridesAndType(outShapeInfo, in, shape::order(in));

  return SHAPELIST(CONSTANT(outShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
