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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 21.08.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_nd)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(scatter_nd, 3, 1, false, 0, 0) {
  auto indices = INPUT_VARIABLE(0);
  auto updates = INPUT_VARIABLE(1);
  auto shape = INPUT_VARIABLE(2);
  if(indices->isEmpty())
    return Status::OK;

  auto output = OUTPUT_VARIABLE(0);

  const bool lock = block.getBArguments()->empty() ? false : B_ARG(0);
  const bool checkIndices = block.getBArguments()->size() <= 1 ? false : B_ARG(1);

  const int indRank = indices->rankOf();
  const int updRank = updates->rankOf();
  const int shapeRank = shape->rankOf();
  const LongType shapeLen = shape->lengthOf();

  REQUIRE_TRUE(shapeRank == 1, 0, "SCATTER_ND OP: the rank of shape array must be 1, but got %i instead !", shapeRank);
  REQUIRE_TRUE(indices->sizeAt(-1) <= shapeLen, 0,
               "SCATTER_ND OP: last dimension of indices array must be <= length of shape array, but got %i and %i "
               "correspondingly !",
               indices->sizeAt(-1), shapeLen);

  REQUIRE_TRUE(
      updRank == (indRank - 1 + shapeLen - indices->sizeAt(-1)), 0,
      "SCATTER_ND OP: the equality updates_rank = (indices_rank - 1 + shape_length - last_indices_dimension) must be "
      "true for input arrays, but got instead: updates_rank = %i, shape_length = %i, last_indices_dimension = %i !",
      updRank, shapeLen, indices->sizeAt(-1));

  std::vector<LongType> outShape = shape->getBufferAsVector<LongType>();
  std::vector<LongType> updShape = updates->getShapeAsVector();
  std::vector<LongType> indShape = indices->getShapeAsVector();
  std::vector<LongType> expectedUpdShape(std::begin(indShape), std::end(indShape) - 1);
  std::move(std::begin(outShape) + indices->sizeAt(-1), std::end(outShape), std::back_inserter(expectedUpdShape));
  REQUIRE_TRUE(expectedUpdShape == updShape, 0,
               "SCATTER_ND OP: wrong shape of updates array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedUpdShape).c_str(), ShapeUtils::shapeAsString(updShape).c_str());

  if (checkIndices) {
    const LongType numOfBadIndx = helpers::checkIndices(block.launchContext(), *indices, *output);
    REQUIRE_TRUE(numOfBadIndx == 0, 0,
                 "SCATTER_ND OP: please check elements of indices-array, total number of wrong elements is %lld!",
                 numOfBadIndx);
  }

  // initial zeroing of output
  *output = 0;

  helpers::scatterND(block.launchContext(), pairwise::Add, *indices, *updates, *output, lock);

  return Status::OK;
}

DECLARE_TYPES(scatter_nd) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}

////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(scatter_nd) {
  auto shape = INPUT_VARIABLE(2);
  auto updShapeInfo = inputShape->at(1);

  LongType *outShapeInfo;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(shape->lengthOf()), sd::LongType);

  outShapeInfo[0] = shape->lengthOf();
  for (int i = 0; i < outShapeInfo[0]; ++i) outShapeInfo[i + 1] = shape->e<LongType>(i);

  ShapeUtils::updateStridesAndType(outShapeInfo, updShapeInfo, shape::order(updShapeInfo));

  auto result = SHAPELIST(CONSTANT(outShapeInfo));
  RELEASE(outShapeInfo, block.getWorkspace());
  return result;
}

}  // namespace ops
}  // namespace sd

#endif
