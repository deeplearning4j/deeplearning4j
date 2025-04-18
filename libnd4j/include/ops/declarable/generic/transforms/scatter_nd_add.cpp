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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.08.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_nd_add)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace sd {
namespace ops {

OP_IMPL(scatter_nd_add, 3, 1, true) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto updates = INPUT_VARIABLE(2);
  if(indices->isEmpty())
    return Status::OK;

  auto output = OUTPUT_VARIABLE(0);

  const bool lock = block.getBArguments()->empty() ? false : B_ARG(0);
  const bool checkIndices = block.getBArguments()->size() <= 1 ? false : B_ARG(1);

  const int inRank = input->rankOf();
  const int indRank = indices->rankOf();
  const int updRank = updates->rankOf();

  const LongType indLastDim = indices->sizeAt(-1);

  REQUIRE_TRUE(
      indLastDim <= inRank, 0,
      "SCATTER_ND_ADD OP: the last dimension of indices array must be <= input_array_rank, but got %i instead !",
      indLastDim);
  REQUIRE_TRUE(
      updRank == (indRank - 1 + inRank - indLastDim), 0,
      "SCATTER_ND_ADD OP: the equality updates_rank = (indices_rank - 1 + input_rank - last_indices_dimension) must be "
      "true for input arrays, but got instead: updates_rank = %i, indices_rank = %i, last_indices_dimension = %i !",
      updRank, indRank, indLastDim);

  std::vector<LongType> inShape = input->getShapeAsVector();
  std::vector<LongType> updShape = updates->getShapeAsVector();
  std::vector<LongType> indShape = indices->getShapeAsVector();
  std::vector<LongType> expectedUpdShape(std::begin(indShape), std::end(indShape) - 1);
  if (inRank > indLastDim)
    std::move(std::begin(inShape) + indLastDim, std::end(inShape), std::back_inserter(expectedUpdShape));
  REQUIRE_TRUE(expectedUpdShape == updShape, 0,
               "SCATTER_ND_ADD OP: wrong shape of updates array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedUpdShape).c_str(), ShapeUtils::shapeAsString(updShape).c_str());

  if (checkIndices) {
    const LongType numOfBadIndx = helpers::checkIndices(block.launchContext(), *indices, *output);
    REQUIRE_TRUE(numOfBadIndx == 0, 0,
                 "SCATTER_ND_ADD OP: please check elements of indices-array, total number of wrong elements is %lld!",
                 numOfBadIndx);
  }

  if (!block.isInplace()) output->assign(input);

  helpers::scatterND(block.launchContext(), pairwise::Add, *indices, *updates, *output, lock);

  return Status::OK;
}

DECLARE_TYPES(scatter_nd_add) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_INTS, ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
