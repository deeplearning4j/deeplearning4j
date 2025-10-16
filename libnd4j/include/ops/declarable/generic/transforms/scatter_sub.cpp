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
// @author Created by raver119 on 24.11.17.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_sub)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace sd {
namespace ops {
OP_IMPL(scatter_sub, 3, 1, true) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto updates = INPUT_VARIABLE(2);
  if(indices->isEmpty())
    return Status::OK;

  auto output = OUTPUT_VARIABLE(0);

  if (!block.isInplace()) output->assign(input);

  const bool lock = block.getBArguments()->empty() ? false : B_ARG(0);
  const bool checkIndices = block.getBArguments()->size() <= 1 ? false : B_ARG(1);

  const int inRank = input->rankOf();
  const int indRank = indices->rankOf();
  const int updRank = updates->rankOf();

  REQUIRE_TRUE(inRank > 0, 0, "SCATTER_SUB OP: input should not be scalar !");

  if (inRank == 1) {
    REQUIRE_TRUE(indices->isSameShape(updates), 0,
                 "SCATTER_SUB OP: when input array has rank = 1 then indices and updates must have the same shapes, "
                 "but got %s and %s correspondingly !",
                 ShapeUtils::shapeAsString(indices).c_str(), ShapeUtils::shapeAsString(updates).c_str());
  } else if (inRank == updRank && indices->isVector()) {
    auto* updShapeVec = updates->getShapeAsVector();
    auto* inShapeVec = input->getShapeAsVector();
    std::vector<LongType> expectedUpdShape = {indices->lengthOf()};
    expectedUpdShape.insert(expectedUpdShape.end(), inShapeVec->begin() + 1, inShapeVec->end());

    REQUIRE_TRUE(expectedUpdShape == *updShapeVec, 0,
                 "SCATTER_SUB OP: wrong shape of updates array, expected is %s, but got %s instead !",
                 ShapeUtils::shapeAsString(expectedUpdShape).c_str(), ShapeUtils::shapeAsString(*updShapeVec).c_str());
    delete updShapeVec;
    delete inShapeVec;
  }

  else {
    REQUIRE_TRUE(updRank == indRank + inRank - 1, 0,
                 "SCATTER_SUB OP: wrong rank of updates array, expected is %i, but got %i instead !",
                 indRank + inRank - 1, updRank);

    auto* updShapeVec = updates->getShapeAsVector();
    auto* inShapeVec = input->getShapeAsVector();
    auto* indShapeVec = indices->getShapeAsVector();
    std::vector<LongType> expectedUpdShape = *indShapeVec;
    expectedUpdShape.insert(expectedUpdShape.end(), inShapeVec->begin() + 1, inShapeVec->end());

    REQUIRE_TRUE(expectedUpdShape == *updShapeVec, 0,
                 "SCATTER_SUB OP: wrong shape of updates array, expected is %s, but got %s instead !",
                 ShapeUtils::shapeAsString(expectedUpdShape).c_str(), ShapeUtils::shapeAsString(*updShapeVec).c_str());
    delete updShapeVec;
    delete inShapeVec;
    delete indShapeVec;
  }

  if (!indices->isEmpty()) {
    if (checkIndices) {
      const LongType numOfBadIndx = helpers::checkIndices(block.launchContext(), *indices, *output, 0);
      REQUIRE_TRUE(numOfBadIndx == 0, 0,
                   "SCATTER_SUB OP: please check elements of indices-array, total number of wrong elements is %lld!",
                   numOfBadIndx);
    }

    // ScatterHelper<T>::template scatterApply<simdOps::Subtract<T>>(output, indices, updates);
    helpers::scatter(block.launchContext(), pairwise::Subtract, *indices, *updates, *output, lock);
  }

  return Status::OK;
}
DECLARE_SYN(ScatterSub, scatter_sub);

DECLARE_TYPES(scatter_sub) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_INTS, ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
