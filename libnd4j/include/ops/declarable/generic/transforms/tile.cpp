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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_tile)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(tile, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int inRank = input->rankOf();
  std::vector<sd::LongType> reps;

  if (block.getIArguments()->size() == static_cast<size_t>(inRank)) {
    reps = ArrayUtils::toLongVector(*(block.getIArguments()));
  } else if (block.width() > 1) {
    auto reps_vector = INPUT_VARIABLE(1);
    REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0,
                 "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !",
                 reps_vector->lengthOf(), inRank);

    reps = reps_vector->template asVectorT<sd::LongType>();
  } else {
    REQUIRE_TRUE(false, 0,
                 "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank "
                 "of input array to be tiled !");
  }

  auto repProd = shape::prodLong(reps.data(), reps.size());
  REQUIRE_TRUE(repProd > 0, 0, "TILE op: reps can't contain 0s");

  input->tile(reps, *output);

  return sd::Status::OK;
}

DECLARE_TYPES(tile) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}

DECLARE_SHAPE_FN(tile) {
  auto inShape = inputShape->at(0);
  const int inRank = inShape[0];
  std::vector<sd::LongType> reps;

  if (block.getIArguments()->size() == static_cast<size_t>(inRank)) {
    reps = ArrayUtils::toLongVector(*(block.getIArguments()));
  } else if (block.width() > 1) {
    auto reps_vector = INPUT_VARIABLE(1);
    REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0,
                 "TILE op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !",
                 reps_vector->lengthOf(), inRank);
    reps = reps_vector->template asVectorT<sd::LongType>();
  } else {
    REQUIRE_TRUE(false, 0,
                 "TILE op: this op requires repeats vector, either as IArgs or second array with length equal to rank "
                 "of input array to be tiled !");
  }

  auto repProd = shape::prodLong(reps.data(), reps.size());
  REQUIRE_TRUE(repProd > 0, 0, "TILE op: reps can't contain 0s");

  std::vector<sd::LongType> shape(inRank);
  for (sd::LongType e = 0; e < shape::rank(inShape); e++) shape[e] = shape::sizeAt(inShape, e) * reps[e];

  auto newShape =
      ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), shape::order(inShape), shape);
  return SHAPELIST(newShape);
}

////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tile_bp, 2, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  const int inRank = input->rankOf();

  std::vector<sd::LongType> reps;

  if (block.getIArguments()->size() == static_cast<size_t>(inRank)) {
    reps = ArrayUtils::toLongVector(*(block.getIArguments()));
  } else if (block.width() > 2) {
    auto reps_vector = INPUT_VARIABLE(1);
    REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0,
                 "TILE_BP op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !",
                 reps_vector->lengthOf(), inRank);

    reps = reps_vector->template asVectorT<sd::LongType>();
    gradO = INPUT_VARIABLE(2);
  } else {
    REQUIRE_TRUE(false, 0,
                 "TILE_BP op: this op requires repeats vector, either as IArgs or second array with length equal to "
                 "rank of input array to be tiled !");
  }

  REQUIRE_TRUE(inRank == gradO->rankOf(), 0,
               "TILE_BP op: the ranks of input array and output's gradients array (next epsilon) must be equal, but "
               "got %i and %i correspondingly !",
               inRank, gradO->rankOf());

  for (int i = 0; i < inRank; ++i)
    REQUIRE_TRUE(gradO->sizeAt(i) == gradI->sizeAt(i) * reps[i], 0,
                 "TILE_BP op: shapes of input array and output's gradients array (next epsilon) are inconsistent !");

  helpers::tileBP(block.launchContext(), *gradO, *gradI, reps);

  return sd::Status::OK;
}

DECLARE_TYPES(tile_bp) {
  getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(1, {ALL_INTS, ALL_FLOATS});
  getOpDescriptor()->setAllowedInputTypes(2, {ALL_FLOATS});

  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(tile_bp) {
  auto inShape = inputShape->at(0);
  auto gradOShape = inputShape->at(1);
  const int inRank = inShape[0];

  std::vector<sd::LongType> reps;

  if (block.getIArguments()->size() == static_cast<size_t>(inRank)) {
    reps = ArrayUtils::toLongVector(*(block.getIArguments()));
  } else if (block.width() > 2) {
    auto reps_vector = INPUT_VARIABLE(1);
    REQUIRE_TRUE(reps_vector->lengthOf() == inRank, 0,
                 "TILE_BP op: repeats vector length should be equal to input rank, but got %i and %i correspondingly !",
                 reps_vector->lengthOf(), inRank);
    reps = reps_vector->template asVectorT<sd::LongType>();
    gradOShape = inputShape->at(2);
  } else {
    REQUIRE_TRUE(false, 0,
                 "TILE_BP op: this op requires repeats vector, either as IArgs or second array with length equal to "
                 "rank of input array to be tiled !");
  }

  REQUIRE_TRUE(inRank == gradOShape[0], 0,
               "TILE_BP op: the ranks of input array and output's gradients array (next epsilon) must be equal, but "
               "got %i and %i correspondingly !",
               inRank, gradOShape[0]);

  for (sd::LongType i = 0; i < inRank; ++i)
    REQUIRE_TRUE(shape::sizeAt(gradOShape, i) == shape::sizeAt(inShape, i) * reps[i], 0,
                 "TILE_BP op: shapes of input array and output's gradients array (next epsilon) are inconsistent !");

  return SHAPELIST(CONSTANT(inShape));
}

}  // namespace ops
}  // namespace sd

#endif
