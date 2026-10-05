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
// Created by george@skymind.io on 9/6/2018.
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/segment.h>
#if NOT_EXCLUDED(OP_unsorted_segment_sum)
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(unsorted_segment_sum, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto idxSegments = INPUT_VARIABLE(1);
  auto segmentedOutput = OUTPUT_VARIABLE(0);
  LongType numOfClasses = block.width() == 3 ? INPUT_VARIABLE(2)->e<LongType>(0) : INT_ARG(0);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "unsorted_segment_sum: the input should have rank >= 1, but it is a scalar.");
  REQUIRE_TRUE(idxSegments->rankOf() >= 1 && idxSegments->lengthOf() >= 1, 0,
               "unsorted_segment_sum: segment indexes array should be a non-empty array, but it has rank %i.",
               idxSegments->rankOf());
  REQUIRE_TRUE(numOfClasses >= 0, 0, "unsorted_segment_sum: the number of segments should not be negative, but it is %lld.",
               static_cast<long long>(numOfClasses));
  REQUIRE_TRUE(idxSegments->lengthOf() == 1 || idxSegments->lengthOf() == input->sizeAt(0), 0,
               "unsorted_segment_sum: segment indexes array length should be equal to the input first dimension, but "
               "%ld != %ld.",
               idxSegments->lengthOf(), input->sizeAt(0));
  REQUIRE_TRUE(segmentedOutput->dataType() == input->dataType(), 0,
               "unsorted_segment_sum: the output type (%s) should be the input type (%s).",
               DataTypeUtils::asString(segmentedOutput->dataType()).c_str(),
               DataTypeUtils::asString(input->dataType()).c_str());

  LongType wrong;
  REQUIRE_TRUE(helpers::unsortedSegmentIndicesValidate(block.launchContext(), idxSegments, numOfClasses, wrong), 0,
               "unsorted_segment_sum: segment indices should be in range [0, %lld), but the id %lld is not.",
               static_cast<long long>(numOfClasses), static_cast<long long>(wrong));
  helpers::unsortedSegmentSumFunctor(block.launchContext(), input, idxSegments, numOfClasses, segmentedOutput);
  return Status::OK;
}

DECLARE_TYPES(unsorted_segment_sum) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setSameMode(false);
}

DECLARE_SHAPE_FN(unsorted_segment_sum) {
  auto in = inputShape->at(0);
  int outRank = shape::rank(in);
  LongType* outputShape = nullptr;
  LongType numOfClasses = block.width() == 3 ? INPUT_VARIABLE(2)->e<LongType>(0) : INT_ARG(0);
  if (numOfClasses < 0) numOfClasses = 0;

  if (shape::rank(in) >= 2) {
    ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);
    outputShape[0] = outRank;
    outputShape[1] = numOfClasses;
    for (LongType i = 1; i < outRank; i++) outputShape[i + 1] = shape::sizeAt(in, i);

    ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));

  } else {
    ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(1), sd::LongType);
    outputShape[0] = 1;
    outputShape[1] = numOfClasses;
    ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));
  }

  return SHAPELIST(CONSTANT(outputShape));
}
CUSTOM_OP_IMPL(unsorted_segment_sum_bp, 3, 2, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto gradOut = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);
  auto outIndices = OUTPUT_VARIABLE(1);
  const LongType numOfClasses = INT_ARG(0);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "unsorted_segment_sum_bp: the input should have rank >= 1, but it is a scalar.");
  REQUIRE_TRUE(indices->lengthOf() == input->sizeAt(0), 0,
               "unsorted_segment_sum_bp: segment indexes array length should be equal to the input first dimension, but "
               "%lld != %lld.",
               static_cast<long long>(indices->lengthOf()), static_cast<long long>(input->sizeAt(0)));
  REQUIRE_TRUE(gradOut->rankOf() == input->rankOf(), 0,
               "unsorted_segment_sum_bp: the gradient should have the rank of the input, but %i != %i.",
               gradOut->rankOf(), input->rankOf());
  for (LongType d = 1; d < input->rankOf(); ++d) {
    REQUIRE_TRUE(gradOut->sizeAt(d) == input->sizeAt(d), 0,
                 "unsorted_segment_sum_bp: the gradient and the input should have equal dimensions after the first, but "
                 "dimension %lld is %lld != %lld.",
                 static_cast<long long>(d), static_cast<long long>(gradOut->sizeAt(d)),
                 static_cast<long long>(input->sizeAt(d)));
  }
  REQUIRE_TRUE(output->dataType() == gradOut->dataType(), 0,
               "unsorted_segment_sum_bp: the output type (%s) should be the gradient type (%s).",
               DataTypeUtils::asString(output->dataType()).c_str(),
               DataTypeUtils::asString(gradOut->dataType()).c_str());
  outIndices->assign(indices);
  return helpers::unsortedSegmentSumFunctorBP(block.launchContext(), input, indices, gradOut, numOfClasses, output);
}

DECLARE_SHAPE_FN(unsorted_segment_sum_bp) {
  auto in = inputShape->at(0);
  auto inIdx = inputShape->at(1);
  return SHAPELIST(CONSTANT(in), CONSTANT(inIdx));
}
DECLARE_TYPES(unsorted_segment_sum_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD | OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(ANY)
      ->setSameMode(false);
}

}  // namespace ops

}  // namespace sd

#endif
