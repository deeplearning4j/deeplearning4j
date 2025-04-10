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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/segment.h>
#if NOT_EXCLUDED(OP_unsorted_segment_sqrt_n)
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(unsorted_segment_sqrt_n, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto reshapedInput = *input;

  auto idxSegments = INPUT_VARIABLE(1);
  auto reshapedSegments = *idxSegments;
  if (!idxSegments->isVector() && idxSegments->rankOf() > 1) {
    std::vector<sd::LongType> shape = {idxSegments->lengthOf()};
    reshapedSegments = idxSegments->reshape('c', shape, false);
  }

  auto segmentedOutput = OUTPUT_NULLIFIED(0);
  LongType numOfClasses = block.width() == 3 ? INPUT_VARIABLE(2)->e<LongType>(0) : INT_ARG(0);
  REQUIRE_TRUE(reshapedSegments.isVector(), 0,
               "unsorted_segment_sqrt_n: segment indexes array should be a vector, but it rank is %i.",
               idxSegments->rankOf());
  REQUIRE_TRUE(reshapedSegments.lengthOf() == input->sizeAt(0), 0,
               "unsorted_segment_sqrt_n: segment indexes array length should be equal to the input first dimension, "
               "but %ld != %ld.",
               reshapedSegments.lengthOf(), input->sizeAt(0));

  LongType wrong;

  REQUIRE_TRUE(helpers::unsortedSegmentIndicesValidate(block.launchContext(), &reshapedSegments, numOfClasses, wrong),
               0, "unsorted_segment_sqrt_n: segment indices should be in range [0, %ld), but %ld != %ld", numOfClasses,
               wrong, numOfClasses);
  helpers::unsortedSegmentSqrtNFunctor(block.launchContext(), &reshapedInput, &reshapedSegments, numOfClasses,
                                       segmentedOutput);

  return Status::OK;
}

DECLARE_SHAPE_FN(unsorted_segment_sqrt_n) {
  auto in = inputShape->at(0);
  int outRank = shape::rank(in);
  LongType* outputShape = nullptr;
  LongType numOfClasses = block.width() == 3 ? INPUT_VARIABLE(2)->e<LongType>(0) : INT_ARG(0);

  if (INPUT_VARIABLE(0)->rankOf() >= 2) {
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
DECLARE_TYPES(unsorted_segment_sqrt_n) {
  getOpDescriptor()
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setSameMode(false);
}

CUSTOM_OP_IMPL(unsorted_segment_sqrt_n_bp, 3, 2, false, 0, 1) {
  return helpers::unsortedSegmentSqrtNFunctorBP(block.launchContext(), INPUT_VARIABLE(0), INPUT_VARIABLE(1),
                                                INPUT_VARIABLE(2), INT_ARG(0), OUTPUT_NULLIFIED(0));
}
DECLARE_TYPES(unsorted_segment_sqrt_n_bp) {
  getOpDescriptor()
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setSameMode(false);
}

DECLARE_SHAPE_FN(unsorted_segment_sqrt_n_bp) {
  auto in = inputShape->at(0);
  auto inIdx = inputShape->at(1);
  return SHAPELIST(CONSTANT(in), CONSTANT(inIdx));
}

}  // namespace ops
}  // namespace sd

#endif
