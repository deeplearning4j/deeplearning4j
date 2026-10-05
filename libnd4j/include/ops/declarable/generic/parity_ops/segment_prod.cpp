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
// Created by george@skymind.io on 2/21/2018.
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/segment.h>
#if NOT_EXCLUDED(OP_segment_prod)
#include <array/NDArrayFactory.h>
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(segment_prod, 2, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto idxSegments = INPUT_VARIABLE(1);
  auto segmentedOutput = OUTPUT_VARIABLE(0);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "segment_prod: the input should have rank >= 1, but it is a scalar.");
  REQUIRE_TRUE(idxSegments->isVector(), 0, "segment_prod: segment indexes array should be a vector, but it rank is %i.",
               idxSegments->rankOf());
  REQUIRE_TRUE(idxSegments->lengthOf() == input->sizeAt(0), 0,
               "segment_prod: segment indexes array length should be equal to the input first dimension, but %i != %i.",
               idxSegments->lengthOf(), input->sizeAt(0));
  REQUIRE_TRUE(segmentedOutput->dataType() == input->dataType(), 0,
               "segment_prod: the output type (%s) should be the input type (%s).",
               DataTypeUtils::asString(segmentedOutput->dataType()).c_str(),
               DataTypeUtils::asString(input->dataType()).c_str());

  LongType previous = 0;
  LongType offending = 0;
  REQUIRE_TRUE(helpers::segmentIndicesValidate(block.launchContext(), idxSegments, previous, offending), 0,
               "segment_prod: segment indices should be non-negative and arranged in ascending order, but the id %lld "
               "follows the id %lld.",
               static_cast<long long>(offending), static_cast<long long>(previous));

  helpers::segmentProdFunctor(block.launchContext(), input, idxSegments, segmentedOutput);
  return Status::OK;
}

DECLARE_SHAPE_FN(segment_prod) {
  auto idxVector = INPUT_VARIABLE(1);

  auto in = inputShape->at(0);
  const LongType inRank = shape::rank(in);
  const LongType outRank = inRank < 1 ? 1 : inRank;
  LongType* outputShape = nullptr;
  // the classes are 0 .. last id (the ids are sorted); no ids, no classes
  LongType numOfClasses = 0;
  const LongType idsLength = shape::length(inputShape->at(1));
  if (idsLength > 0) numOfClasses = idxVector->e<LongType>(idsLength - 1) + 1;
  if (numOfClasses < 0) numOfClasses = 0;

  ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);

  outputShape[0] = outRank;
  outputShape[1] = numOfClasses;
  for (LongType i = 1; i < inRank; ++i) outputShape[i + 1] = shape::sizeAt(in, i);

  ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));

  return SHAPELIST(CONSTANT(outputShape));
}

CUSTOM_OP_IMPL(segment_prod_bp, 3, 2, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto gradOut = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);
  auto outIndices = OUTPUT_VARIABLE(1);
  REQUIRE_TRUE(input->rankOf() >= 1, 0, "segment_prod_bp: the input should have rank >= 1, but it is a scalar.");
  REQUIRE_TRUE(indices->lengthOf() == input->sizeAt(0), 0,
               "segment_prod_bp: segment indexes array length should be equal to the input first dimension, but %i != %i.",
               indices->lengthOf(), input->sizeAt(0));
  REQUIRE_TRUE(gradOut->rankOf() == input->rankOf(), 0,
               "segment_prod_bp: the gradient should have the rank of the input, but %i != %i.", gradOut->rankOf(),
               input->rankOf());
  for (LongType d = 1; d < input->rankOf(); ++d) {
    REQUIRE_TRUE(gradOut->sizeAt(d) == input->sizeAt(d), 0,
                 "segment_prod_bp: the gradient and the input should have equal dimensions after the first, but "
                 "dimension %lld is %lld != %lld.",
                 static_cast<long long>(d), static_cast<long long>(gradOut->sizeAt(d)),
                 static_cast<long long>(input->sizeAt(d)));
  }
  REQUIRE_TRUE(output->dataType() == gradOut->dataType(), 0,
               "segment_prod_bp: the output type (%s) should be the gradient type (%s).",
               DataTypeUtils::asString(output->dataType()).c_str(),
               DataTypeUtils::asString(gradOut->dataType()).c_str());
  REQUIRE_TRUE(input->dataType() == output->dataType(), 0,
               "segment_prod_bp: the input type (%s) should be the gradient type (%s).",
               DataTypeUtils::asString(input->dataType()).c_str(), DataTypeUtils::asString(output->dataType()).c_str());
  outIndices->assign(indices);
  return helpers::segmentProdFunctorBP(block.launchContext(), input, indices, gradOut, output);
}

DECLARE_TYPES(segment_prod) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS})
      ->setSameMode(false);
}

DECLARE_SHAPE_FN(segment_prod_bp) {
  auto in = inputShape->at(0);
  auto inIdx = inputShape->at(1);
  return SHAPELIST(CONSTANT(in), CONSTANT(inIdx));
}

DECLARE_TYPES(segment_prod_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD | OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()
      ->setAllowedInputTypes(ANY)
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->setAllowedOutputTypes(1, {ALL_INTS})
      ->setSameMode(false);
}
}  // namespace ops
#include <array/NDArrayFactory.h>
}  // namespace sd
#endif
