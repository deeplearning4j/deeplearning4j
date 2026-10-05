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
// Created to use with batched tensor by GS <sgazeos@gmail.com> 3/27/2018
//

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/sequence_mask.h>
#if NOT_EXCLUDED(OP_sequence_mask)
namespace sd {
namespace ops {
// mask[..., j] = j < lengths[...]. The width of the mask is decided in one place, the shape function: the second
// input (the maximum length) or, without one, the first integer argument, but never less than the longest length. The
// helpers take the width from the output, so the execution cannot disagree with the shape (it used the position of the
// longest length, or an integer argument that the shape function reads as the data type).
CUSTOM_OP_IMPL(sequence_mask, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const LongType width = output->sizeAt(output->rankOf() - 1);
  REQUIRE_TRUE(output->lengthOf() == input->lengthOf() * width, 0,
               "sequence_mask: the output should have the shape of the input and one more dimension, but its length is "
               "%lld for %lld lengths of width %lld.",
               static_cast<long long>(output->lengthOf()), static_cast<long long>(input->lengthOf()),
               static_cast<long long>(width));

  helpers::sequenceMask(block.launchContext(), input, output, static_cast<int>(width));

  return Status::OK;
}

DECLARE_SHAPE_FN(sequence_mask) {
  LongType* outShapeInfo = nullptr;
  auto in = inputShape->at(0);
  int outRank = shape::rank(in) + 1;
  auto input = INPUT_VARIABLE(0);
  auto dtype = BOOL;
  // the longest length (no length, no negative one, counts)
  LongType max = 0;
  if (input->lengthOf() > 0) {
    auto argMaxInd = input->argMax();
    max = input->e<LongType>(argMaxInd);
    if (max < 0) max = 0;
  }
  LongType maxInd = max;

  if (block.numD() > 0) dtype = D_ARG(0);

  if (block.width() > 1) {
    auto maxlen = INPUT_VARIABLE(1);
    LongType tmaxlen = maxlen->e<LongType>(0);
    if (tmaxlen > max) maxInd = static_cast<LongType>(tmaxlen);
    if (block.numI() > 0) {
      dtype = (DataType)INT_ARG(0);
    }
  } else {
    if (block.numI() > 0) {
      maxInd = INT_ARG(0);
    }
    if (maxInd < max) maxInd = max;
    if (block.numI() > 1) dtype = (DataType)INT_ARG(1);  // to work with legacy code
  }

  const LongType lastDimension = maxInd;
  ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), sd::LongType);
  outShapeInfo[0] = outRank;
  for (LongType i = 0; i < outRank - 1; ++i) outShapeInfo[i + 1] = shape::sizeAt(in, i);
  outShapeInfo[outRank] = lastDimension;

  ShapeUtils::updateStridesAndType(outShapeInfo, dtype, shape::order(in));

  return SHAPELIST(CONSTANT(outShapeInfo));
}

DECLARE_TYPES(sequence_mask) {
  getOpDescriptor()->addTraits(OP_TRAIT_CONSTANT_GENERATION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE | OP_TRAIT_DATA_DEPENDENT);
  getOpDescriptor()->setAllowedInputTypes({ALL_INTS})->setAllowedOutputTypes(ANY);
}
}  // namespace ops
}  // namespace sd
#endif
