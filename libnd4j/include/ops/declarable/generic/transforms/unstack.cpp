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
#if NOT_EXCLUDED(OP_unstack)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/stack.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(unstack, 1, -1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto dim = INT_ARG(0);
  if (dim < 0) dim += input->rankOf();

  // Handle empty input - create empty outputs with correct shapes
  if (input->isEmpty() || input->lengthOf() == 0) {
    LongType numOutputs = input->sizeAt(dim);
    for (LongType i = 0; i < numOutputs; i++) {
      auto output = OUTPUT_VARIABLE(i);
      if (output != nullptr && (output->isEmpty() || output->lengthOf() == 0)) {
        // Output is already empty, nothing to do
        continue;
      }
    }
    return Status::OK;
  }

  REQUIRE_TRUE(dim < input->rankOf(), 0,
               "Unstack dimension should be lower then rank of input %i, but got dimension=%i !", input->rankOf(), dim);
  REQUIRE_TRUE(dim >= 0, 0, "Unstack dimension should be non-negative value, but got %i !", dim);


  std::vector<NDArray*> outArrs(input->sizeAt(dim));
  for (size_t i = 0; i < outArrs.size(); ++i) outArrs[i] = OUTPUT_VARIABLE(i);

  helpers::unstack(block.launchContext(), *input, outArrs, dim);

  return Status::OK;
}

DECLARE_SYN(unpack, unstack);

DECLARE_SHAPE_FN(unstack) {
  auto inShapeInfo = inputShape->at(0);

  auto dim = INT_ARG(0);
  if (dim < 0) dim += shape::rank(inShapeInfo);
  REQUIRE_TRUE(dim >= 0, 0, "UNSTACK op: dimension must be non-negative after normalization, but got %i!", dim);
  REQUIRE_TRUE(dim < shape::rank(inShapeInfo), 0,
               "UNSTACK op: dimension must be < rank (%i), but got %i!", shape::rank(inShapeInfo), dim);

  // Calculate number of outputs from the unstack dimension
  const LongType numTads = block.numI() > 1 ? I_ARG(1) : shape::shapeOf(inShapeInfo)[dim];

  // Build output shape (input shape without the unstack dimension)
  // IMPORTANT: Must read shape values directly and copy them, not use pointers that may become invalid
  std::vector<LongType> outShape;
  outShape.reserve(shape::rank(inShapeInfo) - 1);
  auto* shapeOfInput = shape::shapeOf(inShapeInfo);
  for (LongType i = 0; i < shape::rank(inShapeInfo); ++i) {
    if (i != dim) {
      outShape.push_back(shapeOfInput[i]);
    }
  }

  // Check if input has any zero dimensions (which makes it empty)
  bool inHasZeroDim = false;
  for (int i = 0; i < shape::rank(inShapeInfo); i++) {
    if (shapeOfInput[i] == 0) {
      inHasZeroDim = true;
      break;
    }
  }

  // If input has zero dimensions, all outputs should be empty with ARRAY_EMPTY flag
  if (inHasZeroDim || shape::isEmptyConst(inShapeInfo) || ArrayOptions::arrayType(inShapeInfo) == EMPTY) {
    auto result = SHAPELIST();
    for (LongType i = 0; i < numTads; ++i) {
      // Create shape info directly - don't use caching to avoid issues with shared buffers
      LongType* outShapeInfo = ShapeBuilders::createShapeInfo(ArrayOptions::dataType(inShapeInfo), 'c', outShape.size(), outShape.data(), nullptr, true);
      result->push_back(outShapeInfo);
    }
    // Handle edge case: if numTads is 0, still create one output
    if (numTads < 1) {
      LongType* outShapeInfo = ShapeBuilders::createShapeInfo(ArrayOptions::dataType(inShapeInfo), 'c', outShape.size(), outShape.data(), nullptr, true);
      result->push_back(outShapeInfo);
    }
    return result;
  }

  if(!inHasZeroDim && !shape::isEmptyConst(inShapeInfo)) {
    REQUIRE_TRUE(dim < inShapeInfo[0], 0,
                 "UNSTACK op: dimension should be lower then rank of input %i, but got dimension=%i !", inShapeInfo[0],
                 dim);
    REQUIRE_TRUE(dim >= 0, 0, "UNSTACK op: dimension should be non-negative value, but got %i !", dim);
  }

  std::vector<LongType> dimVec = {dim};
  std::vector<LongType> *dims = ShapeUtils::evalDimsToExclude(inShapeInfo[0], 1, dimVec.data());

  if (dims->size() == 0 && shape::rank(inShapeInfo) == 1) {  // split vector into lengthOf scalars
    auto result = SHAPELIST();
    for (LongType e = 0; e < numTads; e++)
      result->push_back(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inShapeInfo)));
    delete dims;

    return result;
  }

  std::vector<LongType> subArrShape(shape::rank(inShapeInfo) - 1);

  for (LongType j = 0, i = 0; i < shape::rank(inShapeInfo); i++)
    if (i != dim) subArrShape[j++] = shape::shapeOf(inShapeInfo)[i];

  // remove leading and trailing 1
  if (inShapeInfo[0] == 2 && subArrShape.size() == 2) {
    if (subArrShape[0] == 1)
      subArrShape.erase(subArrShape.begin());
    else if (subArrShape[1] == 1)
      subArrShape.erase(subArrShape.end());
  }

  auto result = SHAPELIST();
  for (int e = 0; e < numTads; e++) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShapeInfo),
                                                                       shape::order(inShapeInfo), subArrShape);
    result->push_back(newShape);
  }
  return result;
}

DECLARE_TYPES(unstack) { getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})->setSameMode(true); }

}  // namespace ops
}  // namespace sd

#endif
