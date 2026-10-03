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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#if NOT_EXCLUDED(OP_bincount)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/weights.h>

namespace sd {
namespace ops {
DECLARE_TYPES(bincount) {
  getOpDescriptor()->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_INTS})
      ->setAllowedInputTypes(1, ANY)
      ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
}

CUSTOM_OP_IMPL(bincount, 1, 1, false, 0, 0) {
  // The shape function sizes the output: max(values) + 1 bins, raised to minLength and capped at maxLength (from
  // iArgs or from inputs 1 and 2 when there are three inputs). Values outside the bins add nothing.
  auto input = INPUT_VARIABLE(0);
  auto result = OUTPUT_VARIABLE(0);

  // Input 1 holds the weights with two inputs, and with four (then inputs 2 and 3 are minLength and maxLength). No
  // weights, or an empty array, counts each value once; a scalar weighs every value the same.
  NDArray* weightsIn = (block.width() == 2 || block.width() > 3) ? INPUT_VARIABLE(1) : nullptr;
  if (weightsIn != nullptr && weightsIn->lengthOf() > 1)
    REQUIRE_TRUE(input->isSameShape(weightsIn), 0, "bincount: the input and weights shapes should be equals");

  // The helpers read INT64 values and weights of the output's type
  NDArray* values = input->dataType() == INT64 ? input : input->cast(INT64);
  NDArray* weights = nullptr;
  if (weightsIn != nullptr && weightsIn->lengthOf() == 1) {
    std::vector<LongType> valuesShape(values->shapeOf(), values->shapeOf() + values->rankOf());
    weights = new NDArray('c', valuesShape, result->dataType(), block.launchContext());
    weights->assign(weightsIn);
  } else if (weightsIn != nullptr && weightsIn->lengthOf() > 1) {
    weights = weightsIn->dataType() == result->dataType() ? weightsIn : weightsIn->cast(result->dataType());
  }

  // the helpers write every bin
  helpers::adjustWeights(block.launchContext(), values, weights, result, 0, static_cast<int>(result->lengthOf()));

  if (weights != weightsIn) delete weights;
  if (values != input) delete values;
  return Status::OK;
}

DECLARE_SHAPE_FN(bincount) {
  auto shapeList = SHAPELIST();
  auto in = INPUT_VARIABLE(0);
  DataType dtype = INT64;
  if (block.width() > 1)
    dtype = ArrayOptions::dataType(inputShape->at(1));
  else if (block.numI() > 2)
    dtype = (DataType)INT_ARG(2);

  LongType maxIndex = in->argMax();
  LongType maxLength = in->e<LongType>(maxIndex) + 1;
  LongType outLength = maxLength;

  if (block.numI() > 0) outLength = math::sd_max(maxLength, INT_ARG(0));

  if (block.numI() > 1) outLength = math::sd_min(outLength, INT_ARG(1));

  if (block.width() == 3) {  // the second argument is min and the third is max
    auto min = INPUT_VARIABLE(1)->e<LongType>(0);
    auto max = min;
    if (INPUT_VARIABLE(2)->lengthOf() > 0) {
      max = INPUT_VARIABLE(2)->e<LongType>(0);
    }

    outLength = math::sd_max(maxLength, min);
    outLength = math::sd_min(outLength, max);
  } else if (block.width() > 3) {
    auto min = INPUT_VARIABLE(2);
    auto max = min;
    if (INPUT_VARIABLE(3)->lengthOf() > 0) {
      max = INPUT_VARIABLE(3);
    }
    outLength = math::sd_max(maxLength, min->e<LongType>(0));
    outLength = math::sd_min(outLength, max->e<LongType>(0));
  }

  auto newshape = ConstantShapeHelper::getInstance().vectorShapeInfo(outLength, dtype);

  shapeList->push_back(newshape);
  return shapeList;
}

}  // namespace ops
}  // namespace sd

#endif
