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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_stack)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/stack.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(stack, -1, 1, false, 0, 0) {
  auto output = OUTPUT_VARIABLE(0);

  // no-op in case of empty output array
  if (output->isEmpty()) return Status::OK;

  const int numInputs = block.width();
  if (numInputs == 0) return Status::OK;

  auto input = INPUT_VARIABLE(0);
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dim < 0) dim += input->rankOf() + 1;

  REQUIRE_TRUE(dim >= 0, 0,
               "STACK op: the input dimension parameter must be non-negative after normalization, but got %i!", dim);
  REQUIRE_TRUE(
      dim <= input->rankOf(), 0,
      "STACK op: the input dimension parameter must be <= rank of input arrays shapes (rank=%i), but got %i instead !",
      input->shapeOf(), dim);

  // Collect all input arrays and validate shapes in single pass
  std::vector<NDArray*> inArrs(numInputs);
  inArrs[0] = input;
  const sd::LongType* firstShape = input->shapeInfo();

  // Helper lambda to check if two shapes are compatible for stacking
  // Treats scalar [] and [1] as compatible (common in ONNX models)
  auto shapesCompatibleForStack = [](const sd::LongType* shapeA, const sd::LongType* shapeB) -> bool {
    // If shapes are equal, they're compatible
    if (shape::equalsSoft(shapeA, shapeB)) return true;

    // Check if one is scalar and other is [1] - treat as compatible
    int rankA = shape::rank(shapeA);
    int rankB = shape::rank(shapeB);
    sd::LongType lengthA = shape::length(shapeA);
    sd::LongType lengthB = shape::length(shapeB);

    // Both must have length 1 (scalar or [1] tensor)
    if (lengthA == 1 && lengthB == 1) {
      // Allow scalar vs [1] compatibility
      if ((rankA == 0 && rankB == 1) || (rankA == 1 && rankB == 0)) {
        return true;
      }
    }

    return false;
  };

  for (int i = 1; i < numInputs; ++i) {
    inArrs[i] = INPUT_VARIABLE(i);
    REQUIRE_TRUE(shapesCompatibleForStack(firstShape, inArrs[i]->shapeInfo()), 0,
                 "STACK op: the shapes of all input arrays must be the same or compatible (scalar vs [1])!");
  }

  // empty arrays are a no op
  if (!inArrs[0]->isEmpty())
    helpers::stack(block.launchContext(), inArrs, *output, dim);

  return Status::OK;
}
DECLARE_SYN(pack, stack);
DECLARE_SYN(Pack, stack);

DECLARE_TYPES(stack) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_STACK);
}

DECLARE_SHAPE_FN(stack) {
  // check whether input dimension is within rank range
  auto inShapeInfo = inputShape->at(0);

  int rank = shape::rank(inShapeInfo);
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dim < 0) dim += rank + 1;

  REQUIRE_TRUE(dim >= 0, 0,
               "STACK op: the input dimension parameter must be non-negative after normalization, but got %i!", dim);
  REQUIRE_TRUE(
      dim <= inShapeInfo[0], 0,
      "STACK op: the input dimension parameter must be <= rank of input arrays shapes (rank=%i), but got %i instead !",
      inShapeInfo[0], dim);


  // the rank of output ShapeInfo is larger by one compared to input ShapeInfo
  std::vector<LongType> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);

  // insert (int) block.width() at dim position of input shape to get output shape
  outShape.insert(outShape.begin() + LongType(dim), (LongType)block.width());
  auto ret = SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(ArrayOptions::dataType(inShapeInfo),
                                                                             shape::order(inShapeInfo),
                                                                             outShape)->primary());
  return ret;
}

}  // namespace ops
}  // namespace sd

#endif
