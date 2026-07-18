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
// @author Adam Gibson
// Fused cast and scale operation for mixed precision training
// Combines type casting and scaling into a single kernel pass
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_cast_and_scale)

#include <array/DataTypeUtils.h>
#include <ops/declarable/headers/parity_ops.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(cast_and_scale, 1, 1, false, 1, 1) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get scale factor from T_ARG
  double scale = T_ARG(0);

  if (input->isEmpty()) {
    REQUIRE_TRUE(output->isEmpty(), 0, "cast_and_scale: If input is empty, output must also be empty");
    return sd::Status::OK;
  }

  // Fast path: same data type and scale is 1.0 - just copy
  if (input->dataType() == output->dataType() && scale == 1.0) {
    if (!block.isInplace()) {
      output->assign(input);
    }
    return sd::Status::OK;
  }

  // Fast path: same data type, just scale
  if (input->dataType() == output->dataType()) {
    input->applyScalar(sd::scalar::Multiply, scale, output);
    return sd::Status::OK;
  }

  // Different types: cast then scale in two bulk operations
  // First cast by assigning input to output (handles type conversion),
  // then scale the output in-place. Both use bulk ops, avoiding O(n^2) per-element sync.
  output->assign(input);
  output->applyScalar(sd::scalar::Multiply, scale, output);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(cast_and_scale) {
  auto inShape = inputShape->at(0);

  // Get target data type from integer argument
  auto targetTypeInt = INT_ARG(0);
  DataType targetType = DataTypeUtils::fromInt(targetTypeInt);

  // Create output shape with target data type
  auto newShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(targetType, inShape);

  return SHAPELIST(newShapeInfo);
}

DECLARE_TYPES(cast_and_scale) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes(sd::DataType::ANY);
  getOpDescriptor()->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING | OP_TRAIT_CAST);
}

}  // namespace ops
}  // namespace sd

#endif
