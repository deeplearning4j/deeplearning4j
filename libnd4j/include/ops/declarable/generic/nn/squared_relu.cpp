/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Squared ReLU op definition
// Forward:  out = max(0, x)^2
// Backward: dIn = dOut * 2 * x * (x > 0)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_squared_relu)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/squared_relu.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(squared_relu, 1, 1, true, 0, 0) {
  auto input  = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  helpers::squaredRelu(block.launchContext(), input, output);

  return sd::Status::OK;
}

DECLARE_TYPES(squared_relu) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(squared_relu) {
  auto inShape = inputShape->at(0);
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inShape),
      shape::order(inShape),
      shape::rank(inShape),
      shape::shapeOf(inShape));
  return SHAPELIST(outShape);
}

CUSTOM_OP_IMPL(squared_relu_bp, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto dLdO  = INPUT_VARIABLE(1);
  auto dLdI  = OUTPUT_VARIABLE(0);

  helpers::squaredReluBp(block.launchContext(), input, dLdO, dLdI);

  return sd::Status::OK;
}

DECLARE_TYPES(squared_relu_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(squared_relu_bp) {
  auto inShape = inputShape->at(0);
  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inShape),
      shape::order(inShape),
      shape::rank(inShape),
      shape::shapeOf(inShape));
  return SHAPELIST(outShape);
}

}  // namespace ops
}  // namespace sd

#endif
