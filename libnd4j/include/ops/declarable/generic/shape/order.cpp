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
// Created by raver119 on 12.02.18.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_order)

#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(order, 1, 1, false, 0, 1) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  output->assign(input);

  return Status::OK;
}

DECLARE_TYPES(order) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedOutputTypes({ALL_INTS});
}

DECLARE_SHAPE_FN(order) {
  auto input = inputShape->at(0);

  auto isFOrder = INT_ARG(0) == 1;

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(input), isFOrder ? 'f' : 'c', shape::rank(input), shape::shapeOf(input), -1);
  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd

#endif
