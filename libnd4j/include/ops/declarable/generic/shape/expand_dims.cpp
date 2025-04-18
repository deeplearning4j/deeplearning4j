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
// Created by raver119 on 02.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_expand_dims)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(expand_dims, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  LongType axis = block.numI() > 0 ? INT_ARG(0) : INPUT_VARIABLE(1)->e<LongType>(0);

  if (axis < 0) axis += input->rankOf() + 1;
  if(!input->isEmpty() && !input->isScalar())
  REQUIRE_TRUE(axis >= 0 && axis <= input->rankOf(), 0,
               "ExpandDims: axis should be in range of 0...%i in this case, but got %i instead", input->rankOf() + 1,
               axis);


  //note we used to have a specific copy case here but we should
  //be abstracting away data copy and reshape details like buffer copying
  if(input->isEmpty()) {
    return Status::OK;
  }

  std::vector<sd::LongType> shape = output->getShapeAsVector();
  //the shape was already determined in the calculate shape info, just reshape to the same shape as the output
  auto tmp = input->reshape(input->ordering(), shape,false);
  output->assign(&tmp);
  output->syncToHost();
  return Status::OK;
}

DECLARE_TYPES(expand_dims) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(expand_dims) {
  auto inShape = inputShape->at(0);
  auto rank = shape::rank(inShape);
  // 0D scalar edge case
  if (shape::isScalar(inShape)) {
    if(rank < 1) {
      LongType x = 1;
      auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), 'c', 1, &x, -1);
      return SHAPELIST(newShape);
    } else {
      std::vector<LongType> x = {1, 1};
      auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), 'c', 2, x.data(), -1);
      return SHAPELIST(newShape);
    }

  }

  auto input = INPUT_VARIABLE(0);
  if(input->isEmpty() && input->rankOf() < 1) {
    auto newShape = ConstantShapeHelper::getInstance().emptyShapeInfo(ArrayOptions::dataType(inShape));
    return SHAPELIST(newShape);
  }


  auto x_rank = shape::rank(inShape);
  char order = shape::order(inShape);

  LongType axis = block.numI() > 0 ? INT_ARG(0) : INPUT_VARIABLE(1)->e<LongType>(0);
  if (axis < 0) axis += x_rank + 1;

  REQUIRE_TRUE(axis >= 0 && axis <= input->rankOf(), 0,
               "ExpandDims: axis should be in range of 0...%i in this case, but got %i instead", input->rankOf() + 1,
               axis);

  std::vector<LongType> shape;
  for (LongType e = 0; e < x_rank; e++) shape.emplace_back(shape::shapeOf(inShape)[e]);

  shape.insert(shape.begin() + axis, 1);

  auto newShape = input->isEmpty() ? ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(ArrayOptions::dataType(inShape), shape) :
                                   ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), order, shape);
  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd

#endif
