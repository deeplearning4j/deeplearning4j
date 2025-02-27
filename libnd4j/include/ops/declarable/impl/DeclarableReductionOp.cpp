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
// Created by raver119 on 07.10.2017.
//
#include <array/DataTypeUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/DeclarableReductionOp.h>

namespace sd {
namespace ops {
DeclarableReductionOp::DeclarableReductionOp(int numInputs, int numOutputs, const char* opName, bool allowsInplace,
                                             int tArgs, int iArgs)
    : DeclarableOp(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
  //
}

ShapeList* DeclarableReductionOp::calculateOutputShape(ShapeList* inputShape, Context& block) {
  std::vector<LongType> dims;
  if (inputShape->size() > 1) {
    // the second argument is axis
    auto axis = INPUT_VARIABLE(1);
    for (int e = 0; e < axis->lengthOf(); e++) dims.push_back(axis->e<int>(e));
  } else if (block.getIArguments()->size())
    for (size_t e = 0; e < block.getIArguments()->size(); e++) dims.push_back(INT_ARG(e));
  else if (block.getAxis()->size()) {
    dims = *block.getAxis();
  }

  if (dims.size() > 1) std::sort(dims.begin(), dims.end());

  // special case - output is scalar
  if (dims.size() == 0 || (dims.size() == 1 && dims.at(0) == DataTypeUtils::max<int>())) {
    auto newShape = ConstantShapeHelper::getInstance().scalarShapeInfo(block.dataType());
    return SHAPELIST(newShape);
  }

  auto newShape = ShapeUtils::evalReduceShapeInfo('c', &dims, inputShape->at(0), false, false, block.getWorkspace());
  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd
