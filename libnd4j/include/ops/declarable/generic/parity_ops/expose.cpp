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
// Created by raver119 on 12/11/17.
//

#include <ops/declarable/CustomOperations.h>
#if NOT_EXCLUDED(OP_expose)
namespace sd {
namespace ops {
CUSTOM_OP_IMPL(expose, -2, -2, true, 0, 0) {
  for (int e = 0; e < block.width(); e++) {
    //omit for eager computation, normally array size should be equal to block size
    if(block.getVariableSpace() == nullptr || block.getVariableSpace()->getVariables().size() != block.width()) {
      auto in = INPUT_VARIABLE(e);
      auto out = OUTPUT_VARIABLE(e);
      out->assign(in);
    } else {
      auto inVar = block.variable(e);
      if (inVar->variableType() == NDARRAY) {
        auto in = INPUT_VARIABLE(e);
        auto out = OUTPUT_VARIABLE(e);

        out->assign(in);
      } else if (inVar->variableType() == ARRAY_LIST) {
        auto var = block.ensureVariable(e);
        if (!var->hasNDArrayList()) {
          auto list = inVar->getNDArrayList();

          block.pushNDArrayListToVariableSpace(block.nodeId(), e, list, false);
        }
      }
    }

  }

  return Status::OK;
}
DECLARE_SYN(Enter, expose);
DECLARE_SYN(enter, expose);

DECLARE_TYPES(expose) { getOpDescriptor()->setAllowedInputTypes(ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(expose) {
  auto shapeList = SHAPELIST();

  for (int e = 0; e < block.width(); e++) {
    auto p = block.input(e);
    auto var = block.getVariable(e);
    if (var->variableType() == NDARRAY) {
      auto inShape = inputShape->at(e);
      auto desc = new ShapeDescriptor(inShape);
      shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
      delete desc;
    }
  }

  return shapeList;
}
}  // namespace ops
}  // namespace sd
#endif
