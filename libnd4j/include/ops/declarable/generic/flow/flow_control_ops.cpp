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
// Created by raver119 on 13.10.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(op_Switch)
namespace sd {
namespace ops {
/**
 * This operation is, basically IF statement
 *
 * arg_0 is our "signal"
 * arg_1 is condition that will determine transition
 */
// TODO: make this op a placeholder too
DIVERGENT_OP_IMPL(Switch, 2, 2, true) {
  auto input = INPUT_VARIABLE(0);
  auto condition = INPUT_VARIABLE(1);

  // we'll store signal to both ends
  // STORE_2_RESULTS(*input, *input);

  // but we'll ensure only one node is active, and other is disabled
  auto out0 = OUTPUT_VARIABLE(0);
  auto out1 = OUTPUT_VARIABLE(1);

  if (condition->e<int>(0) == 0) {
    block.setBranch(0);
    if (!out0) {
      this->storeResult(block, 0, new NDArray(input->dup()));
    } else {
      out0->assign(input);
    }
  } else {
    block.setBranch(1);
    if (!out1) {
      this->storeResult(block, 1, new NDArray(input->dup()));
    } else {
      out1->assign(input);
    }
  }

  return sd::Status::OK;
}
DECLARE_SYN(switch, Switch);
DECLARE_SYN(if, Switch);

/**
 *  This op is a placeholder.
 *  Actual WHILE implementation is in GraphExecutioner
 */
LOGIC_OP_IMPL(While);
DECLARE_SYN(while, While);

/**
 *  This op is a placeholder.
 *  Actual OpScope implementation is in Graph and GraphExecutioner
 */
LOGIC_OP_IMPL(OpScope);
DECLARE_SYN(scope, OpScope);

/**
 *  This op is a placeholder.
 *  Actual Conditional implementation is in Graph and GraphExecutioner
 */
LOGIC_OP_IMPL(Conditional);
DECLARE_SYN(cond, Conditional);

/**
 * This op is a placeholder
 * Actual implementation is in LogicReturn class
 */
LOGIC_OP_IMPL(Return);
DECLARE_SYN(return, Return);
}  // namespace ops
}  // namespace sd

#endif
