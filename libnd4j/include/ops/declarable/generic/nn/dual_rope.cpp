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
// Dual RoPE (Gemma 4) op definition
// Standard RoPE for sliding-window layers, proportional RoPE for global layers.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dual_rope)

#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/dual_rope.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dual_rope, 1, 1, false, 0, 0) {
  auto input  = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "dual_rope: input must be rank 4 [batch, seq_len, num_heads, head_dim], got rank %lld",
               input->rankOf());
  REQUIRE_TRUE(input->sizeAt(3) % 2 == 0, 0,
               "dual_rope: head_dim must be even, got %lld", input->sizeAt(3));

  // Int args with defaults
  int attentionType  = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  int positionOffset = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

  REQUIRE_TRUE(attentionType == 0 || attentionType == 1, 0,
               "dual_rope: attentionType must be 0 (local) or 1 (global), got %d", attentionType);

  // Float args with defaults
  double localFreqBase   = block.getTArguments()->size() > 0 ? T_ARG(0) : 10000.0;
  double globalFreqBase  = block.getTArguments()->size() > 1 ? T_ARG(1) : 1000000.0;
  double localFreqScale  = block.getTArguments()->size() > 2 ? T_ARG(2) : 1.0;
  double globalFreqScale = block.getTArguments()->size() > 3 ? T_ARG(3) : 1.0;

  helpers::dualRoPE(block.launchContext(), input, output,
                    attentionType, positionOffset,
                    localFreqBase, globalFreqBase,
                    localFreqScale, globalFreqScale);

  return sd::Status::OK;
}

DECLARE_TYPES(dual_rope) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(dual_rope) {
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
