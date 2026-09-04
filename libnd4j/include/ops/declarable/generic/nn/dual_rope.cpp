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
#include <array/DataTypeUtils.h>

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

  if (block.width() >= 2) {
    // Dynamic-position form: the base position comes from an INT64 tensor input
    // (in-graph KV-cache decode); positionOffset iArg is ignored in this form.
    // MARKER-DUALROPE-DYNAMIC-V2
    auto positionArr = INPUT_VARIABLE(1);
    REQUIRE_TRUE(positionArr->isScalar() || positionArr->lengthOf() == 1, 0,
                 "dual_rope: position input must be a scalar or single-element tensor, got %lld elements",
                 positionArr->lengthOf());
    REQUIRE_TRUE(positionArr->dataType() == DataType::INT64, 0,
                 "dual_rope: position input must be INT64 (MARKER-DUALROPE-DYNAMIC-V2), got %s",
                 DataTypeUtils::asString(positionArr->dataType()).c_str());

    helpers::dualRoPE(block.launchContext(), input, output, positionArr,
                      attentionType, localFreqBase, globalFreqBase,
                      localFreqScale, globalFreqScale);
  } else {
    helpers::dualRoPE(block.launchContext(), input, output,
                      attentionType, positionOffset,
                      localFreqBase, globalFreqBase,
                      localFreqScale, globalFreqScale);
  }

  return sd::Status::OK;
}

DECLARE_TYPES(dual_rope) {
  // Input 0: floating-point activations. Input 1 (optional): position tensor
  // (INT64; some backends accept FLOAT — integer values are exact in fp32 up to 2^24).
  // Global list covers both forms — same convention as fused_rope. The executing
  // impl validates the position scalar's dtype and shape.
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS, ALL_INTS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
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
