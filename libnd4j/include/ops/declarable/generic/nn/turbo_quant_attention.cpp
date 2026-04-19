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

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_turbo_quant_attention)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/turbo_quant_attention.h>

namespace sd {
namespace ops {

// ──────────────────────────────────────────────────────────────────────────
// turbo_quant_attention
//
// Asymmetric attention with TurboQuant compressed keys.
//
// Input 0: Q              [B, H, Sq, D] query
// Input 1: K_mse          [B, H, Sk, D] MSE-reconstructed keys
// Input 2: QJL_signs      [B, H, Sk, D] sign bits (INT8)
// Input 3: residual_norms [B, H, Sk]    residual L2 norms
// Input 4: QJL_matrix     [D, D]        projection matrix
// Input 5: V              [B, H, Sk, D] dequantized values
// Input 6: attention_mask [B, 1, 1, Sk] or compatible
//
// IArgs: 0=numHeads, 1=headDim
// TArgs: 0=scale (0 = auto: 1/sqrt(headDim))
//
// Output 0: [B, H, Sq, D]
// ──────────────────────────────────────────────────────────────────────────
CUSTOM_OP_IMPL(turbo_quant_attention, 7, 1, false, 1, 2) {
  auto query          = INPUT_VARIABLE(0);
  auto kMse           = INPUT_VARIABLE(1);
  auto qjlSigns       = INPUT_VARIABLE(2);
  auto residualNorms  = INPUT_VARIABLE(3);
  auto qjlMatrix      = INPUT_VARIABLE(4);
  auto values         = INPUT_VARIABLE(5);
  auto attentionMask  = INPUT_VARIABLE(6);
  auto output         = OUTPUT_VARIABLE(0);

  int numHeads = INT_ARG(0);
  int headDim  = INT_ARG(1);
  float scale  = static_cast<float>(T_ARG(0));

  // Validate inputs
  REQUIRE_TRUE(query->rankOf() == 4, 0,
               "turbo_quant_attention: query must be rank 4, got %d", query->rankOf());
  REQUIRE_TRUE(kMse->rankOf() == 4, 0,
               "turbo_quant_attention: kMse must be rank 4, got %d", kMse->rankOf());
  REQUIRE_TRUE(qjlSigns->rankOf() == 4, 0,
               "turbo_quant_attention: qjlSigns must be rank 4, got %d", qjlSigns->rankOf());
  REQUIRE_TRUE(residualNorms->rankOf() == 3, 0,
               "turbo_quant_attention: residualNorms must be rank 3, got %d", residualNorms->rankOf());
  REQUIRE_TRUE(qjlMatrix->rankOf() == 2, 0,
               "turbo_quant_attention: qjlMatrix must be rank 2, got %d", qjlMatrix->rankOf());
  REQUIRE_TRUE(values->rankOf() == 4, 0,
               "turbo_quant_attention: values must be rank 4, got %d", values->rankOf());

  helpers::TurboQuantAttentionConfig config;
  config.numHeads = numHeads;
  config.headDim  = headDim;
  config.scale    = scale;

  helpers::turboQuantAttentionForward(
      query, kMse, qjlSigns, residualNorms,
      qjlMatrix, values, attentionMask,
      output, config, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(turbo_quant_attention) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})         // Q
      ->setAllowedInputTypes(1, {ALL_FLOATS})         // K_mse
      ->setAllowedInputTypes(2, {INT8})               // QJL_signs
      ->setAllowedInputTypes(3, {ALL_FLOATS})         // residual_norms
      ->setAllowedInputTypes(4, {ALL_FLOATS})         // QJL_matrix
      ->setAllowedInputTypes(5, {ALL_FLOATS})         // V
      ->setAllowedInputTypes(6, {ALL_FLOATS})         // attention_mask
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(turbo_quant_attention) {
  // Output shape = same as query: [B, H, Sq, D]
  auto queryShape = inputShape->at(0);
  auto rank = shape::rank(queryShape);
  std::vector<sd::LongType> outShape(rank);
  for (int i = 0; i < rank; i++) {
    outShape[i] = shape::sizeAt(queryShape, static_cast<sd::LongType>(i));
  }
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(queryShape), shape::order(queryShape), outShape));
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_turbo_quant_attention)
