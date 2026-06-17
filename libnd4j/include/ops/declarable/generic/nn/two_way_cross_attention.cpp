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
#if NOT_EXCLUDED(OP_two_way_cross_attention)

#include <math/templatemath.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/two_way_cross_attention.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(two_way_cross_attention, 6, 2, false, 0, 0) {
  auto tokenQuery = INPUT_VARIABLE(0);
  auto tokenKey = INPUT_VARIABLE(1);
  auto tokenValue = INPUT_VARIABLE(2);
  auto imageQuery = INPUT_VARIABLE(3);
  auto imageKey = INPUT_VARIABLE(4);
  auto imageValue = INPUT_VARIABLE(5);

  auto tokenOutput = OUTPUT_VARIABLE(0);
  auto imageOutput = OUTPUT_VARIABLE(1);

  // scale: 0 means auto = 1/sqrt(embedDim)
  double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;
  if (scale == 0.0) {
    auto embedDim = tokenQuery->sizeAt(-1);
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(embedDim));
  }

  helpers::twoWayCrossAttention(tokenQuery, tokenKey, tokenValue,
                                 imageQuery, imageKey, imageValue,
                                 tokenOutput, imageOutput,
                                 scale, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(two_way_cross_attention) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(two_way_cross_attention) {
  // tokenOutput has same shape as tokenQuery (input 0)
  // imageOutput has same shape as imageQuery (input 3)
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(3)));
}

//////////////////////////////////////////////////////////////////////////

CUSTOM_OP_IMPL(two_way_cross_attention_bp, 8, 6, false, 0, 0) {
  auto tokenQuery = INPUT_VARIABLE(0);
  auto tokenKey = INPUT_VARIABLE(1);
  auto tokenValue = INPUT_VARIABLE(2);
  auto imageQuery = INPUT_VARIABLE(3);
  auto imageKey = INPUT_VARIABLE(4);
  auto imageValue = INPUT_VARIABLE(5);
  auto gradTokenOut = INPUT_VARIABLE(6);
  auto gradImageOut = INPUT_VARIABLE(7);

  auto dTokenQ = OUTPUT_VARIABLE(0);
  auto dTokenK = OUTPUT_VARIABLE(1);
  auto dTokenV = OUTPUT_VARIABLE(2);
  auto dImageQ = OUTPUT_VARIABLE(3);
  auto dImageK = OUTPUT_VARIABLE(4);
  auto dImageV = OUTPUT_VARIABLE(5);

  double scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;
  if (scale == 0.0) {
    auto embedDim = tokenQuery->sizeAt(-1);
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(embedDim));
  }

  helpers::twoWayCrossAttentionBp(tokenQuery, tokenKey, tokenValue,
                                   imageQuery, imageKey, imageValue,
                                   gradTokenOut, gradImageOut,
                                   dTokenQ, dTokenK, dTokenV,
                                   dImageQ, dImageK, dImageV,
                                   scale, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(two_way_cross_attention_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

DECLARE_SHAPE_FN(two_way_cross_attention_bp) {
  // 6 outputs, same shapes as the first 6 inputs
  return SHAPELIST(CONSTANT(inputShape->at(0)), CONSTANT(inputShape->at(1)),
                   CONSTANT(inputShape->at(2)), CONSTANT(inputShape->at(3)),
                   CONSTANT(inputShape->at(4)), CONSTANT(inputShape->at(5)));
}

}  // namespace ops
}  // namespace sd

#endif
