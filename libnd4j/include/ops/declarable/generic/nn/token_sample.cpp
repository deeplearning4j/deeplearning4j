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
#if NOT_EXCLUDED(OP_token_sample)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/token_sample.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(token_sample, 1, 1, false, 0, 0) {
  auto logits = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto rank = logits->rankOf();
  REQUIRE_TRUE(rank >= 1 && rank <= 3, 0,
               "token_sample: input must be rank 1, 2, or 3, got %d", rank);

  double temperature = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.0;
  double topP = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0;
  int topK = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  LongType seed = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;

  helpers::tokenSample(logits, output, temperature, topK, topP, seed, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(token_sample) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({INT64});
  getOpDescriptor()->addTraits(OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(token_sample) {
  auto inShape = inputShape->at(0);
  auto rank = shape::rank(inShape);

  if (rank == 1) {
    // [vocabSize] -> scalar
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(INT64));
  } else if (rank == 2) {
    // [batch, vocabSize] -> [batch]
    auto batch = shape::sizeAt(inShape, static_cast<LongType>(0));
    return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(batch, INT64));
  } else {
    // [batch, seqLen, vocabSize] -> [batch]
    auto batch = shape::sizeAt(inShape, static_cast<LongType>(0));
    return SHAPELIST(ConstantShapeHelper::getInstance().vectorShapeInfo(batch, INT64));
  }
}

}  // namespace ops
}  // namespace sd

#endif
