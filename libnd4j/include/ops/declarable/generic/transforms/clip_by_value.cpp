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
//  @author raver119@gmail.com
//  @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyvalue)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {
CONFIGURABLE_OP_IMPL(clipbyvalue, -2, 1, true, -2, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (block.getTArguments()->size() > 0) {
    auto left = T_ARG(0);
    auto right = T_ARG(1);

    REQUIRE_TRUE(left < right, 0, "clip_by_value: left bound should be lesser than right. But %f >= %f given.", left,
                 right);
    // input->applyTransform(transform::ClipByValue, output, block.getTArguments()->data());
    helpers::clipByValue(block.launchContext(), *input, left, right, *output);

  } else {
    auto left = INPUT_VARIABLE(1);
    auto right = INPUT_VARIABLE(2);

    switch (input->dataType()) {
      case DOUBLE: {
        auto leftValueDouble = left->e<double>(0);
        auto rightValueDouble = right->e<double>(0);
        helpers::clipByValue(block.launchContext(), *input, leftValueDouble, rightValueDouble, *output);
        break;
      }
      case FLOAT32: {
        auto leftValueFloat = left->e<float>(0);
        auto rightValueFloat = right->e<float>(0);
        helpers::clipByValue(block.launchContext(), *input, leftValueFloat, rightValueFloat, *output);
        break;
      }
      case HALF: {
        auto leftValueFloat16 = left->e<float16>(0);
        auto rightValueFloat16 = right->e<float16>(0);
        helpers::clipByValue(block.launchContext(), *input, leftValueFloat16, rightValueFloat16, *output);
        break;
      }
      case BFLOAT16: {
        auto leftValueBFloat16 = left->e<bfloat16>(0);
        auto rightValueBFloat16 = right->e<bfloat16>(0);
        helpers::clipByValue(block.launchContext(), *input, leftValueBFloat16, rightValueBFloat16, *output);
        break;
      }
    }
  }

  return Status::OK;
}

DECLARE_SYN(ClipByValue, clipbyvalue);

DECLARE_TYPES(clipbyvalue) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}
}  // namespace ops
}  // namespace sd

#endif
