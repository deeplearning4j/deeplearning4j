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

#include <cmath>

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
PLATFORM_IMPL(mish, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // The bundled Arm Compute version does not expose a MISH activation.
  // Keep the operation in the ARM/CPU tier with a stable scalar fallback
  // instead of referencing an optional vendor enum.
  auto inPtr = input->bufferAsT<float>();
  auto outPtr = output->bufferAsT<float>();

  for (sd::LongType i = 0; i < input->lengthOf(); ++i) {
    const float x = inPtr[i];
    const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
    outPtr[i] = x * std::tanh(softplus);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(mish, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE MISH OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
