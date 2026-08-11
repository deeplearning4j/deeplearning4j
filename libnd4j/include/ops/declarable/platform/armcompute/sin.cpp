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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <cmath>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(sin, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // The bundled Arm Compute version exposes no usable unary SIN layer
  // constructor in this build. Keep the ARM/CPU tier functional with the
  // exact scalar operation rather than depending on a vendor API variant.
  const auto* in = input->bufferAsT<float>();
  auto* out = output->bufferAsT<float>();

  for (sd::LongType i = 0; i < input->lengthOf(); ++i) {
    out[i] = std::sin(in[i]);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(sin, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE SIN OP");
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
