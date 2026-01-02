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

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Identity: output = input (just a copy)
PLATFORM_IMPL(identity, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NECopy identity;
  identity.configure(&inTensor, &outTensor);

  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  identity.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(identity, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE IDENTITY OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
