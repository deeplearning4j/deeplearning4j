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
PLATFORM_IMPL(unstack, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);

  // Get axis
  int axis = 0;
  if (block.numI() > 0) {
    axis = INT_ARG(0);
  }

  // Handle negative axis
  if (axis < 0) {
    axis += input->rankOf();
  }

  // ARM Compute uses reversed axis ordering
  unsigned int armAxis = input->rankOf() - 1 - axis;

  // Create tensor info for input
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  Arm_Tensor inTensor;
  inTensor.allocator()->init(inInfo);

  // Get number of outputs
  int numOutputs = input->sizeAt(axis);

  // Create output tensors
  std::vector<Arm_Tensor> outTensors(numOutputs);
  std::vector<arm_compute::ITensor*> outPtrs(numOutputs);
  std::vector<bool> copyOutputs(numOutputs, false);

  for (int i = 0; i < numOutputs; i++) {
    auto output = OUTPUT_VARIABLE(i);
    auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
    outTensors[i].allocator()->init(outInfo);
    outPtrs[i] = &outTensors[i];
  }

  // Configure unstack (using Split)
  arm_compute::NESplit unstack;
  unstack.configure(&inTensor, outPtrs, armAxis);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Import or allocate memory for outputs
  for (int i = 0; i < numOutputs; i++) {
    auto output = OUTPUT_VARIABLE(i);
    if (!output->hasPaddedBuffer() && !outTensors[i].info()->has_padding()) {
      outTensors[i].allocator()->import_memory(output->buffer());
    } else {
      outTensors[i].allocator()->allocate();
      copyOutputs[i] = true;
    }
  }

  // Run unstack
  unstack.run();

  // Copy outputs if needed
  for (int i = 0; i < numOutputs; i++) {
    if (copyOutputs[i]) {
      auto output = OUTPUT_VARIABLE(i);
      copyFromTensor(outTensors[i], *output);
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(unstack, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE UNSTACK OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
