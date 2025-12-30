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
PLATFORM_IMPL(stack, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  // Get axis
  int axis = 0;
  if (block.numI() > 0) {
    axis = INT_ARG(0);
  }

  // Handle negative axis
  if (axis < 0) {
    axis += output->rankOf();
  }

  // ARM Compute uses reversed axis ordering
  unsigned int armAxis = output->rankOf() - 1 - axis;

  // Get number of inputs
  int numInputs = block.width();

  // Create input tensors
  std::vector<Arm_Tensor> inTensors(numInputs);
  std::vector<const arm_compute::ITensor*> inPtrs(numInputs);

  for (int i = 0; i < numInputs; i++) {
    auto input = INPUT_VARIABLE(i);
    // For stack, we need to add an extra dimension, so reshape input first
    auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
    inTensors[i].allocator()->init(inInfo);
    inPtrs[i] = &inTensors[i];
  }

  // Create output tensor
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);
  Arm_Tensor outTensor;
  outTensor.allocator()->init(outInfo);

  // Configure stack
  arm_compute::NEStackLayer stack;
  stack.configure(inPtrs, armAxis, &outTensor);

  // Import or allocate memory for inputs
  for (int i = 0; i < numInputs; i++) {
    auto input = INPUT_VARIABLE(i);
    if (!input->hasPaddedBuffer() && !inTensors[i].info()->has_padding()) {
      inTensors[i].allocator()->import_memory(input->buffer());
    } else {
      inTensors[i].allocator()->allocate();
      copyToTensor(*input, inTensors[i]);
    }
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run stack
  stack.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(stack, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE STACK OP");
  req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 0) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);

  // Check all inputs
  for (int i = 0; i < block.width(); i++) {
    auto input = INPUT_VARIABLE(i);
    req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
        req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
        req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1);
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
