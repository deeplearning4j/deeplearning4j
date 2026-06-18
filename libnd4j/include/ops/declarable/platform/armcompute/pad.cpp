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
PLATFORM_IMPL(pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  // Get padding mode (0 = CONSTANT, 1 = REFLECT, 2 = SYMMETRIC)
  int mode = 0;
  if (block.numI() > 0) {
    mode = INT_ARG(0);
  }

  // Get pad value for constant mode
  float padValue = 0.0f;
  if (block.numT() > 0) {
    padValue = T_ARG(0);
  }

  // Only support constant padding for now
  if (mode != 0) {
    return sd::Status::BAD_ARGUMENTS;
  }

  // Create ARM Compute padding info
  arm_compute::PaddingList padList;
  int rank = input->rankOf();

  for (int i = 0; i < rank; i++) {
    // ARM Compute uses reversed axis ordering
    int armIdx = rank - 1 - i;
    auto padBefore = paddings->e<sd::LongType>(i, 0);
    auto padAfter = paddings->e<sd::LongType>(i, 1);
    padList.push_back(arm_compute::PaddingInfo(static_cast<size_t>(padBefore), static_cast<size_t>(padAfter)));
  }

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // Configure pad
  arm_compute::NEPadLayer pad;
  pad.configure(&inTensor, &outTensor, padList, arm_compute::PixelValue(padValue), arm_compute::PaddingMode::CONSTANT);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run pad
  pad.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get padding mode
  int mode = 0;
  if (block.numI() > 0) {
    mode = INT_ARG(0);
  }

  Requirements req("ARMCOMPUTE PAD OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(mode, "padding mode"), 0) &&  // Only constant padding
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
