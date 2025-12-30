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
PLATFORM_IMPL(reverse, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get axis
  std::vector<int> axes;
  if (block.width() > 1) {
    auto axisArg = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axisArg->lengthOf(); i++) {
      int axis = axisArg->e<int>(i);
      if (axis < 0) {
        axis += input->rankOf();
      }
      axes.push_back(axis);
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      int axis = INT_ARG(i);
      if (axis < 0) {
        axis += input->rankOf();
      }
      axes.push_back(axis);
    }
  } else {
    // Default: reverse all axes
    for (int i = 0; i < input->rankOf(); i++) {
      axes.push_back(i);
    }
  }

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

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

  // Perform reverse for each axis using NEReverse
  Arm_Tensor* currentIn = &inTensor;
  Arm_Tensor tempTensor1, tempTensor2;
  bool useTempTensor1 = true;

  for (size_t i = 0; i < axes.size(); i++) {
    // ARM Compute uses reversed axis ordering
    int armAxis = input->rankOf() - 1 - axes[i];

    Arm_Tensor* currentOut;
    if (i == axes.size() - 1) {
      currentOut = &outTensor;
    } else {
      if (useTempTensor1) {
        tempTensor1.allocator()->init(inInfo);
        tempTensor1.allocator()->allocate();
        currentOut = &tempTensor1;
      } else {
        tempTensor2.allocator()->init(inInfo);
        tempTensor2.allocator()->allocate();
        currentOut = &tempTensor2;
      }
      useTempTensor1 = !useTempTensor1;
    }

    arm_compute::NEReverse reverse;
    // Create a 1D tensor for the axis
    Arm_TensorInfo axisInfo(Arm_TensorShape(1), 1, arm_compute::DataType::U32);
    Arm_Tensor axisTensor;
    axisTensor.allocator()->init(axisInfo);
    axisTensor.allocator()->allocate();
    *reinterpret_cast<uint32_t*>(axisTensor.buffer()) = static_cast<uint32_t>(armAxis);

    reverse.configure(currentIn, &axisTensor, currentOut, false);
    reverse.run();

    currentIn = currentOut;
  }

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(reverse, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE REVERSE OP");
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
