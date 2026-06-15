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
PLATFORM_IMPL(gather, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

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

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  // For indices, we need S32 type
  Arm_TensorShape indicesShape;
  indicesShape.set_num_dimensions(indices->rankOf());
  for (int i = 0; i < indices->rankOf(); i++) {
    indicesShape[indices->rankOf() - 1 - i] = indices->sizeAt(i);
  }
  Arm_TensorInfo indicesInfo(indicesShape, 1, arm_compute::DataType::S32);

  Arm_Tensor inTensor, indicesTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  indicesTensor.allocator()->init(indicesInfo);
  outTensor.allocator()->init(outInfo);

  // Configure gather
  arm_compute::NEGather gather;
  gather.configure(&inTensor, &indicesTensor, &outTensor, armAxis);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Allocate and copy indices (always need to copy due to potential type conversion)
  indicesTensor.allocator()->allocate();
  auto indicesPtr = reinterpret_cast<int32_t*>(indicesTensor.buffer());
  for (sd::LongType i = 0; i < indices->lengthOf(); i++) {
    indicesPtr[i] = static_cast<int32_t>(indices->e<sd::LongType>(i));
  }

  // Import or allocate memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run gather
  gather.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(gather, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto indices = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE GATHER OP");
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
