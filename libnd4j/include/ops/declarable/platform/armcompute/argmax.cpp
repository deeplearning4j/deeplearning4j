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
PLATFORM_IMPL(argmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get axis from dimensions argument or block
  int axis = 0;
  if (block.width() > 1) {
    auto dimensions = INPUT_VARIABLE(1);
    axis = dimensions->e<int>(0);
  } else if (block.numI() > 0) {
    axis = INT_ARG(0);
  }

  // Handle negative axis
  if (axis < 0) {
    axis += input->rankOf();
  }

  // ARM Compute uses reversed axis ordering
  int armAxis = input->rankOf() - 1 - axis;

  // Create tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  Arm_TensorInfo outInfo;

  // ARM Compute argmax output is U32 or S32
  Arm_TensorShape outShape;
  int outRank = input->rankOf() - 1;
  if (outRank == 0) outRank = 1;
  outShape.set_num_dimensions(outRank);
  int outIdx = 0;
  for (int i = 0; i < input->rankOf(); i++) {
    if (i != axis) {
      outShape[outRank - 1 - outIdx] = input->sizeAt(i);
      outIdx++;
    }
  }
  if (outIdx == 0) {
    outShape[0] = 1;
  }
  outInfo = Arm_TensorInfo(outShape, 1, arm_compute::DataType::S32);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // Configure argmax
  arm_compute::NEArgMinMaxLayer argmax;
  argmax.configure(&inTensor, armAxis, &outTensor, arm_compute::ReductionOperation::ARG_IDX_MAX);

  // Import or allocate memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Allocate output tensor
  outTensor.allocator()->allocate();

  // Run argmax
  argmax.run();

  // Copy results to output (handling type conversion)
  auto outPtr = reinterpret_cast<int32_t*>(outTensor.buffer());
  for (sd::LongType i = 0; i < output->lengthOf(); i++) {
    output->p(i, static_cast<sd::LongType>(outPtr[i]));
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(argmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE ARGMAX OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
