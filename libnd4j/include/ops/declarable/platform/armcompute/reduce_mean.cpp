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
PLATFORM_IMPL(reduce_mean, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<int> dimensions;
  if (block.width() > 1) {
    auto axisVector = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axisVector->lengthOf(); i++) {
      dimensions.push_back(axisVector->e<int>(i));
    }
  } else if (block.getIArguments()->size() > 0) {
    // IArguments are LongType on Android, while ARM Compute expects int axes.
    for (auto axis : *block.getIArguments()) {
      dimensions.push_back(static_cast<int>(axis));
    }
  }

  bool keepDims = block.getBArguments()->size() > 0 ? B_ARG(0) : false;

  const int rank = input->rankOf();

  // Handle negative dimensions
  for (auto& dim : dimensions) {
    if (dim < 0) dim += rank;
  }

  // ARM Compute reduction axis (in reverse order from the end)
  arm_compute::Coordinates reductionAxis;
  for (auto dim : dimensions) {
    reductionAxis.set(rank - 1 - dim, 1);
  }

  // Create ARM Compute tensor info
  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  // Configure reduction layer
  arm_compute::NEReduceMean reduceMean;
  reduceMean.configure(&inTensor, reductionAxis, keepDims, &outTensor);

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

  // Run reduction
  reduceMean.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(reduce_mean, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE REDUCE_MEAN OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
