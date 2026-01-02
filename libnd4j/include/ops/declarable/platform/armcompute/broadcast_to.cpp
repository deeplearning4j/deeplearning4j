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
PLATFORM_IMPL(broadcast_to, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get target shape
  std::vector<sd::LongType> targetShape;
  if (block.width() > 1) {
    auto shapeArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < shapeArr->lengthOf(); i++) {
      targetShape.push_back(shapeArr->e<sd::LongType>(i));
    }
  } else {
    for (int i = 0; i < block.numI(); i++) {
      targetShape.push_back(INT_ARG(i));
    }
  }

  // Use tile to broadcast - compute multiples
  std::vector<uint32_t> multiples;
  int inputRank = input->rankOf();
  int outputRank = targetShape.size();

  // Pad input shape with 1s at the beginning if needed
  std::vector<sd::LongType> paddedInputShape(outputRank, 1);
  for (int i = 0; i < inputRank; i++) {
    paddedInputShape[outputRank - inputRank + i] = input->sizeAt(i);
  }

  for (int i = 0; i < outputRank; i++) {
    multiples.push_back(static_cast<uint32_t>(targetShape[i] / paddedInputShape[i]));
  }

  // ARM Compute uses reversed axis ordering
  std::vector<uint32_t> armMultiples(multiples.rbegin(), multiples.rend());

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::NETile tile;
  tile.configure(&inTensor, &outTensor, arm_compute::Multiples(armMultiples));

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

  tile.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(broadcast_to, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE BROADCAST_TO OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectGreater(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
