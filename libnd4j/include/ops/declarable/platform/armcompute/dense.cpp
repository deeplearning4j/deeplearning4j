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
// Dense (Fully Connected) layer
PLATFORM_IMPL(dense, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);   // [bS, iC]
  auto weights = INPUT_VARIABLE(1); // [iC, oC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0); // [bS, oC]

  auto inInfo = getArmTensorInfo(*input, Arm_DataLayout::UNKNOWN);
  auto wInfo = getArmTensorInfo(*weights, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(*output, Arm_DataLayout::UNKNOWN);

  Arm_Tensor inTensor, wTensor, bTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  wTensor.allocator()->init(wInfo);
  outTensor.allocator()->init(outInfo);

  Arm_Tensor* biasPtr = nullptr;
  if (bias) {
    auto bInfo = getArmTensorInfo(*bias, Arm_DataLayout::UNKNOWN);
    bTensor.allocator()->init(bInfo);
    biasPtr = &bTensor;
  }

  arm_compute::NEFullyConnectedLayer fc;
  fc.configure(&inTensor, &wTensor, biasPtr, &outTensor);

  // Allocate and copy input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Allocate and copy weights
  if (!weights->hasPaddedBuffer() && !wTensor.info()->has_padding()) {
    wTensor.allocator()->import_memory(weights->buffer());
  } else {
    wTensor.allocator()->allocate();
    copyToTensor(*weights, wTensor);
  }

  // Allocate and copy bias
  if (bias) {
    if (!bias->hasPaddedBuffer() && !bTensor.info()->has_padding()) {
      bTensor.allocator()->import_memory(bias->buffer());
    } else {
      bTensor.allocator()->allocate();
      copyToTensor(*bias, bTensor);
    }
  }

  // Allocate output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  fc.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(dense, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE DENSE OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 2) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
