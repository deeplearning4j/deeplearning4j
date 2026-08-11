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
#include <limits>

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// 1D Pooling (both max and avg) - implemented as 2D pooling with height=1
PLATFORM_IMPL(maxpool1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);   // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
  auto output = OUTPUT_VARIABLE(0); // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW)

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = INT_ARG(3);
  int paddingMode = INT_ARG(4);
  int isNCW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 1;

  sd::LongType bS = input->sizeAt(0);
  sd::LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  sd::LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  sd::LongType oW = isNCW ? output->sizeAt(2) : output->sizeAt(1);

  // Reshape for 2D pooling [bS, iC, 1, iW] or [bS, 1, iW, iC]
  std::vector<sd::LongType> inputShape =
      isNCW ? std::vector<sd::LongType>{bS, iC, 1, iW}
            : std::vector<sd::LongType>{bS, 1, iW, iC};
  std::vector<sd::LongType> outputShape =
      isNCW ? std::vector<sd::LongType>{bS, iC, 1, oW}
            : std::vector<sd::LongType>{bS, 1, oW, iC};
  auto input4D = input->reshape(input->ordering(), inputShape);
  auto output4D = output->reshape(output->ordering(), outputShape);

  Arm_DataLayout layout = isNCW ? Arm_DataLayout::NCHW : Arm_DataLayout::NHWC;

  auto inInfo = getArmTensorInfo(*input4D, layout);
  auto outInfo = getArmTensorInfo(*output4D, layout);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::PoolingLayerInfo poolInfo(arm_compute::PoolingType::MAX,
                                         arm_compute::Size2D(kW, 1),
                                         layout,
                                         arm_compute::PadStrideInfo(sW, 1, pW, 0));

  arm_compute::NEPoolingLayer pool;
  pool.configure(&inTensor, &outTensor, poolInfo);

  if (!input4D->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input4D->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input4D, inTensor);
  }

  bool copyOutput = false;
  if (!output4D->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output4D->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  pool.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output4D);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE MAXPOOL1D OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 3) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = INT_ARG(3);
  int paddingMode = INT_ARG(4);
  int isNCW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 1;

  sd::LongType bS = input->sizeAt(0);
  sd::LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  sd::LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  sd::LongType oW = isNCW ? output->sizeAt(2) : output->sizeAt(1);

  // Reshape for 2D pooling
  std::vector<sd::LongType> inputShape =
      isNCW ? std::vector<sd::LongType>{bS, iC, 1, iW}
            : std::vector<sd::LongType>{bS, 1, iW, iC};
  std::vector<sd::LongType> outputShape =
      isNCW ? std::vector<sd::LongType>{bS, iC, 1, oW}
            : std::vector<sd::LongType>{bS, 1, oW, iC};
  auto input4D = input->reshape(input->ordering(), inputShape);
  auto output4D = output->reshape(output->ordering(), outputShape);

  Arm_DataLayout layout = isNCW ? Arm_DataLayout::NCHW : Arm_DataLayout::NHWC;

  auto inInfo = getArmTensorInfo(*input4D, layout);
  auto outInfo = getArmTensorInfo(*output4D, layout);

  Arm_Tensor inTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::PoolingLayerInfo poolInfo(arm_compute::PoolingType::AVG,
                                         arm_compute::Size2D(kW, 1),
                                         layout,
                                         arm_compute::PadStrideInfo(sW, 1, pW, 0));

  arm_compute::NEPoolingLayer pool;
  pool.configure(&inTensor, &outTensor, poolInfo);

  if (!input4D->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input4D->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input4D, inTensor);
  }

  bool copyOutput = false;
  if (!output4D->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output4D->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  pool.run();

  if (copyOutput) {
    copyFromTensor(outTensor, *output4D);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE AVGPOOL1D OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 3) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
