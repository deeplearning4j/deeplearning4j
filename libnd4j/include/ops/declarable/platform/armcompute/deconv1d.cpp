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
// Deconv1D (Transposed Conv1D) implemented as Deconv2D with height=1
PLATFORM_IMPL(deconv1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);   // [bS, iW, iC] or [bS, iC, iW]
  auto weights = INPUT_VARIABLE(1); // [kW, oC, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0); // [bS, oW, oC] or [bS, oC, oW]

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = INT_ARG(3);
  int paddingMode = INT_ARG(4);
  int isNCW = block.getIArguments()->size() > 5 ? INT_ARG(5) : 1;

  sd::LongType bS = input->sizeAt(0);
  sd::LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  sd::LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  sd::LongType oC = isNCW ? output->sizeAt(1) : output->sizeAt(2);
  sd::LongType oW = isNCW ? output->sizeAt(2) : output->sizeAt(1);

  // Reshape for 2D deconvolution
  auto input4D = input->reshape(input->ordering(),
                                isNCW ? std::vector<sd::LongType>{bS, iC, 1, iW}
                                      : std::vector<sd::LongType>{bS, 1, iW, iC});
  auto output4D = output->reshape(output->ordering(),
                                  isNCW ? std::vector<sd::LongType>{bS, oC, 1, oW}
                                        : std::vector<sd::LongType>{bS, 1, oW, oC});

  // Weights: [kW, oC, iC] -> [1, kW, oC, iC]
  auto weights4D = weights->reshape(weights->ordering(), {1, kW, oC, iC});

  Arm_DataLayout layout = isNCW ? Arm_DataLayout::NCHW : Arm_DataLayout::NHWC;

  auto inInfo = getArmTensorInfo(input4D, layout);
  auto wInfo = getArmTensorInfo(weights4D, Arm_DataLayout::UNKNOWN);
  auto outInfo = getArmTensorInfo(output4D, layout);

  Arm_Tensor inTensor, wTensor, bTensor, outTensor;
  inTensor.allocator()->init(inInfo);
  wTensor.allocator()->init(wInfo);
  outTensor.allocator()->init(outInfo);

  arm_compute::PadStrideInfo padInfo(1, sW, 0, pW, 0, pW,
                                     arm_compute::DimensionRoundingType::FLOOR);

  Arm_Tensor* biasPtr = nullptr;
  if (bias) {
    auto bInfo = getArmTensorInfo(*bias, Arm_DataLayout::UNKNOWN);
    bTensor.allocator()->init(bInfo);
    biasPtr = &bTensor;
  }

  arm_compute::NEDeconvolutionLayer deconv;
  deconv.configure(&inTensor, &wTensor, biasPtr, &outTensor, padInfo);

  // Allocate and copy
  if (!input4D.hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input4D.buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(input4D, inTensor);
  }

  if (!weights4D.hasPaddedBuffer() && !wTensor.info()->has_padding()) {
    wTensor.allocator()->import_memory(weights4D.buffer());
  } else {
    wTensor.allocator()->allocate();
    copyToTensor(weights4D, wTensor);
  }

  if (bias) {
    if (!bias->hasPaddedBuffer() && !bTensor.info()->has_padding()) {
      bTensor.allocator()->import_memory(bias->buffer());
    } else {
      bTensor.allocator()->allocate();
      copyToTensor(*bias, bTensor);
    }
  }

  bool copyOutput = false;
  if (!output4D.hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output4D.buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  deconv.run();

  if (copyOutput) {
    copyFromTensor(outTensor, output4D);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(deconv1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("ARMCOMPUTE DECONV1D OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
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
