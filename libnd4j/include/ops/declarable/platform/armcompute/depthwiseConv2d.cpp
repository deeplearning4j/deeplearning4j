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
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(depthwise_conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC] = iC*mC

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

  sd::LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) height
  sd::LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) width
  sd::LongType sH = INT_ARG(2);                                                          // strides height
  sd::LongType sW = INT_ARG(3);                                                          // strides width
  sd::LongType pH = INT_ARG(4);                                                          // paddings height
  sd::LongType pW = INT_ARG(5);                                                          // paddings width
  sd::LongType dH = INT_ARG(6);                                                          // dilations height
  sd::LongType dW = INT_ARG(7);                                                          // dilations width
  int paddingMode = INT_ARG(8);                                                 // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

  sd::LongType bS, iC, iH, iW, mC, oC, oH, oW;
  sd::LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  // Calculate individual paddings
  sd::LongType padLeft = pW;
  sd::LongType padTop = pH;
  sd::LongType padRight = (oW - 1) * sW - iW + kW - pW;
  sd::LongType padBottom = (oH - 1) * sH - iH + kH - pH;

  auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

  arm_compute::PadStrideInfo padInfo(sW, sH, padLeft, padRight, padTop, padBottom,
                                     arm_compute::DimensionRoundingType::FLOOR);
  arm_compute::Size2D dilation(dW, dH);

  // Create ARM Compute tensors
  Arm_Tensor inTensor, wTensor, outTensor, bTensor;

  auto inInfo = getArmTensorInfo(*input, dataLayout);
  auto wInfo = getArmTensorInfo(*weights, dataLayout);
  auto outInfo = getArmTensorInfo(*output, dataLayout);

  inTensor.allocator()->init(inInfo);
  wTensor.allocator()->init(wInfo);
  outTensor.allocator()->init(outInfo);

  Arm_Tensor* biasPtr = nullptr;
  if (bias != nullptr) {
    auto bInfo = getArmTensorInfo(*bias, Arm_DataLayout::UNKNOWN);
    bTensor.allocator()->init(bInfo);
    biasPtr = &bTensor;
  }

  // Configure depthwise convolution
  arm_compute::NEDepthwiseConvolutionLayer depthwise;
  depthwise.configure(&inTensor, &wTensor, biasPtr, &outTensor, padInfo, mC,
                      arm_compute::ActivationLayerInfo(), dilation);

  // Allocate and import memory for input
  if (!input->hasPaddedBuffer() && !inTensor.info()->has_padding()) {
    inTensor.allocator()->import_memory(input->buffer());
  } else {
    inTensor.allocator()->allocate();
    copyToTensor(*input, inTensor);
  }

  // Allocate and import memory for weights
  if (!weights->hasPaddedBuffer() && !wTensor.info()->has_padding()) {
    wTensor.allocator()->import_memory(weights->buffer());
  } else {
    wTensor.allocator()->allocate();
    copyToTensor(*weights, wTensor);
  }

  // Allocate and import memory for bias if present
  if (bias != nullptr) {
    if (!bias->hasPaddedBuffer() && !bTensor.info()->has_padding()) {
      bTensor.allocator()->import_memory(bias->buffer());
    } else {
      bTensor.allocator()->allocate();
      copyToTensor(*bias, bTensor);
    }
  }

  // Allocate and import memory for output
  bool copyOutput = false;
  if (!output->hasPaddedBuffer() && !outTensor.info()->has_padding()) {
    outTensor.allocator()->import_memory(output->buffer());
  } else {
    outTensor.allocator()->allocate();
    copyOutput = true;
  }

  // Run depthwise convolution
  depthwise.run();

  // Copy output if needed
  if (copyOutput) {
    copyFromTensor(outTensor, *output);
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(depthwise_conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  const int dH = INT_ARG(6);
  const int dW = INT_ARG(7);

  Requirements req("ARMCOMPUTE DEPTHWISE_CONV2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dH, "dilation#H"), 1) &&
      req.expectEq(makeInfoVariable(dW, "dilation#W"), 1) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectLessEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(weights->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(weights->stridesOf()[weights->rankOf() - 1], "input1#lastStride"), 1) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);

  if (bias != nullptr) {
    req.expectEq(makeInfoVariable(bias->dataType(), "bias#type"), DataType::FLOAT32);
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
