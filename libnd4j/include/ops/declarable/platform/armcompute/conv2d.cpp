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

// Created by Abdelrauf (rauf@konduit.ai) 2020

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  sd::LongType sH = INT_ARG(2);                                                // strides height
  sd::LongType sW = INT_ARG(3);                                                // strides width
  sd::LongType pH = INT_ARG(4);                                                // paddings height
  sd::LongType pW = INT_ARG(5);                                                // paddings width
  sd::LongType dH = INT_ARG(6);                                                // dilations height
  sd::LongType dW = INT_ARG(7);                                                // dilations width
  int paddingMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  sd::LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<sd::LongType>(weights->sizeAt(0));  // filter(kernel) height
  sd::LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<sd::LongType>(weights->sizeAt(1));  // filter(kernel) width

  // Calculate individual paddings
   sd::LongType padLeft, padTop, padRight, padBottom;
  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);
  int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
                                            : pW;  // dH == 1 for causal mode in conv1d
   padLeft = pW;
   padTop = pH;
  padRight = (oW - 1) * sW - iW + kW - pWSame;
 padBottom = (oH - 1) * sH - iH + kH - pH;

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D ARMCOMPUTE OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CONV2D ARMCOMPUTE OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());



  auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
  // check weight input datalayout match
  bool dataLayoutMatch = (isNCHW && wFormat == 1) || (!isNCHW && wFormat == 2);
  arm_compute::PermutationVector permuteVector;
  if (!dataLayoutMatch) {
    // lets premute
    if (wFormat == 0) {
      if (isNCHW) {

        // reshape
        permuteVector = arm_compute::PermutationVector(2U, 3U, 1U, 0U);
      } else {

        // reshape
        permuteVector = arm_compute::PermutationVector(1U, 2U, 3U, 0U);
      }
    } else if (wFormat == 1) {

      permuteVector = arm_compute::PermutationVector(2U, 0U, 1U, 3U);
    } else {

      permuteVector = arm_compute::PermutationVector(1U, 2U, 0U, 3U);
    }
  } else {

    // set 0
    permuteVector.set_num_dimensions(0);
  }

  Arm_WeightsInfo wInfo(false, kW, kH, 1);
  arm_compute::Size2D dilation(dW, dH);
  arm_compute::PadStrideInfo pad(sW, sH, padLeft, padRight, padTop, padBottom,
                                 arm_compute::DimensionRoundingType::FLOOR);
  ArmFunctionWeighted<arm_compute::NEConvolutionLayer> conv;
  conv.configure(input, weights, bias, output, dataLayout, permuteVector, pad, wInfo, dilation);
  conv.run();  // run function
  return sd::Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  // Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
  Requirements req("ARMCOMPUTE CONV2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input0#lastStride"), 1) &&
      req.expectLessEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(weights->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(weights->stridesOf()[weights->rankOf() - 1], "input1#lastStride"), 1) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
