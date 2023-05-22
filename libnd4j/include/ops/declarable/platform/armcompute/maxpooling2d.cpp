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

// Created by Abdelrauf 2020

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "armcomputeUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(maxpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "MAXPOOL2D ARMCOMPUTE  OP: input array should have rank of 4, but got %i instead", input->rankOf());

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same
  // mode;
  const sd::LongType kH = INT_ARG(0);
  const sd::LongType kW = INT_ARG(1);
  const sd::LongType sH = INT_ARG(2);
  const sd::LongType sW = INT_ARG(3);
  sd::LongType pH = INT_ARG(4);
  sd::LongType pW = INT_ARG(5);
  const sd::LongType dH = INT_ARG(6);
  const sd::LongType dW = INT_ARG(7);
  const int paddingMode = INT_ARG(8);
  // const int extraParam0 = INT_ARG(9);
  const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 1-NHWC, 0-NCHW

  auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

  // Calculate individual paddings
  sd::LongType padLeft, padTop, padRight, padBottom;
  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  if (paddingMode) {
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);
  }
  padLeft = pW;
  padTop = pH;
  padRight = (oW - 1) * sW - iW + kW - pW;
  padBottom = (oH - 1) * sH - iH + kH - pH;

  auto poolPad = arm_compute::PadStrideInfo(sW, sH, padLeft, padRight, padTop, padBottom,
                                            arm_compute::DimensionRoundingType::FLOOR);
  auto poolInfo =
      arm_compute::PoolingLayerInfo(arm_compute::PoolingType::MAX, arm_compute::Size2D(kW, kH), dataLayout, poolPad);
  ArmFunction<arm_compute::NEPoolingLayer> pool;

  pool.configure(input, output, dataLayout, poolInfo);

  pool.run();  // run function

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  const int dH = INT_ARG(6);
  const int dW = INT_ARG(7);
  // Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
  // for now, we will ignore F16 as it shoulde be preconditioned for pool size 2,3 and arm64-v8.2-a architecture
  Requirements req("ARMCOMPUTE MAXPOOL2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dH, "dilation#H"), 1) && req.expectEq(makeInfoVariable(dW, "dilation#W"), 1) &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input#lastStride"), 1) &&
      req.expectLessEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), arm_compute::MAX_DIMS) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
