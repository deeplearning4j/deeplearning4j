/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

 // Created by Abdelrauf (rauf@konduit.ai) 2020

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h> 
#include <ops/declarable/helpers/convolutions.h>


#include "armcomputeUtils.h"


namespace sd      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d, ENGINE_CPU) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;

    const auto kH = INT_ARG(0);
    const auto kW = INT_ARG(1);
    const auto sH = INT_ARG(2);
    const auto sW = INT_ARG(3);
          auto pH = INT_ARG(4);
          auto pW = INT_ARG(5);
    const auto dH = INT_ARG(6);
    const auto dW = INT_ARG(7);
    const auto paddingMode = INT_ARG(8);
    const auto extraParam0 = INT_ARG(9);
    const int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // INT_ARG(10): 0-NCHW, 1-NHWC

    REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D ARMCOMPUTE op: input should have rank of 4, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D ARMCOMPUTE op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    bool excludePadding= (extraParam0 == 0) ? true : false;

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;

    // Calculate individual paddings
    unsigned int padLeft, padTop, padRight, padBottom;
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    if(paddingMode){ 
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW); 
    }
    padLeft   = pW;
    padTop    = pH;
    padRight  = (oW - 1) * sW - iW + kW - pW ;
    padBottom = (oH - 1) * sH - iH + kH - pH ; 

#if 0
    nd4j_printf("avgpool kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d exclude pad %d \n" , kH , kW , sH , sW  , pH 
     , pW , dH , dW , paddingMode,isNCHW?1:0 ,excludePadding?1:0);
#endif
    auto poolPad = arm_compute::PadStrideInfo(sW, sH, padLeft, padRight, padTop, padBottom, arm_compute::DimensionRoundingType::FLOOR);
    auto poolInfo = arm_compute::PoolingLayerInfo(arm_compute::PoolingType::AVG, arm_compute::Size2D(kW, kH), dataLayout, poolPad, excludePadding);
    ArmFunction<arm_compute::NEPoolingLayer> pool;    
    pool.configure(input,output, dataLayout, poolInfo);
     
    pool.run(); // run function

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool2d, ENGINE_CPU) { 
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);
    const int dH = INT_ARG(6);
    const int dW = INT_ARG(7);
    // Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32
    auto dTypeInput = getArmType(input->dataType());
    auto dTypeOutput = getArmType(output->dataType());  
    bool isSupported = dH==1 && dW==1 && isArmcomputeFriendly(*input) && isArmcomputeFriendly(*output)
            && (dTypeInput ==Arm_DataType::F32) 
            && (dTypeOutput ==Arm_DataType::F32); 
    return  isSupported; 
}



}
}
}
