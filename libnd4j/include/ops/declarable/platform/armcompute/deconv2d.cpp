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


#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h> 
#include <ops/declarable/helpers/convolutions.h>


#include "armcomputeUtils.h"


namespace sd      {
namespace ops       {
namespace platforms {


 

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D ARMCOMPUTE OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D ARMCOMPUTE OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 0-NCHW,  1-NHWC
    int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;         // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

 
    // Calculate individual paddings
    unsigned int padLeft, padTop, padRight, padBottom;
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV2D ARMCOMPUTE OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D ARMCOMPUTE OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());
   
   if(paddingMode){ 
    //Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);
    }
    padLeft   = pW;
    padTop    = pH;
    padRight  = (iW - 1) * sW - oW + kW - pW;
    padBottom = (iH - 1) * sH - oH + kH - pH;
    //deconv2dMKLDNN(input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat);
#if 0
    nd4j_printf("deconv2d  bS = %d,  iH =%d, iW = %d,  oH=%d, oW=%d  kH=%d, kW=%d wformat=%d, iC =%d, , oC=%d\n",
       bS, iH, iW, oH, oW, kH, kW, wFormat, iC, oC
     );
    nd4j_printf("deconv2d kH = %d, kW = %d, sH = %d, sW = %d  , pH = %d  , pW = %d, dH = %d, dW = %d, paddingMode = %d , isNCHW %d \n" , kH , kW , sH , sW  , pH 
     , pW , dH , dW , paddingMode,isNCHW?1:0 );
#endif

    auto dataLayout = isNCHW ? arm_compute::DataLayout::NCHW : arm_compute::DataLayout::NHWC;
    //check weight input datalayout match
    bool dataLayoutMatch = (isNCHW && wFormat == 1) || (!isNCHW && wFormat == 2);
    arm_compute::PermutationVector permuteVector;
    //unlike in cov2d for weights iC and oC permutted : for example  {oC, iC, kH, kW}, {iC, oC, kH, kW}
    //but we need it normal way for arm
    if (!dataLayoutMatch) {
        //lets premute  
        if (wFormat == 0) {
            if (isNCHW) {
#if 0
                nd4j_printf("perm choise %d\n", 0);
#endif                    
                //reshape
                permuteVector = arm_compute::PermutationVector(2U, 3U, 0U, 1U);
            }
            else {
#if 0
                nd4j_printf("perm choise %d\n", 1);
#endif                         
                //reshape
                permuteVector = arm_compute::PermutationVector(0U, 2U, 3U, 1U);
            }
        }
        else if (wFormat == 1) {
#if 0
            nd4j_printf("perm choise %d\n", 2);
#endif                     
            permuteVector = arm_compute::PermutationVector(3U, 0U, 1U, 2U);
        }
        else {
#if 0
            nd4j_printf("perm choise %d\n", 3);
#endif                     
            permuteVector = arm_compute::PermutationVector(1U, 2U, 3U, 0U);
        }
    }
    else {
//fix weight
        if(isNCHW){
#if 0
        nd4j_printf("perm choise %d\n", 4);
#endif
           permuteVector = arm_compute::PermutationVector(0U, 1U, 3U, 2U);
        }else{
#if 0
        nd4j_printf("perm choise %d\n", 5);
#endif           
           permuteVector = arm_compute::PermutationVector(3U, 1U, 2U, 0U);
        } 
    }

    Arm_WeightsInfo wInfo(false, kW, kH, 1); 
    arm_compute::PadStrideInfo pad(sW, sH, padLeft,padRight, padTop, padBottom,  arm_compute::DimensionRoundingType::FLOOR);
    ArmFunctionWeighted<arm_compute::NEDeconvolutionLayer> deconv;
    deconv.configure( input, weights, bias, output, dataLayout, permuteVector, pad);     
    deconv.run(); // run function
    return Status::OK();
}


PLATFORM_CHECK(deconv2d, ENGINE_CPU) {

    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0); 
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);
    // Data types supported: QASYMM8/QASYMM8_SIGNED/F16/F32.
    auto dTypeInput = getArmType(input->dataType());
    auto dTypeWeight = getArmType(weights->dataType());
    auto dTypeOutput = getArmType(output->dataType());  
    
    bool isSupported = dW==1 && dH==1
            && isArmcomputeFriendly(*input)
            && isArmcomputeFriendly(*weights)             
            && isArmcomputeFriendly(*output)
            && (dTypeInput == Arm_DataType::F32 /*||  dTypeInput == Arm_DataType::F16*/)
            && (dTypeWeight == dTypeInput)            
            && (dTypeOutput == dTypeInput); 

#if 0
nd4j_printf("deconv2d isSupported %d : isArmcomputeFriendly(*input) = %d , isArmcomputeFriendly(*weights) = %d, isArmcomputeFriendly(*output) %d\n",
isSupported, isArmcomputeFriendly(*input),isArmcomputeFriendly(*weights),isArmcomputeFriendly(*output));
#endif
    return  isSupported;            
}



}
}
}
