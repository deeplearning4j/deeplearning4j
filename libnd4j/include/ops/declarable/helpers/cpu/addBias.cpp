/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
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

//
// @author Yurii Shyrma, created on 26.02.2018
//


#include<ops/declarable/helpers/addBias.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void addBias_(NDArray& input, const NDArray& bias, const bool isNCHW) {

    // input  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    // bias   [oC]

          X* inBuff   = input.bufferAsT<X>();
    const Y* biasBuff = bias.bufferAsT<Y>();    

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    bS = input.sizeAt(0);    
    const Nd4jLong stride0 = input.stridesOf()[0];
    const Nd4jLong stride1 = input.stridesOf()[1];
    const Nd4jLong stride2 = input.stridesOf()[2];

    uint biasShapeInfoCast[MAX_RANK];    
    bool canCastBias = nd4j::DataTypeUtils::castShapeInfo(bias.getShapeInfo(), biasShapeInfoCast);
    
    if(isNCHW) {
        
        oC = input.sizeAt(1);
        oH = input.sizeAt(2);
        oW = input.sizeAt(3);

        const int oHoW = oH*oW;

        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int i = 0; i < bS; ++i) {
            for (int c = 0; c < oC; ++c) {
                
                auto biasOffset = shape::indexOffset(c, bias.getShapeInfo(), biasShapeInfoCast, oC, canCastBias);
                auto inOffset = i * stride0 + c * stride1;

                PRAGMA_OMP_SIMD
                for (uint k = 0; k < oHoW; ++k)
                    inBuff[inOffset + k] += static_cast<X>(biasBuff[biasOffset]);
            }
        }
    }
    else {
        
        oC = input.sizeAt(3);
        oH = input.sizeAt(1);
        oW = input.sizeAt(2);

        PRAGMA_OMP_PARALLEL_FOR
        for (int i = 0; i < bS*oH*oW; ++i) {

            PRAGMA_OMP_SIMD
            for (int c = 0; c < oC; ++c) {
                auto biasOffset = shape::indexOffset(c, bias.getShapeInfo(), biasShapeInfoCast, oC, canCastBias);
                inBuff[i * oC + c] += static_cast<X>(biasBuff[biasOffset]);
            }                            
        }
    }        
}

//////////////////////////////////////////////////////////////////////////
void addBias(NDArray& input, const NDArray& bias, const bool isNCHW) {

    BUILD_DOUBLE_SELECTOR(input.dataType(), bias.dataType(), addBias_, (input, bias, isNCHW), FLOAT_TYPES, FLOAT_TYPES);
}


BUILD_DOUBLE_TEMPLATE(template void addBias_, (NDArray& input, const NDArray& bias, const bool isNCHW), FLOAT_TYPES, FLOAT_TYPES);

}
}
}

