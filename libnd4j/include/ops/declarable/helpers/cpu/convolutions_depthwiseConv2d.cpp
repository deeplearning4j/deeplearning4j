/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//

#include <ops/declarable/helpers/convolutions.h>
#include<ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops  {


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2d_(sd::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {

            // input     [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            // weights   [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
            // bias      [oC] = iC*mC
            // output    [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

            // kH           filter(kernel) height
            // kW           filter(kernel) width
            // sH           strides height
            // sW           strides width
            // pH           paddings height
            // pW           paddings width
            // dH           dilations height
            // dW           dilations width
            // paddingMode  0-VALID, 1-SAME
            // isNCHW       0-NCHW,  1-NHWC

            int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
            int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
            mC = weights->sizeAt(indWmC);                           // channels multiplier

            std::vector<std::vector<Nd4jLong>> modifColumns = {{1,0,4,5,2,3}, {iC,bS*oH*oW,kH*kW}};  // [bS,iC,kH,kW,oH,oW] -> [iC,bS,oH,oW,kH,kW] -> [iC,bS*oH*oW,kH*kW]
            std::vector<std::vector<Nd4jLong>> modifOutput, modifWeights;
            std::vector<Nd4jLong> outReShape;

            if(!isNCHW) {
                outReShape = {bS, oH, oW, iC, mC};                                              // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
                modifOutput = {{3,0,1,2,4},{iC, bS*oH*oW, mC}};                                 // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
                input = new NDArray(input->permute({0, 3, 1, 2}));                              // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
            }
            else {
                outReShape = {bS, iC, mC, oH, oW};                                              // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
                modifOutput = {{1,0,3,4,2},{iC, bS*oH*oW, mC}};                                 // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
            }

            if(0 == wFormat)
                modifWeights = {{2,0,1,3},{iC,kH*kW,mC}};
            else if(1 == wFormat)
                modifWeights = {{1,2,3,0},{iC,kH*kW,mC}};
            else
                modifWeights = {{3,1,2,0},{iC,kH*kW,mC}};

            if(paddingMode == 1)                       // SAME
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());
            NDArray outputReshaped = output->reshape(output->ordering(), outReShape, false);

            helpers::im2col(*output->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
            MmulHelper::tensorDot(&columns, weights, &outputReshaped, modifColumns, modifWeights, modifOutput);              // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]

            if(bias)
                // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
                helpers::addBias(block, *output, *bias, *output, isNCHW);

            if(!isNCHW)
                delete input;
        }

void ConvolutionUtils::depthwiseConv2d(sd::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), depthwiseConv2d_, (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat), FLOAT_TYPES);
        }

}
}
