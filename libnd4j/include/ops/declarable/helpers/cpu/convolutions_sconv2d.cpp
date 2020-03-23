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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//

#include <ops/declarable/helpers/convolutions.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops  {


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void sconv2d_(sd::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {

            // input         [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
            // weightsDepth  [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
            // weightsPoint  [1, 1, iC*mC, oC], [oC, iC*mC, 1, 1], [oC, 1, 1, iC*mC]
            // bias          [oC], oC = iC*mC if weightsPoint=nullptr
            // output is     [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)

            //  kH         filter(kernel) height
            //  kW         filter(kernel) width
            //  sH         strides height
            //  sW         strides width
            //  pH         paddings height
            //  pW         paddings width
            //  dH         dilations height
            //  dW         dilations width
            //  paddingMode 0-VALID, 1-SAME
            //  isNCHW      1-NCHW,  0-NHWC

            int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
            int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
            mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

            NDArray* outputDepth = output;
            if(weightsPoint)                        // if pointwise convolution is expected
                outputDepth = new NDArray(output->ordering(), !isNCHW ? std::vector<Nd4jLong>({bS, oH, oW, iC*mC}) : std::vector<Nd4jLong>({bS, iC*mC, oH, oW}), input->dataType(), input->getContext());

            // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //
            ConvolutionUtils::depthwiseConv2d(block, input, weightsDepth, weightsPoint ? nullptr : bias, outputDepth, kH,kW, sH,sW, pH,pW, dH,dW, paddingMode, isNCHW, wFormat);

            // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
            if (weightsPoint) {
                ConvolutionUtils::conv2d(block, outputDepth, weightsPoint, bias, output, 1,1, 1,1, 0,0, 1,1, paddingMode, isNCHW, wFormat);             // in this case oH=iH, oW=iW
                delete outputDepth;
            }
        }

void ConvolutionUtils::sconv2d(sd::graph::Context& block, const NDArray* input, const NDArray* weightsDepth, const NDArray* weightsPoint, const NDArray* bias,  NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {
            BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), sconv2d_, (block, input, weightsDepth, weightsPoint, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat), FLOAT_TYPES);
        }

}
}
