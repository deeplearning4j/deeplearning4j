/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include<ops/declarable/helpers/addBias.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>

namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2dBP_(sd::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
    // bias    [oC]
    // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

    // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    // gradW    [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
    // gradB    [oC]

    // kH         filter(kernel) height
    // kW         filter(kernel) width
    // sH         strides height
    // sW         strides width
    // pH         paddings height
    // pW         paddings width
    // dH         dilations height
    // dW         dilations width
    // paddingMode 0-VALID, 1-SAME
    // isNCHW     0-NHWC, 1-NCHW

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

    std::vector<int> gradOaxesForDot;

    if(!isNCHW) {
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW
        input = new NDArray(input->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        gradI = new NDArray(gradI->permute({0, 3, 1, 2}));                      // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    } else {
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
    }

    std::vector<int> wPermut, colPermut;
    if(0 == wFormat) {
        wPermut   = {2, 0, 1, 3};
        colPermut = {2, 3, 1, 0, 4, 5};
    }
    else if(1 == wFormat) {
        wPermut   = {1, 2, 3, 0};
        colPermut = {1, 2, 3, 0, 4, 5};
    }
    else {
        wPermut   = {3, 1, 2, 0};
        colPermut = {2, 3, 1, 0, 4, 5};
    }

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->dataType(), input->getContext());

    // ----- calculation of gradW ----- //
    if(gradW) {
        auto ctx = block.launchContext();
        helpers::im2col(*ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));   // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
        sd::MmulHelper::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, wPermut);       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
    }

    // ----- calculation of gradB ----- //
    if(gradB) {
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2)
            gradBR = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}));
        gradO->reduceAlongDimension(reduce::Sum, *gradBR, gradOaxesForDot, false);                          // sum over bS, oH, oW
        if(gradBR != gradB)
            delete gradBR;
    }

    //----- calculation of gradI -----//
    // [kH, kW, iC, oC] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]
    // [oC, iC, kH, kW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, bS, oH, oW]
    // [oC, kH, kW, iC] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]
    sd::MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, colPermut);  // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]

    helpers::col2im(*block.launchContext(), columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                          // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::conv2dBP(sd::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int paddingMode, const int isNCHW, const int wFormat) {
    BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2dBP_, (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat), FLOAT_TYPES);
}

}
}
