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
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#if NOT_EXCLUDED(OP_col2im) && NOT_EXCLUDED(OP_im2col)

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2dBP_(NDArray* input, NDArray* weights, NDArray* bias, NDArray* gradO,
                               NDArray* gradI, NDArray* gradW, NDArray* gradB, const LongType kH, const LongType kW, const LongType sH,
                               const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                               const int isNCHW, const int wFormat) {
  // input    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
  // weights  [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  // bias     [oC] = [iC*mC]
  // gradO    [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
  // gradI    [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
  // gradW    [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  // gradB    [oC]

  //  kH          filter(kernel) height
  //  kW          filter(kernel) width
  //  sH          strides height
  //  sW          strides width
  //  pH          paddings height
  //  pW          paddings width
  //  dH          dilations height
  //  dW          dilations width
  //  paddingMode 0-VALID, 1-SAME
  //  isNCHW      0-NHWC, 1-NCHW

  LongType bS, iC, iH, iW, mC, oC, oH, oW;  // batch size, input channels, input height/width, channels multiplier(oC =
  // iC*mC), output channels, output height/width
  LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);  // channels multiplier

  std::vector<std::vector<sd::LongType>> modifColumns = {
      {1, 2, 3, 0, 4, 5}, {iC, kH * kW, bS * oH * oW}};  // [bS,iC,kH,kW,oH,oW] -> [iC, kH*kW, bS*oH*oW]
  std::vector<std::vector<sd::LongType>> modifGradO1, modifGradO2, modifWeights;
  std::vector<sd::LongType> gradOreShape;

  if (!isNCHW) {
    gradOreShape = {bS, oH, oW, iC, mC};  // [bS,oH,oW,iC*mC] -> [bS,oH,oW,iC,mC]
    modifGradO1 = {{3, 0, 1, 2, 4},
                   {iC, bS * oH * oW, mC}};                // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    modifGradO2 = {{3, 0, 1, 2}, {iC, mC, bS * oH * oW}};  // [bS,oH,oW,iC*mC] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
    std::vector<sd::LongType> perm = {0,3,1,2};
    input = new NDArray(input->permute(perm, false, false));     // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
    gradI = new NDArray(gradI->permute(perm, false, false));     // [bS,iH,iW,iC]    -> [bS,iC,iH,iW]
  } else {
    gradOreShape = {bS, iC, mC, oH, oW};  // [bS,iC*mC,oH,oW] -> [bS,iC,mC,oH,oW]
    modifGradO1 = {{1, 0, 3, 4, 2},
                   {iC, bS * oH * oW, mC}};                // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC] -> [iC,bS*oH*oW,mC]
    modifGradO2 = {{1, 0, 2, 3}, {iC, mC, bS * oH * oW}};  // [bS,iC*mC,oH,oW] -> [iC*mC,bS,oH,oW] -> [iC,mC,bS*oH*oW]
  }

  if (0 == wFormat)
    modifWeights = {{2, 0, 1, 3}, {iC, kH * kW, mC}};
  else if (1 == wFormat)
    modifWeights = {{1, 2, 3, 0}, {iC, kH * kW, mC}};
  else
    modifWeights = {{3, 1, 2, 0}, {iC, kH * kW, mC}};

  if (paddingMode == 1)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  std::vector<LongType> colShape = {bS, iC, kH, kW, oH, oW};
  NDArray columns(input->ordering(), colShape, input->dataType(), input->getContext());
  NDArray gradOreshaped = gradO->reshape(gradO->ordering(), gradOreShape);

  // ----- calculation of gradW and gradB ----- //
  NDArray zero = NDArrayFactory::create(0.f, input->getContext());
  helpers::im2col(
      *input->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW,
  zero);  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
  sd::MmulHelper::tensorDot(&columns, &gradOreshaped, gradW, modifColumns, modifGradO1,
                            modifWeights);  // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

  // ----- calculation of gradB ----- //
  if (gradB) {
    NDArray* gradBR = gradB;
    std::vector<LongType> shape = {gradB->lengthOf()};
    if (gradB->rankOf() == 2) gradBR = new NDArray(gradB->reshape(gradB->ordering(), shape, false));
    std::vector<sd::LongType> axes = {0, indOoH, indOoH + 1};
    gradO->reduceAlongDimension(reduce::Sum, gradBR, &axes);  // sum over bS, oH, oW

    if (gradBR != gradB) delete gradBR;
  }

  //----- calculation of gradI -----//
  sd::MmulHelper::tensorDot(weights, gradO, &columns, modifWeights, modifGradO2,
                            modifColumns);  // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]
  helpers::col2im(*input->getContext(), &columns, gradI, sH, sW, pH, pW, iH, iW, dH,
                  dW);  // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

  if (!isNCHW) {
    delete input;
    delete gradI;
  }
}

void ConvolutionUtils::depthwiseConv2dBP(graph::Context& block, NDArray* input, NDArray* weights,
                                         NDArray* bias, NDArray* gradO, NDArray* gradI, NDArray* gradW,
                                         NDArray* gradB, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH,
                                         LongType pW, const LongType dH, const LongType dW, const int paddingMode, const int isNCHW,
                                         const int wFormat) {
  BUILD_SINGLE_SELECTOR_TWICE(
      input->dataType(), depthwiseConv2dBP_,
      (input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat),
      SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
#endif