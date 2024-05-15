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
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#if NOT_EXCLUDED(OP_col2im) && NOT_EXCLUDED(OP_im2col)

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2d_(sd::graph::Context& block, NDArray* input, NDArray* weights, NDArray* bias,
                    NDArray* output, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH, LongType pW,
                    const LongType dH, const LongType dW, const int paddingMode, const int isNCHW, const int wFormat) {

  // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  // weights [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  // bias    [oC]
  // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  // kH  filter(kernel) height
  // kW  filter(kernel) width
  // sH  strides height
  // sW  strides width
  // pH  paddings height
  // pW  paddings width
  // dH  dilations height
  // dW  dilations width
  // paddingMode 0-VALID, 1-SAME
  // isNCHW      1-NCHW,  0-NHWC


  LongType bS = input->sizeAt(0);
  LongType   iC = ConvolutionUtils::inChannels(weights->shapeInfo(), wFormat);
  LongType    oC = ConvolutionUtils::outChannels(weights->shapeInfo(), wFormat);
  LongType iH = ConvolutionUtils::inputHeight(input->shapeInfo(), isNCHW);
  LongType iW = ConvolutionUtils::inputWidth(input->shapeInfo(), isNCHW);
  LongType    oH = ConvolutionUtils::calcOutDimConv(iH, kH, sH, pH, dH, paddingMode);
  LongType   oW = ConvolutionUtils::calcOutDimConv(iW,kW,sW,pW,dW,paddingMode);  // batch size, input channels, input height/width, output channels, output height/width;

  if (!isNCHW)
    input = new NDArray(input->permute({0, 3, 1, 2}, false));  // NHWC to NCHW


  NDArray col('c', {bS, oH, oW, iC, kH, kW}, input->dataType(), input->getContext());
  std::vector<sd::LongType> permute = {0, 3, 4, 5, 1, 2};
  NDArray* col2 = new NDArray(col.permute(permute, false));  // {bS, iC, kH, kW, oH, oW}

  NDArray* im2ColIn = new NDArray(input->cast(col2->dataType()));

  auto ctx = block.launchContext();
  helpers::im2col(*ctx, *im2ColIn, *col2, kH, kW, sH, sW, pH, pW, dH, dW, NDArrayFactory::create(0.f, input->getContext()));
  block.pushIntermediateResult(col2);

  std::vector<LongType> permuteW = {3,2,1,0};
  NDArray permutedW = weights->permute(permuteW, true);
  std::vector<LongType> newShape = {kW * kH * iC, oC};
  NDArray *reshapedW =  new NDArray(permutedW.reshape(permutedW.ordering(),newShape,true));
  NDArray im2col2d = col.reshape('c', {bS * oH * oW, iC * kH * kW}, true);
  if(output->ordering() != 'f') {
    NDArray mmulResult('f', {bS * oH * oW, oC}, output->dataType(), output->getContext());
    MmulHelper::matmul(&im2col2d,reshapedW,&mmulResult,false,false);
    if (bias) {
      helpers::addBias(block, mmulResult, *bias, mmulResult, true);
    }

    if (isNCHW) {
      mmulResult.reshapei({bS, oH, oW, oC});
      mmulResult.permutei({0, 3, 1, 2}, false);  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    }

    //NOTE: WE DO THIS BECAUSE OF GEMM/BLAS OPERATING PURELY ON LINEAR BUFFERS. IT DOES NOT KNOW WHAT STRIDES ARE
    //THE CORRECT ORDER HERE IS TO COPY THE DATA OVER TO THE OUTPUT BUFFER
    output->dataBuffer()->copyBufferFrom(*mmulResult.dataBuffer(), mmulResult.lengthOf() * mmulResult.sizeOfT());


  } else {
    NDArray mmulResult = output->reshape(output->ordering(), {bS * oH * oW, oC},false);
    MmulHelper::matmul(&im2col2d,reshapedW,&mmulResult,false,false);
    if (bias) {
      helpers::addBias(block, mmulResult, *bias, mmulResult, isNCHW);
    }

    if (isNCHW) {
      mmulResult.reshapei({bS, oH, oW, oC});
      mmulResult.permutei({0, 3, 1, 2}, false);  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    }

    //NOTE: WE DO THIS BECAUSE OF GEMM/BLAS OPERATING PURELY ON LINEAR BUFFERS. IT DOES NOT KNOW WHAT STRIDES ARE
    //THE CORRECT ORDER HERE IS TO COPY THE DATA OVER TO THE OUTPUT BUFFER
    output->dataBuffer()->copyBufferFrom(*mmulResult.dataBuffer(), mmulResult.lengthOf() * mmulResult.sizeOfT());


  }


  if (!isNCHW) {
    delete input;
    delete im2ColIn;
  }
}

void ConvolutionUtils::conv2d(sd::graph::Context& block, NDArray* input, NDArray* weights,
                              NDArray* bias, NDArray* output, const LongType kH, const LongType kW, const LongType sH,
                              const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                              const int isNCHW, const int wFormat) {
  BUILD_SINGLE_SELECTOR_TWICE(
      input->dataType(), conv2d_,
      (block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat),
      SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
#endif