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

  LongType bS = input->sizeAt(0);
  LongType iC = ConvolutionUtils::inChannels(weights->shapeInfo(), wFormat);
  LongType oC = ConvolutionUtils::outChannels(weights->shapeInfo(), wFormat);
  LongType iH = ConvolutionUtils::inputHeight(input->shapeInfo(), isNCHW);
  LongType iW = ConvolutionUtils::inputWidth(input->shapeInfo(), isNCHW);
  LongType oH = ConvolutionUtils::calcOutDimConv(iH, kH, sH, pH, dH, paddingMode);
  LongType oW = ConvolutionUtils::calcOutDimConv(iW, kW, sW, pW, dW, paddingMode);
  std::vector<LongType> wAxes;
  if (0 == wFormat)
    wAxes = {0, 1, 2};
  else if (1 == wFormat)
    wAxes = {2, 3, 1};
  else
    wAxes = {1, 2, 3};


  std::vector<sd::LongType> colShape = {bS, oH, oW, kH, kW, iC};
  std::vector<sd::LongType> perm = {0, 3, 4, 5, 1, 2};
  NDArray *col = new NDArray('c', colShape, input->dataType(), input->getContext());
  col->printShapeInfo("COL SHAPE INFO:");
  NDArray &colPFrom = col->permute(perm, false, false);
  colPFrom.printShapeInfo("COLP FROM SHAPE INFO:");
  NDArray *colP = new NDArray(colPFrom);  // {bS, iC, kH, kW, oH, oW}
  colP->printShapeInfo("COLP SHAPE INFO:");
  std::vector<sd::LongType> mmulResultShape = {bS * oH * oW, oC};
  NDArray mmulResult('f', mmulResultShape, output->dataType(), output->getContext());
  std::vector<LongType> permuteForOutput = {0, 3, 1, 2};

  //----- calculation of output -----//
  auto ctx = block.launchContext();


  NDArray zeroVal =  NDArrayFactory::create(0.f, input->getContext());
  if (isNCHW) {
    input->printBufferRaw("im2col input:");
    input->printShapeInfo("im2col input shape info:");
    helpers::im2col(*ctx, *input, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    zeroVal);
  } else {
    std::vector<sd::LongType> permute = {0, 3, 1, 2};
    // For NHWC, we need to permute the input to NCHW before im2col
    NDArray* inputNchw = new NDArray(input->permute(permute, false,false));
    helpers::im2col(*ctx, *inputNchw, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    zeroVal);
  }

  col->printBufferRaw("COL BUFFER RAW:");

  block.pushIntermediateResult(colP);

  weights->printBufferRaw("WEIGHTS PRE PERMUTE:");
  std::vector<sd::LongType> shape = {bS * oH * oW, kH * kW * iC};
  NDArray &colReshaped = colP->reshape('c', shape, false);
  colP->printBufferRaw("colPReshaped buffer raw:");
  std::vector<sd::LongType> perm2 = {3,2,1,0};
   /**
    * TODO; get the weights back in the right format.
    * The im2col output seems to work now.
    */
  NDArray &weightsPermuted = weights->permute(perm2, false, false);
  weights->printBufferRaw("\nweights buffer raw:\n");
  colP->printBufferRaw("colPReshaped 2 buffer raw:");

  weightsPermuted.printBufferRaw("\nweights permuted buffer raw:\n");
  std::vector<sd::LongType> wShape = {iC * kH * kW, oC};
  NDArray &reshapedW = weightsPermuted.reshape('f',wShape, false);
  colReshaped.printBufferRaw("colPReshaped 3 buffer raw:");
  NDArray *colpPReshapedAddr = &colReshaped;
  colpPReshapedAddr->printBufferRaw("colPReshaped ADDR AFTER 3 buffer raw:");

  reshapedW.printBufferRaw("\nweights permuted reshaped buffer raw:\n");
  colReshaped.printBufferRaw("colPReshaped going in for matmul");
  reshapedW.printBufferRaw("\nweights permuted reshaped buffer raw:\n");
  NDArray *reshapedWAddr = &reshapedW;
  colpPReshapedAddr->printBufferRaw("colPReshaped buffer raw:");
  reshapedWAddr->printBufferRaw("\nweights permuted reshaped buffer raw:\n");
  MmulHelper::matmul(colpPReshapedAddr, reshapedWAddr, &mmulResult, false, false, 1.0, 0.0);
  mmulResult.printBufferRaw("mmulResult buffer raw:");

  std::vector<sd::LongType>lastShape = {oH,oW,bS,oC};
  NDArray &reshaped = mmulResult.reshape('f', lastShape, false);
  mmulResult.printBufferRaw("mmulResult reshaped buffer raw:");
  std::vector<sd::LongType> permute2 = {2,3,1,0};
  NDArray &permuted = reshaped.permute(permute2, false, false);
  mmulResult.printBufferRaw("mmulResult reshaped permuted buffer raw:");

  // Reshape and copy result to output
  if (isNCHW) {
    output->assign(permuted);
  } else {
    std::vector<sd::LongType> perm3 = {0,2,3,1};
    permuted = permuted.permute(perm3, false, false);
    output->assign(permuted);
  }

  //----- add biases if required -----//
  if (bias) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
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