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
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>

#include "helpers/ShapeUtils.h"
#if NOT_EXCLUDED(OP_col2im) && NOT_EXCLUDED(OP_im2col)

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////


template <typename X, typename Y>
static void conv2dBP_(sd::graph::Context& block, NDArray* input, NDArray* weights, NDArray* bias,
                      NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const LongType kH, const LongType kW,
                      const LongType sH, const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                      const int isNCHW, const int wFormat) {

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
  // isNCHW      0-NHWC, 1-NCHW
  const LongType bS = input->sizeAt(0);  // batch size
  const LongType iH = ConvolutionUtils::inputHeight(input->shapeInfo(), isNCHW);    // input height
  const LongType iW = ConvolutionUtils::inputWidth(input->shapeInfo(), isNCHW);    // input width
  const LongType iC = ConvolutionUtils::inChannels(weights->shapeInfo(), wFormat);  // input channels
  const LongType oC = ConvolutionUtils::outChannels(weights->shapeInfo(), wFormat);  // output channels
  LongType oH = ConvolutionUtils::calcOutDimConv(iH, kH, sH, pH, dH, paddingMode);
  LongType oW = ConvolutionUtils::calcOutDimConv(iW, kW, sW, pW, dW, paddingMode);  // batch size, input channels, input height/width, output channels, output height/width;

  NDArray *inputPermuted, *gradIPermuted, *gradOPermuted;
  if (!isNCHW) {
    inputPermuted = new NDArray(input->permute({0, 3, 1, 2}, true));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradIPermuted = new NDArray(gradI->permute({0, 3, 1, 2}, true));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradOPermuted = const_cast<NDArray*>(gradO);
  } else {
    inputPermuted = const_cast<NDArray*>(input);
    gradIPermuted = const_cast<NDArray*>(gradI);
    gradOPermuted = new NDArray(gradO->permute({1, 0, 2, 3}, true));
  }

  NDArray* columns;
  if (block.hasIntermediateResults()) {
    columns = block.intermediateResult(0);
    if(columns->rankOf() < 6) {
      columns->reshapei({bS, iC, kH, kW, oH, oW});
    }

  } else {
    columns = new NDArray(inputPermuted->ordering(), {bS, iC, kH, kW, oH, oW}, inputPermuted->dataType(), inputPermuted->getContext());
  }

  columns->printIndexedBuffer("conv2dBP_ columns: \n");

  // ----- calculation of gradW ----- //
  if (gradW) {
    auto ctx = block.launchContext();
    if (!block.hasIntermediateResults()) {
      helpers::im2col(*ctx, *inputPermuted, *columns, kH, kW, sH, sW, pH, pW, dH, dW,
                      NDArrayFactory::create<double>(0., inputPermuted->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]

    }

    columns->printIndexedBuffer("conv2dBP_ columns after: \n");


    /**
     * NOTE ON THIS LOGIC here.
     * Be VERY careful with views and knowing buffer order.
     * Due to how GEMM works it sometimes will produce very strange results.
     */



    NDArray columns2d = columns->reshape('c', {iC * kH * kW,bS * oH * oW}, true);
    NDArray gradO2d = gradOPermuted->reshape('f', {bS * oH * oW,oC}, true);
    NDArray gradW2d = gradW->reshape('c', {iC * kH * kW,oC}, false);
    printf("bS %lld oH %lld oW %lld iC %lld kH %lld kW %lld\n", bS, oH, oW, iC, kH, kW);
    fflush(stdout);
    printf("Reshaped columns to: %lld %lld\n", bS * oH * oW, iC * kH * kW);
    fflush(stdout);
    printf("Reshaped gradO to: %lld %lld\n", oC, bS * oH * oW);
    fflush(stdout);
    printf("Reshaped gradW to: %lld %lld\n", iC * kH * kW, oC);

    columns2d.printShapeInfo("columns2d shape");
    gradO2d.printShapeInfo("gradO2d shape");
    gradW2d.printShapeInfo("gradW2d shape");
    fflush(stdout);
    sd::MmulHelper::matmul(&columns2d, &gradO2d, &gradW2d, false, false, 1.0, 0.0);
    gradW->printIndexedBuffer("conv2dBP_ GRAD W: \n");

  }

  // ----- calculation of gradB ----- //
  if (gradB) {
    if (!isNCHW) {
      std::vector<sd::LongType> axes = {0, 1, 2};
      gradOPermuted->reduceAlongDimension(reduce::Sum, *gradB, &axes);  // sum over bS, oH, oW
      gradB->printIndexedBuffer("conv2dBP_ GRAD B: \n");

    } else {
      std::vector<sd::LongType> axes = {1, 2, 3};
      gradOPermuted->reduceAlongDimension(reduce::Sum, *gradB, &axes);  // sum over bS, oH, oW
      gradB->printIndexedBuffer("conv2dBP_ GRAD B: \n");

    }
  }

  //----- calculation of gradI -----//
  NDArray weights2d = weights->permute({0, 3, 1, 2}, false).reshape(weights->ordering(), {oC, iC * kH * kW});

  NDArray gradO2d = gradOPermuted->reshape(gradOPermuted->ordering(), {bS * oH * oW, oC});
  NDArray columns2d = NDArray(columns->ordering(), {iC * kH * kW, bS * oH * oW}, columns->dataType(), columns->getContext());
  sd::MmulHelper::matmul(&weights2d, &gradO2d, &columns2d, true, true, 1.0, 0.0);



  std::vector<sd::LongType> columnsShape = {bS, iC, kH, kW, oH, oW};
  columns->assign(columns2d.reshape(columns2d.ordering(), columnsShape));

  helpers::col2im(*block.launchContext(), columns, gradIPermuted, sH, sW, pH, pW, iH, iW, dH,
                  dW);  // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]


  if (!isNCHW) {
    gradI->assign(gradIPermuted->permute({0, 2, 3, 1}, false));  // [bS, iC, iH, iW] -> [bS, iH, iW, iC]
  }
}

void ConvolutionUtils::conv2dBP(sd::graph::Context& block, NDArray* input, NDArray* weights,
                                NDArray* bias, NDArray* gradO, NDArray* gradI, NDArray* gradW,
                                NDArray* gradB, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH, LongType pW,
                                const LongType dH, const LongType dW, const int paddingMode, const int isNCHW,
                                const int wFormat) {
  BUILD_SINGLE_SELECTOR_TWICE(input->dataType(), conv2dBP_,
                              (block, input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW,
                                  paddingMode, isNCHW, wFormat),
                              SD_FLOAT_TYPES);
}

}  // namespace ops
}  // namespace sd
#endif