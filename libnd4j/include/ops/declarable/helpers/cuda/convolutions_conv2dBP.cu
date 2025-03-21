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

//
// @author Yurii Shyrma (iuriish@yahoo.com)
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
                      const LongType sH, const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW,
                      const int paddingMode, const int isNCHW, const int wFormat) {

  // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  // weights [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  // bias    [oC]
  // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  // gradI   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  // gradW   [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  // gradB   [oC]

  const LongType bS = input->sizeAt(0);  // batch size
  const LongType iC = isNCHW ? input->sizeAt(1) : input->sizeAt(3);  // input channels
  const LongType iH = isNCHW ? input->sizeAt(2) : input->sizeAt(1);  // input height
  const LongType iW = isNCHW ? input->sizeAt(3) : input->sizeAt(2);  // input width

  const LongType oC = isNCHW ? gradO->sizeAt(1) : gradO->sizeAt(3);  // output channels
  const LongType oH = isNCHW ? gradO->sizeAt(2) : gradO->sizeAt(1);  // output height
  const LongType oW = isNCHW ? gradO->sizeAt(3) : gradO->sizeAt(2);  // output width
  NDArray *inputPermuted, *gradOPermuted, *gradIPermuted;
  if (!isNCHW) {
    std::vector<sd::LongType> permute = {0, 3, 1, 2};
    inputPermuted = new NDArray(input->permute(permute,false,false));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradOPermuted = new NDArray(gradO->permute(permute,false,false));  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    gradIPermuted = new NDArray(gradI->permute(permute,false,false));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
  } else {
    inputPermuted = input;
    gradOPermuted = gradO;
    gradIPermuted = gradI;
  }

  std::vector<sd::LongType> reshape = {oC,bS * oH * oW};
  // Reshape gradO to 2D: [oC, bS * oH * oW]
  NDArray gradO2d = gradOPermuted->reshape(gradOPermuted->ordering(), reshape,false);

  // Perform im2col
  NDArray* columns;
  if (block.hasIntermediateResults()) {
    columns = block.intermediateResult(0);
    if (columns->rankOf() < 6) {
      columns->reshapei({bS, iC, kH, kW, oH, oW});
    }
  } else {
    std::vector<sd::LongType> colShape = {bS, iC, kH, kW, oH, oW};
    columns = new NDArray(inputPermuted->ordering(), colShape, inputPermuted->dataType(), inputPermuted->getContext());
    auto ctx = block.launchContext();
    NDArray zero =   NDArrayFactory::create<double>(0., inputPermuted->getContext());
    helpers::im2col(*ctx, *inputPermuted, *columns, kH, kW, sH, sW, pH, pW, dH, dW,
                   zero);
  }

  // Calculate gradW
  if (gradW) {
    std::vector<sd::LongType> columnsShape = {bS * oH * oW, iC * kH * kW};
    std::vector<sd::LongType> wShape = {oC, iC * kH * kW};
    NDArray columns2d = columns->reshape('c', columnsShape,false);
    std::vector<sd::LongType > permu = {1,0};
    NDArray &gradW2d = gradW->reshape('f', wShape,false).permute(permu,false,false);

    MmulHelper::matmul( &columns2d,&gradO2d, &gradW2d, true, true, 1.0, 0.0, &gradW2d);
    gradW->assign(gradW2d);

  }

  // Calculate gradB
  if (gradB) {
    std::vector<LongType> axes = {1};  // Sum over bS, oH, oW
    gradO2d.reduceAlongDimension(reduce::Sum, gradB, &axes);
  }

  // Calculate gradI
  NDArray weights2d;
  if (wFormat == 0) {
    std::vector<sd::LongType> permute = {3,2,1,0};
    std::vector<sd::LongType> shape2 = {iC * kH * kW, oC};
    weights2d = weights->permute(permute,false,false).reshape('f', shape2);
  } else if (wFormat == 1) {
    std::vector<sd::LongType> shape2 = {iC * kH * kW, oC};
    weights2d = weights->reshape('f', shape2);
  } else {
    std::vector<sd::LongType> shape2 = {iC * kH * kW, oC};
    std::vector<sd::LongType> permute3 = {0, 2, 3, 1};
    weights2d = weights->permute(permute3,false,false).reshape('f', shape2);
  }

  std::vector<sd::LongType> colShape2 = {iC * kH * kW, bS * oH * oW};
  NDArray columns2d('c', colShape2, columns->dataType(), columns->getContext());


  MmulHelper::matmul(&weights2d, &gradO2d, &columns2d, false, false, 1.0, 0.0);
  //Calculate epsilonNext by doing im2col reduction.
  //Current col2im implementation expects input with order: [miniBatch,channels,kH,kW,outH,outW]
  //currently have [kH,kW,inDepth,outW,outH,miniBatch] -> permute first
  auto eps6d = columns2d.newShapeNoCopy({kH, kW,iC, oW, oH, bS }, 'f');
  std::vector<sd::LongType> epsPerm = {5,2,1,0,4,3};
  auto permuted = eps6d->permute(epsPerm,false,false);

  // Perform col2im
  auto ctx = block.launchContext();
  helpers::col2im(*ctx, &permuted, gradIPermuted, sH, sW, pH, pW, iH, iW, dH, dW);
  // Handle NHWC format if necessary
  if (!isNCHW) {
    std::vector<sd::LongType> permute = {0, 2,3,1};
    gradI->assign(gradIPermuted->permute(permute,false,false));  // [bS, iC, iH, iW] -> [bS, iH, iW, iC]
  }

  // Clean up
  if (!isNCHW) {
    delete inputPermuted;
    delete gradOPermuted;
    delete gradIPermuted;
  }
  if (!block.hasIntermediateResults()) {
    delete columns;
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
