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
    inputPermuted = input->permute(permute, false, false);  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradOPermuted = gradO->permute(permute, false, false);  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    gradIPermuted = gradI->permute(permute, false, false);  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
  } else {
    inputPermuted = input;
    gradOPermuted = gradO;
    gradIPermuted = gradI;
  }

  std::vector<sd::LongType> gradOShape = {oC, bS * oH * oW};
  // Reshape gradO to 2D: [oC, bS * oH * oW]
  NDArray *gradO2d = gradOPermuted->reshape(gradOPermuted->ordering(), gradOShape,false);

  // For SAME/CAUSAL padding modes, recalculate the actual padding used in the forward pass.
  // The config stores pH/pW=0 for SAME mode; we must recompute from the input/output sizes
  // so that both im2col (gradW) and col2im (gradI) use the same padding as the forward conv2d.
  if (paddingMode == 1) {
    // SAME: centered padding, matching convolutions_conv2d.cpp forward pass logic
    const LongType eKH = (kH - 1) * dH + 1;
    const LongType eKW = (kW - 1) * dW + 1;
    LongType totalPadH = std::max((LongType)0, (oH - 1) * sH + eKH - iH);
    LongType totalPadW = std::max((LongType)0, (oW - 1) * sW + eKW - iW);
    pH = totalPadH / 2;
    pW = totalPadW / 2;
  } else if (paddingMode == 2) {
    // CAUSAL: left-only padding
    pH = (kH - 1) * dH;
    pW = (kW - 1) * dW;
  }

  // Perform im2col or retrieve from intermediate results
  // Forward pass stores: index 0 = col (owner), index 1 = colP (view with shape {bS, iC, kH, kW, oH, oW})
  NDArray* columns;
  if (block.hasIntermediateResults()) {
    // Intermediate result at index 1 is colP (the view we use)
    // Index 0 is col (the owner) - framework handles cleanup
    columns = block.intermediateResult(1);
    if (columns->rankOf() < 6) {
      columns->reshapei({bS, iC, kH, kW, oH, oW});
    }
  } else {
    std::vector<sd::LongType> colShape = {bS, iC, kH, kW, oH, oW};
    columns = new NDArray(inputPermuted->ordering(), colShape, inputPermuted->dataType(), inputPermuted->getContext());
    auto ctx = block.launchContext();
    NDArray *zeroVal = NDArrayFactory::create<double>(0., inputPermuted->getContext());
    helpers::im2col(*ctx, *inputPermuted, *columns, kH, kW, sH, sW, pH, pW, dH, dW,
                    *zeroVal);
    delete zeroVal;
  }

  // Calculate gradW
  if (gradW) {
    std::vector<sd::LongType> colShape = {bS * oH * oW, iC * kH * kW};
    std::vector<sd::LongType> wShape = {oC, iC * kH * kW};
    NDArray *columns2d = columns->reshape('c',colShape,false);
    std::vector<sd::LongType> permute = {1,0};
    NDArray *gradWReshaped = gradW->reshape('f', wShape, false);
    NDArray *gradW2d = gradWReshaped->permute(permute, false, false);

    MmulHelper::matmul( columns2d,gradO2d, gradW2d, true, true, 1.0, 0.0, gradW2d);
    gradW->assign(gradW2d);
    delete gradW2d;
    delete gradWReshaped;
    delete columns2d;

  }

  // Calculate gradB
  if (gradB) {
    std::vector<LongType> axes = {1};  // Sum over bS, oH, oW
    gradO2d->reduceAlongDimension(reduce::Sum, gradB, &axes);
  }

  // Calculate gradI
  NDArray *weights2d;
  NDArray *weightsPermuted = nullptr;
  if (wFormat == 0) {
    std::vector<sd::LongType> perm = {3,2,1,0};
    std::vector<sd::LongType> wShape = {iC * kH * kW,oC};
    weightsPermuted = weights->permute(perm, false, false);
    weights2d = weightsPermuted->reshape('f', wShape);
  } else if (wFormat == 1) {
    std::vector<sd::LongType> wShape2 = {iC * kH * kW,oC};
    weights2d = weights->reshape('f', wShape2);
  } else {
    std::vector<sd::LongType> wPermute = {0,2,3,1};
    std::vector<sd::LongType> weights2dShape = {iC * kH * kW,oC};
    weightsPermuted = weights->permute(wPermute, false, false);
    weights2d = weightsPermuted->reshape('f', weights2dShape);
  }

  std::vector<sd::LongType> columns2dShape = {iC * kH * kW, bS * oH * oW};
  NDArray columns2d('c', columns2dShape, columns->dataType(), columns->getContext());


  MmulHelper::matmul(weights2d, gradO2d, &columns2d, false, false, 1.0, 0.0);
  if (weightsPermuted) delete weightsPermuted;
  delete weights2d;
  //Calculate epsilonNext by doing im2col reduction.
  //Current col2im implementation expects input with order: [miniBatch,channels,kH,kW,outH,outW]
  //currently have [kH,kW,inDepth,outW,outH,miniBatch] -> permute first
  auto eps6d = columns2d.newShapeNoCopy({kH, kW,iC, oW, oH, bS }, 'f');
  std::vector<sd::LongType> epsPermute = {5,2,1,0,4,3};
  auto permuted = eps6d->permute(epsPermute, false, false);

  // Perform col2im
  auto ctx = block.launchContext();
  helpers::col2im(*ctx, permuted, gradIPermuted, sH, sW, pH, pW, iH, iW, dH, dW);
  // Handle NHWC format if necessary
  if (!isNCHW) {
    std::vector<sd::LongType> perm = {0,2,3,1};
    NDArray *permutedGradI = gradIPermuted->permute(perm, false, false);
    gradI->assign(permutedGradI);  // [bS, iC, iH, iW] -> [bS, iH, iW, iC]
    delete permutedGradI;
  }

  delete permuted;
  delete eps6d;
  delete gradO2d;
  // Clean up
  if (!isNCHW) {
    delete inputPermuted;
    delete gradOPermuted;
    delete gradIPermuted;
  }

  // Only delete columns if we created it fresh (not from intermediate results)
  // When from intermediate results, the framework handles cleanup automatically
  if (!block.hasIntermediateResults()) {
    delete columns;
  }
  // Note: intermediate results (col owner and colP view) are cleaned up by the framework
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