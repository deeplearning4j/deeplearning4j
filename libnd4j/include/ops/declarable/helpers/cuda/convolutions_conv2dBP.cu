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
#include <helpers/DebugHelper.h>
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
    inputPermuted = input->permute(permute, false, false);
    gradOPermuted = gradO->permute(permute, false, false);
    gradIPermuted = gradI->permute(permute, false, false);
  } else {
    inputPermuted = input;
    gradOPermuted = gradO;
    gradIPermuted = gradI;
  }

  // Prepare gradO as 2D: permute gradOPermuted [bS, oC, oH, oW] -> [oC, bS, oH, oW],
  // then dup contiguous and reshape to [oC, bS*oH*oW].
  // This ensures the bS*oH*oW dimension decomposes as (bS, oH, oW) in C-order.
  std::vector<sd::LongType> gradOPerm = {1, 0, 2, 3};
  NDArray* gradOPermuted2 = gradOPermuted->permute(gradOPerm, false, false);
  NDArray* gradO2d = gradOPermuted2->dup('c');
  delete gradOPermuted2;
  std::vector<sd::LongType> gradO2dShape = {oC, bS * oH * oW};
  gradO2d->reshapei('c', gradO2dShape);

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
  NDArray* zero = nullptr;
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
    zero = NDArrayFactory::create<double>(0., inputPermuted->getContext());
    helpers::im2col(*ctx, *inputPermuted, *columns, kH, kW, sH, sW, pH, pW, dH, dW,
                   *zero);
  }

  // Calculate gradW directly in the output's physical weight format. This avoids
  // an asynchronous GEMM into an inner-scope owner followed by a queued permute/
  // assign: the owner's destructor previously returned its device storage to the
  // pool before the assign kernel consumed it.
  NDArray* gradW2dView = nullptr;
  if (gradW) {
    const LongType kernelElements = iC * kH * kW;
    const LongType sampleElements = bS * oH * oW;
    // Formats 0 and 2 flatten their kernel axis as (kH,kW,iC); format 1 uses
    // (iC,kH,kW). Match that order before flattening the im2col operand.
    std::vector<sd::LongType> colPerm = wFormat == 1
        ? std::vector<sd::LongType>{1, 2, 3, 0, 4, 5}
        : std::vector<sd::LongType>{2, 3, 1, 0, 4, 5};
    NDArray* colPermuted = columns->permute(colPerm, false, false);
    NDArray* colContig = colPermuted->dup('c');
    delete colPermuted;
    colContig->reshapei('c', {kernelElements, sampleElements});

    std::vector<sd::LongType> gradW2dShape;
    if (wFormat == 0) {
      // [kH,kW,iC,oC] is physically [K,oC].
      gradW2dShape = {kernelElements, oC};
    } else {
      // [oC,iC,kH,kW] and [oC,kH,kW,iC] are physically [oC,K].
      gradW2dShape = {oC, kernelElements};
    }
    gradW2dView = gradW->reshape('c', gradW2dShape, false);

    if (wFormat == 0) {
      MmulHelper::matmul(colContig, gradO2d, gradW2dView, false, true, 1.0, 0.0);
    } else {
      MmulHelper::matmul(gradO2d, colContig, gradW2dView, false, true, 1.0, 0.0);
    }
    // cuBLAS consumes colContig asynchronously. Retire it on the execution
    // stream so the pool cannot recycle it before the GEMM has read it.
    MmulHelper::deleteTemporary(colContig);
  }

  // Calculate gradB
  if (gradB) {
    std::vector<LongType> axes = {1};  // Sum over bS*oH*oW dimension
    // gradO2d is [oC, bS*oH*oW]; reduction along axis 1 produces shape [oC] (1D).
    // gradB may be allocated as [1, oC] (2D) when the bias was [1, oC], so we must
    // reshape it to 1D before the reduction to avoid a shape mismatch.
    std::vector<LongType> gradBShape1d = {gradB->lengthOf()};
    NDArray *gradB1d = gradB->reshape(gradB->ordering(), gradBShape1d, false);
    gradO2d->reduceAlongDimension(reduce::Sum, gradB1d, &axes);
    delete gradB1d;
  }

  // Calculate gradI: permute weights to [iC, kH, kW, oC], dup contiguous,
  // then matmul as [iC*kH*kW, oC] * [oC, bS*oH*oW] = [iC*kH*kW, bS*oH*oW]
  NDArray* weightsPermuted4d = nullptr;
  if (wFormat == 0) {
    std::vector<sd::LongType> permute = {2, 0, 1, 3};
    weightsPermuted4d = weights->permute(permute, false, false);
  } else if (wFormat == 1) {
    std::vector<sd::LongType> permute = {1, 2, 3, 0};
    weightsPermuted4d = weights->permute(permute, false, false);
  } else {
    std::vector<sd::LongType> permute3 = {3, 1, 2, 0};
    weightsPermuted4d = weights->permute(permute3, false, false);
  }

  NDArray* weightsContig = weightsPermuted4d->dup('c');
  delete weightsPermuted4d;
  std::vector<sd::LongType> wShape2d = {iC * kH * kW, oC};
  weightsContig->reshapei('c', wShape2d);

  std::vector<sd::LongType> colShape2 = {iC * kH * kW, bS * oH * oW};
  NDArray* columns2d = new NDArray('c', colShape2, columns->dataType(), columns->getContext());
  MmulHelper::matmul(weightsContig, gradO2d, columns2d, false, false, 1.0, 0.0);

  std::vector<sd::LongType> eps6dShape = {iC, kH, kW, bS, oH, oW};
  columns2d->reshapei('c', eps6dShape);
  std::vector<sd::LongType> epsPerm = {3, 0, 1, 2, 4, 5};
  NDArray* permuted = columns2d->permute(epsPerm, false, false);

  // Perform col2im
  auto ctx = block.launchContext();
  helpers::col2im(*ctx, permuted, gradIPermuted, sH, sW, pH, pW, iH, iW, dH, dW);
  // Handle NHWC format if necessary
  if (!isNCHW) {
    std::vector<sd::LongType> permute = {0, 2, 3, 1};
    NDArray* permutedGradI = gradIPermuted->permute(permute, false, false);  // [bS, iC, iH, iW] -> [bS, iH, iW, iC]
    gradI->assign(permutedGradI);  // [bS, iC, iH, iW] -> [bS, iH, iW, iC]
    delete permutedGradI;
  }

  // Clean up. Queue owned GPU-buffer retirement behind each temporary's final
  // stream consumer so asynchronous GEMM/reduction/col2im work remains valid
  // without a host-side stream synchronization.
  delete permuted;
  MmulHelper::deleteTemporary(columns2d);
  MmulHelper::deleteTemporary(weightsContig);
  if (gradW2dView) delete gradW2dView;
  MmulHelper::deleteTemporary(gradO2d);
  if (zero) MmulHelper::deleteTemporary(zero);
  if (!isNCHW) {
    delete inputPermuted;
    delete gradOPermuted;
    delete gradIPermuted;
  }

  // Only delete columns if we created it fresh (not from intermediate results)
  // When from intermediate results, the framework handles cleanup automatically
  if (!block.hasIntermediateResults()) {
    MmulHelper::deleteTemporary(columns);
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
