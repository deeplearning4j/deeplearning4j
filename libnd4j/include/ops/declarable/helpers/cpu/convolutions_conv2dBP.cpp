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
  // gradW[iC,kH,kW,oC] = sum_{bS,oH,oW} columns[bS,iC,kH,kW,oH,oW] * gradO[bS,oC,oH,oW]
  // Permute columns to [iC, kH, kW, bS, oH, oW], dup contiguous, reshape to [iC*kH*kW, bS*oH*oW]
  // Then matmul: [iC*kH*kW, bS*oH*oW] * [bS*oH*oW, oC] = [iC*kH*kW, oC]
  if (gradW) {
    std::vector<sd::LongType> colPerm = {1, 2, 3, 0, 4, 5};
    NDArray* colPermuted = columns->permute(colPerm, false, false);
    NDArray* colContig = colPermuted->dup('c');
    delete colPermuted;
    std::vector<sd::LongType> col2dShape = {iC * kH * kW, bS * oH * oW};
    colContig->reshapei('c', col2dShape);

    // gradO2d^T = [bS*oH*oW, oC]
    std::vector<sd::LongType> gradW2dShape = {iC * kH * kW, oC};
    NDArray gradW2d('c', gradW2dShape, gradW->dataType(), gradW->getContext());
    MmulHelper::matmul(colContig, gradO2d, &gradW2d, false, true, 1.0, 0.0);
    delete colContig;

    // Reshape to [iC, kH, kW, oC] then permute to match gradW format
    std::vector<sd::LongType> gradW4dShape = {iC, kH, kW, oC};
    NDArray *gradW4d = gradW2d.reshape('c', gradW4dShape, false);
    if (wFormat == 0) {
      // gradW is [kH, kW, iC, oC] — permute from [iC, kH, kW, oC]
      std::vector<sd::LongType> perm = {1, 2, 0, 3};
      NDArray *gradWPermuted = gradW4d->permute(perm, false, false);
      gradW->assign(gradWPermuted);
      delete gradWPermuted;
    } else if (wFormat == 1) {
      // gradW is [oC, iC, kH, kW] — permute from [iC, kH, kW, oC]
      std::vector<sd::LongType> perm = {3, 0, 1, 2};
      NDArray *gradWPermuted = gradW4d->permute(perm, false, false);
      gradW->assign(gradWPermuted);
      delete gradWPermuted;
    } else {
      // gradW is [oC, kH, kW, iC] — permute from [iC, kH, kW, oC]
      std::vector<sd::LongType> perm = {3, 1, 2, 0};
      NDArray *gradWPermuted = gradW4d->permute(perm, false, false);
      gradW->assign(gradWPermuted);
      delete gradWPermuted;
    }
    delete gradW4d;
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

  // Calculate gradI using the same im2col/col2im approach as gradW:
  // Compute weights^T * gradO in the columns space, then col2im to get gradI.
  //
  // We need: columns[bS, iC, kH, kW, oH, oW] = sum_oC weights[iC,kH,kW,oC] * gradO[oC, bS*oH*oW]
  // Use tensorDot-style approach: permute weights to [iC, kH, kW, oC], make contiguous,
  // then matmul as [iC*kH*kW, oC] * [oC, bS*oH*oW] = [iC*kH*kW, bS*oH*oW]
  NDArray *weightsPermuted4d = nullptr;
  if (wFormat == 0) {
    // weights [kH, kW, iC, oC] -> [iC, kH, kW, oC]
    std::vector<sd::LongType> perm = {2, 0, 1, 3};
    weightsPermuted4d = weights->permute(perm, false, false);
  } else if (wFormat == 1) {
    // weights [oC, iC, kH, kW] -> [iC, kH, kW, oC]
    std::vector<sd::LongType> perm = {1, 2, 3, 0};
    weightsPermuted4d = weights->permute(perm, false, false);
  } else {
    // weights [oC, kH, kW, iC] -> [iC, kH, kW, oC]
    std::vector<sd::LongType> perm = {3, 1, 2, 0};
    weightsPermuted4d = weights->permute(perm, false, false);
  }

  // Make a contiguous C-order copy of the permuted weights, then reshape to 2D
  NDArray* weightsContig = weightsPermuted4d->dup('c');  // [iC, kH, kW, oC] contiguous
  delete weightsPermuted4d;
  std::vector<sd::LongType> wShape2d = {iC * kH * kW, oC};
  weightsContig->reshapei('c', wShape2d);  // [iC*kH*kW, oC] — rows decompose as (iC, kH, kW)

  // matmul: weights2d * gradO2d = columns2d
  // [iC*kH*kW, oC] * [oC, bS*oH*oW] = [iC*kH*kW, bS*oH*oW]
  std::vector<sd::LongType> columns2dShape = {iC * kH * kW, bS * oH * oW};
  NDArray columns2d('c', columns2dShape, columns->dataType(), columns->getContext());
  MmulHelper::matmul(weightsContig, gradO2d, &columns2d, false, false, 1.0, 0.0);

  // Reshape columns2d to 6D and permute to col2im expected order [bS, iC, kH, kW, oH, oW]
  // columns2d rows are C-order (iC, kH, kW), columns are C-order (bS, oH, oW)
  std::vector<sd::LongType> eps6dShape = {iC, kH, kW, bS, oH, oW};
  columns2d.reshapei('c', eps6dShape);
  std::vector<sd::LongType> epsPermute = {3, 0, 1, 2, 4, 5};
  NDArray *permuted = columns2d.permute(epsPermute, false, false);

  // Perform col2im
  auto ctx = block.launchContext();
  if (!isNCHW) {
    // For NHWC: allocate a fresh contiguous NCHW buffer for col2im so that += accumulation
    // happens in contiguous memory (matching the NCHW accumulation order). Accumulating
    // directly into a non-contiguous permuted view causes floating-point precision differences
    // in FLOAT that do not appear in DOUBLE.
    std::vector<sd::LongType> gradINchwShape = {bS, iC, iH, iW};
    NDArray gradINchw('c', gradINchwShape, gradI->dataType(), gradI->getContext());
    helpers::col2im(*ctx, permuted, &gradINchw, sH, sW, pH, pW, iH, iW, dH, dW);
    // Permute contiguous NCHW result [bS, iC, iH, iW] -> [bS, iH, iW, iC] and assign to gradI
    std::vector<sd::LongType> perm = {0, 2, 3, 1};
    NDArray *gradINchwPermuted = gradINchw.permute(perm, false, false);
    gradI->assign(gradINchwPermuted);
    delete gradINchwPermuted;
  } else {
    helpers::col2im(*ctx, permuted, gradIPermuted, sH, sW, pH, pW, iH, iW, dH, dW);
  }

  delete permuted;
  delete weightsContig;
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
