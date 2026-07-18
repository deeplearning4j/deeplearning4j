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
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>


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

  // For SAME/CAUSAL padding modes, compute padding values.
  // Without this, im2col with pH=0/pW=0 produces right-biased padding which
  // gives different results from TF/Keras for odd kernel sizes (k=3, k=5, etc.).
  // NOTE: Do NOT use calcPadding2D here — its "symmetry adjustment" can produce
  // negative padding values for stride>1 cases. Compute directly using TF formula.
  if (paddingMode == 1 || paddingMode == 2) {
    const LongType eKH = (kH - 1) * dH + 1;
    const LongType eKW = (kW - 1) * dW + 1;
    LongType totalPadH = std::max((LongType)0, (oH - 1) * sH + eKH - iH);
    LongType totalPadW = std::max((LongType)0, (oW - 1) * sW + eKW - iW);
    if (paddingMode == 1) {
      // SAME: centered padding, extra goes to bottom/right (TF convention)
      pH = totalPadH / 2;
      pW = totalPadW / 2;
    } else {
      // CAUSAL: explicit left-only padding = (kernel-1) * dilation
      // TF pads input with this many zeros on the left, then does valid conv
      pH = (kH - 1) * dH;
      pW = (kW - 1) * dW;
    }
  }

  // Create col in C-order {bS, oH, oW, kH, kW, iC}.
  // im2col writes through colP strides, filling col so that
  // col[b, oh, ow, kh, kw, ic] = input patch value at that position.
  // C-order reshape of col to [bS*oH*oW, kH*kW*iC] then gives:
  //   row = spatial position (b, oh, ow)
  //   col = kernel+channel position (kH, kW, iC) in C-order
  std::vector<sd::LongType> colShape = {bS, oH, oW, kH, kW, iC};
  // Permute col [bS, oH, oW, kH, kW, iC] -> colP [bS, iC, kH, kW, oH, oW]
  // so im2col (which expects output [bS, iC, kH, kW, oH, oW]) writes correctly.
  // dim mapping: new0=old0(bS), new1=old5(iC), new2=old3(kH), new3=old4(kW), new4=old1(oH), new5=old2(oW)
  std::vector<sd::LongType> perm = {0, 5, 3, 4, 1, 2};
  NDArray *col = new NDArray('c', colShape, input->dataType(), input->getContext());

  // colP is a VIEW of col with shape [bS, iC, kH, kW, oH, oW] and permuted strides.
  // im2col indexes using colP's strides, which maps channel/kernel/spatial writes
  // into the correct positions in col's contiguous C-order buffer.
  NDArray *colP = col->permute(perm, false, false);

  // Push col (the owner) as intermediate result first - framework handles cleanup
  // colP (the view) will be pushed later and used by backward pass
  block.pushIntermediateResult(col);

  //----- calculation of output -----//
  auto ctx = block.launchContext();

  NDArray *inputNchw = nullptr;
  NDArray *zeroVal = NDArrayFactory::create(0.f, input->getContext());
  if (isNCHW) {
    helpers::im2col(*ctx, *input, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    *zeroVal);
  } else {
    std::vector<sd::LongType> permute = {0, 3, 1, 2};
    // For NHWC, we need to permute the input to NCHW before im2col
    inputNchw = input->permute(permute, false, false);
    helpers::im2col(*ctx, *inputNchw, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    *zeroVal);
  }

  MmulHelper::deleteTemporary(zeroVal);
  block.pushIntermediateResult(colP);

  // Reshape col to [bS*oH*oW, kH*kW*iC] in C-order.
  // col is contiguous C-order [bS, oH, oW, kH, kW, iC], so this is zero-copy.
  std::vector<sd::LongType> colShape2d = {bS * oH * oW, kH * kW * iC};
  NDArray *colReshaped = col->reshape('c', colShape2d, false);

  // Prepare weights as [kH*kW*iC, oC] with rows in C-order of (kH, kW, iC)
  // to match the im2col column ordering.
  NDArray *weightsKWIO = nullptr;
  std::vector<sd::LongType> wShape = {kH * kW * iC, oC};
  if (wFormat == 0) {
    // [kH, kW, iC, oC] - already target layout
    weightsKWIO = weights->dup('c');
  } else if (wFormat == 1) {
    // [oC, iC, kH, kW] -> [kH, kW, iC, oC]
    std::vector<sd::LongType> wPerm = {2, 3, 1, 0};
    NDArray *wp = weights->permute(wPerm, false, false);
    weightsKWIO = wp->dup('c');
    delete wp;
  } else {
    // [oC, kH, kW, iC] -> [kH, kW, iC, oC]
    std::vector<sd::LongType> wPerm = {1, 2, 3, 0};
    NDArray *wp = weights->permute(wPerm, false, false);
    weightsKWIO = wp->dup('c');
    delete wp;
  }
  // The packed weights feed an asynchronous GEMM. Keep the owning buffer under
  // the op context instead of retiring it while the device may still read it.
  block.pushIntermediateResult(weightsKWIO);
  NDArray *reshapedW = weightsKWIO->reshape('c', wShape, false);

  // Matmul: [bS*oH*oW, kH*kW*iC] x [kH*kW*iC, oC] -> [bS*oH*oW, oC]
  std::vector<sd::LongType> mmulResultShape = {bS * oH * oW, oC};
  NDArray *mmulResult = new NDArray('c', mmulResultShape, output->dataType(), output->getContext());
  MmulHelper::matmul(colReshaped, reshapedW, mmulResult, false, false, 1.0, 0.0);

  delete colReshaped;
  delete reshapedW;

  // Reshape mmulResult to [bS, oH, oW, oC] (NHWC layout) in C-order.
  std::vector<sd::LongType> nhwcShape = {bS, oH, oW, oC};
  NDArray *outputNHWC = mmulResult->reshape('c', nhwcShape, false);

  if (isNCHW) {
    // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    std::vector<sd::LongType> nhwcToNchw = {0, 3, 1, 2};
    NDArray *outputNCHW = outputNHWC->permute(nhwcToNchw, false, false);
    output->assign(outputNCHW);
    delete outputNCHW;
  } else {
    output->assign(outputNHWC);
  }
  delete outputNHWC;
  MmulHelper::deleteTemporary(mmulResult);

  // Clean up NHWC input permutation if created
  if (inputNchw != nullptr) {
    delete inputNchw;
  }

  //----- add biases if required -----//
  if (bias) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
  }
}

//////////////////////////////////////////////////////////////////////////
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
