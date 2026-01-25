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

  // Tripwire: validate Context at entry to conv2d_
  auto ctxEntry = block.launchContext();
  uintptr_t ctxAddrEntry = reinterpret_cast<uintptr_t>(ctxEntry);
  if (ctxAddrEntry < 0x10000 || (ctxAddrEntry & 0x7) != 0) {
    THROW_EXCEPTION("conv2d_: Context._context corrupted at ENTRY");
  }

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


  // Create col with memory layout optimized for later reshape: {bS, oH, oW, kH, kW, iC}
  std::vector<sd::LongType> colShape = {bS, oH, oW, kH, kW, iC};
  std::vector<LongType> colPermute = {0, 3, 4, 5, 1, 2};
  NDArray *col = new NDArray('c', colShape, input->dataType(), input->getContext());

  // Tripwire: after col creation
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after col creation");

  // colP is a VIEW of col - it shares col's DataBuffer with a different shape/strides.
  // We must NOT delete col while colP is in use, as that would free the underlying memory.
  // Both col and colP will be stored as intermediate results for the backward pass to manage.
  NDArray *colP = col->permute(colPermute, false, false);

  // Tripwire: after colP permute
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after colP permute");

  // Push col (the owner) as intermediate result first - backward pass will clean it up
  // colP (the view) will be pushed later and used by backward pass
  block.pushIntermediateResult(col);

  std::vector<sd::LongType> mmulResShape = {bS * oH * oW, oC};
  NDArray mmulResult('f', mmulResShape, output->dataType(), output->getContext());

  // Tripwire: after mmulResult creation
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after mmulResult creation");

  std::vector<LongType> permuteForOutput = {0, 3, 1, 2};

  //----- calculation of output -----//
  auto ctx = block.launchContext();


  NDArray* zero = NDArrayFactory::create(0.f, input->getContext());

  // Tripwire: after zero creation
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after zero creation");

  if (isNCHW) {
    helpers::im2col(*ctx, *input, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    *zero);
  } else {
    std::vector<sd::LongType> permute = {0, 3, 1, 2};
    // For NHWC, we need to permute the input to NCHW before im2col
    NDArray* inputNchw = input->permute(permute, false, false);
    helpers::im2col(*ctx, *inputNchw, *colP, kH, kW, sH, sW, pH, pW, dH, dW,
                    *zero);
    delete inputNchw;  // Clean up permuted copy
  }

  // Tripwire: after im2col
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after im2col");

  delete zero;

  // Tripwire: check before pushIntermediateResult
  auto ctxBeforePush = block.launchContext();
  uintptr_t addrBeforePush = reinterpret_cast<uintptr_t>(ctxBeforePush);
  if (addrBeforePush < 0x10000 || (addrBeforePush & 0x7) != 0) {
    THROW_EXCEPTION("conv2d_: Context._context corrupted BEFORE pushIntermediateResult");
  }

  // Push colP (the view) to intermediate results for use in backward pass
  // colP has shape {bS, iC, kH, kW, oH, oW} and is a view of col (pushed at index 0)
  // Index 0 = col (owner), Index 1 = colP (view)
  block.pushIntermediateResult(colP);

  // Tripwire: check after pushIntermediateResult
  auto ctxAfterPush = block.launchContext();
  if (ctxAfterPush != ctxBeforePush) {
    THROW_EXCEPTION("conv2d_: Context._context CHANGED during pushIntermediateResult");
  }

  std::vector<sd::LongType> shape = {bS * oH * oW, kH * kW * iC};
  NDArray* colReshaped = colP->reshape('c', shape, false);  // Use colP (permuted), view not copy

  // Tripwire: after colReshaped
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after colReshaped");

  NDArray* weightsPermuted = weights->permute(permuteForOutput, false, false);

  // Tripwire: after weightsPermuted
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after weightsPermuted");

  std::vector<LongType> weightShape = {iC * kH * kW, oC};
  NDArray* reshapedW = weightsPermuted->reshape('f', weightShape, false);

  // Tripwire: after reshapedW
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after reshapedW");

  MmulHelper::matmul(colReshaped, reshapedW, &mmulResult, false, false, 1.0, 0.0);

  // Tripwire: after matmul
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after matmul");


  std::vector<LongType> mmulResultShape = {oH, oW, bS, oC};
  NDArray* reshaped = mmulResult.reshape('f', mmulResultShape, false);

  // Tripwire: after reshaped
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after reshaped");

  std::vector<sd::LongType> permutedShape = {2, 3, 1,0};
  NDArray* permuted = reshaped->permute(permutedShape, false, false);

  // Tripwire: after permuted
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after permuted");

  // Reshape and copy result to output
  if (isNCHW) {
    output->assign(permuted);
  } else {
    std::vector<sd::LongType> otherPermute = {0,2,3,1};
    NDArray* permuted2 = permuted->permute(otherPermute, false, false);
    output->assign(permuted2);
    delete permuted2;
  }

  // Tripwire: after output assign
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after output assign");

  // Synchronize CUDA stream before cleanup to ensure all async operations complete
  cudaStreamSynchronize(*ctx->getCudaStream());

  // Tripwire: after cudaStreamSynchronize
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after cudaStreamSynchronize");

  delete permuted;
  // Tripwire: after delete permuted
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after delete permuted");

  delete reshaped;
  // Tripwire: after delete reshaped
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after delete reshaped");

  delete reshapedW;
  // Tripwire: after delete reshapedW
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after delete reshapedW");

  delete weightsPermuted;
  // Tripwire: after delete weightsPermuted
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after delete weightsPermuted");

  delete colReshaped;
  // Tripwire: after delete colReshaped
  if (block.launchContext() != ctxEntry) THROW_EXCEPTION("conv2d_: _context changed after delete colReshaped");

  // NOTE: colP is NOT deleted here - it's stored in intermediate results for backward pass
  // The backward pass is responsible for cleaning it up

  //----- add biases if required -----//
  if (bias) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
  }

  // Tripwire: validate Context at exit from conv2d_
  auto ctxExit = block.launchContext();
  if (ctxExit != ctxEntry) {
    THROW_EXCEPTION("conv2d_: Context._context CHANGED during conv2d_ execution");
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
