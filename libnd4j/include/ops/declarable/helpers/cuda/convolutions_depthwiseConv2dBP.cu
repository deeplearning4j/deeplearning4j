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
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>


namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void depthwiseConv2dBP_(NDArray* input, NDArray* weights, NDArray* bias, NDArray* gradO,
                               NDArray* gradI, NDArray* gradW, NDArray* gradB, const LongType kH, const LongType kW, const LongType sH,
                               const LongType sW, LongType pH, LongType pW, const LongType dH, const LongType dW, const int paddingMode,
                               const int isNCHW, const int wFormat) {
  // input    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  // weights  [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  // bias     [oC] = [iC*mC]
  // gradO    [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  // gradW    [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  // gradB    [oC]

  LongType bS, iC, iH, iW, mC, oC, oH, oW;
  LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);

  // Permute to NCHW if needed
  NDArray *inputPermuted, *gradIPermuted;
  if (!isNCHW) {
    std::vector<sd::LongType> perm = {0, 3, 1, 2};
    inputPermuted = input->permute(perm, false, false);  // [bS,iH,iW,iC] -> [bS,iC,iH,iW]
    gradIPermuted = gradI->permute(perm, false, false);  // [bS,iH,iW,iC] -> [bS,iC,iH,iW]
  } else {
    inputPermuted = input;
    gradIPermuted = gradI;
  }

  if (paddingMode == 1)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  // im2col: [bS, iC, iH, iW] -> [bS, iC, kH, kW, oH, oW]
  std::vector<sd::LongType> colShape = {bS, iC, kH, kW, oH, oW};
  NDArray columns(inputPermuted->ordering(), colShape, inputPermuted->dataType(), inputPermuted->getContext());
  NDArray *zero = NDArrayFactory::create(0.f, inputPermuted->getContext());
  helpers::im2col(*inputPermuted->getContext(), *inputPermuted, columns, kH, kW, sH, sW, pH, pW, dH, dW, *zero);
  delete zero;

  // Prepare weights as 3D: [iC, kH*kW, mC]
  NDArray* weightsPermuted4d = nullptr;
  if (wFormat == 0) {
    std::vector<sd::LongType> perm = {2, 0, 1, 3};
    weightsPermuted4d = weights->permute(perm, false, false);
  } else if (wFormat == 1) {
    std::vector<sd::LongType> perm = {1, 2, 3, 0};
    weightsPermuted4d = weights->permute(perm, false, false);
  } else {
    std::vector<sd::LongType> perm = {3, 1, 2, 0};
    weightsPermuted4d = weights->permute(perm, false, false);
  }
  NDArray* weights3d = weightsPermuted4d->dup('c');
  delete weightsPermuted4d;
  std::vector<sd::LongType> w3dShape = {iC, kH * kW, mC};
  weights3d->reshapei('c', w3dShape);

  // Prepare columns as 3D for gradW: [iC, kH*kW, bS*oH*oW]
  std::vector<sd::LongType> colPerm = {1, 2, 3, 0, 4, 5};
  NDArray* colPermuted = columns.permute(colPerm, false, false);
  NDArray* col3d = colPermuted->dup('c');
  delete colPermuted;
  std::vector<sd::LongType> col3dShape = {iC, kH * kW, bS * oH * oW};
  col3d->reshapei('c', col3dShape);

  // Prepare gradO as 5D then 3D for gradW: [iC, bS*oH*oW, mC]
  NDArray* gradO5d = nullptr;
  if (isNCHW) {
    std::vector<sd::LongType> shape5d = {bS, iC, mC, oH, oW};
    gradO5d = gradO->reshape(gradO->ordering(), shape5d);
  } else {
    std::vector<sd::LongType> shape5d = {bS, oH, oW, iC, mC};
    gradO5d = gradO->reshape(gradO->ordering(), shape5d);
  }

  NDArray* gradO3d_forW = nullptr;
  {
    std::vector<sd::LongType> perm5d;
    if (isNCHW) {
      perm5d = {1, 0, 3, 4, 2};  // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC]
    } else {
      perm5d = {3, 0, 1, 2, 4};  // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC]
    }
    NDArray* gradOPerm5d = gradO5d->permute(perm5d, false, false);
    gradO3d_forW = gradOPerm5d->dup('c');
    delete gradOPerm5d;
    std::vector<sd::LongType> go3dShape = {iC, bS * oH * oW, mC};
    gradO3d_forW->reshapei('c', go3dShape);
  }

  // ----- calculation of gradW ----- //
  std::vector<sd::LongType> gw3dShape = {iC, kH * kW, mC};
  NDArray gradW3d('c', gw3dShape, gradW->dataType(), gradW->getContext());
  MmulHelper::mmul(col3d, gradO3d_forW, &gradW3d, 1.0, 0.0);

  std::vector<sd::LongType> gw4dShape = {iC, kH, kW, mC};
  NDArray* gradW4d = gradW3d.reshape('c', gw4dShape, false);
  if (wFormat == 0) {
    std::vector<sd::LongType> perm = {1, 2, 0, 3};
    NDArray* p = gradW4d->permute(perm, false, false);
    gradW->assign(p);
    delete p;
  } else if (wFormat == 1) {
    std::vector<sd::LongType> perm = {3, 0, 1, 2};
    NDArray* p = gradW4d->permute(perm, false, false);
    gradW->assign(p);
    delete p;
  } else {
    std::vector<sd::LongType> perm = {3, 1, 2, 0};
    NDArray* p = gradW4d->permute(perm, false, false);
    gradW->assign(p);
    delete p;
  }
  delete gradW4d;

  // ----- calculation of gradB ----- //
  if (gradB) {
    NDArray* gradBR = gradB;
    if (gradB->rankOf() == 2) {
      std::vector<sd::LongType> lenShape = {gradB->lengthOf()};
      gradBR = gradB->reshape(gradB->ordering(), lenShape, false);
    }
    std::vector<LongType> dims = {0, indOoH, indOoH + 1};
    gradO->reduceAlongDimension(reduce::Sum, gradBR, &dims, false);
    if (gradBR != gradB) delete gradBR;
  }

  // ----- calculation of gradI ----- //
  NDArray* gradO3d_forI = nullptr;
  {
    std::vector<sd::LongType> perm5d;
    if (isNCHW) {
      perm5d = {1, 2, 0, 3, 4};  // [bS,iC,mC,oH,oW] -> [iC,mC,bS,oH,oW]
    } else {
      perm5d = {3, 4, 0, 1, 2};  // [bS,oH,oW,iC,mC] -> [iC,mC,bS,oH,oW]
    }
    NDArray* gradOPerm5d = gradO5d->permute(perm5d, false, false);
    gradO3d_forI = gradOPerm5d->dup('c');
    delete gradOPerm5d;
    std::vector<sd::LongType> go3dShape = {iC, mC, bS * oH * oW};
    gradO3d_forI->reshapei('c', go3dShape);
  }

  std::vector<sd::LongType> colRes3dShape = {iC, kH * kW, bS * oH * oW};
  NDArray colResult3d('c', colRes3dShape, columns.dataType(), columns.getContext());
  MmulHelper::mmul(weights3d, gradO3d_forI, &colResult3d, 1.0, 0.0);

  std::vector<sd::LongType> col6dShape = {iC, kH, kW, bS, oH, oW};
  colResult3d.reshapei('c', col6dShape);
  std::vector<sd::LongType> colResultPerm = {3, 0, 1, 2, 4, 5};
  NDArray* colForCol2im = colResult3d.permute(colResultPerm, false, false);

  helpers::col2im(*inputPermuted->getContext(), colForCol2im, gradIPermuted, sH, sW, pH, pW, iH, iW, dH, dW);

  if (!isNCHW) {
    std::vector<sd::LongType> perm = {0, 2, 3, 1};
    NDArray* permutedGradI = gradIPermuted->permute(perm, false, false);
    gradI->assign(permutedGradI);
    delete permutedGradI;
  }

  // During CUDA graph capture, stream sync is illegal. Stream ordering guarantees correctness.
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    cudaStreamSynchronize(*inputPermuted->getContext()->getCudaStream());
  }

  // Cleanup
  delete colForCol2im;
  delete gradO3d_forI;
  delete gradO3d_forW;
  delete gradO5d;
  delete col3d;
  delete weights3d;
  if (!isNCHW) {
    delete inputPermuted;
    delete gradIPermuted;
  }
}

//////////////////////////////////////////////////////////////////////////
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
