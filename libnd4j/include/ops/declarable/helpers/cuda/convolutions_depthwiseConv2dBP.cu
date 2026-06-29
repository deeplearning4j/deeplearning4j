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

  //  kH          filter(kernel) height
  //  kW          filter(kernel) width
  //  sH          strides height
  //  sW          strides width
  //  pH          paddings height
  //  pW          paddings width
  //  dH          dilations height
  //  dW          dilations width
  //  paddingMode  0-VALID, 1-SAME
  //  isNCHW      0-NHWC, 1-NCHW

  LongType bS, iC, iH, iW, mC, oC, oH, oW;  // batch size, input channels, input height/width, channels multiplier(oC =
                                       // iC*mC), output channels, output height/width
  LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);  // channels multiplier

  // For NHWC: keep the original gradI pointer for the final copy-back.
  NDArray* gradIOriginal = gradI;
  NDArray* gradINchw = nullptr;  // fresh contiguous NCHW buffer for col2im (NHWC only)

  if (!isNCHW) {
    std::vector<sd::LongType> permuteVec = {0, 3, 1, 2};
    input = input->permute(permuteVec, false, false);
    // Allocate a fresh contiguous NCHW buffer for col2im.
    std::vector<sd::LongType> gradINchwShape = {bS, iC, iH, iW};
    gradINchw = new NDArray('c', gradINchwShape, gradI->dataType(), gradI->getContext());
    gradI = gradINchw;
  }

  if (paddingMode == 1)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  // im2col: [bS, iC, iH, iW] -> [bS, iC, kH, kW, oH, oW]
  std::vector<sd::LongType> colShape = {bS, iC, kH, kW, oH, oW};
  NDArray columns('c', colShape, input->dataType(), input->getContext());
  NDArray* zero = NDArrayFactory::create(0.f, input->getContext());
  helpers::im2col(*input->getContext(), *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, *zero);
  delete zero;

  // ---- weights3d: [iC, kH*kW, mC] contiguous ----
  // weights permute depends on wFormat:
  //   wFormat=0: [kH,kW,iC,mC] permute {2,0,1,3} -> [iC,kH,kW,mC] reshape -> [iC,kH*kW,mC]
  //   wFormat=1: [mC,iC,kH,kW] permute {1,2,3,0} -> [iC,kH,kW,mC] reshape -> [iC,kH*kW,mC]
  //   wFormat=2: [mC,kH,kW,iC] permute {3,1,2,0} -> [iC,kH,kW,mC] reshape -> [iC,kH*kW,mC]
  // permute gives a non-contiguous view; reshape() materializes a contiguous copy when needed.
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
  // reshape() materializes a contiguous copy iff the permuted source is non-contiguous —
  // the correct primitive (NDArray::reshapei now also handles non-contiguity internally).
  std::vector<sd::LongType> w3dShape = {iC, kH * kW, mC};
  NDArray* weights4dc = weightsPermuted4d->reshape('c', w3dShape, false);
  delete weightsPermuted4d;

  // ---- col3d: [iC, kH*kW, bS*oH*oW] contiguous ----
  // columns [bS,iC,kH,kW,oH,oW] permute {1,2,3,0,4,5} -> [iC,kH,kW,bS,oH,oW]
  // dup then reshapei to [iC,kH*kW,bS*oH*oW]
  std::vector<sd::LongType> colPerm = {1, 2, 3, 0, 4, 5};
  NDArray* colPermuted = columns.permute(colPerm, false, false);
  std::vector<sd::LongType> col3dShape = {iC, kH * kW, bS * oH * oW};
  NDArray* col3dc = colPermuted->reshape('c', col3dShape, false);
  delete colPermuted;

  // ---- gradO3d_forW: [iC, bS*oH*oW, mC] contiguous ----
  // NCHW: gradO [bS,oC,oH,oW] reshape5d [bS,iC,mC,oH,oW] permute {1,0,3,4,2} -> [iC,bS,oH,oW,mC]
  // NHWC: gradO [bS,oH,oW,oC] reshape5d [bS,oH,oW,iC,mC] permute {3,0,1,2,4} -> [iC,bS,oH,oW,mC]
  // dup then reshapei to [iC, bS*oH*oW, mC]
  NDArray* gradO5d = nullptr;
  NDArray* gradO3d_forW = nullptr;
  {
    std::vector<sd::LongType> go3dShape = {iC, bS * oH * oW, mC};
    if (isNCHW) {
      std::vector<sd::LongType> shape5d = {bS, iC, mC, oH, oW};
      gradO5d = gradO->reshape(gradO->ordering(), shape5d);
      std::vector<sd::LongType> perm5d = {1, 0, 3, 4, 2};  // [bS,iC,mC,oH,oW] -> [iC,bS,oH,oW,mC]
      NDArray* gradOPerm5d = gradO5d->permute(perm5d, false, false);
      gradO3d_forW = gradOPerm5d->reshape('c', go3dShape, false);
      delete gradOPerm5d;
    } else {
      std::vector<sd::LongType> shape5d = {bS, oH, oW, iC, mC};
      gradO5d = gradO->reshape(gradO->ordering(), shape5d);
      std::vector<sd::LongType> perm5d = {3, 0, 1, 2, 4};  // [bS,oH,oW,iC,mC] -> [iC,bS,oH,oW,mC]
      NDArray* gradOPerm5d = gradO5d->permute(perm5d, false, false);
      gradO3d_forW = gradOPerm5d->reshape('c', go3dShape, false);
      delete gradOPerm5d;
    }
  }

  // ----- calculation of gradW ----- //
  // GEMM: col3dc [iC,kH*kW,bS*oH*oW] x gradO3d_forW [iC,bS*oH*oW,mC] → [iC,kH*kW,mC]
  // Note: MmulHelper::mmul uses cuBLAS strided batched if 3D arrays with matching batch dim.
  std::vector<sd::LongType> gw3dShape = {iC, kH * kW, mC};
  NDArray gradW3d('c', gw3dShape, gradW->dataType(), gradW->getContext());
  MmulHelper::mmul(col3dc, gradO3d_forW, &gradW3d, 1.0, 0.0);

  // Copy gradW3d [iC,kH*kW,mC] -> gradW via reshaping to [iC,kH,kW,mC] then permuting to target layout
  {
    std::vector<sd::LongType> gw4dShape = {iC, kH, kW, mC};
    NDArray* gradW4d = gradW3d.reshape('c', gw4dShape, false);  // safe: gradW3d is contiguous
    std::vector<sd::LongType> invPerm;
    if (wFormat == 0)
      invPerm = {1, 2, 0, 3};  // [iC,kH,kW,mC] -> [kH,kW,iC,mC]
    else if (wFormat == 1)
      invPerm = {3, 0, 1, 2};  // [iC,kH,kW,mC] -> [mC,iC,kH,kW]
    else
      invPerm = {3, 1, 2, 0};  // [iC,kH,kW,mC] -> [mC,kH,kW,iC]
    NDArray* gradW4dPerm = gradW4d->permute(invPerm, false, false);
    gradW->assign(gradW4dPerm);
    delete gradW4dPerm;
    if (gradW4d != &gradW3d) delete gradW4d;
  }

  delete col3dc;
  delete gradO3d_forW;
  delete gradO5d;

  // ----- calculation of gradB ----- //
  if (gradB) {
    NDArray* gradBR = gradB;
    if (gradB->rankOf() == 2) {
      std::vector<sd::LongType> lenShape = {gradB->lengthOf()};
      gradBR = gradB->reshape(gradB->ordering(), lenShape, false);
    }
    std::vector<LongType> dims = {0, indOoH, indOoH + 1};
    gradO->reduceAlongDimension(reduce::Sum, gradBR, &dims, false);  // sum over bS, oH, oW
    if (gradBR != gradB) delete gradBR;
  }

  //----- calculation of gradI -----//
  // gradO3d_forI: [iC, mC, bS*oH*oW] contiguous
  // NCHW: gradO [bS,oC,oH,oW] reshape5d [bS,iC,mC,oH,oW] permute {1,2,0,3,4} -> [iC,mC,bS,oH,oW]
  // NHWC: gradO [bS,oH,oW,oC] reshape5d [bS,oH,oW,iC,mC] permute {3,4,0,1,2} -> [iC,mC,bS,oH,oW]
  // dup then reshapei to [iC, mC, bS*oH*oW]
  NDArray* gradO5d_forI = nullptr;
  NDArray* gradO3d_forI = nullptr;
  {
    std::vector<sd::LongType> go3dShape = {iC, mC, bS * oH * oW};
    if (isNCHW) {
      std::vector<sd::LongType> shape5d = {bS, iC, mC, oH, oW};
      gradO5d_forI = gradO->reshape(gradO->ordering(), shape5d);
      std::vector<sd::LongType> perm5d = {1, 2, 0, 3, 4};  // [bS,iC,mC,oH,oW] -> [iC,mC,bS,oH,oW]
      NDArray* gradOPerm5d = gradO5d_forI->permute(perm5d, false, false);
      gradO3d_forI = gradOPerm5d->reshape('c', go3dShape, false);
      delete gradOPerm5d;
    } else {
      std::vector<sd::LongType> shape5d = {bS, oH, oW, iC, mC};
      gradO5d_forI = gradO->reshape(gradO->ordering(), shape5d);
      std::vector<sd::LongType> perm5d = {3, 4, 0, 1, 2};  // [bS,oH,oW,iC,mC] -> [iC,mC,bS,oH,oW]
      NDArray* gradOPerm5d = gradO5d_forI->permute(perm5d, false, false);
      gradO3d_forI = gradOPerm5d->reshape('c', go3dShape, false);
      delete gradOPerm5d;
    }
  }

  // GEMM: weights4dc [iC,kH*kW,mC] x gradO3d_forI [iC,mC,bS*oH*oW] → colResult3d [iC,kH*kW,bS*oH*oW]
  std::vector<sd::LongType> colRes3dShape = {iC, kH * kW, bS * oH * oW};
  NDArray colResult3d('c', colRes3dShape, columns.dataType(), columns.getContext());
  MmulHelper::mmul(weights4dc, gradO3d_forI, &colResult3d, 1.0, 0.0);

  delete weights4dc;
  delete gradO3d_forI;
  delete gradO5d_forI;

  // Copy colResult3d [iC,kH*kW,bS*oH*oW] -> columns [bS,iC,kH,kW,oH,oW]
  // Reshape to [iC,kH,kW,bS,oH,oW], permute {3,0,1,2,4,5} -> [bS,iC,kH,kW,oH,oW], assign to columns
  {
    std::vector<sd::LongType> col6dShape = {iC, kH, kW, bS, oH, oW};
    NDArray* colResult6d = colResult3d.reshape('c', col6dShape, false);  // safe: colResult3d is contiguous
    std::vector<sd::LongType> colResultPerm = {3, 0, 1, 2, 4, 5};
    NDArray* colForCol2im = colResult6d->permute(colResultPerm, false, false);
    columns.assign(colForCol2im);
    delete colForCol2im;
    if (colResult6d != &colResult3d) delete colResult6d;
  }

  helpers::col2im(*input->getContext(), &columns, gradI, sH, sW, pH, pW, iH, iW, dH,
                  dW);  // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

  // During CUDA graph capture, stream sync is illegal. Stream ordering guarantees correctness.
  // Use the context from the (possibly permuted) input array before any cleanup.
  auto* ctx = input->getContext();

  if (!isNCHW) {
    // gradI now points to gradINchw (fresh contiguous NCHW buffer).
    // Permute [bS, iC, iH, iW] -> [bS, iH, iW, iC] and assign to original NHWC gradI.
    std::vector<sd::LongType> perm = {0, 2, 3, 1};
    NDArray* gradINchwPermuted = gradINchw->permute(perm, false, false);
    gradIOriginal->assign(gradINchwPermuted);
    delete gradINchwPermuted;
    delete gradINchw;
    delete input;
  }

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    cudaStreamSynchronize(*ctx->getCudaStream());
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
