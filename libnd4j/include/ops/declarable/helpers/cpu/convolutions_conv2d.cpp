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
#if NOT_EXCLUDED(OP_col2im) && NOT_EXCLUDED(OP_im2col)

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void conv2d_(sd::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias,
                    NDArray* output, const LongType kH, const LongType kW, const LongType sH, const LongType sW, LongType pH, LongType pW,
                    const LongType dH, const LongType dW, const int paddingMode, const int isNCHW, const int wFormat) {
  // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  // weights [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  // bias    [oC]
  // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  // kH  filter(kernel) height
  // kW  filter(kernel) width
  // sH  strides height
  // sW  strides width
  // pH  paddings height
  // pW  paddings width
  // dH  dilations height
  // dW  dilations width
  // paddingMode 0-VALID, 1-SAME
  // isNCHW      1-NCHW,  0-NHWC

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  sd_debug("ONEDNN is not used for conv2d!\n", 0);

  std::vector<sd::LongType> permutForOutput;

  if (isNCHW)
    permutForOutput = {0, 3, 1, 2};  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
  else
    input = new NDArray(input->permute({0, 3, 1, 2}));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC

  std::vector<sd::LongType> wAxes;
  if (0 == wFormat)
    wAxes = {0, 1, 2};
  else if (1 == wFormat)
    wAxes = {2, 3, 1};
  else
    wAxes = {1, 2, 3};

  NDArray col('c', {bS, oH, oW, kH, kW, iC}, input->dataType(), input->getContext());
  NDArray colP = col.permute({0, 5, 3, 4, 1, 2});  // {bS, iC, kH, kW, oH, oW}
  colP.printIndexedBuffer("colP initial:");
  printf("colP initial end\n");

  NDArray mmulResult('f', {bS * oH * oW, oC}, output->dataType(), output->getContext());

  //----- calculation of output -----//
  auto ctx = block.launchContext();
  helpers::im2col(
      *ctx, *input, colP, kH, kW, sH, sW, pH, pW, dH, dW,
      NDArrayFactory::create(0.f, input->getContext()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
  MmulHelper::tensorDot(&col, weights, &mmulResult, {3, 4, 5}, wAxes,
                        {});  // [bS, oH, oW, kH, kW, iC] x [kH, kW, iC, oC] = [bS, oH, oW, oC]




  //----- assign outTemp to output  -----//
  if (isNCHW) {
    mmulResult.reshapei({bS, oH, oW, oC});
    mmulResult.permutei(permutForOutput);
  }

  output->assign(mmulResult);


  //----- add biases if required -----//
  if (bias) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
    output->printIndexedBuffer("output post bias");
    printf("output post bias end\n");


  }
  if (!isNCHW) delete input;
}

void ConvolutionUtils::conv2d(sd::graph::Context& block, const NDArray* input, const NDArray* weights,
                              const NDArray* bias, NDArray* output, const LongType kH, const LongType kW, const LongType sH,
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