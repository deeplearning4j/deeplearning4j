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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void depthwiseConv2dCUDNN(const LaunchContext* context, const NDArray* input, const NDArray* weights,
                                 const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH,
                                 const int sW, const int pH, const int pW, const int dH, const int dW,
                                 const int paddingMode, const bool isNCHW) {
  // cudnn supports only following case: mC = 1, oC = iC (groupCount == iC)

  // input [bS, iC, iH, iW] nchw or [bS, iH, iW, iC] nhwc
  // weights [iC, mC, kH, kW]
  // bias [oC], may be nullptr
  // output [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc
  // oC = iC*mC

  int bS, iC, iH, iW, mC, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(1);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  PointersManager manager(context, __func__);
  // input descriptor
  CudnnTensor x;
  if (input->ews() == 1 && input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // weights descriptor
  FilterDesc w;
  w.set4D(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, iC, mC, kH, kW);

  // output descriptor
  CudnnTensor z;
  if (output->ews() == 1 && output->ordering() == 'c')
    z.set4D(format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
  else
    z.set4DEx(cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC),
              output->strideAt(indOoH), output->strideAt(indOoH + 1));

  // description of convolution
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSetConvolutionGroupCount),
      cudnnSetConvolutionGroupCount(
          conv, iC));  // set number of groups (depthwise mode) in description of convolution, groupCount == iC

  // algorithm description
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;
  // CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardAlgorithm), cudnnGetConvolutionForwardAlgorithm(
  // *handle, x, w, conv, z, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                          cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  if (count == 0)
    throw sd::cuda_exception::build("depthwiseConv2dCUDNN: cudnnGetConvolutionForwardAlgorithm failed", 0);
  algo = algoPerf.algo;

  // allocate auxiliary device memory, abbreviation ws means workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardWorkspaceSize),
                          cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, weights, bias});

  // run calculation
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionForward),
      cudnnConvolutionForward(*handle, alpha, x, input->specialBuffer(), w, weights->specialBuffer(), conv, algo,
                              wsData, wsSize, beta, z, output->specialBuffer()));

  // add bias if it is present
  if (bias != nullptr) {
    CudnnTensor b;
    // b.set( format, cudnnDataType(bias->dataType()), 1, isNCHW ? bias->lengthOf() : 1, 1, isNCHW ? 1:
    // bias->lengthOf());
    b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);

    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor), cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha,
                                                                      z, output->specialBuffer()));
  }

  // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  // if (cudaErr != 0)
  //     throw cuda_exception::build("depthwiseConv2dCUDNN: cudaStreamSynchronize failed !", cudaErr);
}

//////////////////////////////////////////////////////////////////////////
static void depthwiseConv2dBpCUDNN(const LaunchContext* context, const NDArray* input, const NDArray* weights,
                                   const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH,
                                   const int kW, const int sH, const int sW, const int pH, const int pW, const int dH,
                                   const int dW, const int paddingMode, const bool isNCHW) {
  // cudnn supports only following case: mC = 1, oC = iC (groupCount == iC)

  // input, gradI [bS, iC, iH, iW] nchw or [bS, iH, iW, iC] nhwc
  // weights, gradW [iC, mC, kH, kW]
  // gradB [oC], may be nullptr
  // gradO [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc
  // oC = iC*mC

  int bS, iC, iH, iW, mC, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(1);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  PointersManager manager(context, __func__);
  // input descriptor
  CudnnTensor x;
  if (input->ews() == 1 && input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // gradO descriptor
  CudnnTensor dz;
  if (gradO->ews() == 1 && gradO->ordering() == 'c')
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
  else
    dz.set4DEx(cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC),
               gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));

  // gradI descriptor
  CudnnTensor dx;
  if (gradI->ews() == 1 && gradI->ordering() == 'c')
    dx.set4D(format, cudnnDataType(gradI->dataType()), bS, iC, iH, iW);
  else
    dx.set4DEx(cudnnDataType(gradI->dataType()), bS, iC, iH, iW, gradI->strideAt(0), gradI->strideAt(indIOioC),
               gradI->strideAt(indIiH), gradI->strideAt(indIiH + 1));

  // gradW descriptor
  FilterDesc dw;
  dw.set4D(cudnnDataType(gradW->dataType()), CUDNN_TENSOR_NCHW, iC, mC, kH, kW);

  // description of convolution
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSetConvolutionGroupCount),
      cudnnSetConvolutionGroupCount(
          conv, iC));  // set number of groups (depthwise mode) in description of convolution, groupCount == iC

  // gradW algorithm description
  cudnnConvolutionBwdFilterAlgo_t algoGradW;
  cudnnConvolutionBwdFilterAlgoPerf_t algoGradWPerf;
  int count = 0;
  // CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionBackwardFilterAlgorithm),
  // cudnnGetConvolutionBackwardFilterAlgorithm( *handle, x, dz, conv, dw, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
  // 0, &algoGradW));
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
      cudnnFindConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf));
  if (count == 0)
    throw sd::cuda_exception::build(
        "depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed as the count is 0 ", 0);
  algoGradW = algoGradWPerf.algo;

  // gradI algorithm description
  cudnnConvolutionBwdDataAlgo_t algoGradI;
  cudnnConvolutionBwdDataAlgoPerf_t algoGradIPerf;
  // CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionBackwardDataAlgorithm),
  // cudnnGetConvolutionBackwardDataAlgorithm( *handle, dw, dz, conv, x, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
  // &algoGradI));
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardDataAlgorithm),
      cudnnFindConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, 1, &count, &algoGradIPerf));
  if (count == 0)
    throw sd::cuda_exception::build(
        "depthwiseConv2dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed as the count is 0 ", 0);
  algoGradI = algoGradIPerf.algo;

  // allocate auxiliary device memory for gradW calculation, abbreviation ws means workspace
  size_t wsGradWSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardFilterWorkspaceSize),
      cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, x, dz, conv, dw, algoGradW, &wsGradWSize));
  void* wsGradWData = manager.allocateDevMem(wsGradWSize);

  // allocate auxiliary device memory for gradI calculation, abbreviation ws means workspace
  size_t wsGradISize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardDataWorkspaceSize),
      cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, dw, dz, conv, dx, algoGradI, &wsGradISize));
  void* wsGradIData = manager.allocateDevMem(wsGradISize);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

  // run calculation for gradB (if not nullptr)
  if (gradB != nullptr) {
    CudnnTensor db;
    // db.set( format, cudnnDataType(gradB->dataType()), 1, isNCHW ? gradB->lengthOf() : 1, 1, isNCHW ? 1:
    // gradB->lengthOf());
    db.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(gradB->dataType()), 1, oC, 1, 1);

    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnConvolutionBackwardBias),
        cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->specialBuffer(), beta, db, gradB->specialBuffer()));
  }

  // run calculation for gradW
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardFilter),
      cudnnConvolutionBackwardFilter(*handle, alpha, x, input->specialBuffer(), dz, gradO->specialBuffer(), conv,
                                     algoGradW, wsGradWData, wsGradWSize, beta, dw, gradW->specialBuffer()));

  // run calculation for gradI
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardData),
      cudnnConvolutionBackwardData(*handle, alpha, dw, weights->specialBuffer(), dz, gradO->specialBuffer(), conv,
                                   algoGradI, wsGradIData, wsGradISize, beta, dx, gradI->specialBuffer()));

  // cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  // if (cudaErr != 0)
  //     throw cuda_exception::build("depthwiseConv2dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(depthwise_conv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC] = iC*mC

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "DEPTHWISECONV2D CUDNN OP: rank of input array must be equal to 4, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "DEPTHWISECONV2D CUDNN OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));  // filter(kernel) width
  int sH = INT_ARG(2);                                                          // strides height
  int sW = INT_ARG(3);                                                          // strides width
  int pH = INT_ARG(4);                                                          // paddings height
  int pW = INT_ARG(5);                                                          // paddings width
  int dH = INT_ARG(6);                                                          // dilations height
  int dW = INT_ARG(7);                                                          // dilations width
  int paddingMode = INT_ARG(8);                                                 // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

  int bS, iC, iH, iW, mC, oC, oH, oW;  // batch size, input channels, input height/width, channels multiplier(oC =
                                       // iC*mC), output channels, output height/width
  int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);  // channels multiplier

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, mC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "DEPTHWISECONV2D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  REQUIRE_TRUE(
      output->sizeAt(indIOioC) == iC * mC, 0,
      "DEPTHWISECONV2D CUDNN OP: the output_channels must be equal to input_channels * channels_multiplier = %i !",
      iC * mC);
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "DEPTHWISECONV2D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got "
                 "%i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  std::vector<int> wPermut;  // cudnn support format {oC, iC/groupCount, kH, kW} only, mC = 1, oC = iC (groupCount ==
                             // iC) that is {iC, mC, kH, kW} in our case
  if (0 == wFormat)
    wPermut = {2, 3, 0, 1};  // kH, kW, iC, mC -> iC, mC, kH, kW
  else if (1 == wFormat)
    wPermut = {1, 0, 2, 3};  // mC, iC, kH, kW -> iC, mC, kH, kW
  else
    wPermut = {3, 0, 1, 2};  // mC, kH, kW, iC -> iC, mC, kH, kW

  std::unique_ptr<NDArray> uNewWeights(
      new NDArray(weights->ordering(), {iC, mC, kH, kW}, weights->dataType(), weights->getContext()));
  uNewWeights->assign(weights->permute(wPermut));
  std::unique_ptr<NDArray> tmpInput = {};

  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv2dCUDNNPadAsymmetric(input, nullptr, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);
    tmpInput = std::move(std::get<0>(ret));
    if (tmpInput) input = tmpInput.get();
  }
  depthwiseConv2dCUDNN(block.launchContext(), input, uNewWeights.get(), bias, output, kH, kW, sH, sW, pH, pW, dH, dW,
                       paddingMode, isNCHW);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(depthwise_conv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC] = iC*mC

  const int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME, 2-CAUSAL
  const int wFormat = block.getIArguments()->size() > 10
                          ? INT_ARG(10)
                          : 0;  // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

  Requirements req("CUDNN DEPTHWISE_CONV2d OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
      req.expectEq(makeInfoVariable(weights->sizeAt(0 == wFormat ? 3 : 0), "weights#mC"), 1) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE}) &&
      req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                   {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE});
  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"),
                 {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE});
  }
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(depthwise_conv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC] = [iC*mC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "DEPTHWISECONV2D_BP CUDNN OP: rank of input array must be equal to 4, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "DEPTHWISECONV2D_BP CUDNN OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 4, 0,
               "DEPTHWISECONV2D_BP CUDNN OP: rank of output gradients (next epsilon) array must be equal to 4, but got "
               "%i instead !",
               gradO->rankOf());

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));  // filter(kernel) width
  int sH = INT_ARG(2);                                                          // strides height
  int sW = INT_ARG(3);                                                          // strides width
  int pH = INT_ARG(4);                                                          // paddings height
  int pW = INT_ARG(5);                                                          // paddings width
  int dH = INT_ARG(6);                                                          // dilations height
  int dW = INT_ARG(7);                                                          // dilations width
  int paddingMode = INT_ARG(8);                                                 // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

  int bS, iC, iH, iW, mC, oC, oH, oW;  // batch size, input channels, input height/width, channels multiplier(oC =
                                       // iC*mC), output channels, output height/width
  int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weights->sizeAt(indWmC);  // channels multiplier

  int trueoH, trueoW;  // correct output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, mC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "DEPTHWISECONV2D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but "
                 "got %i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  std::vector<int> wPermut, gradWPermut;  // cudnn support format {oC, iC/groupCount, kH, kW} only, mC = 1, oC = iC
                                          // (groupCount == iC) that is {iC, mC, kH, kW}
  if (0 == wFormat) {
    wPermut = {2, 3, 0, 1};      // kH, kW, iC, mC -> iC, mC, kH, kW
    gradWPermut = {2, 3, 0, 1};  // iC, mC, kH, kW -> kH, kW, iC, mC
  } else if (1 == wFormat) {
    wPermut = {1, 0, 2, 3};      // mC, iC, kH, kW -> iC, mC, kH, kW
    gradWPermut = {1, 0, 2, 3};  // iC, mC, kH, kW -> mC, iC, kH, kW
  } else {
    wPermut = {3, 0, 1, 2};      // mC, kH, kW, iC -> iC, mC, kH, kW
    gradWPermut = {1, 2, 3, 0};  // iC, mC, kH, kW -> mC, kH, kW, iC
  }

  std::unique_ptr<NDArray> tmpGradI = {}, tmpInput = {};
  std::unique_ptr<NDArray> uNewGradW(
      new NDArray(gradW->ordering(), {iC, mC, kH, kW}, gradW->dataType(), gradW->getContext()));
  std::unique_ptr<NDArray> uNewWeights(
      new NDArray(weights->ordering(), {iC, mC, kH, kW}, weights->dataType(), weights->getContext()));

  uNewWeights->assign(weights->permute(wPermut));

  NDArray* newInput = input;
  NDArray* newGradI = gradI;
  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv2dCUDNNPadAsymmetric(input, gradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);
    tmpInput = std::move(std::get<0>(ret));
    tmpGradI = std::move(std::get<1>(ret));
    if (tmpInput) newInput = tmpInput.get();
    if (tmpGradI) newGradI = tmpGradI.get();
  }
  depthwiseConv2dBpCUDNN(block.launchContext(), newInput, uNewWeights.get(), gradO, newGradI, uNewGradW.get(), gradB,
                         kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW);

  uNewGradW->permutei(gradWPermut);
  gradW->assign(uNewGradW.get());

  if (newInput != input) {
    if (isNCHW)
      gradI->assign((*newGradI)({0, 0, 0, 0, 0, gradI->sizeAt(2), 0, gradI->sizeAt(3)}));
    else
      gradI->assign((*newGradI)({0, 0, 0, gradI->sizeAt(1), 0, gradI->sizeAt(2), 0, 0}));
  }

  return sd::Status::OK;
}

PLATFORM_CHECK(depthwise_conv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC] = [iC*mC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

  const int paddingMode = INT_ARG(8);                                      // 0-VALID, 1-SAME, 2-CAUSAL
  const int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  const int wFormat = block.getIArguments()->size() > 10
                          ? INT_ARG(10)
                          : 0;  // 0 - [kH, kW, iC, mC], 1 - [mC, iC, kH, kW], 2 - [mC, kH, kW, iC]

  Requirements req("CUDNN DEPTHWISE_CONV2d_BP OP");
  const auto inType = input->dataType();
  const auto wType = weights->dataType();
  const auto gType = gradO->dataType();
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
      req.expectTrue(makeInfoVariable(isNCHW, "isNCHW")) &&
      req.expectEq(makeInfoVariable(weights->sizeAt(0 == wFormat ? 3 : 0), "weights#mC"), 1) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE}) &&
      req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                   {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE});
  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"),
                 {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE}) &&
        req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT3),
                     {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE});
  } else {
    req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2),
                 {DataType::HALF, DataType::FLOAT32, DataType::DOUBLE});
  }
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
