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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void conv2dCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights, NDArray* bias,
                        NDArray* output, const int kH, const LongType kW, const LongType sH, const LongType sW, const LongType pH,
                        const LongType pW, const LongType dH, const LongType dW, const int paddingMode, const bool isNCHW,
                        const int wFormat) {
  // cudnn support only two formats for weights {oC,iC,kH,kW} and {oC,kH,kW,iC}

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

  // input descriptor
  CudnnTensor x;

  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // weights descriptor
  FilterDesc w;
  w.set4D(cudnnDataType(weights->dataType()), formatW, oC, iC, kH, kW);

  // output descriptor
  CudnnTensor z;

  if (output->ordering() == 'c')
    z.set4D(format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
  else
    z.set4DEx(cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC),
              output->strideAt(indOoH), output->strideAt(indOoH + 1));

  // description of convolution
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));

  // algorithm description
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;
  // err = cudnnGetConvolutionForwardAlgorithm(*handle, x, w, conv, z, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                          cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  if (count == 0)
    throw cuda_exception::build("conv2dCUDNN: cudnnGetConvolutionForwardAlgorithm failed as the count is 0", 0);
  algo = algoPerf.algo;

  PointersManager manager(context, __func__);
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


    b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor), cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha,
                                                                      z, output->specialBuffer()));
  }


  NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void conv2dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights,
                          NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const LongType kH,
                          const LongType kW, const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH,
                          const LongType dW, const LongType paddingMode, const bool isNCHW, const int wFormat) {
  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);
  PointersManager manager(context, __func__);
  // input descriptor, gradO descriptor, gradI descriptor
  CudnnTensor x, dz, dx;

  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  if (gradO->ordering() == 'c')
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
  else
    dz.set4DEx(cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC),
               gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));

  if (gradI->ordering() == 'c')
    dx.set4D(format, cudnnDataType(gradI->dataType()), bS, iC, iH, iW);
  else
    dx.set4DEx(cudnnDataType(gradI->dataType()), bS, iC, iH, iW, gradI->strideAt(0), gradI->strideAt(indIOioC),
               gradI->strideAt(indIiH), gradI->strideAt(indIiH + 1));

  // gradW descriptor
  FilterDesc dw;
  dw.set4D(cudnnDataType(gradW->dataType()), formatW, oC, iC, kH, kW);

  // description of convolution
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));

  // gradW algorithm description
  cudnnConvolutionBwdFilterAlgo_t algoGradW;
  cudnnConvolutionBwdFilterAlgoPerf_t algoGradWPerf;
  int count = 0;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
      cudnnFindConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf));
  if (count == 0)
    throw cuda_exception::build(
        "conv2dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed as the count is 0", 0);
  algoGradW = algoGradWPerf.algo;

  // gradI algorithm description
  cudnnConvolutionBwdDataAlgo_t algoGradI;
  cudnnConvolutionBwdDataAlgoPerf_t algoGradIPerf;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardDataAlgorithm),
      cudnnFindConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, 1, &count, &algoGradIPerf));
  if (count == 0)
    throw cuda_exception::build("conv2dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed as the count is 0",
                                    0);
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


  NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  LongType sH = INT_ARG(2);                                                // strides height
  LongType sW = INT_ARG(3);                                                // strides width
  LongType pH = INT_ARG(4);                                                // paddings height
  LongType pW = INT_ARG(5);                                                // paddings width
  LongType dH = INT_ARG(6);                                                // dilations height
  LongType dW = INT_ARG(7);                                                // dilations width
  int paddingMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) width

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "CUSTOM CONV2D CUDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "CUSTOM CONV2D CUDNN OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV2D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias) {
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM CONV2D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got "
                 "%i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());
    REQUIRE_TRUE((bias->rankOf() == 1 && bias->strideAt(0) == 1) ||
                     (bias->rankOf() == 2 && bias->sizeAt(0) == 1 && bias->strideAt(1) == 1) ||
                     (bias->rankOf() == 2 && bias->sizeAt(1) == 1 && bias->strideAt(0) == 1),
                 0, "CUSTOM CONV2D CUDNN OP: bias array should be contiguous in memory !");
  }
  std::unique_ptr<NDArray> tmpWeight = {}, tmpInput = {};
  NDArray* newWeights = weights;  // cudnn support only two formats {oC,iC,kH,kW} and {oC,kH,kW,iC}
  if (0 == wFormat) {
    // Create named vectors as lvalues
    std::vector<LongType> nchwShape = {oC, iC, kH, kW};
    std::vector<LongType> nhwcShape = {oC, kH, kW, iC};

    // Use the appropriate one for the weight reset
    tmpWeight.reset(
        new NDArray(weights->ordering(),
                    isNCHW ? nchwShape : nhwcShape,
                    weights->dataType(), weights->getContext()));
    newWeights = tmpWeight.get();
    // Create named vectors as lvalues
    std::vector<LongType> nchwDims = {3, 2, 0, 1};
    std::vector<LongType> nhwcDims = {3, 0, 1, 2};

    // Use the appropriate one in the call
    NDArray assign = weights->permute(
        isNCHW ? nchwDims : nhwcDims,
        true,   // copyToNewBuff
        true);  // resetStrides
    newWeights->assign(&assign);  // (kH, kW, iC, oC  --> oC, iC, kH, kW) or (kH, kW, iC, oC  --> oC, kH, kW, iC)
  }

  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv2dCUDNNPadAsymmetric(input, nullptr, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);
    tmpInput = std::move(std::get<0>(ret));  // prolong life
    if (tmpInput) input = tmpInput.get();
  }
  conv2dCUDNN(block.launchContext(), input, newWeights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode,
              isNCHW, wFormat);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC] always
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  const int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME, 2-CAUSAL

  const bool badInputType = input->dataType() != DOUBLE && input->dataType() != FLOAT32 &&
                            input->dataType() != HALF;
  const bool badWeightsType = weights->dataType() != DOUBLE && weights->dataType() != FLOAT32 &&
                              weights->dataType() != HALF;
  const bool badBiasType = bias == nullptr
                               ? false
                               : (bias->dataType() != DOUBLE && bias->dataType() != FLOAT32 &&
                                  bias->dataType() != HALF);

  return paddingMode != 2 && !badInputType && !badWeightsType && !badBiasType;
  Requirements req("CUDNN CONV2d OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   {HALF, FLOAT32, DOUBLE}) &&
      req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                   {HALF, FLOAT32, DOUBLE});
  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"),
                 {HALF, FLOAT32, DOUBLE});
  }
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  LongType kH = INT_ARG(0);                                               // filter(kernel) height
  LongType kW = INT_ARG(1);                                               // filter(kernel) width
  LongType sH = INT_ARG(2);                                               // strides height
  LongType sW = INT_ARG(3);                                               // strides width
  LongType pH = INT_ARG(4);                                               // paddings height
  LongType pW = INT_ARG(5);                                               // paddings width
  LongType dH = INT_ARG(6);                                               // dilations height
  LongType dW = INT_ARG(7);                                               // dilations width
  int paddingMode = INT_ARG(8);                                      // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "CUSTOM CONV2D_BP CUDNN OP: rank of input array must be equal to 4, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "CUSTOM CONV2D_BP CUDNN OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 4, 0,
               "CUSTOM CONV2D_BP CUDNN OP: rank of output's gradients (next epsilon) array must be equal to 4, but got "
               "%i instead !",
               gradO->rankOf());

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM CONV2D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM CONV2D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM CONV2D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got "
                 "%i, %i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  std::unique_ptr<NDArray> tmpGradI = {}, tmpInput = {}, tmpWeights = {}, tmpGradW = {};
  NDArray *newWeights = weights, *newGradW = gradW;  // cudnn support only two formats {oC,iC,kH,kW} and {oC,kH,kW,iC}
  if (0 == wFormat) {
    // Create named vectors as lvalues
    std::vector<LongType> nchwGradShape = {oC, iC, kH, kW};
    std::vector<LongType> nhwcGradShape = {oC, kH, kW, iC};

    // Use the appropriate one for the gradW reset
    tmpGradW.reset(
        new NDArray(gradW->ordering(),
                    isNCHW ? nchwGradShape : nhwcGradShape,
                    gradW->dataType(), gradW->getContext()));

    // Use the same vectors for the weights reset
    tmpWeights.reset(
        new NDArray(weights->ordering(),
                    isNCHW ? nchwGradShape : nhwcGradShape,
                    weights->dataType(), weights->getContext()));
    newGradW = tmpGradW.get();
    newWeights = tmpWeights.get();
    // Create named vectors as lvalues
    std::vector<LongType> nchwDims = {3, 2, 0, 1};
    std::vector<LongType> nhwcDims = {3, 0, 1, 2};
    NDArray assign = weights->permute(
                                isNCHW ? nchwDims : nhwcDims,
                                true,   // copyToNewBuff
                                true);
    // Use the appropriate one in the call
    newWeights->assign(&assign);
  }

  NDArray* newInput = input;
  NDArray* newGradI = gradI;

  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv2dCUDNNPadAsymmetric(input, gradI, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW);
    tmpInput = std::move(std::get<0>(ret));
    tmpGradI = std::move(std::get<1>(ret));
    if (tmpInput) newInput = tmpInput.get();
    if (tmpGradI) newGradI = tmpGradI.get();
  }
  conv2dBpCUDNN(block.launchContext(), newInput, newWeights, gradO, newGradI, newGradW, gradB, kH, kW, sH, sW, pH, pW,
                dH, dW, paddingMode, isNCHW, wFormat);

  if (0 == wFormat) {
    // Create named vectors as lvalues
    std::vector<LongType> nchwPermute = {2, 3, 1, 0};
    std::vector<LongType> nhwcPermute = {1, 2, 3, 0};

    // Use the appropriate one in the permutei call
    newGradW->permutei(
        isNCHW ? nchwPermute : nhwcPermute,false,false);  // (oC, iC, kH, kW --> kH, kW, iC, oC) or (oC, kH, kW, iC --> kH, kW, iC, oC) iC, oC)
    gradW->assign(newGradW);
  }

  if (newInput != input) {
    if (isNCHW) {
       NDArray assign = (*newGradI)({0, 0, 0, 0, 0, gradI->sizeAt(2), 0, gradI->sizeAt(3)});
      gradI->assign(&assign);
    } else {
      NDArray assign = (*newGradI)({0, 0, 0, gradI->sizeAt(1), 0, gradI->sizeAt(2), 0, 0});
      gradI->assign(&assign);
    }
  }

  return Status::OK;
}

PLATFORM_CHECK(conv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC] always
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  const int paddingMode = INT_ARG(8);                                      // 0-VALID, 1-SAME, 2-CAUSAL
  const int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC

  Requirements req("CUDNN CONV2d_BP OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
      req.expectTrue(makeInfoVariable(isNCHW, "isNCHW")) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   {HALF, FLOAT32, DOUBLE}) &&
      req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1),
                   {HALF, FLOAT32, DOUBLE});
  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"),
                 {HALF, FLOAT32, DOUBLE}) &&
        req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT3),
                     {HALF, FLOAT32, DOUBLE});
  } else {
    req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2),
                 {HALF, FLOAT32, DOUBLE});
  }
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
