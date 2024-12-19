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
static void conv3dCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights, NDArray* bias,
                        NDArray* output, const LongType kD, const LongType kH, const LongType kW, const LongType sD, const LongType sH,
                        const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH,
                        const LongType dW, const int paddingMode, const bool isNCDHW, const int wFormat) {
  // cudnn support only one format for weights {oC,iC,kD,kH,kW}

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const std::vector<int> pads = {static_cast<int>(pD), static_cast<int>(pH), static_cast<int>(pW)};
  const std::vector<int> filtStrides = {static_cast<int>(sD), static_cast<int>(sH), static_cast<int>(sW)};
  const std::vector<int> dilations = {static_cast<int>(dD), static_cast<int>(dH), static_cast<int>(dW)};

  const std::vector<int> xShape = {static_cast<int>(bS), static_cast<int>(iC), static_cast<int>(iD), static_cast<int>(iH), static_cast<int>(iW)};
  const std::vector<int> zShape = {static_cast<int>(bS), static_cast<int>(oC), static_cast<int>(oD), static_cast<int>(oH), static_cast<int>(oW)};
  const std::vector<int> wShape = {static_cast<int>(oC), static_cast<int>(iC), static_cast<int>(kD), static_cast<int>(kH), static_cast<int>(kW)};
  const std::vector<int> bShape = {1, static_cast<int>(oC), 1, 1, 1};

  const std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)), static_cast<int>(input->strideAt(2)),
                                     static_cast<int>(input->strideAt(3)), static_cast<int>(input->strideAt(4))};
  const std::vector<int> zStrides = {static_cast<int>(output->strideAt(0)), static_cast<int>(output->strideAt(1)), static_cast<int>(output->strideAt(2)),
                                     static_cast<int>(output->strideAt(3)), static_cast<int>(output->strideAt(4))};
  cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  PointersManager manager(context, __func__);
  // input descriptor
  CudnnTensor x;
    x.set(cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());

  // weights descriptor
  FilterDesc w;
  w.set(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, numDims, wShape.data());

  // output descriptor
  CudnnTensor z;
    z.set(cudnnDataType(output->dataType()), numDims, zShape.data(), zStrides.data());

  // description of convolution
  ConvolutionDesc conv;
  conv.set(numDims - 2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION,
           cudnnDataType(output->dataType()));

  // algorithm description
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;

  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                          cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  if (count == 0)
    throw cuda_exception::build("conv3dCUDNN: cudnnGetConvolutionForwardAlgorithm failed as the count is 0", 0);
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
    b.setEx(/*format*/ CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), numDims, bShape.data());

    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor), cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha,
                                                                      z, output->specialBuffer()));
  }


  NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void conv3dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights,
                          NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kD,
                          const LongType kH, const LongType kW, const LongType sD, const LongType sH, const LongType sW, const LongType pD,
                          const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW, const int paddingMode,
                          const bool isNCDHW, const int wFormat) {
  // cudnn supports only two formats {oC,iC,kD,kH,kW} and {oC,kD,kH,kW,iC} for weights/gradW

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));
  const std::vector<int> pads = {static_cast<int>(pD), static_cast<int>(pH), static_cast<int>(pW)};
  const std::vector<int> filtStrides = {static_cast<int>(sD), static_cast<int>(sH), static_cast<int>(sW)};
  const std::vector<int> dilations = {static_cast<int>(dD), static_cast<int>(dH), static_cast<int>(dW)};

  const std::vector<int> xShape = {static_cast<int>(bS), static_cast<int>(iC), static_cast<int>(iD), static_cast<int>(iH), static_cast<int>(iW)};
  const std::vector<int> dzShape = {static_cast<int>(bS), static_cast<int>(oC), static_cast<int>(oD), static_cast<int>(oH), static_cast<int>(oW)};
  const std::vector<int> wShape = {static_cast<int>(oC), static_cast<int>(iC), static_cast<int>(kD), static_cast<int>(kH), static_cast<int>(kW)};
  const std::vector<int> dbShape = {1, static_cast<int>(isNCDHW ? oC : 1), 1, 1, (int)(isNCDHW ? 1 : oC)};

  const std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)), static_cast<int>(input->strideAt(2)),
                                     static_cast<int>(input->strideAt(3)), static_cast<int>(input->strideAt(4))};
  const std::vector<int> dxStrides = {static_cast<int>(gradI->strideAt(0)), static_cast<int>(gradI->strideAt(1)), static_cast<int>(gradI->strideAt(2)),
                                      static_cast<int>(gradI->strideAt(3)), static_cast<int>(gradI->strideAt(4))};
  const std::vector<int> dzStrides = {static_cast<int>(gradO->strideAt(0)), static_cast<int>(gradO->strideAt(1)), static_cast<int>(gradO->strideAt(2)),
                                      static_cast<int>(gradO->strideAt(3)), static_cast<int>(gradO->strideAt(4))};
  cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);
  PointersManager manager(context, __func__);
  // input descriptor, gradO descriptor, gradI descriptor
  CudnnTensor x, dz, dx;
    x.set(cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());

    dz.set(cudnnDataType(gradO->dataType()), numDims, dzShape.data(), dzStrides.data());

    dx.set(cudnnDataType(gradI->dataType()), numDims, xShape.data(), dxStrides.data());

  // gradW descriptor
  FilterDesc dw;
  dw.set(cudnnDataType(gradW->dataType()), formatW, numDims, wShape.data());

  // description of convolution
  ConvolutionDesc conv;
  conv.set(numDims - 2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION,
           cudnnDataType(gradO->dataType()));

  // gradW algorithm description
  cudnnConvolutionBwdFilterAlgo_t algoGradW;
  cudnnConvolutionBwdFilterAlgoPerf_t algoGradWPerf;
  int count = 0;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
      cudnnFindConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf));
  if (count == 0)
    throw cuda_exception::build(
        "conv3dBpCUDNN: cudnnGetConvolutionBackwardFilterAlgorithm failed as the count is 0", 0);
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
    throw cuda_exception::build("conv3dBpCUDNN: cudnnGetConvolutionBackwardDataAlgorithm failed  as the count is 0",
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
    db.setEx(format, cudnnDataType(gradB->dataType()), numDims, dbShape.data());

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
PLATFORM_IMPL(conv3dnew, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

  REQUIRE_TRUE(input->rankOf() == 5, 0, "CONV3D CUDNN OP: rank of input array must be equal to 5, but got %i instead !",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CONV3D CUDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                          // strides depth
  LongType sH = INT_ARG(4);                                                          // strides height
  LongType sW = INT_ARG(5);                                                          // strides width
  LongType pD = INT_ARG(6);                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                          // paddings height
  LongType pW = INT_ARG(8);                                                          // paddings width
  LongType dD = INT_ARG(9);                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                         // dilations height
  LongType dW = INT_ARG(11);                                                         // dilations width
  int paddingMode = INT_ARG(12);                                                // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

  REQUIRE_TRUE(paddingMode < 2, 0,
               "CONV3D CUDNN OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV3D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(
        bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
        "CONV3D CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !",
        oC, bias->rankOf(), bias->lengthOf());

  std::unique_ptr<NDArray> tmpWeight = {}, tmpInput = {};
  NDArray* newWeights = weights;  // cudnn support only one format {oC,iC,kD,kH,kW}
  if (1 != wFormat) {
    tmpWeight.reset(new NDArray(weights->ordering(), {oC, iC, kD, kH, kW}, weights->dataType(), weights->getContext()));
    newWeights = tmpWeight.get();
    newWeights->assign(weights->permute(
        0 == wFormat
            ? std::vector<LongType>({4, 3, 0, 1, 2})
            : std::vector<LongType>(
                  {0, 4, 1, 2,
                   3})));  // kD, kH, kW, iC, oC  --> oC, iC, kD, kH, kW   or oC, kD, kH, kW, iC  --> oC, iC, kD, kH, kW
  }

  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv3dCUDNNPadAsymmetric(input, nullptr, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
                                             dD, dH, dW, isNCDHW);
    tmpInput = std::move(std::get<0>(ret));  // prolong life
    if (tmpInput) input = tmpInput.get();
  }
  conv3dCUDNN(block.launchContext(), input, newWeights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
              paddingMode, isNCDHW, wFormat);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv3dnew, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  int paddingMode = INT_ARG(12);  // 0-SAME,  1-VALID

  Requirements req("CUDNN CONV3d OP");
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
PLATFORM_IMPL(conv3dnew_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CONV3D_BP CUDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CONV3D_BP CUDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());
  REQUIRE_TRUE(
      gradO->rankOf() == 5, 0,
      "CONV3D_BP CUDNN OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !",
      gradO->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                          // strides depth
  LongType sH = INT_ARG(4);                                                          // strides height
  LongType sW = INT_ARG(5);                                                          // strides width
  LongType pD = INT_ARG(6);                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                          // paddings height
  LongType pW = INT_ARG(8);                                                          // paddings width
  LongType dD = INT_ARG(9);                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                         // dilations height
  LongType dW = INT_ARG(11);                                                         // dilations width
  int paddingMode = INT_ARG(12);                                                // 1-SAME,  0-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWiC, indWoC, indWkD);

  LongType trueoD, trueoH, trueoW;  // true output depth/height/width
  ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
                                      iW, paddingMode);

  REQUIRE_TRUE(paddingMode < 2, 0,
               "CONV3D_BP CUDNN OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");

  std::vector<LongType> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx(
      {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "CONV3D_BP CUDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(gradW->isSameShape(expectedWeightsShape), 0,
               "CONV3D_BP CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CONV3D_BP CUDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i "
                 "instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

  std::unique_ptr<NDArray> tmpGradI = {}, tmpInput = {}, tmpWeights = {}, tmpGradW = {};
  NDArray *newWeights = weights,
          *newGradW = gradW;  // cudnn support only two formats {oC,iC,kD,kH,kW} and {oC,kD,kH,kW,iC}
  if (0 == wFormat) {
    tmpGradW.reset(new NDArray(
        gradW->ordering(),
        isNCDHW ? std::vector<LongType>({oC, iC, kD, kH, kW}) : std::vector<LongType>({oC, kD, kH, kW, iC}),
        gradW->dataType(), gradW->getContext()));
    tmpWeights.reset(new NDArray(
        weights->ordering(),
        isNCDHW ? std::vector<LongType>({oC, iC, kD, kH, kW}) : std::vector<LongType>({oC, kD, kH, kW, iC}),
        weights->dataType(), weights->getContext()));
    newGradW = tmpGradW.get();
    newWeights = tmpWeights.get();
    newWeights->assign(weights->permute(
        isNCDHW ? std::vector<LongType>({4, 3, 0, 1, 2})
                : std::vector<LongType>({4, 0, 1, 2, 3})));  // (kD, kH, kW, iC, oC  --> oC, iC, kD, kH, kW) or (kD, kH, kW,
                                                        // iC, oC  --> oC, kD, kH, kW, iC)
  }

  NDArray* newInput = input;
  NDArray* newGradI = gradI;
  if (paddingMode == 1) {  // in same paddingMode cudnn doesn't support asymmetric left/right top/bottopm paddings
    auto ret = checkConv3dCUDNNPadAsymmetric(input, gradI, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
                                             dD, dH, dW, isNCDHW);
    tmpInput = std::move(std::get<0>(ret));
    tmpGradI = std::move(std::get<1>(ret));
    if (tmpInput) newInput = tmpInput.get();
    if (tmpGradI) newGradI = tmpGradI.get();
  }
  conv3dBpCUDNN(block.launchContext(), newInput, newWeights, gradO, newGradI, newGradW, gradB, kD, kH, kW, sD, sH, sW,
                pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW, wFormat);

  if (0 == wFormat) {
    newGradW->permutei(isNCDHW ? std::vector<LongType>({2, 3, 4, 1, 0})
                               : std::vector<LongType>({1, 2, 3, 4, 0}));  // (oC, iC, kD, kH, kW --> kD, kH, kW, iC, oC) or
                                                                      // (oC, kD, kH, kW, iC --> kD, kH, kW, iC, oC)
    gradW->assign(newGradW);
  }

  if (newInput != input) {
    if (isNCDHW)
      gradI->assign((*newGradI)({0, 0, 0, 0, 0, gradI->sizeAt(2), 0, gradI->sizeAt(3), 0, gradI->sizeAt(4)}));
    else
      gradI->assign((*newGradI)({0, 0, 0, gradI->sizeAt(1), 0, gradI->sizeAt(2), 0, gradI->sizeAt(3), 0, 0}));
  }

  return Status::OK;
}

PLATFORM_CHECK(conv3dnew_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  LongType paddingMode = INT_ARG(12);                                        // 1-SAME,  0-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;  // INT_ARG(13): 1-NDHWC, 0-NCDHW

  Requirements req("CUDNN CONV3d_BP OP");
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
      req.expectTrue(makeInfoVariable(isNCDHW, "isNCDHW")) &&
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
