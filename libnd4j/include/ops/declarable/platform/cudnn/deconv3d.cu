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
// @author Adam Gibson
//

#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/addBias.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void deconv3dCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights, NDArray* bias,
                          NDArray* output, const LongType kD, const LongType kH, const LongType kW,
                          const LongType sD, const LongType sH, const LongType sW,
                          const LongType pD, const LongType pH, const LongType pW,
                          const LongType dD, const LongType dH, const LongType dW,
                          const int paddingMode, const bool isNCDHW, const int wFormat) {
  // input  [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  // weights [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  // output [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH, oW;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const std::vector<int> pads = {static_cast<int>(pD), static_cast<int>(pH), static_cast<int>(pW)};
  const std::vector<int> filtStrides = {static_cast<int>(sD), static_cast<int>(sH), static_cast<int>(sW)};
  const std::vector<int> dilations = {static_cast<int>(dD), static_cast<int>(dH), static_cast<int>(dW)};

  // For deconv, input and output roles are swapped compared to conv
  const std::vector<int> xShape = {static_cast<int>(bS), static_cast<int>(iC), static_cast<int>(iD), static_cast<int>(iH), static_cast<int>(iW)};
  const std::vector<int> zShape = {static_cast<int>(bS), static_cast<int>(oC), static_cast<int>(oD), static_cast<int>(oH), static_cast<int>(oW)};
  const std::vector<int> wShape = {static_cast<int>(iC), static_cast<int>(oC), static_cast<int>(kD), static_cast<int>(kH), static_cast<int>(kW)};
  const std::vector<int> bShape = {1, static_cast<int>(oC), 1, 1, 1};

  const std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)),
                                     static_cast<int>(input->strideAt(2)), static_cast<int>(input->strideAt(3)),
                                     static_cast<int>(input->strideAt(4))};
  const std::vector<int> zStrides = {static_cast<int>(output->strideAt(0)), static_cast<int>(output->strideAt(1)),
                                     static_cast<int>(output->strideAt(2)), static_cast<int>(output->strideAt(3)),
                                     static_cast<int>(output->strideAt(4))};

  cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  PointersManager manager(context, __func__);

  // Input tensor descriptor
  CudnnTensor x;
  x.set(cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());

  // Output tensor descriptor
  CudnnTensor z;
  z.set(cudnnDataType(output->dataType()), numDims, zShape.data(), zStrides.data());

  // Weights descriptor
  FilterDesc w;
  w.set(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, numDims, wShape.data());

  // Convolution descriptor
  ConvolutionDesc conv;
  conv.set(numDims - 2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION,
           cudnnDataType(output->dataType()));

  // Find algorithm for backward data (which is equivalent to deconv forward)
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnConvolutionBwdDataAlgoPerf_t algoPerf;
  int count = 0;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardDataAlgorithm),
      cudnnFindConvolutionBackwardDataAlgorithm(*handle, w, x, conv, z, 1, &count, &algoPerf));
  if (count == 0)
    throw cuda_exception::build("deconv3dCUDNN: cudnnFindConvolutionBackwardDataAlgorithm failed", 0);
  algo = algoPerf.algo;

  // Allocate workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardDataWorkspaceSize),
      cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, w, x, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // Scaling parameters
  const float alpha32 = 1.0f, beta32 = 0.0f;
  const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, weights, bias});

  // Run deconvolution (backward data)
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardData),
      cudnnConvolutionBackwardData(*handle, alpha, w, weights->specialBuffer(), x, input->specialBuffer(),
                                   conv, algo, wsData, wsSize, beta, z, output->specialBuffer()));

  // Add bias if present
  if (bias != nullptr) {
    CudnnTensor b;
    b.setEx(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), numDims, bShape.data());
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnAddTensor),
        cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha, z, output->specialBuffer()));
  }

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("deconv3dCUDNN: cudaStreamSynchronize failed!", cudaErr);

  NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void deconv3dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights,
                            NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                            const LongType kD, const LongType kH, const LongType kW,
                            const LongType sD, const LongType sH, const LongType sW,
                            const LongType pD, const LongType pH, const LongType pW,
                            const LongType dD, const LongType dH, const LongType dW,
                            const int paddingMode, const bool isNCDHW, const int wFormat) {

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH, oW;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const std::vector<int> pads = {static_cast<int>(pD), static_cast<int>(pH), static_cast<int>(pW)};
  const std::vector<int> filtStrides = {static_cast<int>(sD), static_cast<int>(sH), static_cast<int>(sW)};
  const std::vector<int> dilations = {static_cast<int>(dD), static_cast<int>(dH), static_cast<int>(dW)};

  const std::vector<int> xShape = {static_cast<int>(bS), static_cast<int>(iC), static_cast<int>(iD), static_cast<int>(iH), static_cast<int>(iW)};
  const std::vector<int> dzShape = {static_cast<int>(bS), static_cast<int>(oC), static_cast<int>(oD), static_cast<int>(oH), static_cast<int>(oW)};
  const std::vector<int> wShape = {static_cast<int>(iC), static_cast<int>(oC), static_cast<int>(kD), static_cast<int>(kH), static_cast<int>(kW)};
  const std::vector<int> dbShape = {1, static_cast<int>(oC), 1, 1, 1};

  const std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)),
                                     static_cast<int>(input->strideAt(2)), static_cast<int>(input->strideAt(3)),
                                     static_cast<int>(input->strideAt(4))};
  const std::vector<int> dxStrides = {static_cast<int>(gradI->strideAt(0)), static_cast<int>(gradI->strideAt(1)),
                                      static_cast<int>(gradI->strideAt(2)), static_cast<int>(gradI->strideAt(3)),
                                      static_cast<int>(gradI->strideAt(4))};
  const std::vector<int> dzStrides = {static_cast<int>(gradO->strideAt(0)), static_cast<int>(gradO->strideAt(1)),
                                      static_cast<int>(gradO->strideAt(2)), static_cast<int>(gradO->strideAt(3)),
                                      static_cast<int>(gradO->strideAt(4))};

  PointersManager manager(context, __func__);

  // Tensor descriptors
  CudnnTensor x, dz, dx;
  x.set(cudnnDataType(input->dataType()), numDims, xShape.data(), xStrides.data());
  dz.set(cudnnDataType(gradO->dataType()), numDims, dzShape.data(), dzStrides.data());
  dx.set(cudnnDataType(gradI->dataType()), numDims, xShape.data(), dxStrides.data());

  // Weights descriptor
  FilterDesc w, dw;
  w.set(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, numDims, wShape.data());
  dw.set(cudnnDataType(gradW->dataType()), CUDNN_TENSOR_NCHW, numDims, wShape.data());

  // Convolution descriptor
  ConvolutionDesc conv;
  conv.set(numDims - 2, pads.data(), filtStrides.data(), dilations.data(), CUDNN_CROSS_CORRELATION,
           cudnnDataType(gradO->dataType()));

  // Scaling parameters
  const float alpha32 = 1.0f, beta32 = 0.0f;
  const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

  // Calculate gradI using forward convolution
  cudnnConvolutionFwdAlgo_t algoFwd;
  cudnnConvolutionFwdAlgoPerf_t algoFwdPerf;
  int count = 0;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
      cudnnFindConvolutionForwardAlgorithm(*handle, dz, w, conv, dx, 1, &count, &algoFwdPerf));
  if (count == 0)
    throw cuda_exception::build("deconv3dBpCUDNN: cudnnFindConvolutionForwardAlgorithm failed", 0);
  algoFwd = algoFwdPerf.algo;

  size_t wsFwdSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionForwardWorkspaceSize),
      cudnnGetConvolutionForwardWorkspaceSize(*handle, dz, w, conv, dx, algoFwd, &wsFwdSize));
  void* wsFwdData = manager.allocateDevMem(wsFwdSize);

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionForward),
      cudnnConvolutionForward(*handle, alpha, dz, gradO->specialBuffer(), w, weights->specialBuffer(),
                              conv, algoFwd, wsFwdData, wsFwdSize, beta, dx, gradI->specialBuffer()));

  // Calculate gradW using backward filter
  cudnnConvolutionBwdFilterAlgo_t algoFilter;
  cudnnConvolutionBwdFilterAlgoPerf_t algoFilterPerf;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
      cudnnFindConvolutionBackwardFilterAlgorithm(*handle, dz, x, conv, dw, 1, &count, &algoFilterPerf));
  if (count == 0)
    throw cuda_exception::build("deconv3dBpCUDNN: cudnnFindConvolutionBackwardFilterAlgorithm failed", 0);
  algoFilter = algoFilterPerf.algo;

  size_t wsFilterSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardFilterWorkspaceSize),
      cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, dz, x, conv, dw, algoFilter, &wsFilterSize));
  void* wsFilterData = manager.allocateDevMem(wsFilterSize);

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardFilter),
      cudnnConvolutionBackwardFilter(*handle, alpha, dz, gradO->specialBuffer(), x, input->specialBuffer(),
                                     conv, algoFilter, wsFilterData, wsFilterSize, beta, dw, gradW->specialBuffer()));

  // Calculate gradB if present
  if (gradB != nullptr) {
    CudnnTensor db;
    db.setEx(CUDNN_TENSOR_NCHW, cudnnDataType(gradB->dataType()), numDims, dbShape.data());
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnConvolutionBackwardBias),
        cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->specialBuffer(), beta, db, gradB->specialBuffer()));
  }

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("deconv3dBpCUDNN: cudaStreamSynchronize failed!", cudaErr);

  NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv3d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "DECONV3D CUDNN OP: rank of input array must be equal to 5, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "DECONV3D CUDNN OP: rank of weights array must be equal to 5, but got %i instead!", weights->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));
  LongType sD = INT_ARG(3);
  LongType sH = INT_ARG(4);
  LongType sW = INT_ARG(5);
  LongType pD = INT_ARG(6);
  LongType pH = INT_ARG(7);
  LongType pW = INT_ARG(8);
  LongType dD = INT_ARG(9);
  LongType dH = INT_ARG(10);
  LongType dW = INT_ARG(11);
  int paddingMode = INT_ARG(12);
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;
  int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;

  LongType bS, iC, iD, iH, iW, oC, oD, oH, oW;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  if (paddingMode)
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "DECONV3D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead!",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  std::unique_ptr<NDArray> tmpWeights;
  NDArray* newWeights = weights;

  // Convert weights format if needed
  if (0 == wFormat) {
    std::vector<LongType> newShape = {iC, oC, kD, kH, kW};
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    newWeights = tmpWeights.get();
    std::vector<LongType> permDims = {4, 3, 0, 1, 2};
    NDArray permuted = weights->permute(permDims, true, true);
    newWeights->assign(&permuted);
  }

  deconv3dCUDNN(block.launchContext(), input, newWeights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
                paddingMode, isNCDHW, wFormat);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(deconv3d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

  int paddingMode = INT_ARG(12);

  Requirements req("CUDNN DECONV3D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv3d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "DECONV3D_BP CUDNN OP: rank of input array must be equal to 5, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "DECONV3D_BP CUDNN OP: rank of weights array must be equal to 5, but got %i instead!", weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 5, 0,
               "DECONV3D_BP CUDNN OP: rank of gradO array must be equal to 5, but got %i instead!", gradO->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));
  LongType sD = INT_ARG(3);
  LongType sH = INT_ARG(4);
  LongType sW = INT_ARG(5);
  LongType pD = INT_ARG(6);
  LongType pH = INT_ARG(7);
  LongType pW = INT_ARG(8);
  LongType dD = INT_ARG(9);
  LongType dH = INT_ARG(10);
  LongType dW = INT_ARG(11);
  int paddingMode = INT_ARG(12);
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;
  int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;

  LongType bS, iC, iD, iH, iW, oC, oD, oH, oW;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  if (paddingMode)
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  std::unique_ptr<NDArray> tmpWeights, tmpGradW;
  NDArray *newWeights = weights, *newGradW = gradW;

  if (0 == wFormat) {
    std::vector<LongType> newShape = {iC, oC, kD, kH, kW};
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    tmpGradW.reset(new NDArray(gradW->ordering(), newShape, gradW->dataType(), gradW->getContext()));
    newWeights = tmpWeights.get();
    newGradW = tmpGradW.get();
    std::vector<LongType> permDims = {4, 3, 0, 1, 2};
    NDArray permuted = weights->permute(permDims, true, true);
    newWeights->assign(&permuted);
  }

  deconv3dBpCUDNN(block.launchContext(), input, newWeights, gradO, gradI, newGradW, gradB,
                  kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW, wFormat);

  if (0 == wFormat) {
    std::vector<LongType> permDims = {2, 3, 4, 1, 0};
    newGradW->permutei(permDims, false, false);
    gradW->assign(newGradW);
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(deconv3d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  int paddingMode = INT_ARG(12);

  Requirements req("CUDNN DECONV3D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
