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
// Deconvolution (transposed convolution) is mathematically equivalent to
// the backward data pass of a regular convolution
static void deconv2dCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights, NDArray* bias,
                          NDArray* output, const LongType kH, const LongType kW, const LongType sH, const LongType sW,
                          const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                          const int paddingMode, const bool isNCHW, const int wFormat) {
  // input  [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  // weights [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  // output [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  LongType bS, iC, iH, iW, oC, oH, oW;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW,
                                             indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  // cuDNN expects weights in format [oC, iC, kH, kW] for NCHW or [oC, kH, kW, iC] for NHWC
  // But for deconv, the roles of iC and oC are swapped compared to regular conv
  cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

  // Input tensor descriptor (this is the "gradO" equivalent in backward data)
  CudnnTensor x;
  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW,
              input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // Output tensor descriptor (this is the "gradI" equivalent in backward data)
  CudnnTensor z;
  if (output->ordering() == 'c')
    z.set4D(format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
  else
    z.set4DEx(cudnnDataType(output->dataType()), bS, oC, oH, oW,
              output->strideAt(0), output->strideAt(indIOioC),
              output->strideAt(indOoH), output->strideAt(indOoH + 1));

  // Weights descriptor - for deconv, we need [iC, oC, kH, kW] format
  // because the weight tensor has shape [kH, kW, oC, iC] in wFormat=0
  FilterDesc w;
  w.set4D(cudnnDataType(weights->dataType()), formatW, iC, oC, kH, kW);

  // Convolution descriptor
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));

  // Find algorithm for backward data (which is equivalent to deconv forward)
  cudnnConvolutionBwdDataAlgo_t algo;
  cudnnConvolutionBwdDataAlgoPerf_t algoPerf;
  int count = 0;
  // During CUDA graph capture, use heuristic _v7 to avoid capture invalidation.
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnGetConvolutionBackwardDataAlgorithm_v7),
        cudnnGetConvolutionBackwardDataAlgorithm_v7(*handle, w, x, conv, z, 1, &count, &algoPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnFindConvolutionBackwardDataAlgorithm),
        cudnnFindConvolutionBackwardDataAlgorithm(*handle, w, x, conv, z, 1, &count, &algoPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("deconv2dCUDNN: cudnnFindConvolutionBackwardDataAlgorithm failed");
  algo = algoPerf.algo;

  PointersManager manager(context, __func__);

  // Allocate workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardDataWorkspaceSize),
      cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, w, x, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
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
    b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnAddTensor),
        cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha, z, output->specialBuffer()));
  }

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "deconv2dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void deconv2dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights,
                            NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                            const LongType kH, const LongType kW, const LongType sH, const LongType sW,
                            const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                            const int paddingMode, const bool isNCHW, const int wFormat) {
  // For deconv backward:
  // - gradI is computed using cudnnConvolutionForward (forward conv)
  // - gradW is computed using cudnnConvolutionBackwardFilter

  LongType bS, iC, iH, iW, oC, oH, oW;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW,
                                             indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  cudnnTensorFormat_t formatW = 0 == wFormat ? format : (1 == wFormat ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);

  PointersManager manager(context, __func__);

  // Tensor descriptors
  CudnnTensor x, dz, dx;

  // Input descriptor
  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW,
              input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // gradO descriptor
  if (gradO->ordering() == 'c')
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
  else
    dz.set4DEx(cudnnDataType(gradO->dataType()), bS, oC, oH, oW,
               gradO->strideAt(0), gradO->strideAt(indIOioC),
               gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));

  // gradI descriptor
  if (gradI->ordering() == 'c')
    dx.set4D(format, cudnnDataType(gradI->dataType()), bS, iC, iH, iW);
  else
    dx.set4DEx(cudnnDataType(gradI->dataType()), bS, iC, iH, iW,
               gradI->strideAt(0), gradI->strideAt(indIOioC),
               gradI->strideAt(indIiH), gradI->strideAt(indIiH + 1));

  // Weights descriptor
  FilterDesc w, dw;
  w.set4D(cudnnDataType(weights->dataType()), formatW, iC, oC, kH, kW);
  dw.set4D(cudnnDataType(gradW->dataType()), formatW, iC, oC, kH, kW);

  // Convolution descriptor
  ConvolutionDesc conv;
  conv.set2D(pH, pW, sH, sW, dH, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

  // Calculate gradI using forward convolution
  // (For deconv, the backward pass for input is a forward conv)
  cudnnConvolutionFwdAlgo_t algoFwd;
  cudnnConvolutionFwdAlgoPerf_t algoFwdPerf;
  int count = 0;
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnGetConvolutionForwardAlgorithm_v7),
        cudnnGetConvolutionForwardAlgorithm_v7(*handle, dz, w, conv, dx, 1, &count, &algoFwdPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
        cudnnFindConvolutionForwardAlgorithm(*handle, dz, w, conv, dx, 1, &count, &algoFwdPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("deconv2dBpCUDNN: cudnnFindConvolutionForwardAlgorithm failed");
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
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnGetConvolutionBackwardFilterAlgorithm_v7),
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(*handle, dz, x, conv, dw, 1, &count, &algoFilterPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
        cudnnFindConvolutionBackwardFilterAlgorithm(*handle, dz, x, conv, dw, 1, &count, &algoFilterPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("deconv2dBpCUDNN: cudnnFindConvolutionBackwardFilterAlgorithm failed");
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
    db.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(gradB->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnConvolutionBackwardBias),
        cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->specialBuffer(), beta, db, gradB->specialBuffer()));
  }

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "deconv2dBpCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "DECONV2D CUDNN OP: rank of input array must be equal to 4, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "DECONV2D CUDNN OP: rank of weights array must be equal to 4, but got %i instead!", weights->rankOf());

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));
  LongType sH = INT_ARG(2);
  LongType sW = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  LongType dH = INT_ARG(6);
  LongType dW = INT_ARG(7);
  int paddingMode = INT_ARG(8);
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  LongType bS, iC, iH, iW, oC, oH, oW;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW,
                                             indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

  if (paddingMode)
    ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);

  std::vector<LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "DECONV2D CUDNN OP: wrong shape of weights array, expected is %s, but got %s instead!",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  std::unique_ptr<NDArray> tmpWeights;
  NDArray* newWeights = weights;

  // cuDNN requires weights in specific format, convert if needed
  if (0 == wFormat) {
    std::vector<LongType> newShape = isNCHW ? std::vector<LongType>({iC, oC, kH, kW}) : std::vector<LongType>({iC, kH, kW, oC});
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    newWeights = tmpWeights.get();
    std::vector<LongType> permDims = isNCHW ? std::vector<LongType>({3, 2, 0, 1}) : std::vector<LongType>({3, 0, 1, 2});
    NDArray* permuted = weights->permute(permDims, true, true);
    newWeights->assign(permuted);
    delete permuted;
  }

  deconv2dCUDNN(block.launchContext(), input, newWeights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW,
                paddingMode, isNCHW, wFormat);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(deconv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

  int paddingMode = INT_ARG(8);

  Requirements req("CUDNN DECONV2D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "DECONV2D_BP CUDNN OP: rank of input array must be equal to 4, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "DECONV2D_BP CUDNN OP: rank of weights array must be equal to 4, but got %i instead!", weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 4, 0,
               "DECONV2D_BP CUDNN OP: rank of gradO array must be equal to 4, but got %i instead!", gradO->rankOf());

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));
  LongType sH = INT_ARG(2);
  LongType sW = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  LongType dH = INT_ARG(6);
  LongType dW = INT_ARG(7);
  int paddingMode = INT_ARG(8);
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  LongType bS, iC, iH, iW, oC, oH, oW;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW,
                                             indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

  if (paddingMode)
    ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);

  std::unique_ptr<NDArray> tmpWeights, tmpGradW;
  NDArray *newWeights = weights, *newGradW = gradW;

  if (0 == wFormat) {
    std::vector<LongType> newShape = isNCHW ? std::vector<LongType>({iC, oC, kH, kW}) : std::vector<LongType>({iC, kH, kW, oC});
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    tmpGradW.reset(new NDArray(gradW->ordering(), newShape, gradW->dataType(), gradW->getContext()));
    newWeights = tmpWeights.get();
    newGradW = tmpGradW.get();
    std::vector<LongType> permDims = isNCHW ? std::vector<LongType>({3, 2, 0, 1}) : std::vector<LongType>({3, 0, 1, 2});
    NDArray* permuted = weights->permute(permDims, true, true);
    newWeights->assign(permuted);
    delete permuted;
  }

  deconv2dBpCUDNN(block.launchContext(), input, newWeights, gradO, gradI, newGradW, gradB,
                  kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat);

  if (0 == wFormat) {
    std::vector<LongType> permDims = isNCHW ? std::vector<LongType>({2, 3, 1, 0}) : std::vector<LongType>({1, 2, 3, 0});
    newGradW->permutei(permDims, false, false);
    gradW->assign(newGradW);
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(deconv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  int paddingMode = INT_ARG(8);

  Requirements req("CUDNN DECONV2D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
