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
// Conv1D implementation using cuDNN by treating it as Conv2D with height=1
//

#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void conv1dCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights, NDArray* bias,
                        NDArray* output, const LongType kW, const LongType sW, const LongType pW,
                        const LongType dW, const int paddingMode, const bool isNCW, const int wFormat) {
  // Conv1D is implemented as Conv2D with height=1
  // input:  [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW) -> reshape to 4D
  // weights: [kW, iC, oC], [oC, iC, kW], [oC, kW, iC] -> reshape to 4D
  // output: [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW) -> reshape to 4D

  const LongType bS = input->sizeAt(0);
  const LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  const LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  const LongType oC = isNCW ? output->sizeAt(1) : output->sizeAt(2);
  const LongType oW = isNCW ? output->sizeAt(2) : output->sizeAt(1);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // Input descriptor: [bS, iC, 1, iW] for NCHW or [bS, 1, iW, iC] for NHWC
  CudnnTensor x;
  if (isNCW)
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, 1, iW);
  else
    x.set4D(format, cudnnDataType(input->dataType()), bS, 1, iW, iC);

  // Weights descriptor: [oC, iC, 1, kW]
  FilterDesc w;
  w.set4D(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, oC, iC, 1, kW);

  // Output descriptor: [bS, oC, 1, oW] for NCHW or [bS, 1, oW, oC] for NHWC
  CudnnTensor z;
  if (isNCW)
    z.set4D(format, cudnnDataType(output->dataType()), bS, oC, 1, oW);
  else
    z.set4D(format, cudnnDataType(output->dataType()), bS, 1, oW, oC);

  // Convolution descriptor: padding height=0, stride height=1, dilation height=1
  ConvolutionDesc conv;
  conv.set2D(0, pW, 1, sW, 1, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));

  // Find algorithm
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;
  // During CUDA graph capture, use heuristic _v7 to avoid capture invalidation.
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardAlgorithm_v7),
                            cudnnGetConvolutionForwardAlgorithm_v7(*handle, x, w, conv, z, 1, &count, &algoPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                            cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("conv1dCUDNN: cudnnFindConvolutionForwardAlgorithm failed");
  algo = algoPerf.algo;

  PointersManager manager(context, __func__);

  // Allocate workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardWorkspaceSize),
                          cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, weights, bias});

  // Run convolution
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionForward),
      cudnnConvolutionForward(*handle, alpha, x, input->specialBuffer(), w, weights->specialBuffer(),
                              conv, algo, wsData, wsSize, beta, z, output->specialBuffer()));

  // Add bias if present
  if (bias != nullptr) {
    CudnnTensor b;
    b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor),
                            cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha, z, output->specialBuffer()));
  }

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "conv1dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input, weights, bias});
}

//////////////////////////////////////////////////////////////////////////
static void conv1dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* weights,
                          NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                          const LongType kW, const LongType sW, const LongType pW, const LongType dW,
                          const int paddingMode, const bool isNCW, const int wFormat) {

  const LongType bS = input->sizeAt(0);
  const LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  const LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  const LongType oC = isNCW ? gradO->sizeAt(1) : gradO->sizeAt(2);
  const LongType oW = isNCW ? gradO->sizeAt(2) : gradO->sizeAt(1);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  PointersManager manager(context, __func__);

  // Tensor descriptors
  CudnnTensor x, dz, dx;
  if (isNCW) {
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, 1, iW);
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, oC, 1, oW);
    dx.set4D(format, cudnnDataType(gradI->dataType()), bS, iC, 1, iW);
  } else {
    x.set4D(format, cudnnDataType(input->dataType()), bS, 1, iW, iC);
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, 1, oW, oC);
    dx.set4D(format, cudnnDataType(gradI->dataType()), bS, 1, iW, iC);
  }

  // Weights descriptor
  FilterDesc w, dw;
  w.set4D(cudnnDataType(weights->dataType()), CUDNN_TENSOR_NCHW, oC, iC, 1, kW);
  dw.set4D(cudnnDataType(gradW->dataType()), CUDNN_TENSOR_NCHW, oC, iC, 1, kW);

  // Convolution descriptor
  ConvolutionDesc conv;
  conv.set2D(0, pW, 1, sW, 1, dW, CUDNN_CROSS_CORRELATION, cudnnDataType(gradO->dataType()));

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});

  // gradW algorithm
  cudnnConvolutionBwdFilterAlgo_t algoGradW;
  cudnnConvolutionBwdFilterAlgoPerf_t algoGradWPerf;
  int count = 0;
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnGetConvolutionBackwardFilterAlgorithm_v7),
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnFindConvolutionBackwardFilterAlgorithm),
        cudnnFindConvolutionBackwardFilterAlgorithm(*handle, x, dz, conv, dw, 1, &count, &algoGradWPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("conv1dBpCUDNN: cudnnFindConvolutionBackwardFilterAlgorithm failed");
  algoGradW = algoGradWPerf.algo;

  // gradI algorithm
  cudnnConvolutionBwdDataAlgo_t algoGradI;
  cudnnConvolutionBwdDataAlgoPerf_t algoGradIPerf;
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnGetConvolutionBackwardDataAlgorithm_v7),
        cudnnGetConvolutionBackwardDataAlgorithm_v7(*handle, dw, dz, conv, x, 1, &count, &algoGradIPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnFindConvolutionBackwardDataAlgorithm),
        cudnnFindConvolutionBackwardDataAlgorithm(*handle, dw, dz, conv, x, 1, &count, &algoGradIPerf));
  }
  if (count == 0)
    THROW_EXCEPTION("conv1dBpCUDNN: cudnnFindConvolutionBackwardDataAlgorithm failed");
  algoGradI = algoGradIPerf.algo;

  // Allocate workspace for gradW
  size_t wsGradWSize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardFilterWorkspaceSize),
      cudnnGetConvolutionBackwardFilterWorkspaceSize(*handle, x, dz, conv, dw, algoGradW, &wsGradWSize));
  void* wsGradWData = manager.allocateDevMem(wsGradWSize);

  // Allocate workspace for gradI
  size_t wsGradISize;
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnGetConvolutionBackwardDataWorkspaceSize),
      cudnnGetConvolutionBackwardDataWorkspaceSize(*handle, dw, dz, conv, dx, algoGradI, &wsGradISize));
  void* wsGradIData = manager.allocateDevMem(wsGradISize);

  // Calculate gradB if present
  if (gradB != nullptr) {
    CudnnTensor db;
    db.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(gradB->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnConvolutionBackwardBias),
        cudnnConvolutionBackwardBias(*handle, alpha, dz, gradO->specialBuffer(), beta, db, gradB->specialBuffer()));
  }

  // Calculate gradW
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardFilter),
      cudnnConvolutionBackwardFilter(*handle, alpha, x, input->specialBuffer(), dz, gradO->specialBuffer(),
                                     conv, algoGradW, wsGradWData, wsGradWSize, beta, dw, gradW->specialBuffer()));

  // Calculate gradI
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionBackwardData),
      cudnnConvolutionBackwardData(*handle, alpha, dw, weights->specialBuffer(), dz, gradO->specialBuffer(),
                                   conv, algoGradI, wsGradIData, wsGradISize, beta, dx, gradI->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "conv1dBpCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({gradI, gradW, gradB}, {input, weights, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv1d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 3, 0,
               "CONV1D CUDNN OP: rank of input array must be equal to 3, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 3, 0,
               "CONV1D CUDNN OP: rank of weights array must be equal to 3, but got %i instead!", weights->rankOf());

  LongType kW = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType sW = INT_ARG(1);
  LongType pW = INT_ARG(2);
  LongType dW = INT_ARG(3);
  int paddingMode = INT_ARG(4);
  int isNCW = block.getIArguments()->size() > 5 ? (INT_ARG(5) == 0 ? 1 : 0) : 1;
  int wFormat = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;

  const LongType bS = input->sizeAt(0);
  const LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  const LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  const LongType oC = isNCW ? output->sizeAt(1) : output->sizeAt(2);
  const LongType oW = isNCW ? output->sizeAt(2) : output->sizeAt(1);

  if (paddingMode)
    {
      const LongType eKW = (kW - 1) * dW + 1;
      pW = ((oW - 1) * sW + eKW - iW) / 2;
    }

  // Reshape weights to cuDNN format [oC, iC, 1, kW]
  std::unique_ptr<NDArray> tmpWeights;
  NDArray* newWeights = weights;

  if (0 == wFormat) {
    // [kW, iC, oC] -> [oC, iC, kW]
    std::vector<LongType> newShape = {oC, iC, kW};
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    newWeights = tmpWeights.get();
    std::vector<LongType> permDims = {2, 1, 0};
    NDArray* permuted = weights->permute(permDims, true, true);
    newWeights->assign(permuted);
    delete permuted;
  }

  conv1dCUDNN(block.launchContext(), input, newWeights, bias, output, kW, sW, pW, dW, paddingMode, isNCW, wFormat);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv1d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

  int paddingMode = INT_ARG(4);

  Requirements req("CUDNN CONV1D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv1d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  REQUIRE_TRUE(input->rankOf() == 3, 0,
               "CONV1D_BP CUDNN OP: rank of input array must be equal to 3, but got %i instead!", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 3, 0,
               "CONV1D_BP CUDNN OP: rank of weights array must be equal to 3, but got %i instead!", weights->rankOf());
  REQUIRE_TRUE(gradO->rankOf() == 3, 0,
               "CONV1D_BP CUDNN OP: rank of gradO array must be equal to 3, but got %i instead!", gradO->rankOf());

  LongType kW = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));
  LongType sW = INT_ARG(1);
  LongType pW = INT_ARG(2);
  LongType dW = INT_ARG(3);
  int paddingMode = INT_ARG(4);
  int isNCW = block.getIArguments()->size() > 5 ? (INT_ARG(5) == 0 ? 1 : 0) : 1;
  int wFormat = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;

  const LongType bS = input->sizeAt(0);
  const LongType iC = isNCW ? input->sizeAt(1) : input->sizeAt(2);
  const LongType iW = isNCW ? input->sizeAt(2) : input->sizeAt(1);
  const LongType oC = isNCW ? gradO->sizeAt(1) : gradO->sizeAt(2);
  const LongType oW = isNCW ? gradO->sizeAt(2) : gradO->sizeAt(1);

  if (paddingMode)
    {
      const LongType eKW = (kW - 1) * dW + 1;
      pW = ((oW - 1) * sW + eKW - iW) / 2;
    }

  std::unique_ptr<NDArray> tmpWeights, tmpGradW;
  NDArray *newWeights = weights, *newGradW = gradW;

  if (0 == wFormat) {
    std::vector<LongType> newShape = {oC, iC, kW};
    tmpWeights.reset(new NDArray(weights->ordering(), newShape, weights->dataType(), weights->getContext()));
    tmpGradW.reset(new NDArray(gradW->ordering(), newShape, gradW->dataType(), gradW->getContext()));
    newWeights = tmpWeights.get();
    newGradW = tmpGradW.get();
    std::vector<LongType> permDims = {2, 1, 0};
    NDArray* permuted = weights->permute(permDims, true, true);
    newWeights->assign(permuted);
    delete permuted;
  }

  conv1dBpCUDNN(block.launchContext(), input, newWeights, gradO, gradI, newGradW, gradB,
                kW, sW, pW, dW, paddingMode, isNCW, wFormat);

  if (0 == wFormat) {
    std::vector<LongType> permDims = {2, 1, 0};
    newGradW->permutei(permDims, false, false);
    gradW->assign(newGradW);
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(conv1d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  int paddingMode = INT_ARG(4);

  Requirements req("CUDNN CONV1D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2), {HALF, FLOAT32, DOUBLE}) &&
  req.expectNotEq(makeInfoVariable(paddingMode, "paddingMode"), 2) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 3);

  if (bias) {
    req.expectIn(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "#bias"), {HALF, FLOAT32, DOUBLE});
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
