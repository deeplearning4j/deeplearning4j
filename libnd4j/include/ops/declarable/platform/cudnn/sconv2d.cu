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

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Depthwise convolution helper for separable conv
static void sconv2dDepthwiseCUDNN(const LaunchContext* context, NDArray* input, NDArray* weightsDepth,
                                   NDArray* output, const LongType kH, const LongType kW, const LongType sH,
                                   const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                                   const LongType paddingMode, const bool isNCHW, const int wFormat) {
  // input [bS, iC, iH, iW] nchw or [bS, iH, iW, iC] nhwc
  // weightsDepth [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  // output [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc
  // oC = iC*mC

  LongType bS, iC, iH, iW, mC, oC, oH, oW;
  LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWmC, indWkH, indOoH);
  mC = weightsDepth->sizeAt(indWmC);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  PointersManager manager(context, __func__);

  // input descriptor
  CudnnTensor x;
  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // For depthwise conv, cuDNN expects weights in [oC, iC/groups, kH, kW] format
  // where groups = iC for depthwise. So effectively [iC*mC, 1, kH, kW] when mC=1
  // We need to permute weights to match cuDNN's expected format
  NDArray weightsPermuted;
  {
    std::vector<LongType> dims;
    if (wFormat == 0) {
      dims = {2, 3, 0, 1};  // [kH, kW, iC, mC] -> [iC, mC, kH, kW]
    } else if (wFormat == 1) {
      dims = {1, 0, 2, 3};  // [mC, iC, kH, kW] -> [iC, mC, kH, kW]
    } else {
      dims = {3, 0, 1, 2};  // [mC, kH, kW, iC] -> [iC, mC, kH, kW]
    }
    auto* pPtr = weightsDepth->permute(dims, false, false);
    auto* dPtr = pPtr->dup('c');
    weightsPermuted = std::move(*dPtr);
    delete dPtr;
    delete pPtr;
  }

  // weights descriptor - cuDNN depthwise expects [iC, mC, kH, kW]
  FilterDesc w;
  w.set4D(cudnnDataType(weightsPermuted.dataType()), CUDNN_TENSOR_NCHW, iC, mC, kH, kW);

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
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetConvolutionGroupCount), cudnnSetConvolutionGroupCount(conv, iC));

  // algorithm description
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;
  // During CUDA graph capture, use heuristic _v7 (no GPU benchmarks) to avoid capture invalidation.
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardAlgorithm_v7),
                            cudnnGetConvolutionForwardAlgorithm_v7(*handle, x, w, conv, z, 1, &count, &algoPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                            cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  }
  if (count == 0)
    throw cuda_exception::build("sconv2dDepthwiseCUDNN: cudnnFindConvolutionForwardAlgorithm failed", 0);
  algo = algoPerf.algo;

  // allocate workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardWorkspaceSize),
                          cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // scaling parameters
  static const float alpha32(1), beta32(0);
  static const double alpha64(1), beta64(0);
  const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, &weightsPermuted});

  // run calculation
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionForward),
      cudnnConvolutionForward(*handle, alpha, x, input->specialBuffer(), w, weightsPermuted.specialBuffer(), conv, algo,
                              wsData, wsSize, beta, z, output->specialBuffer()));

  NDArray::registerSpecialUse({output}, {input, &weightsPermuted});
}

//////////////////////////////////////////////////////////////////////////
// Pointwise (1x1) convolution helper for separable conv
static void sconv2dPointwiseCUDNN(const LaunchContext* context, NDArray* input, NDArray* weightsPoint,
                                   NDArray* bias, NDArray* output, const bool isNCHW, const int wFormat) {
  // input [bS, iC*mC, oH, oW] nchw or [bS, oH, oW, iC*mC] nhwc
  // weightsPoint [1, 1, iC*mC, oC], [oC, iC*mC, 1, 1], [oC, 1, 1, iC*mC]
  // bias [oC], may be nullptr
  // output [bS, oC, oH, oW] nchw or [bS, oH, oW, oC] nhwc

  const int indIOioC = isNCHW ? 1 : 3;
  const int indIiH = isNCHW ? 2 : 1;

  const LongType bS = input->sizeAt(0);
  const LongType iC = input->sizeAt(indIOioC);  // iC*mC from depthwise output
  const LongType iH = input->sizeAt(indIiH);
  const LongType iW = input->sizeAt(indIiH + 1);
  const LongType oC = output->sizeAt(indIOioC);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
  PointersManager manager(context, __func__);

  // input descriptor
  CudnnTensor x;
  x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);

  // Permute weights to cuDNN format [oC, iC, kH, kW]
  NDArray weightsPermuted;
  if (wFormat == 1) {
    // [oC, iC*mC, 1, 1] -> already in correct format
    auto* dPtr = weightsPoint->dup('c');
    weightsPermuted = std::move(*dPtr);
    delete dPtr;
  } else {
    std::vector<LongType> dims;
    if (wFormat == 0) {
      dims = {3, 2, 0, 1};  // [1, 1, iC*mC, oC] -> [oC, iC*mC, 1, 1]
    } else {
      dims = {0, 3, 1, 2};  // [oC, 1, 1, iC*mC] -> [oC, iC*mC, 1, 1]
    }
    auto* pPtr = weightsPoint->permute(dims, false, false);
    auto* dPtr = pPtr->dup('c');
    weightsPermuted = std::move(*dPtr);
    delete dPtr;
    delete pPtr;
  }

  // weights descriptor [oC, iC, 1, 1]
  FilterDesc w;
  w.set4D(cudnnDataType(weightsPermuted.dataType()), CUDNN_TENSOR_NCHW, oC, iC, 1, 1);

  // output descriptor
  CudnnTensor z;
  z.set4D(format, cudnnDataType(output->dataType()), bS, oC, iH, iW);

  // description of convolution (1x1, no padding, stride 1)
  ConvolutionDesc conv;
  conv.set2D(0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, cudnnDataType(output->dataType()));

  // algorithm description
  cudnnConvolutionFwdAlgo_t algo;
  cudnnConvolutionFwdAlgoPerf_t algoPerf;
  int count = 0;
  // During CUDA graph capture, use heuristic _v7 (no GPU benchmarks) to avoid capture invalidation.
  if (tl_graphExecutionActive) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardAlgorithm_v7),
                            cudnnGetConvolutionForwardAlgorithm_v7(*handle, x, w, conv, z, 1, &count, &algoPerf));
  } else {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnFindConvolutionForwardAlgorithm),
                            cudnnFindConvolutionForwardAlgorithm(*handle, x, w, conv, z, 1, &count, &algoPerf));
  }
  if (count == 0)
    throw cuda_exception::build("sconv2dPointwiseCUDNN: cudnnFindConvolutionForwardAlgorithm failed", 0);
  algo = algoPerf.algo;

  // allocate workspace
  size_t wsSize;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnGetConvolutionForwardWorkspaceSize),
                          cudnnGetConvolutionForwardWorkspaceSize(*handle, x, w, conv, z, algo, &wsSize));
  void* wsData = manager.allocateDevMem(wsSize);

  // scaling parameters
  static const float alpha32(1), beta32(0);
  static const double alpha64(1), beta64(0);
  const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, &weightsPermuted, bias});

  // run calculation
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnConvolutionForward),
      cudnnConvolutionForward(*handle, alpha, x, input->specialBuffer(), w, weightsPermuted.specialBuffer(), conv, algo,
                              wsData, wsSize, beta, z, output->specialBuffer()));

  // add bias if present
  if (bias != nullptr) {
    CudnnTensor b;
    b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor), cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha,
                                                                      z, output->specialBuffer()));
  }

  NDArray::registerSpecialUse({output}, {input, &weightsPermuted, bias});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(sconv2d, ENGINE_CUDA) {
  NDArray* input = INPUT_VARIABLE(0);         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  NDArray* weightsDepth = INPUT_VARIABLE(1);  // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  NDArray* weightsPoint = nullptr;            // [1, 1, iC*mC, oC], [oC, iC*mC, 1, 1], [oC, 1, 1, iC*mC]
  NDArray* bias = nullptr;                    // [oC], if weightsPoint=nullptr then oC = iC*mC

  NDArray* output = OUTPUT_NULLIFIED(0);      // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  if (block.width() == 3) {
    if ((INPUT_VARIABLE(2))->rankOf() == 4)
      weightsPoint = INPUT_VARIABLE(2);
    else
      bias = INPUT_VARIABLE(2);
  } else if (block.width() == 4) {
    weightsPoint = INPUT_VARIABLE(2);
    bias = INPUT_VARIABLE(3);
  }

  LongType kH = INT_ARG(0);
  LongType kW = INT_ARG(1);
  LongType sH = INT_ARG(2);
  LongType sW = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  LongType dH = INT_ARG(6);
  LongType dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  LongType bS, iC, iH, iW, mC, oC, oH, oW;
  LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWmC, indWkH, indOoH);
  mC = weightsDepth->sizeAt(indWmC);

  // Handle SAME padding
  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, isSameMode);

  // If iC == 1, this is equivalent to standard conv2d
  if (iC == 1) {
    ConvolutionUtils::conv2d(block, input, weightsDepth, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode,
                             isNCHW, wFormat);
    return Status::OK;
  }

  auto* contextPtr = block.launchContext();

  if (weightsPoint == nullptr) {
    // Only depthwise convolution, output = iC*mC channels
    sconv2dDepthwiseCUDNN(contextPtr, input, weightsDepth, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW, wFormat);

    // Add bias if present
    if (bias != nullptr) {
      auto handle = reinterpret_cast<cudnnHandle_t*>(contextPtr->getCuDnnHandle());
      auto stream = cudnnCaptureAwareStream(contextPtr->getCudaStream());
      CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

      static const float alpha32(1);
      static const double alpha64(1);
      const void* alpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);

      CudnnTensor b;
      b.set4D(CUDNN_TENSOR_NCHW, cudnnDataType(bias->dataType()), 1, oC, 1, 1);

      cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;
      CudnnTensor z;
      z.set4D(format, cudnnDataType(output->dataType()), bS, oC, oH, oW);

      NDArray::prepareSpecialUse({output}, {bias});
      CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnAddTensor), cudnnAddTensor(*handle, alpha, b, bias->specialBuffer(), alpha,
                                                                        z, output->specialBuffer()));
      NDArray::registerSpecialUse({output}, {bias});
    }
  } else {
    // Full separable convolution: depthwise + pointwise
    // Create intermediate buffer for depthwise output
    auto depthwiseOutShape = isNCHW ? std::vector<LongType>({bS, iC * mC, oH, oW})
                                    : std::vector<LongType>({bS, oH, oW, iC * mC});
    auto depthwiseOut = NDArrayFactory::create_(output->ordering(), depthwiseOutShape, output->dataType(), contextPtr);

    // Depthwise convolution
    sconv2dDepthwiseCUDNN(contextPtr, input, weightsDepth, depthwiseOut, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW, wFormat);

    // Pointwise (1x1) convolution
    sconv2dPointwiseCUDNN(contextPtr, depthwiseOut, weightsPoint, bias, output, isNCHW, wFormat);

    delete depthwiseOut;
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(sconv2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto weightsDepth = INPUT_VARIABLE(1);
  NDArray* weightsPoint = nullptr;
  NDArray* bias = nullptr;

  if (block.width() == 3) {
    if ((INPUT_VARIABLE(2))->rankOf() == 4)
      weightsPoint = INPUT_VARIABLE(2);
    else
      bias = INPUT_VARIABLE(2);
  } else if (block.width() == 4) {
    weightsPoint = INPUT_VARIABLE(2);
    bias = INPUT_VARIABLE(3);
  }

  auto output = OUTPUT_VARIABLE(0);

  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  const int indIOioC = isNCHW ? 1 : 3;
  const LongType iC = input->sizeAt(indIOioC);
  const int indWmC = wFormat == 0 ? 3 : 0;
  const LongType mC = weightsDepth->sizeAt(indWmC);

  Requirements req("CUDNN SCONV2D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectIn(makeInfoVariable(weightsDepth->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(weightsDepth->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(weightsDepth->rankOf(), RANK_MSG_INPUT1), 4) &&
      req.expectEq(makeInfoVariable(mC, "channel multiplier (mC)"), 1);  // cuDNN depthwise requires mC=1

  if (weightsPoint) {
    req.expectEq(makeInfoVariable(weightsPoint->dataType(), TYPE_MSG_INPUT2),
                 makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0)) &&
        req.expectEq(makeInfoVariable(weightsPoint->rankOf(), RANK_MSG_INPUT2), 4);
  }

  if (bias) {
    req.expectEq(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT_ "bias"),
                 makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0));
  }

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(sconv2d_bp, ENGINE_CUDA) {
  NDArray* input = INPUT_VARIABLE(0);         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  NDArray* gradO = INPUT_VARIABLE(1);         // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
  NDArray* weightsDepth = INPUT_VARIABLE(2);  // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  NDArray* weightsPoint = nullptr;            // [1, 1, iC*mC, oC], [oC, iC*mC, 1, 1], [oC, 1, 1, iC*mC]
  NDArray* bias = nullptr;                    // [oC]

  NDArray* gradI = OUTPUT_NULLIFIED(0);       // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  NDArray* gradWD = OUTPUT_NULLIFIED(1);      // [kH, kW, iC, mC], [mC, iC, kH, kW], [mC, kH, kW, iC]
  NDArray* gradWP = nullptr;                  // [1, 1, iC*mC, oC], [oC, iC*mC, 1, 1], [oC, 1, 1, iC*mC]
  NDArray* gradB = nullptr;                   // [oC]

  if (block.width() == 4) {
    if ((INPUT_VARIABLE(3))->rankOf() == 4) {
      weightsPoint = INPUT_VARIABLE(3);
      gradWP = OUTPUT_NULLIFIED(2);
    } else {
      bias = INPUT_VARIABLE(3);
      gradB = OUTPUT_NULLIFIED(2);
    }
  } else if (block.width() == 5) {
    weightsPoint = INPUT_VARIABLE(3);
    bias = INPUT_VARIABLE(4);
    gradWP = OUTPUT_NULLIFIED(2);
    gradB = OUTPUT_NULLIFIED(3);
  }

  LongType kH = INT_ARG(0);
  LongType kW = INT_ARG(1);
  LongType sH = INT_ARG(2);
  LongType sW = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  LongType dH = INT_ARG(6);
  LongType dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  // For backward pass, fall back to helper implementation
  // A full cuDNN implementation would require implementing backward passes for both stages
  if (weightsPoint) {
    LongType bS, iC, iH, iW, mC, oC, oH, oW;
    LongType indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                               indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weightsDepth->sizeAt(indWmC);

    auto resultFFShape = isNCHW ? std::vector<LongType>({bS, mC * iC, oH, oW}) : std::vector<LongType>({bS, oH, oW, mC * iC});
    auto resultFF = NDArrayFactory::create_(input->ordering(), resultFFShape, input->dataType(), block.launchContext());
    ConvolutionUtils::sconv2d(block, input, weightsDepth, nullptr, nullptr, resultFF, kH, kW, sH, sW, pH, pW, dH, dW,
                              isSameMode, isNCHW, wFormat);

    auto gradIDepthShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC * mC, oH, oW, 0, indIOioC, indIiH, indIiH + 1});
    auto gradIDepth = NDArrayFactory::create_(resultFF->ordering(), gradIDepthShape, resultFF->dataType(), block.launchContext());

    ConvolutionUtils::conv2dBP(block, resultFF, weightsPoint, bias, gradO, gradIDepth, gradWP, gradB, 1, 1, 1, 1, 0, 0,
                               1, 1, isSameMode, isNCHW, wFormat);

    gradO = gradIDepth;
    bias = gradB = nullptr;

    delete resultFF;

    ConvolutionUtils::depthwiseConv2dBP(block, input, weightsDepth, bias, gradO, gradI, gradWD, gradB, kH, kW, sH, sW, pH,
                                        pW, dH, dW, isSameMode, isNCHW, wFormat);

    delete gradO;
  } else {
    ConvolutionUtils::depthwiseConv2dBP(block, input, weightsDepth, bias, gradO, gradI, gradWD, gradB, kH, kW, sH, sW, pH,
                                        pW, dH, dW, isSameMode, isNCHW, wFormat);
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(sconv2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto weightsDepth = INPUT_VARIABLE(2);

  int wFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;
  const int indWmC = wFormat == 0 ? 3 : 0;
  const LongType mC = weightsDepth->sizeAt(indWmC);

  Requirements req("CUDNN SCONV2D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(weightsDepth->dataType(), TYPE_MSG_INPUT2)) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(mC, "channel multiplier (mC)"), 1);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
