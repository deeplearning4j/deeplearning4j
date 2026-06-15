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
// cuDNN-based global pooling operations
// Global Average Pooling and Global Max Pooling
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 2D using cuDNN
static void globalAvgPool2dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, bool isNCHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(3));
  const int H = isNCHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int W = isNCHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));

  PointersManager manager(context, __func__);

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // Input tensor descriptor
  CudnnTensor xDesc;
  xDesc.set4D(format, dataType, bS, C, H, W);

  // Output tensor descriptor - [bS, C, 1, 1]
  CudnnTensor yDesc;
  yDesc.set4D(format, dataType, bS, C, 1, 1);

  // Create pooling descriptor - global pooling means kernel size equals spatial dims
  PoolingDesc poolDesc;
  poolDesc.set2D(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_PROPAGATE_NAN, H, W, 0, 0, 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, poolDesc, alpha, xDesc, input->specialBuffer(),
                          beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalAvgPool2dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 2D using cuDNN
static void globalMaxPool2dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, bool isNCHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(3));
  const int H = isNCHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int W = isNCHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));

  PointersManager manager(context, __func__);

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // Input tensor descriptor
  CudnnTensor xDesc;
  xDesc.set4D(format, dataType, bS, C, H, W);

  // Output tensor descriptor - [bS, C, 1, 1]
  CudnnTensor yDesc;
  yDesc.set4D(format, dataType, bS, C, 1, 1);

  // Create pooling descriptor - global pooling means kernel size equals spatial dims
  PoolingDesc poolDesc;
  poolDesc.set2D(CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, H, W, 0, 0, 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, poolDesc, alpha, xDesc, input->specialBuffer(),
                          beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalMaxPool2dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 2D backward
static void globalAvgPool2dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                                    NDArray* output, NDArray* gradI, bool isNCHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(3));
  const int H = isNCHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int W = isNCHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));

  PointersManager manager(context, __func__);

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // Input tensor descriptor [bS, C, H, W]
  CudnnTensor xDesc;
  xDesc.set4D(format, dataType, bS, C, H, W);

  // Output tensor descriptor [bS, C, 1, 1]
  CudnnTensor yDesc;
  yDesc.set4D(format, dataType, bS, C, 1, 1);

  // Gradient tensor descriptors
  CudnnTensor dxDesc, dyDesc;
  dxDesc.set4D(format, dataType, bS, C, H, W);
  dyDesc.set4D(format, dataType, bS, C, 1, 1);

  // Create pooling descriptor
  PoolingDesc poolDesc;
  poolDesc.set2D(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_PROPAGATE_NAN, H, W, 0, 0, 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI}, {input, gradO, output});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingBackward),
      cudnnPoolingBackward(*handle, poolDesc, alpha,
                           yDesc, output->specialBuffer(),
                           dyDesc, gradO->specialBuffer(),
                           xDesc, input->specialBuffer(),
                           beta, dxDesc, gradI->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalAvgPool2dBpCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({gradI}, {input, gradO, output});
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 2D backward
static void globalMaxPool2dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                                    NDArray* output, NDArray* gradI, bool isNCHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(3));
  const int H = isNCHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int W = isNCHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));

  PointersManager manager(context, __func__);

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // Input tensor descriptor [bS, C, H, W]
  CudnnTensor xDesc;
  xDesc.set4D(format, dataType, bS, C, H, W);

  // Output tensor descriptor [bS, C, 1, 1]
  CudnnTensor yDesc;
  yDesc.set4D(format, dataType, bS, C, 1, 1);

  // Gradient tensor descriptors
  CudnnTensor dxDesc, dyDesc;
  dxDesc.set4D(format, dataType, bS, C, H, W);
  dyDesc.set4D(format, dataType, bS, C, 1, 1);

  // Create pooling descriptor
  PoolingDesc poolDesc;
  poolDesc.set2D(CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, H, W, 0, 0, 1, 1);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI}, {input, gradO, output});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingBackward),
      cudnnPoolingBackward(*handle, poolDesc, alpha,
                           yDesc, output->specialBuffer(),
                           dyDesc, gradO->specialBuffer(),
                           xDesc, input->specialBuffer(),
                           beta, dxDesc, gradI->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalMaxPool2dBpCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({gradI}, {input, gradO, output});
}

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 3D using cuDNN
static void globalAvgPool3dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, bool isNCDHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCDHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(4));
  const int D = isNCDHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int H = isNCDHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));
  const int W = isNCDHW ? static_cast<int>(input->sizeAt(4)) : static_cast<int>(input->sizeAt(3));

  PointersManager manager(context, __func__);

  // Input tensor descriptor 5D
  std::vector<int> xDims = {bS, C, D, H, W};
  std::vector<int> xStrides = {C * D * H * W, D * H * W, H * W, W, 1};
  if (!isNCDHW) {
    xDims = {bS, D, H, W, C};
    xStrides = {D * H * W * C, H * W * C, W * C, C, 1};
  }

  CudnnTensor xDesc;
  xDesc.set(dataType, 5, xDims.data(), xStrides.data());

  // Output tensor descriptor 5D - [bS, C, 1, 1, 1]
  std::vector<int> yDims = {bS, C, 1, 1, 1};
  std::vector<int> yStrides = {C, 1, 1, 1, 1};
  if (!isNCDHW) {
    yDims = {bS, 1, 1, 1, C};
    yStrides = {C, C, C, C, 1};
  }

  CudnnTensor yDesc;
  yDesc.set(dataType, 5, yDims.data(), yStrides.data());

  // Create pooling descriptor - 3D global pooling
  PoolingDesc poolDesc;
  int windowDims[3] = {D, H, W};
  int padding[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  poolDesc.set(CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_PROPAGATE_NAN, 3, &windowDims[0], &padding[0], &stride[0]);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, poolDesc, alpha, xDesc, input->specialBuffer(),
                          beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalAvgPool3dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 3D using cuDNN
static void globalMaxPool3dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, bool isNCDHW) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // Get dimensions
  const int bS = static_cast<int>(input->sizeAt(0));
  const int C = isNCDHW ? static_cast<int>(input->sizeAt(1)) : static_cast<int>(input->sizeAt(4));
  const int D = isNCDHW ? static_cast<int>(input->sizeAt(2)) : static_cast<int>(input->sizeAt(1));
  const int H = isNCDHW ? static_cast<int>(input->sizeAt(3)) : static_cast<int>(input->sizeAt(2));
  const int W = isNCDHW ? static_cast<int>(input->sizeAt(4)) : static_cast<int>(input->sizeAt(3));

  PointersManager manager(context, __func__);

  // Input tensor descriptor 5D
  std::vector<int> xDims = {bS, C, D, H, W};
  std::vector<int> xStrides = {C * D * H * W, D * H * W, H * W, W, 1};
  if (!isNCDHW) {
    xDims = {bS, D, H, W, C};
    xStrides = {D * H * W * C, H * W * C, W * C, C, 1};
  }

  CudnnTensor xDesc;
  xDesc.set(dataType, 5, xDims.data(), xStrides.data());

  // Output tensor descriptor 5D - [bS, C, 1, 1, 1]
  std::vector<int> yDims = {bS, C, 1, 1, 1};
  std::vector<int> yStrides = {C, 1, 1, 1, 1};
  if (!isNCDHW) {
    yDims = {bS, 1, 1, 1, C};
    yStrides = {C, C, C, C, 1};
  }

  CudnnTensor yDesc;
  yDesc.set(dataType, 5, yDims.data(), yStrides.data());

  // Create pooling descriptor - 3D global pooling
  PoolingDesc poolDesc;
  int windowDims[3] = {D, H, W};
  int padding[3] = {0, 0, 0};
  int stride[3] = {1, 1, 1};
  poolDesc.set(CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 3, &windowDims[0], &padding[0], &stride[0]);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, poolDesc, alpha, xDesc, input->specialBuffer(),
                          beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "globalMaxPool3dCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 2D platform implementation
PLATFORM_IMPL(avgpool2d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  globalAvgPool2dCUDNN(block.launchContext(), input, output, isNCHW);

  return Status::OK;
}

PLATFORM_CHECK(avgpool2d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN GLOBAL_AVGPOOL2D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 2D platform implementation
PLATFORM_IMPL(maxpool2d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  globalMaxPool2dCUDNN(block.launchContext(), input, output, isNCHW);

  return Status::OK;
}

PLATFORM_CHECK(maxpool2d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN GLOBAL_MAXPOOL2D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 2D backward platform implementation
PLATFORM_IMPL(avgpool2d_global_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  const bool isNCHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  // Compute forward pass result for backward
  NDArray output(gradO->shapeInfo(), false, block.launchContext(), true);
  globalAvgPool2dCUDNN(block.launchContext(), input, &output, isNCHW);

  globalAvgPool2dBpCUDNN(block.launchContext(), input, gradO, &output, gradI, isNCHW);

  return Status::OK;
}

PLATFORM_CHECK(avgpool2d_global_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  Requirements req("CUDNN GLOBAL_AVGPOOL2D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 2D backward platform implementation
PLATFORM_IMPL(maxpool2d_global_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  const bool isNCHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  // Compute forward pass result for backward
  NDArray output(gradO->shapeInfo(), false, block.launchContext(), true);
  globalMaxPool2dCUDNN(block.launchContext(), input, &output, isNCHW);

  globalMaxPool2dBpCUDNN(block.launchContext(), input, gradO, &output, gradI, isNCHW);

  return Status::OK;
}

PLATFORM_CHECK(maxpool2d_global_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  Requirements req("CUDNN GLOBAL_MAXPOOL2D_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Global Average Pooling 3D platform implementation
PLATFORM_IMPL(avgpool3d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCDHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  globalAvgPool3dCUDNN(block.launchContext(), input, output, isNCDHW);

  return Status::OK;
}

PLATFORM_CHECK(avgpool3d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN GLOBAL_AVGPOOL3D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Global Max Pooling 3D platform implementation
PLATFORM_IMPL(maxpool3d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const bool isNCDHW = block.getIArguments()->size() > 0 ? I_ARG(0) == 0 : true;

  globalMaxPool3dCUDNN(block.launchContext(), input, output, isNCDHW);

  return Status::OK;
}

PLATFORM_CHECK(maxpool3d_global, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN GLOBAL_MAXPOOL3D OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 5) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
