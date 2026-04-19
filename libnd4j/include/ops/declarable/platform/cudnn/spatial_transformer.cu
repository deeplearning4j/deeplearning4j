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
// cuDNN-based Spatial Transformer Network operations
// Uses cudnnSpatialTfGridGeneratorForward and cudnnSpatialTfSamplerForward
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// RAII wrapper for Spatial Transformer descriptor
struct SpatialTransformerDesc {
  MOVEONLY_DESC_FULL_IMPL(SpatialTransformerDesc, SpatialTransformerDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetSpatialTransformerNdDescriptor),
                            cudnnSetSpatialTransformerNdDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
// Generate sampling grid from affine transformation matrix
// theta: [N, 2, 3] affine transformation matrix
// grid: [N, H, W, 2] sampling grid (x, y coordinates)
static void gridGeneratorCUDNN(const LaunchContext* context, NDArray* theta, NDArray* grid) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(theta->dataType());

  const int N = static_cast<int>(grid->sizeAt(0));
  const int H = static_cast<int>(grid->sizeAt(1));
  const int W = static_cast<int>(grid->sizeAt(2));

  PointersManager manager(context, __func__);

  // Create spatial transformer descriptor
  SpatialTransformerDesc stDesc;
  int dims[4] = {N, 1, H, W};  // [N, C, H, W] - C=1 for grid generation
  stDesc.set(CUDNN_SAMPLER_BILINEAR, dataType, 4, dims);

  NDArray::prepareSpecialUse({grid}, {theta});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSpatialTfGridGeneratorForward),
      cudnnSpatialTfGridGeneratorForward(*handle, stDesc, theta->specialBuffer(), grid->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("gridGeneratorCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({grid}, {theta});
}

//////////////////////////////////////////////////////////////////////////
// Backward pass for grid generator
static void gridGeneratorBpCUDNN(const LaunchContext* context, NDArray* gradGrid, NDArray* gradTheta) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(gradGrid->dataType());

  const int N = static_cast<int>(gradGrid->sizeAt(0));
  const int H = static_cast<int>(gradGrid->sizeAt(1));
  const int W = static_cast<int>(gradGrid->sizeAt(2));

  PointersManager manager(context, __func__);

  // Create spatial transformer descriptor
  SpatialTransformerDesc stDesc;
  int dims[4] = {N, 1, H, W};
  stDesc.set(CUDNN_SAMPLER_BILINEAR, dataType, 4, dims);

  NDArray::prepareSpecialUse({gradTheta}, {gradGrid});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSpatialTfGridGeneratorBackward),
      cudnnSpatialTfGridGeneratorBackward(*handle, stDesc, gradGrid->specialBuffer(), gradTheta->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("gridGeneratorBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradTheta}, {gradGrid});
}

//////////////////////////////////////////////////////////////////////////
// Sample from input using grid (bilinear interpolation)
// input: [N, C, H_in, W_in]
// grid: [N, H_out, W_out, 2]
// output: [N, C, H_out, W_out]
static void gridSamplerCUDNN(const LaunchContext* context, NDArray* input, NDArray* grid, NDArray* output) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int N = static_cast<int>(input->sizeAt(0));
  const int C = static_cast<int>(input->sizeAt(1));
  const int H_in = static_cast<int>(input->sizeAt(2));
  const int W_in = static_cast<int>(input->sizeAt(3));
  const int H_out = static_cast<int>(grid->sizeAt(1));
  const int W_out = static_cast<int>(grid->sizeAt(2));

  PointersManager manager(context, __func__);

  // Input tensor descriptor
  CudnnTensor xDesc;
  xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H_in, W_in);

  // Output tensor descriptor
  CudnnTensor yDesc;
  yDesc.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H_out, W_out);

  // Create spatial transformer descriptor
  SpatialTransformerDesc stDesc;
  int dims[4] = {N, C, H_out, W_out};
  stDesc.set(CUDNN_SAMPLER_BILINEAR, dataType, 4, dims);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, grid});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSpatialTfSamplerForward),
      cudnnSpatialTfSamplerForward(*handle, stDesc, alpha,
                                    xDesc, input->specialBuffer(),
                                    grid->specialBuffer(),
                                    beta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("gridSamplerCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input, grid});
}

//////////////////////////////////////////////////////////////////////////
// Backward pass for grid sampler
static void gridSamplerBpCUDNN(const LaunchContext* context,
                                NDArray* input, NDArray* grid, NDArray* gradO,
                                NDArray* gradI, NDArray* gradGrid) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int N = static_cast<int>(input->sizeAt(0));
  const int C = static_cast<int>(input->sizeAt(1));
  const int H_in = static_cast<int>(input->sizeAt(2));
  const int W_in = static_cast<int>(input->sizeAt(3));
  const int H_out = static_cast<int>(grid->sizeAt(1));
  const int W_out = static_cast<int>(grid->sizeAt(2));

  PointersManager manager(context, __func__);

  // Tensor descriptors
  CudnnTensor xDesc, dxDesc, dyDesc;
  xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H_in, W_in);
  dxDesc.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H_in, W_in);
  dyDesc.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H_out, W_out);

  // Create spatial transformer descriptor
  SpatialTransformerDesc stDesc;
  int dims[4] = {N, C, H_out, W_out};
  stDesc.set(CUDNN_SAMPLER_BILINEAR, dataType, 4, dims);

  // Scaling parameters
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradGrid}, {input, grid, gradO});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnSpatialTfSamplerBackward),
      cudnnSpatialTfSamplerBackward(*handle, stDesc,
                                     alpha, xDesc, input->specialBuffer(),
                                     beta, dxDesc, gradI->specialBuffer(),
                                     alpha, dyDesc, gradO->specialBuffer(),
                                     grid->specialBuffer(),
                                     beta, gradGrid->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("gridSamplerBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradI, gradGrid}, {input, grid, gradO});
}

//////////////////////////////////////////////////////////////////////////
// Affine Grid Generator - generates sampling grid from affine matrix
PLATFORM_IMPL(affine_grid, ENGINE_CUDA) {
  auto theta = INPUT_VARIABLE(0);  // [N, 2, 3] affine transformation matrix
  auto output = OUTPUT_VARIABLE(0);  // [N, H, W, 2] sampling grid

  // Output size from integer arguments or input
  int H = I_ARG(0);
  int W = I_ARG(1);

  gridGeneratorCUDNN(block.launchContext(), theta, output);

  return Status::OK;
}

PLATFORM_CHECK(affine_grid, ENGINE_CUDA) {
  auto theta = INPUT_VARIABLE(0);

  Requirements req("CUDNN AFFINE_GRID OP");
  req.expectIn(makeInfoVariable(theta->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(theta->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(theta->sizeAt(1), "theta dim 1"), 2) &&
      req.expectEq(makeInfoVariable(theta->sizeAt(2), "theta dim 2"), 3) &&
      req.expectEq(makeInfoVariable(theta->ordering(), ORDERING_MSG_INPUT0), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Affine Grid Generator Backward
PLATFORM_IMPL(affine_grid_bp, ENGINE_CUDA) {
  auto theta = INPUT_VARIABLE(0);
  auto gradGrid = INPUT_VARIABLE(1);

  auto gradTheta = OUTPUT_VARIABLE(0);

  gridGeneratorBpCUDNN(block.launchContext(), gradGrid, gradTheta);

  return Status::OK;
}

PLATFORM_CHECK(affine_grid_bp, ENGINE_CUDA) {
  auto theta = INPUT_VARIABLE(0);
  auto gradGrid = INPUT_VARIABLE(1);

  Requirements req("CUDNN AFFINE_GRID_BP OP");
  req.expectIn(makeInfoVariable(theta->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(gradGrid->rankOf(), RANK_MSG_INPUT1), 4);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Grid Sampler - samples from input using grid coordinates
PLATFORM_IMPL(grid_sample, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);  // [N, C, H_in, W_in]
  auto grid = INPUT_VARIABLE(1);   // [N, H_out, W_out, 2]

  auto output = OUTPUT_VARIABLE(0);  // [N, C, H_out, W_out]

  gridSamplerCUDNN(block.launchContext(), input, grid, output);

  return Status::OK;
}

PLATFORM_CHECK(grid_sample, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto grid = INPUT_VARIABLE(1);

  Requirements req("CUDNN GRID_SAMPLE OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(grid->rankOf(), RANK_MSG_INPUT1), 4) &&
      req.expectEq(makeInfoVariable(grid->sizeAt(3), "grid last dim"), 2) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(grid->ordering(), ORDERING_MSG_INPUT1), 'c');

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Grid Sampler Backward
PLATFORM_IMPL(grid_sample_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto grid = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradGrid = OUTPUT_VARIABLE(1);

  gridSamplerBpCUDNN(block.launchContext(), input, grid, gradO, gradI, gradGrid);

  return Status::OK;
}

PLATFORM_CHECK(grid_sample_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto grid = INPUT_VARIABLE(1);

  Requirements req("CUDNN GRID_SAMPLE_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(grid->rankOf(), RANK_MSG_INPUT1), 4);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
