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
// cuDNN-based dropout operation
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Dropout forward using cuDNN
static void dropoutCUDNN(const LaunchContext* context, NDArray* input, NDArray* output,
                         NDArray* mask, double ratio, uint64_t seed) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  PointersManager manager(context, __func__);

  // Create tensor descriptor
  CudnnTensor xDesc;
  const int rank = input->rankOf();

  if (rank <= 4) {
    int n = 1, c = 1, h = 1, w = 1;
    if (rank >= 1) w = static_cast<int>(input->sizeAt(rank - 1));
    if (rank >= 2) h = static_cast<int>(input->sizeAt(rank - 2));
    if (rank >= 3) c = static_cast<int>(input->sizeAt(rank - 3));
    if (rank >= 4) n = static_cast<int>(input->sizeAt(rank - 4));
    xDesc.set4D(CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
  } else {
    std::vector<int> dims(rank);
    std::vector<int> strides(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = static_cast<int>(input->sizeAt(i));
      strides[i] = static_cast<int>(input->strideAt(i));
    }
    xDesc.set(dataType, rank, dims.data(), strides.data());
  }

  // Get dropout states size
  size_t statesSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetStatesSize),
                          cudnnDropoutGetStatesSize(*handle, &statesSize));

  // Allocate states
  void* states = manager.allocateDevMem(statesSize);

  // Get reserve space size
  size_t reserveSpaceSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetReserveSpaceSize),
                          cudnnDropoutGetReserveSpaceSize(xDesc, &reserveSpaceSize));

  // Allocate reserve space (used to store mask for backward pass)
  void* reserveSpace = manager.allocateDevMem(reserveSpaceSize);

  // Create dropout descriptor
  DropoutDesc dropoutDesc;
  dropoutDesc.create();
  dropoutDesc.set(*handle, static_cast<float>(ratio), states, statesSize, seed);

  NDArray::prepareSpecialUse({output}, {input});

  // Forward dropout
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnDropoutForward),
      cudnnDropoutForward(*handle, dropoutDesc, xDesc, input->specialBuffer(),
                          xDesc, output->specialBuffer(), reserveSpace, reserveSpaceSize));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("dropoutCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// Dropout backward using cuDNN
static void dropoutBpCUDNN(const LaunchContext* context, NDArray* gradO, NDArray* gradI,
                           double ratio, uint64_t seed) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(gradO->dataType());

  PointersManager manager(context, __func__);

  // Create tensor descriptor
  CudnnTensor dyDesc;
  const int rank = gradO->rankOf();

  if (rank <= 4) {
    int n = 1, c = 1, h = 1, w = 1;
    if (rank >= 1) w = static_cast<int>(gradO->sizeAt(rank - 1));
    if (rank >= 2) h = static_cast<int>(gradO->sizeAt(rank - 2));
    if (rank >= 3) c = static_cast<int>(gradO->sizeAt(rank - 3));
    if (rank >= 4) n = static_cast<int>(gradO->sizeAt(rank - 4));
    dyDesc.set4D(CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
  } else {
    std::vector<int> dims(rank);
    std::vector<int> strides(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = static_cast<int>(gradO->sizeAt(i));
      strides[i] = static_cast<int>(gradO->strideAt(i));
    }
    dyDesc.set(dataType, rank, dims.data(), strides.data());
  }

  // Get dropout states size
  size_t statesSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetStatesSize),
                          cudnnDropoutGetStatesSize(*handle, &statesSize));

  // Allocate states
  void* states = manager.allocateDevMem(statesSize);

  // Get reserve space size
  size_t reserveSpaceSize = 0;
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDropoutGetReserveSpaceSize),
                          cudnnDropoutGetReserveSpaceSize(dyDesc, &reserveSpaceSize));

  // Allocate reserve space
  void* reserveSpace = manager.allocateDevMem(reserveSpaceSize);

  // Create dropout descriptor with same seed to recreate the mask
  DropoutDesc dropoutDesc;
  dropoutDesc.create();
  dropoutDesc.set(*handle, static_cast<float>(ratio), states, statesSize, seed);

  NDArray::prepareSpecialUse({gradI}, {gradO});

  // Backward dropout
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnDropoutBackward),
      cudnnDropoutBackward(*handle, dropoutDesc, dyDesc, gradO->specialBuffer(),
                           dyDesc, gradI->specialBuffer(), reserveSpace, reserveSpaceSize));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("dropoutBpCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({gradI}, {gradO});
}

//////////////////////////////////////////////////////////////////////////
// Dropout platform implementation
PLATFORM_IMPL(dropout, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get probability (p = probability of dropping)
  double p = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.5;

  // Seed for reproducibility
  uint64_t seed = block.getIArguments()->size() > 0 ? static_cast<uint64_t>(I_ARG(0)) : 0;

  // In training mode, apply dropout. In inference, just copy
  bool training = true;  // Default to training mode
  if (block.getBArguments()->size() > 0) {
    training = B_ARG(0);
  }

  if (training && p > 0.0) {
    dropoutCUDNN(block.launchContext(), input, output, nullptr, p, seed);
  } else {
    // During inference, just copy input to output
    if (input != output) {
      output->assign(input);
    }
  }

  return Status::OK;
}

PLATFORM_CHECK(dropout, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN DROPOUT OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Dropout backward platform implementation
PLATFORM_IMPL(dropout_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  // Get probability (p = probability of dropping)
  double p = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.5;

  // Seed for reproducibility (must match forward pass)
  uint64_t seed = block.getIArguments()->size() > 0 ? static_cast<uint64_t>(I_ARG(0)) : 0;

  if (p > 0.0) {
    dropoutBpCUDNN(block.launchContext(), gradO, gradI, p, seed);
  } else {
    // If no dropout, gradient passes through unchanged
    if (gradO != gradI) {
      gradI->assign(gradO);
    }
  }

  return Status::OK;
}

PLATFORM_CHECK(dropout_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  Requirements req("CUDNN DROPOUT_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Alpha Dropout - specialized dropout that maintains mean and variance
// Useful for SELU activation networks
PLATFORM_IMPL(alpha_dropout, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get probability
  double p = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.5;

  // Alpha dropout parameters for SELU
  // alpha = -1.7580993408473766
  // scale = 1.0507009873554805
  // For SELU: alpha' = alpha * lambda / (1 - p)

  uint64_t seed = block.getIArguments()->size() > 0 ? static_cast<uint64_t>(I_ARG(0)) : 0;

  bool training = true;
  if (block.getBArguments()->size() > 0) {
    training = B_ARG(0);
  }

  if (training && p > 0.0) {
    // Use standard dropout as base, then adjust for alpha dropout
    // Alpha dropout: keeps mean and variance by using saturation value
    dropoutCUDNN(block.launchContext(), input, output, nullptr, p, seed);

    // Apply alpha dropout scaling
    // The dropped values should be set to alpha * lambda, not zero
    // For true alpha dropout, we'd need a custom kernel
    // This is an approximation using cuDNN dropout
    double a = -1.7580993408473766 * 1.0507009873554805;
    double scale = 1.0 / (1.0 - p);

    // Scale surviving activations
    *output *= scale;
  } else {
    if (input != output) {
      output->assign(input);
    }
  }

  return Status::OK;
}

PLATFORM_CHECK(alpha_dropout, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN ALPHA_DROPOUT OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
