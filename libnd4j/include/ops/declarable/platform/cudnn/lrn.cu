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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// LRN descriptor RAII wrapper
struct LRNDesc {
  MOVEONLY_DESC_FULL_IMPL(LRNDesc, LRNDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetLRNDescriptor),
                            cudnnSetLRNDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
static void lrnCUDNN(const LaunchContext* context, NDArray* input, NDArray* output,
                     const int depth, const double bias, const double alpha, const double beta) {
  // cuDNN LRN expects 4D tensor in NCHW format
  // Input shape should be [batch, channels, height, width]

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // cuDNN LRN uses lrnN = 2 * depth + 1 (size of the normalization window)
  // The formula is: y = x / (bias + (alpha/lrnN) * sum(x^2))^beta
  // But cuDNN expects: y = x / (k + alpha * sum(x^2))^beta
  // So we need to adjust alpha accordingly
  const unsigned int lrnN = 2 * depth + 1;
  const double lrnAlpha = alpha;  // cuDNN handles the division by lrnN internally
  const double lrnBeta = beta;
  const double lrnK = bias;

  // Get tensor dimensions
  const int N = static_cast<int>(input->sizeAt(0));
  const int C = static_cast<int>(input->sizeAt(1));
  const int H = static_cast<int>(input->sizeAt(2));
  const int W = static_cast<int>(input->sizeAt(3));

  // Set up tensor descriptors
  CudnnTensor x, y;
  x.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);
  y.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);

  // Set up LRN descriptor
  LRNDesc lrnDesc;
  lrnDesc.set(lrnN, lrnAlpha, lrnBeta, lrnK);

  // Scaling factors
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnLRNCrossChannelForward),
      cudnnLRNCrossChannelForward(*handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                  ptrAlpha, x, input->specialBuffer(),
                                  ptrBeta, y, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "lrnCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
static void lrnBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO,
                       NDArray* gradI, const int depth, const double bias,
                       const double alpha, const double beta) {
  // cuDNN LRN backward

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const unsigned int lrnN = 2 * depth + 1;
  const double lrnAlpha = alpha;
  const double lrnBeta = beta;
  const double lrnK = bias;

  // Get tensor dimensions
  const int N = static_cast<int>(input->sizeAt(0));
  const int C = static_cast<int>(input->sizeAt(1));
  const int H = static_cast<int>(input->sizeAt(2));
  const int W = static_cast<int>(input->sizeAt(3));

  // Set up tensor descriptors
  CudnnTensor x, y, dy, dx;
  x.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);
  y.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);
  dy.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);
  dx.set4D(CUDNN_TENSOR_NCHW, dataType, N, C, H, W);

  // Set up LRN descriptor
  LRNDesc lrnDesc;
  lrnDesc.set(lrnN, lrnAlpha, lrnBeta, lrnK);

  // Scaling factors
  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* ptrAlpha = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta = gradI->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  // We need the forward output for backward pass
  // Create a temporary array to hold the forward pass result
  NDArray lrnOutput(input->shapeInfo(), false, const_cast<LaunchContext*>(context), true);

  NDArray::prepareSpecialUse({&lrnOutput}, {input});

  // First, run forward pass to get output (needed for backward)
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnLRNCrossChannelForward),
      cudnnLRNCrossChannelForward(*handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                  ptrAlpha, x, input->specialBuffer(),
                                  ptrBeta, y, lrnOutput.specialBuffer()));

  NDArray::registerSpecialUse({&lrnOutput}, {input});

  NDArray::prepareSpecialUse({gradI}, {input, gradO, &lrnOutput});

  // Now run backward pass
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnLRNCrossChannelBackward),
      cudnnLRNCrossChannelBackward(*handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                   ptrAlpha, y, lrnOutput.specialBuffer(),
                                   dy, gradO->specialBuffer(),
                                   x, input->specialBuffer(),
                                   ptrBeta, dx, gradI->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "lrnBpCUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({gradI}, {input, gradO, &lrnOutput});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lrn, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "LRN CUDNN OP: Input rank of 4 expected, but got %i instead", input->rankOf());

  const double bias = T_ARG(0);
  const double alpha = T_ARG(1);
  const double beta = T_ARG(2);
  const int depth = INT_ARG(0);

  lrnCUDNN(block.launchContext(), input, output, depth, bias, alpha, beta);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(lrn, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int depth = INT_ARG(0);

  Requirements req("CUDNN LRN OP");
  // cuDNN LRN supports FLOAT16, FLOAT32, FLOAT64
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
  // cuDNN requires lrnN (2*depth+1) to be in range [1, 16]
  req.expectGreaterEq(makeInfoVariable(depth, "depth"), 0) &&
  req.expectLessEq(makeInfoVariable(2 * depth + 1, "lrnN"), 16);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lrn_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "LRN_BP CUDNN OP: Input rank of 4 expected, but got %i instead", input->rankOf());
  REQUIRE_TRUE(input->isSameShape(gradO), 0,
               "LRN_BP CUDNN OP: Both input and grad_output should have the same shape!");

  const double bias = T_ARG(0);
  const double alpha = T_ARG(1);
  const double beta = T_ARG(2);
  const int depth = INT_ARG(0);

  lrnBpCUDNN(block.launchContext(), input, gradO, gradI, depth, bias, alpha, beta);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(lrn_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  const int depth = INT_ARG(0);

  Requirements req("CUDNN LRN_BP OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), {HALF, FLOAT32, DOUBLE}) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
  req.expectGreaterEq(makeInfoVariable(depth, "depth"), 0) &&
  req.expectLessEq(makeInfoVariable(2 * depth + 1, "lrnN"), 16);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
