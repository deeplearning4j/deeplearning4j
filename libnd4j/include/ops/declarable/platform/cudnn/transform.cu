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
// cuDNN-based tensor transformation operations
// Includes: scale, identity/copy, neg (negate)
//

#include <helpers/PointersManager.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// Activation descriptor RAII wrapper
struct ActivationDesc {
  MOVEONLY_DESC_FULL_IMPL(ActivationDesc, ActivationDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetActivationDescriptor),
                            cudnnSetActivationDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
// Helper to create tensor descriptor for arbitrary rank tensors
static void createTensorDescriptor(CudnnTensor& desc, NDArray* arr, cudnnDataType_t dataType) {
  const int rank = arr->rankOf();
  if (rank <= 4) {
    // Use 4D descriptor
    int n = 1, c = 1, h = 1, w = 1;
    if (rank >= 1) w = static_cast<int>(arr->sizeAt(rank - 1));
    if (rank >= 2) h = static_cast<int>(arr->sizeAt(rank - 2));
    if (rank >= 3) c = static_cast<int>(arr->sizeAt(rank - 3));
    if (rank >= 4) n = static_cast<int>(arr->sizeAt(rank - 4));
    desc.set4D(CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
  } else {
    // Use ND descriptor
    std::vector<int> dims(rank);
    std::vector<int> strides(rank);
    for (int i = 0; i < rank; i++) {
      dims[i] = static_cast<int>(arr->sizeAt(i));
      strides[i] = static_cast<int>(arr->strideAt(i));
    }
    desc.set(dataType, rank, dims.data(), strides.data());
  }
}

//////////////////////////////////////////////////////////////////////////
// Scale tensor: output = alpha * input
static void scaleTensorCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, double alpha) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // First copy input to output if they're different
  if (input->specialBuffer() != output->specialBuffer()) {
    NDArray::prepareSpecialUse({output}, {input});
    cudaMemcpyAsync(output->specialBuffer(), input->specialBuffer(),
                    input->lengthOf() * input->sizeOfT(), cudaMemcpyDeviceToDevice,
                    stream);
    NDArray::registerSpecialUse({output}, {input});
  }

  CudnnTensor yDesc;
  createTensorDescriptor(yDesc, output, dataType);

  // Scaling parameter
  const float alpha32 = static_cast<float>(alpha);
  const double alpha64 = alpha;
  const void* pAlpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);

  NDArray::prepareSpecialUse({output}, {});

  // Scale the tensor in-place: y = alpha * y
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnScaleTensor),
      cudnnScaleTensor(*handle, yDesc, output->specialBuffer(), pAlpha));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("scaleTensorCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {});
}

//////////////////////////////////////////////////////////////////////////
// Transform tensor: output = alpha * input + beta * output (with possible layout change)
static void transformTensorCUDNN(const LaunchContext* context, NDArray* input, NDArray* output,
                                  double alpha = 1.0, double beta = 0.0) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(context->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc, yDesc;
  createTensorDescriptor(xDesc, input, dataType);
  createTensorDescriptor(yDesc, output, dataType);

  // Scaling parameters
  const float alpha32 = static_cast<float>(alpha), beta32 = static_cast<float>(beta);
  const double alpha64 = alpha, beta64 = beta;
  const void* pAlpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* pBeta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // Transform: y = alpha * x + beta * y
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnTransformTensor),
      cudnnTransformTensor(*handle, pAlpha, xDesc, input->specialBuffer(),
                           pBeta, yDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("transformTensorCUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
// identity operation: output = input (copy)
PLATFORM_IMPL(identity, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  if (input->specialBuffer() == output->specialBuffer()) {
    return Status::OK;  // In-place, nothing to do
  }

  transformTensorCUDNN(block.launchContext(), input, output, 1.0, 0.0);

  return Status::OK;
}

PLATFORM_CHECK(identity, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN IDENTITY OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0)) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// neg operation: output = -input
PLATFORM_IMPL(neg, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  transformTensorCUDNN(block.launchContext(), input, output, -1.0, 0.0);

  return Status::OK;
}

PLATFORM_CHECK(neg, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN NEG OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT0)) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// abs operation using OpTensor NOT
// cuDNN doesn't have a direct abs, but we can use the formula: abs(x) = sqrt(x^2)
// However, this is less efficient. For now, we'll skip abs.

//////////////////////////////////////////////////////////////////////////
// Leaky ReLU activation using cuDNN activation descriptor
PLATFORM_IMPL(leakyrelu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 0.01;

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createTensorDescriptor(xDesc, input, dataType);

  // Create activation descriptor for leaky ReLU (CUDNN_ACTIVATION_RELU with coef)
  // Note: cuDNN's CLIPPED_RELU and ELU don't match leaky ReLU exactly
  // Using a custom approach with OpTensor might be needed
  // For cuDNN 8+, there's better support

  // For now, use the formula: leaky_relu(x) = max(x, alpha*x)
  // This requires two operations which isn't ideal

  // Alternative: Use cudnnActivationForward with appropriate mode
  // cuDNN doesn't have native leaky ReLU in older versions

  // Fallback to transform approach: we'll compute it manually
  // leaky_relu(x, alpha) = x if x >= 0, alpha*x otherwise

  // For simplicity, let's use a workaround with available cuDNN ops
  // This is a placeholder - the actual impl would need more work

  // Copy input to output first
  transformTensorCUDNN(block.launchContext(), input, output, 1.0, 0.0);

  return Status::OK;
}

PLATFORM_CHECK(leakyrelu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  // Disable cuDNN for leaky ReLU as it doesn't have native support
  // and the workaround is complex
  Requirements req("CUDNN LEAKYRELU OP");
  req.expectFalse(makeInfoVariable(true, "cuDNN leaky ReLU not natively supported"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// SELU activation - cuDNN supports this in newer versions
PLATFORM_IMPL(selu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // SELU: scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
  // scale ≈ 1.0507, alpha ≈ 1.6733

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createTensorDescriptor(xDesc, input, dataType);

  // cuDNN 7+ has CUDNN_ACTIVATION_ELU but not SELU directly
  // SELU = scale * ELU(x, alpha) where scale ≈ 1.0507 and alpha ≈ 1.6733

  // First apply ELU
  ActivationDesc actDesc;
  actDesc.set(CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, 1.6732632423543772);  // alpha for SELU

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, input->specialBuffer(),
                             beta, xDesc, output->specialBuffer()));

  // Then scale by SELU scale factor
  const float scale32 = 1.0507009873554804f;
  const double scale64 = 1.0507009873554804;
  const void* scale = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&scale32) : reinterpret_cast<const void*>(&scale64);

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnScaleTensor),
      cudnnScaleTensor(*handle, xDesc, output->specialBuffer(), scale));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("selu CUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}

PLATFORM_CHECK(selu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN SELU OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Swish activation: x * sigmoid(x) - cuDNN 8+ has this as CUDNN_ACTIVATION_SWISH
#if CUDNN_VERSION >= 8000
PLATFORM_IMPL(swish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createTensorDescriptor(xDesc, input, dataType);

  ActivationDesc actDesc;
  actDesc.set(CUDNN_ACTIVATION_SWISH, CUDNN_PROPAGATE_NAN, 1.0);

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, input->specialBuffer(),
                             beta, xDesc, output->specialBuffer()));

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) throw cuda_exception::build("swish CUDNN: cudaStreamSynchronize failed!", cudaErr);
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}

PLATFORM_CHECK(swish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN SWISH OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 8);

  req.logTheSuccess();
  return req;
}
#endif

}  // namespace platforms
}  // namespace ops
}  // namespace sd
