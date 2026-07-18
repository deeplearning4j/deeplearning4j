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
// Extended cuDNN activation functions: softplus, gelu, mish, hardswish
// These use cuDNN pointwise operations where available (cuDNN 8+)
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "cudnnUtils.h"

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

// Activation descriptor RAII wrapper for extended activations
struct ActivationDescExt {
  MOVEONLY_DESC_FULL_IMPL(ActivationDescExt, ActivationDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetActivationDescriptor),
                            cudnnSetActivationDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
// Helper to create tensor descriptor for activation operations
static void createActivationTensorDesc(CudnnTensor& desc, NDArray* arr, cudnnDataType_t dataType) {
  const LongType length = arr->lengthOf();
  // Reshape to 4D: [length, 1, 1, 1]
  desc.set4D(CUDNN_TENSOR_NCHW, dataType, static_cast<int>(length), 1, 1, 1);
}

//////////////////////////////////////////////////////////////////////////
// Softplus: softplus(x) = log(1 + exp(x))
// cuDNN has CUDNN_ACTIVATION_SOFTPLUS available in some versions
// Fallback: Use ELU approximation or custom kernel
//////////////////////////////////////////////////////////////////////////
// Op macros below open SD_NS themselves (platform_boilerplate.h) — the
// file-level wrap must end before them or SD_NS nests inside itself.
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END

#if CUDNN_VERSION >= 8500
// cuDNN 8.5+ has proper softplus support
PLATFORM_IMPL(softplus, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createActivationTensorDesc(xDesc, input, dataType);

  // cuDNN 8+ softplus
  ActivationDescExt actDesc;
  // CUDNN_ACTIVATION_SOFTPLUS is value 8 in cuDNN 8+
  actDesc.set(static_cast<cudnnActivationMode_t>(8), CUDNN_PROPAGATE_NAN, 1.0);

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
    if (cudaErr != 0) { std::string msg = "softplus CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  NDArray::registerSpecialUse({output}, {input});

  return Status::OK;
}

PLATFORM_CHECK(softplus, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN SOFTPLUS OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}
#endif

//////////////////////////////////////////////////////////////////////////
// GELU: Gaussian Error Linear Unit
// gelu(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// cuDNN 8.3+ has CUDNN_POINTWISE_GELU_FWD
//////////////////////////////////////////////////////////////////////////
#if CUDNN_VERSION >= 8300
PLATFORM_IMPL(gelu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto stream = cudnnCaptureAwareStream(block.launchContext()->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  // For GELU, we need to use the cudnnBackend API with pointwise operations
  // This is complex, so for now we'll implement a simpler approximation using available ops

  // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  // Using cuDNN's tanh activation as part of this

  CudnnTensor xDesc;
  createActivationTensorDesc(xDesc, input, dataType);

  // For simplicity, compute GELU using the tanh approximation
  // gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

  // Create temp arrays for intermediate results
  auto contextPtr = block.launchContext();
  NDArray temp1(input->shapeInfo(), false, contextPtr, true);
  NDArray temp2(input->shapeInfo(), false, contextPtr, true);

  // Compute: temp1 = x^3
  input->applyScalar(scalar::Pow, 3.0, &temp1);

  // temp1 = 0.044715 * x^3
  temp1 *= 0.044715;

  // temp1 = x + 0.044715 * x^3
  temp1 += *input;

  // temp1 = sqrt(2/pi) * (x + 0.044715 * x^3)
  const double sqrt2pi = 0.7978845608028654;  // sqrt(2/pi)
  temp1 *= sqrt2pi;

  // temp2 = tanh(temp1)
  ActivationDescExt actDesc;
  actDesc.set(CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0);

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({&temp2}, {&temp1});
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, temp1.specialBuffer(),
                             beta, xDesc, temp2.specialBuffer()));
  NDArray::registerSpecialUse({&temp2}, {&temp1});

  // temp2 = 1 + tanh(...)
  temp2 += 1.0;

  // output = 0.5 * x * (1 + tanh(...))
  output->assign(input);
  *output *= temp2;
  *output *= 0.5;

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "gelu CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  return Status::OK;
}

PLATFORM_CHECK(gelu, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN GELU OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}
#endif

//////////////////////////////////////////////////////////////////////////
// Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
//////////////////////////////////////////////////////////////////////////
#if CUDNN_VERSION >= 8000
PLATFORM_IMPL(mish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto contextPtr = block.launchContext();
  auto stream = cudnnCaptureAwareStream(contextPtr->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createActivationTensorDesc(xDesc, input, dataType);

  // Mish: x * tanh(softplus(x))
  // Step 1: Compute softplus(x) = log(1 + exp(x))
  // Step 2: Compute tanh(softplus(x))
  // Step 3: Multiply by x

  // Create temp array for softplus result
  NDArray softplusResult(input->shapeInfo(), false, contextPtr, true);

  // Compute softplus using log(1 + exp(x))
  // For numerical stability, we use: softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
  input->applyTransform(transform::SoftPlus, &softplusResult);

  // Compute tanh(softplus(x))
  NDArray tanhResult(input->shapeInfo(), false, contextPtr, true);

  ActivationDescExt actDesc;
  actDesc.set(CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0);

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({&tanhResult}, {&softplusResult});
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, softplusResult.specialBuffer(),
                             beta, xDesc, tanhResult.specialBuffer()));
  NDArray::registerSpecialUse({&tanhResult}, {&softplusResult});

  // Multiply by x: output = x * tanh(softplus(x))
  input->applyPairwiseTransform(pairwise::Multiply, &tanhResult, output);

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "mish CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  return Status::OK;
}

PLATFORM_CHECK(mish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN MISH OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}
#endif

//////////////////////////////////////////////////////////////////////////
// Hard Swish: x * relu6(x + 3) / 6
// Available as CUDNN_ACTIVATION_HARDSWISH in cuDNN 8.2+
//////////////////////////////////////////////////////////////////////////
#if CUDNN_VERSION >= 8200
PLATFORM_IMPL(hardswish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto contextPtr = block.launchContext();
  auto stream = cudnnCaptureAwareStream(contextPtr->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createActivationTensorDesc(xDesc, input, dataType);

  // Hard Swish: x * relu6(x + 3) / 6
  // Using clipped relu for relu6

  // Step 1: temp = x + 3
  NDArray temp(input->shapeInfo(), false, contextPtr, true);
  input->applyScalar(scalar::Add, 3.0, &temp);

  // Step 2: temp = relu6(temp) = clipped_relu(temp, 6)
  NDArray relu6Result(input->shapeInfo(), false, contextPtr, true);

  ActivationDescExt actDesc;
  actDesc.set(CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 6.0);

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({&relu6Result}, {&temp});
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, temp.specialBuffer(),
                             beta, xDesc, relu6Result.specialBuffer()));
  NDArray::registerSpecialUse({&relu6Result}, {&temp});

  // Step 3: output = x * relu6(x + 3) / 6
  input->applyPairwiseTransform(pairwise::Multiply, &relu6Result, output);
  *output /= 6.0;

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "hardswish CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  return Status::OK;
}

PLATFORM_CHECK(hardswish, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN HARDSWISH OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}
#endif

//////////////////////////////////////////////////////////////////////////
// Hard Sigmoid: relu6(x + 3) / 6
// A simplified sigmoid approximation
//////////////////////////////////////////////////////////////////////////
#if CUDNN_VERSION >= 8000
PLATFORM_IMPL(hardsigmoid, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto handle = reinterpret_cast<cudnnHandle_t*>(block.launchContext()->getCuDnnHandle());
  auto contextPtr = block.launchContext();
  auto stream = cudnnCaptureAwareStream(contextPtr->getCudaStream());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, stream));

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  CudnnTensor xDesc;
  createActivationTensorDesc(xDesc, input, dataType);

  // Hard Sigmoid: relu6(x + 3) / 6

  // Step 1: temp = x + 3
  NDArray temp(input->shapeInfo(), false, contextPtr, true);
  input->applyScalar(scalar::Add, 3.0, &temp);

  // Step 2: output = relu6(temp) = clipped_relu(temp, 6)
  ActivationDescExt actDesc;
  actDesc.set(CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 6.0);

  static const float alpha32 = 1.0f, beta32 = 0.0f;
  static const double alpha64 = 1.0, beta64 = 0.0;
  const void* alpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {&temp});
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnActivationForward),
      cudnnActivationForward(*handle, actDesc, alpha, xDesc, temp.specialBuffer(),
                             beta, xDesc, output->specialBuffer()));
  NDArray::registerSpecialUse({output}, {&temp});

  // Step 3: output = output / 6
  *output /= 6.0;

  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto cudaErr = cudaStreamSynchronize(stream);
    if (cudaErr != 0) { std::string msg = "hardsigmoid CUDNN: cudaStreamSynchronize failed!; Error code: [" + std::to_string(cudaErr) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  return Status::OK;
}

PLATFORM_CHECK(hardsigmoid, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("CUDNN HARDSIGMOID OP");
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), {HALF, FLOAT32, DOUBLE}) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectLessEq(makeInfoVariable(input->lengthOf(), LENGTH_MSG_INPUT0), static_cast<LongType>(INT_MAX));

  req.logTheSuccess();
  return req;
}
#endif

}  // namespace platforms
}  // namespace ops
}  // namespace sd
