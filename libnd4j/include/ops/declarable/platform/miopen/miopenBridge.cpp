/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "miopenBridge.h"

// This is an isolated AMD-native translation unit in the CUDA/ZLUDA artifact.
// It intentionally includes no ND4J or CUDA headers. The final shared library
// may contain CUDA and ROCm objects; their type systems must not share a TU.
#if defined(__HIP_PLATFORM_NVIDIA__)
#error "miopenBridge.cpp must use the AMD HIP ABI"
#endif
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace sd {
namespace ops {
namespace platforms {
namespace miopen_bridge {
namespace {

thread_local std::string gLastError;

int internalFailure(const char* operation, const std::string& detail,
                    int status = -1) {
  std::ostringstream message;
  message << operation << ": " << detail;
  gLastError = message.str();
  return status;
}

int hipFailure(const char* operation, hipError_t status) {
  if (status == hipSuccess) return 0;
  const char* detail = hipGetErrorString(status);
  return internalFailure(operation, detail != nullptr ? detail : "unknown HIP error",
                         -1000 - static_cast<int>(status));
}

int miopenFailure(const char* operation, miopenStatus_t status) {
  if (status == miopenStatusSuccess) return 0;
  const char* detail = miopenGetErrorString(status);
  return internalFailure(operation,
                         detail != nullptr ? detail : "unknown MIOpen error",
                         static_cast<int>(status) == 0
                             ? -2000
                             : static_cast<int>(status));
}

miopenDataType_t toMIOpenDataType(DataType dataType) {
  switch (dataType) {
    case DataType::FLOAT32:
      return miopenFloat;
    case DataType::FLOAT16:
      return miopenHalf;
    case DataType::BFLOAT16:
      return miopenBFloat16;
    case DataType::INT8:
      return miopenInt8;
    case DataType::INT32:
      return miopenInt32;
  }
  return miopenFloat;
}

miopenActivationMode_t toMIOpenActivation(ActivationMode mode) {
  switch (mode) {
    case ActivationMode::RELU:
      return miopenActivationRELU;
    case ActivationMode::CLIPPED_RELU:
      return miopenActivationCLIPPEDRELU;
    case ActivationMode::LOGISTIC:
      return miopenActivationLOGISTIC;
    case ActivationMode::TANH:
      return miopenActivationTANH;
    case ActivationMode::ELU:
      return miopenActivationELU;
    case ActivationMode::SOFT_RELU:
      return miopenActivationSOFTRELU;
  }
  return miopenActivationRELU;
}

size_t elementSize(DataType dataType) {
  switch (dataType) {
    case DataType::FLOAT32:
    case DataType::INT32:
      return 4;
    case DataType::FLOAT16:
    case DataType::BFLOAT16:
      return 2;
    case DataType::INT8:
      return 1;
  }
  return 0;
}

size_t tensorBytes(const Tensor4D& tensor) {
  const size_t itemSize = elementSize(tensor.dataType);
  if (itemSize == 0 || tensor.n < 0 || tensor.c < 0 ||
      tensor.h < 0 || tensor.w < 0) {
    return 0;
  }
  size_t elements = 1;
  for (int dimension : {tensor.n, tensor.c, tensor.h, tensor.w}) {
    const size_t value = static_cast<size_t>(dimension);
    if (value != 0 && elements > std::numeric_limits<size_t>::max() / value) {
      return 0;
    }
    elements *= value;
  }
  if (elements != 0 &&
      itemSize > std::numeric_limits<size_t>::max() / elements) {
    return 0;
  }
  return elements * itemSize;
}

struct DeviceContext {
  explicit DeviceContext(int requestedDevice) : deviceId(requestedDevice) {}

  ~DeviceContext() {
    hipSetDevice(deviceId);
    if (handle != nullptr) miopenDestroy(handle);
    if (stream != nullptr) hipStreamDestroy(stream);
  }

  int initialize() {
    int status = hipFailure("hipSetDevice", hipSetDevice(deviceId));
    if (status != 0) return status;
    status = hipFailure("hipStreamCreateWithFlags",
                        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    if (status != 0) return status;
    const auto createStatus = miopenCreateWithStream(&handle, stream);
    if (createStatus != miopenStatusSuccess) {
      return miopenFailure("miopenCreateWithStream", createStatus);
    }
    return 0;
  }

  int synchronize() {
    return hipFailure("hipStreamSynchronize", hipStreamSynchronize(stream));
  }

  int deviceId;
  hipStream_t stream = nullptr;
  miopenHandle_t handle = nullptr;
};

struct DeviceContexts {
  std::unordered_map<int, std::unique_ptr<DeviceContext>> values;
};

thread_local DeviceContexts gDeviceContexts;

DeviceContext* deviceContext(int deviceId, int& status) {
  auto found = gDeviceContexts.values.find(deviceId);
  if (found != gDeviceContexts.values.end()) {
    status = hipFailure("hipSetDevice", hipSetDevice(deviceId));
    return status == 0 ? found->second.get() : nullptr;
  }

  auto created = std::make_unique<DeviceContext>(deviceId);
  status = created->initialize();
  if (status != 0) return nullptr;
  auto* result = created.get();
  gDeviceContexts.values.emplace(deviceId, std::move(created));
  return result;
}

struct TensorDescriptor {
  explicit TensorDescriptor(const Tensor4D& tensor) {
    auto status = miopenCreateTensorDescriptor(&value);
    if (status == miopenStatusSuccess) {
      status = miopenSet4dTensorDescriptor(
          value, toMIOpenDataType(tensor.dataType),
          tensor.n, tensor.c, tensor.h, tensor.w);
    }
    if (status != miopenStatusSuccess) {
      error = miopenFailure("MIOpen tensor descriptor", status);
    }
  }

  ~TensorDescriptor() {
    if (value != nullptr) miopenDestroyTensorDescriptor(value);
  }

  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor& operator=(const TensorDescriptor&) = delete;

  miopenTensorDescriptor_t value = nullptr;
  int error = 0;
};

struct ConvolutionDescriptor {
  explicit ConvolutionDescriptor(const Convolution2D& convolution) {
    auto status = miopenCreateConvolutionDescriptor(&value);
    if (status == miopenStatusSuccess) {
      status = miopenInitConvolutionDescriptor(
          value, miopenConvolution,
          convolution.padH, convolution.padW,
          convolution.strideH, convolution.strideW,
          convolution.dilationH, convolution.dilationW);
    }
    if (status != miopenStatusSuccess) {
      error = miopenFailure("MIOpen convolution descriptor", status);
    }
  }

  ~ConvolutionDescriptor() {
    if (value != nullptr) miopenDestroyConvolutionDescriptor(value);
  }

  ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
  ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

  miopenConvolutionDescriptor_t value = nullptr;
  int error = 0;
};

struct ActivationDescriptor {
  ActivationDescriptor(ActivationMode mode, double alpha,
                       double beta, double gamma) {
    auto status = miopenCreateActivationDescriptor(&value);
    if (status == miopenStatusSuccess) {
      status = miopenSetActivationDescriptor(
          value, toMIOpenActivation(mode), alpha, beta, gamma);
    }
    if (status != miopenStatusSuccess) {
      error = miopenFailure("MIOpen activation descriptor", status);
    }
  }

  ~ActivationDescriptor() {
    if (value != nullptr) miopenDestroyActivationDescriptor(value);
  }

  ActivationDescriptor(const ActivationDescriptor&) = delete;
  ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;

  miopenActivationDescriptor_t value = nullptr;
  int error = 0;
};

struct Workspace {
  explicit Workspace(size_t requestedBytes) : bytes(requestedBytes) {
    if (bytes > 0) error = hipFailure("hipMalloc(workspace)", hipMalloc(&data, bytes));
  }

  ~Workspace() {
    if (data != nullptr) hipFree(data);
  }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  void* data = nullptr;
  size_t bytes = 0;
  int error = 0;
};

int requireDescriptors(std::initializer_list<int> errors) {
  for (int error : errors) {
    if (error != 0) return error;
  }
  return 0;
}

int requireAlgorithm(const char* operation, int count) {
  return count > 0
             ? 0
             : internalFailure(operation, "MIOpen returned no usable algorithm",
                               -3000);
}

}  // namespace

const char* lastError() {
  return gLastError.c_str();
}

int activationForward(int deviceId, const Tensor4D& tensor,
                      const void* input, void* output,
                      ActivationMode mode, double alpha,
                      double beta, double gamma) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor descriptor(tensor);
  ActivationDescriptor activation(mode, alpha, beta, gamma);
  status = requireDescriptors({descriptor.error, activation.error});
  if (status != 0) return status;

  float scale = 1.0f;
  float shift = 0.0f;
  auto callStatus = miopenActivationForward(
      context->handle, activation.value,
      &scale, descriptor.value, input,
      &shift, descriptor.value, output);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenActivationForward", callStatus);
  }
  return context->synchronize();
}

int batchNormForwardInference(int deviceId, const Tensor4D& tensor,
                              const Tensor4D& parameterTensor,
                              const void* input, void* output,
                              const void* scale, const void* bias,
                              const void* mean, const void* variance,
                              double epsilon) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor dataDescriptor(tensor);
  TensorDescriptor parameterDescriptor(parameterTensor);
  status = requireDescriptors({dataDescriptor.error, parameterDescriptor.error});
  if (status != 0) return status;

  float alpha = 1.0f;
  float beta = 0.0f;
  auto callStatus = miopenBatchNormalizationForwardInference(
      context->handle, miopenBNSpatial,
      &alpha, &beta,
      dataDescriptor.value, input,
      dataDescriptor.value, output,
      parameterDescriptor.value,
      const_cast<void*>(scale), const_cast<void*>(bias),
      const_cast<void*>(mean), const_cast<void*>(variance), epsilon);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenBatchNormalizationForwardInference", callStatus);
  }
  return context->synchronize();
}

int batchNormBackward(int deviceId, const Tensor4D& tensor,
                      const Tensor4D& parameterTensor,
                      const void* input, const void* gradOutput,
                      const void* scale, const void* mean,
                      const void* variance, void* gradInput,
                      void* gradScale, void* gradBias,
                      double epsilon) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor dataDescriptor(tensor);
  TensorDescriptor parameterDescriptor(parameterTensor);
  status = requireDescriptors({dataDescriptor.error, parameterDescriptor.error});
  if (status != 0) return status;

  float alphaData = 1.0f;
  float betaData = 0.0f;
  float alphaParameters = 1.0f;
  float betaParameters = 0.0f;
  auto callStatus = miopenBatchNormalizationBackward(
      context->handle, miopenBNSpatial,
      &alphaData, &betaData, &alphaParameters, &betaParameters,
      dataDescriptor.value, input,
      dataDescriptor.value, gradOutput,
      dataDescriptor.value, gradInput,
      parameterDescriptor.value,
      const_cast<void*>(scale), gradScale, gradBias,
      epsilon, const_cast<void*>(mean), const_cast<void*>(variance));
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenBatchNormalizationBackward", callStatus);
  }
  return context->synchronize();
}

int softmaxForward(int deviceId, const Tensor4D& tensor,
                   const void* input, void* output, bool logSoftmax) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor descriptor(tensor);
  if (descriptor.error != 0) return descriptor.error;

  float alpha = 1.0f;
  float beta = 0.0f;
  const miopenSoftmaxAlgorithm_t algorithm =
      logSoftmax ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE;
  auto callStatus = miopenSoftmaxForward_V2(
      context->handle, &alpha,
      descriptor.value, input,
      &beta, descriptor.value, output,
      algorithm, MIOPEN_SOFTMAX_MODE_CHANNEL);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenSoftmaxForward_V2", callStatus);
  }
  return context->synchronize();
}

int softmaxBackward(int deviceId, const Tensor4D& tensor,
                    const void* input, const void* gradOutput,
                    void* gradInput, bool logSoftmax) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor descriptor(tensor);
  if (descriptor.error != 0) return descriptor.error;

  const size_t forwardBytes = tensorBytes(tensor);
  if (forwardBytes == 0) {
    return internalFailure("softmaxBackward", "invalid tensor byte size", -3001);
  }
  Workspace forwardOutput(forwardBytes);
  if (forwardOutput.error != 0) return forwardOutput.error;

  float alpha = 1.0f;
  float beta = 0.0f;
  const miopenSoftmaxAlgorithm_t algorithm =
      logSoftmax ? MIOPEN_SOFTMAX_LOG : MIOPEN_SOFTMAX_ACCURATE;
  auto callStatus = miopenSoftmaxForward_V2(
      context->handle, &alpha,
      descriptor.value, input,
      &beta, descriptor.value, forwardOutput.data,
      algorithm, MIOPEN_SOFTMAX_MODE_CHANNEL);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenSoftmaxForward_V2(backward preparation)",
                         callStatus);
  }

  callStatus = miopenSoftmaxBackward_V2(
      context->handle, &alpha,
      descriptor.value, forwardOutput.data,
      descriptor.value, gradOutput,
      &beta, descriptor.value, gradInput,
      algorithm, MIOPEN_SOFTMAX_MODE_CHANNEL);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenSoftmaxBackward_V2", callStatus);
  }
  return context->synchronize();
}

int convolutionForward(int deviceId,
                       const Tensor4D& inputTensor,
                       const Tensor4D& weightsTensor,
                       const Tensor4D& outputTensor,
                       const Convolution2D& convolution,
                       const void* input, const void* weights,
                       void* output, const Tensor4D* biasTensor,
                       const void* bias) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor inputDescriptor(inputTensor);
  TensorDescriptor weightsDescriptor(weightsTensor);
  TensorDescriptor outputDescriptor(outputTensor);
  ConvolutionDescriptor convolutionDescriptor(convolution);
  status = requireDescriptors({
      inputDescriptor.error, weightsDescriptor.error,
      outputDescriptor.error, convolutionDescriptor.error});
  if (status != 0) return status;

  size_t workspaceBytes = 0;
  auto callStatus = miopenConvolutionForwardGetWorkSpaceSize(
      context->handle, weightsDescriptor.value, inputDescriptor.value,
      convolutionDescriptor.value, outputDescriptor.value, &workspaceBytes);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenConvolutionForwardGetWorkSpaceSize", callStatus);
  }

  Workspace workspace(workspaceBytes);
  if (workspace.error != 0) return workspace.error;

  int algorithmCount = 0;
  miopenConvAlgoPerf_t performance{};
  callStatus = miopenFindConvolutionForwardAlgorithm(
      context->handle,
      inputDescriptor.value, input,
      weightsDescriptor.value, weights,
      convolutionDescriptor.value,
      outputDescriptor.value, output,
      1, &algorithmCount, &performance,
      workspace.data, workspace.bytes, false);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenFindConvolutionForwardAlgorithm", callStatus);
  }
  status = requireAlgorithm("miopenFindConvolutionForwardAlgorithm",
                            algorithmCount);
  if (status != 0) return status;

  float alpha = 1.0f;
  float beta = 0.0f;
  callStatus = miopenConvolutionForward(
      context->handle, &alpha,
      inputDescriptor.value, input,
      weightsDescriptor.value, weights,
      convolutionDescriptor.value, performance.fwd_algo,
      &beta, outputDescriptor.value, output,
      workspace.data, workspace.bytes);
  if (callStatus != miopenStatusSuccess) {
    return miopenFailure("miopenConvolutionForward", callStatus);
  }

  if (biasTensor != nullptr && bias != nullptr) {
    TensorDescriptor biasDescriptor(*biasTensor);
    if (biasDescriptor.error != 0) return biasDescriptor.error;
    float biasBeta = 1.0f;
    callStatus = miopenConvolutionForwardBias(
        context->handle, &alpha,
        biasDescriptor.value, bias,
        &biasBeta, outputDescriptor.value, output);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure("miopenConvolutionForwardBias", callStatus);
    }
  }

  return context->synchronize();
}

int convolutionBackward(int deviceId,
                        const Tensor4D& inputTensor,
                        const Tensor4D& weightsTensor,
                        const Tensor4D& gradOutputTensor,
                        const Convolution2D& convolution,
                        const void* input, const void* weights,
                        const void* gradOutput, void* gradInput,
                        void* gradWeights, const Tensor4D* gradBiasTensor,
                        void* gradBias) {
  gLastError.clear();
  int status = 0;
  auto* context = deviceContext(deviceId, status);
  if (context == nullptr) return status;

  TensorDescriptor inputDescriptor(inputTensor);
  TensorDescriptor weightsDescriptor(weightsTensor);
  TensorDescriptor gradOutputDescriptor(gradOutputTensor);
  ConvolutionDescriptor convolutionDescriptor(convolution);
  status = requireDescriptors({
      inputDescriptor.error, weightsDescriptor.error,
      gradOutputDescriptor.error, convolutionDescriptor.error});
  if (status != 0) return status;

  float alpha = 1.0f;
  float beta = 0.0f;

  if (gradInput != nullptr) {
    size_t workspaceBytes = 0;
    auto callStatus = miopenConvolutionBackwardDataGetWorkSpaceSize(
        context->handle,
        gradOutputDescriptor.value, weightsDescriptor.value,
        convolutionDescriptor.value, inputDescriptor.value,
        &workspaceBytes);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure(
          "miopenConvolutionBackwardDataGetWorkSpaceSize", callStatus);
    }
    Workspace workspace(workspaceBytes);
    if (workspace.error != 0) return workspace.error;

    int algorithmCount = 0;
    miopenConvAlgoPerf_t performance{};
    callStatus = miopenFindConvolutionBackwardDataAlgorithm(
        context->handle,
        gradOutputDescriptor.value, gradOutput,
        weightsDescriptor.value, weights,
        convolutionDescriptor.value,
        inputDescriptor.value, gradInput,
        1, &algorithmCount, &performance,
        workspace.data, workspace.bytes, false);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure(
          "miopenFindConvolutionBackwardDataAlgorithm", callStatus);
    }
    status = requireAlgorithm("miopenFindConvolutionBackwardDataAlgorithm",
                              algorithmCount);
    if (status != 0) return status;

    callStatus = miopenConvolutionBackwardData(
        context->handle, &alpha,
        gradOutputDescriptor.value, gradOutput,
        weightsDescriptor.value, weights,
        convolutionDescriptor.value, performance.bwd_data_algo,
        &beta, inputDescriptor.value, gradInput,
        workspace.data, workspace.bytes);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure("miopenConvolutionBackwardData", callStatus);
    }
  }

  if (gradWeights != nullptr) {
    size_t workspaceBytes = 0;
    auto callStatus = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        context->handle,
        gradOutputDescriptor.value, inputDescriptor.value,
        convolutionDescriptor.value, weightsDescriptor.value,
        &workspaceBytes);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure(
          "miopenConvolutionBackwardWeightsGetWorkSpaceSize", callStatus);
    }
    Workspace workspace(workspaceBytes);
    if (workspace.error != 0) return workspace.error;

    int algorithmCount = 0;
    miopenConvAlgoPerf_t performance{};
    callStatus = miopenFindConvolutionBackwardWeightsAlgorithm(
        context->handle,
        gradOutputDescriptor.value, gradOutput,
        inputDescriptor.value, input,
        convolutionDescriptor.value,
        weightsDescriptor.value, gradWeights,
        1, &algorithmCount, &performance,
        workspace.data, workspace.bytes, false);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure(
          "miopenFindConvolutionBackwardWeightsAlgorithm", callStatus);
    }
    status = requireAlgorithm("miopenFindConvolutionBackwardWeightsAlgorithm",
                              algorithmCount);
    if (status != 0) return status;

    callStatus = miopenConvolutionBackwardWeights(
        context->handle, &alpha,
        gradOutputDescriptor.value, gradOutput,
        inputDescriptor.value, input,
        convolutionDescriptor.value, performance.bwd_weights_algo,
        &beta, weightsDescriptor.value, gradWeights,
        workspace.data, workspace.bytes);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure("miopenConvolutionBackwardWeights", callStatus);
    }
  }

  if (gradBias != nullptr && gradBiasTensor != nullptr) {
    TensorDescriptor biasDescriptor(*gradBiasTensor);
    if (biasDescriptor.error != 0) return biasDescriptor.error;
    auto callStatus = miopenConvolutionBackwardBias(
        context->handle, &alpha,
        gradOutputDescriptor.value, gradOutput,
        &beta, biasDescriptor.value, gradBias);
    if (callStatus != miopenStatusSuccess) {
      return miopenFailure("miopenConvolutionBackwardBias", callStatus);
    }
  }

  return context->synchronize();
}

}  // namespace miopen_bridge
}  // namespace platforms
}  // namespace ops
}  // namespace sd
