/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef SD_MIOPEN_UTILS_H
#define SD_MIOPEN_UTILS_H

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include "miopenBridge.h"

#include <array/DataType.h>
#include <cuda_runtime_api.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <string>

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {

// Convolution operations
DECLARE_PLATFORM(conv2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv3dnew_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_ZLUDA_AMD);

// Pooling operations
DECLARE_PLATFORM(avgpool2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_ZLUDA_AMD);

// Normalization operations
DECLARE_PLATFORM(batchnorm, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_ZLUDA_AMD);

// Activation operations
DECLARE_PLATFORM(relu, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(relu6, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(sigmoid, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(tanh, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(elu, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(softplus, ENGINE_ZLUDA_AMD);

// Softmax operations
DECLARE_PLATFORM(softmax, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(softmax_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(log_softmax, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(log_softmax_bp, ENGINE_ZLUDA_AMD);

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

inline miopen_bridge::DataType miopenBridgeDataType(DataType dataType) {
  switch (dataType) {
    case sd::DataType::FLOAT32:
      return miopen_bridge::DataType::FLOAT32;
    case sd::DataType::HALF:
      return miopen_bridge::DataType::FLOAT16;
    case sd::DataType::BFLOAT16:
      return miopen_bridge::DataType::BFLOAT16;
    case sd::DataType::INT8:
      return miopen_bridge::DataType::INT8;
    case sd::DataType::INT32:
      return miopen_bridge::DataType::INT32;
    default: {
      const std::string message =
          "Unsupported data type for the ZLUDA MIOpen bridge: " +
          std::to_string(static_cast<int>(dataType));
      THROW_EXCEPTION(message.c_str());
    }
  }
}

inline bool isMIOpenSupportedType(DataType dataType) {
  switch (dataType) {
    case sd::DataType::FLOAT32:
    case sd::DataType::HALF:
    case sd::DataType::BFLOAT16:
    case sd::DataType::INT8:
    case sd::DataType::INT32:
      return true;
    default:
      return false;
  }
}

inline miopen_bridge::Tensor4D miopenTensor4D(
    DataType dataType, int n, int c, int h, int w) {
  return {miopenBridgeDataType(dataType), n, c, h, w};
}

inline void checkMIOpenBridge(int status, const char* operation) {
  if (status == 0) return;

  std::string message = "ZLUDA MIOpen bridge failure";
  if (operation != nullptr && operation[0] != '\0') {
    message += " in ";
    message += operation;
  }
  message += " (status ";
  message += std::to_string(status);
  message += ")";

  const char* detail = miopen_bridge::lastError();
  if (detail != nullptr && detail[0] != '\0') {
    message += ": ";
    message += detail;
  }
  THROW_EXCEPTION(message.c_str());
}

// ND4J prepares special buffers on its CUDA/ZLUDA stream. The native MIOpen
// bridge owns a separate AMD HIP stream because CUDA and AMD HIP types cannot
// coexist in one translation unit. Complete the producer work before handing
// the raw device pointers across that ABI boundary.
inline void synchronizeZludaForMIOpen(const LaunchContext* context) {
  cudaStream_t stream = nullptr;
  if (context != nullptr) {
    auto* streamPointer = context->getCudaStream();
    if (streamPointer != nullptr) stream = *streamPointer;
  }

  const cudaError_t status = cudaStreamSynchronize(stream);
  if (status == cudaSuccess) return;

  std::string message = "Unable to synchronize the CUDA/ZLUDA stream before MIOpen";
  const char* detail = cudaGetErrorString(status);
  if (detail != nullptr && detail[0] != '\0') {
    message += ": ";
    message += detail;
  }
  THROW_EXCEPTION(message.c_str());
}

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN

#endif  // SD_MIOPEN_UTILS_H
