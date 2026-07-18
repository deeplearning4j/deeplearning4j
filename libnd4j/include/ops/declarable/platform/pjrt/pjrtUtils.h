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

#ifndef SD_PJRTUTILS_H
#define SD_PJRTUTILS_H

#include <array/DataType.h>
#include <array/NDArray.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations for PJRT types (when PJRT headers are available)
#if defined(HAVE_PJRT) && HAVE_PJRT
#include <pjrt_c_api.h>
#endif

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
// Op macros open SD_NS themselves (platform_boilerplate.h) — the DECLARE list
// must sit outside the file-level wrap or SD_NS nests inside itself.

//////////////////////////////////////////////////////////////////////////
// Platform declarations for TPU operations
//////////////////////////////////////////////////////////////////////////

// Core linear algebra operations
DECLARE_PLATFORM(matmul, ENGINE_TPU);
DECLARE_PLATFORM(mmul, ENGINE_TPU);

// Element-wise operations
DECLARE_PLATFORM(add, ENGINE_TPU);
DECLARE_PLATFORM(subtract, ENGINE_TPU);
DECLARE_PLATFORM(multiply, ENGINE_TPU);
DECLARE_PLATFORM(divide, ENGINE_TPU);
DECLARE_PLATFORM(negate, ENGINE_TPU);
DECLARE_PLATFORM(abs, ENGINE_TPU);
DECLARE_PLATFORM(square, ENGINE_TPU);
DECLARE_PLATFORM(sqrt, ENGINE_TPU);
DECLARE_PLATFORM(exp, ENGINE_TPU);
DECLARE_PLATFORM(log, ENGINE_TPU);
DECLARE_PLATFORM(Pow, ENGINE_TPU);

// Shape operations
DECLARE_PLATFORM(transpose, ENGINE_TPU);
DECLARE_PLATFORM(reshape, ENGINE_TPU);
DECLARE_PLATFORM(concat, ENGINE_TPU);
DECLARE_PLATFORM(slice, ENGINE_TPU);
DECLARE_PLATFORM(tile, ENGINE_TPU);

// Reduction operations
DECLARE_PLATFORM(reduce_sum, ENGINE_TPU);
DECLARE_PLATFORM(reduce_mean, ENGINE_TPU);
DECLARE_PLATFORM(reduce_max, ENGINE_TPU);
DECLARE_PLATFORM(reduce_min, ENGINE_TPU);
DECLARE_PLATFORM(reduce_prod, ENGINE_TPU);
DECLARE_PLATFORM(argmax, ENGINE_TPU);
DECLARE_PLATFORM(argmin, ENGINE_TPU);

// Convolution operations
DECLARE_PLATFORM(conv2d, ENGINE_TPU);
DECLARE_PLATFORM(conv2d_bp, ENGINE_TPU);
DECLARE_PLATFORM(conv3dnew, ENGINE_TPU);
DECLARE_PLATFORM(conv3dnew_bp, ENGINE_TPU);

// Batch normalization
DECLARE_PLATFORM(batchnorm, ENGINE_TPU);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_TPU);

// Activation functions
DECLARE_PLATFORM(relu, ENGINE_TPU);
DECLARE_PLATFORM(sigmoid, ENGINE_TPU);
DECLARE_PLATFORM(tanh, ENGINE_TPU);
DECLARE_PLATFORM(softmax, ENGINE_TPU);
DECLARE_PLATFORM(log_softmax, ENGINE_TPU);

// Pooling operations
DECLARE_PLATFORM(avgpool2d, ENGINE_TPU);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_TPU);
DECLARE_PLATFORM(maxpool2d, ENGINE_TPU);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_TPU);
DECLARE_PLATFORM(avgpool3dnew, ENGINE_TPU);
DECLARE_PLATFORM(maxpool3dnew, ENGINE_TPU);

// Normalization
DECLARE_PLATFORM(layer_norm, ENGINE_TPU);

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

//////////////////////////////////////////////////////////////////////////
// Error handling utilities
//////////////////////////////////////////////////////////////////////////

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x

#if defined(HAVE_PJRT) && HAVE_PJRT

inline void throwIfPjrtFailed(PJRT_Error* error, const PJRT_Api* api,
                              const char* message = "PJRT error: ",
                              const char* prefix = nullptr) {
  if (error != nullptr) {
    std::string err_message;
    if (prefix) err_message = std::string(prefix) + ": ";
    err_message += std::string(message);

    // Get error message from PJRT
    PJRT_Error_Message_Args msg_args;
    msg_args.struct_size = sizeof(PJRT_Error_Message_Args);
    msg_args.error = error;
    api->PJRT_Error_Message(&msg_args);
    err_message += std::string(msg_args.message, msg_args.message_size);

    // Get error code
    PJRT_Error_GetCode_Args code_args;
    code_args.struct_size = sizeof(PJRT_Error_GetCode_Args);
    code_args.error = error;
    api->PJRT_Error_GetCode(&code_args);

    // Destroy the error
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = sizeof(PJRT_Error_Destroy_Args);
    destroy_args.error = error;
    api->PJRT_Error_Destroy(&destroy_args);

    std::string pjrt_msg = "PJRT error " + std::to_string(static_cast<int>(code_args.code)) + ": " + err_message;
    THROW_EXCEPTION(pjrt_msg.c_str());
  }
}

#define CHECK_PJRT_FAILURE(api, error) throwIfPjrtFailed(error, api, "")
#define CHECK_PJRT_FAILURE_MSG(api, error, custom_message) \
  throwIfPjrtFailed(error, api, custom_message, __func__)

//////////////////////////////////////////////////////////////////////////
// RAII Wrappers for PJRT resources
//////////////////////////////////////////////////////////////////////////

/**
 * RAII wrapper for PJRT Client.
 * Manages the lifecycle of a PJRT client connection to TPU.
 */
class PjrtClient {
 public:
  PjrtClient();
  ~PjrtClient();

  // Move-only semantics
  PjrtClient(const PjrtClient&) = delete;
  PjrtClient& operator=(const PjrtClient&) = delete;
  PjrtClient(PjrtClient&& other) noexcept;
  PjrtClient& operator=(PjrtClient&& other) noexcept;

  PJRT_Client* get() const { return client_; }
  const PJRT_Api* api() const { return api_; }
  int deviceCount() const;
  bool isValid() const { return client_ != nullptr && api_ != nullptr; }

 private:
  void destroy();

  PJRT_Client* client_ = nullptr;
  const PJRT_Api* api_ = nullptr;
};

/**
 * RAII wrapper for PJRT Buffer.
 * Manages device memory buffers on TPU.
 */
class PjrtBuffer {
 public:
  PjrtBuffer(PjrtClient& client, const NDArray* arr);
  ~PjrtBuffer();

  // Move-only semantics
  PjrtBuffer(const PjrtBuffer&) = delete;
  PjrtBuffer& operator=(const PjrtBuffer&) = delete;
  PjrtBuffer(PjrtBuffer&& other) noexcept;
  PjrtBuffer& operator=(PjrtBuffer&& other) noexcept;

  PJRT_Buffer* get() const { return buffer_; }
  bool isValid() const { return buffer_ != nullptr; }

  // Copy data back to host
  void toHost(NDArray* arr);

 private:
  void destroy();

  PJRT_Buffer* buffer_ = nullptr;
  const PJRT_Api* api_ = nullptr;
};

/**
 * RAII wrapper for PJRT Loaded Executable.
 * Manages compiled HLO programs on TPU.
 */
class PjrtLoadedExecutable {
 public:
  PjrtLoadedExecutable(PjrtClient& client, const char* hloModule, size_t size);
  ~PjrtLoadedExecutable();

  // Move-only semantics
  PjrtLoadedExecutable(const PjrtLoadedExecutable&) = delete;
  PjrtLoadedExecutable& operator=(const PjrtLoadedExecutable&) = delete;
  PjrtLoadedExecutable(PjrtLoadedExecutable&& other) noexcept;
  PjrtLoadedExecutable& operator=(PjrtLoadedExecutable&& other) noexcept;

  PJRT_LoadedExecutable* get() const { return executable_; }
  bool isValid() const { return executable_ != nullptr; }

  // Execute the program
  void execute(const std::vector<PjrtBuffer*>& inputs, std::vector<PjrtBuffer*>& outputs);

 private:
  void destroy();

  PJRT_LoadedExecutable* executable_ = nullptr;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
};

/**
 * LRU Cache for compiled HLO executables.
 * Prevents recompilation of frequently used programs.
 */
class HLOExecutableCache {
 public:
  static HLOExecutableCache& getInstance();

  std::shared_ptr<PjrtLoadedExecutable> getOrCompile(
      PjrtClient& client, const std::string& hloKey, const std::string& hloModule);

  void clear();
  void setMaxSize(size_t size) { maxSize_ = size; }
  size_t size() const { return cache_.size(); }

 private:
  HLOExecutableCache() = default;
  ~HLOExecutableCache() = default;

  std::unordered_map<std::string, std::shared_ptr<PjrtLoadedExecutable>> cache_;
  std::mutex mutex_;
  size_t maxSize_ = 1000;
};

#endif  // HAVE_PJRT

//////////////////////////////////////////////////////////////////////////
// Utility functions
//////////////////////////////////////////////////////////////////////////

/**
 * Convert ND4J DataType to PJRT buffer type.
 */
int pjrtBufferType(DataType dtype);

/**
 * Convert PJRT buffer type to ND4J DataType.
 */
DataType fromPjrtBufferType(int pjrtType);

/**
 * Check if a data type is supported on TPU.
 */
bool isTpuSupportedType(DataType dtype);

/**
 * Get HLO type string for a given data type.
 */
std::string getHLOTypeString(DataType dtype);

/**
 * Convert shape to HLO dimension string.
 */
std::string shapeToString(const std::vector<sd::LongType>& shape);

/**
 * Generate HLO module string for matrix multiplication.
 */
std::string buildMatmulHLO(const std::vector<sd::LongType>& aShape,
                           const std::vector<sd::LongType>& bShape,
                           DataType dtype, bool transA = false, bool transB = false);

/**
 * Generate HLO module string for element-wise addition.
 */
std::string buildAddHLO(const std::vector<sd::LongType>& shape, DataType dtype);

/**
 * Generate HLO module string for 2D convolution.
 */
std::string buildConv2dHLO(const std::vector<sd::LongType>& inputShape,
                           const std::vector<sd::LongType>& filterShape,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding,
                           DataType dtype);

/**
 * Get the global PJRT client instance.
 * Thread-safe singleton access.
 */
#if defined(HAVE_PJRT) && HAVE_PJRT
PjrtClient& getGlobalPjrtClient();
#endif

SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_PJRTUTILS_H
