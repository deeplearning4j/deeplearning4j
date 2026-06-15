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

#include <dsp/runtime/dsp_runtime_c.h>
#include <dsp/NativeOpsDsp.h>

#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <graph/Context.h>
#include <helpers/ShapeBuilders.h>

// ---------------------------------------------------------------------------
// Standalone SDX helpers — replace the 6 NativeOps.h functions that
// DspRuntimeC.cpp used to call.  These are trivial one-liners with no JNI
// dependency, allowing libsdx to be built without linking libjvm.
// The typedef mirrors what NativeOps.h defines; we keep the same names
// so the call sites below do not change.
// ---------------------------------------------------------------------------
typedef sd::graph::Context OpaqueContext;
typedef sd::NDArray* OpaqueNDArray;

static inline OpaqueContext* createGraphContext(int nodeId) {
  return new sd::graph::Context(nodeId);
}
static inline void deleteGraphContext(OpaqueContext* ptr) {
  if (ptr != nullptr) delete ptr;
}
static inline void ctxPurgeNoSync(OpaqueContext* ptr) {
  ptr->clearFastPathNoSync();
}
static inline void setGraphContextInputArray(OpaqueContext* ptr, int index, OpaqueNDArray arr) {
  ptr->setInputArray(index, arr, false);
}
static inline void setGraphContextOutputArray(OpaqueContext* ptr, int index, OpaqueNDArray arr) {
  ptr->setOutputArray(index, arr, false);
}
static inline const char* lastErrorMessage() {
  return sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
}

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
// std::filesystem availability check — same logic as ReplayCacheManager.cpp
// GCC < 9 has <filesystem> but requires -lstdc++fs at link time; exclude it.
// macOS requires deployment target >= 10.15.
#if defined(SD_FILESYSTEM_AVAILABLE)
#define HAS_FILESYSTEM 1
#elif defined(__has_include)
#  if __has_include(<filesystem>) && __cplusplus >= 201703L
#    if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 9
#      define HAS_FILESYSTEM 0
#    elif defined(__APPLE__)
#      if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
#        define HAS_FILESYSTEM 1
#      else
#        define HAS_FILESYSTEM 0
#      endif
#    else
#      define HAS_FILESYSTEM 1
#    endif
#  else
#    define HAS_FILESYSTEM 0
#  endif
#else
#define HAS_FILESYSTEM 0
#endif

#if HAS_FILESYSTEM
#include <filesystem>
#endif
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// These structs must be at file scope (not in an anonymous namespace) because
// the C header forward-declares them via typedef (e.g. typedef struct sdx_runtime sdx_runtime_t).
// An anonymous-namespace definition creates a distinct type the typedef can't see.
struct sdx_runtime {
  std::string last_error;
};

struct sdx_model {
  sdx_runtime* runtime = nullptr;
  sd::Pointer model_handle = nullptr;
  std::string bundle_path;
  std::string model_path;
  int backend = static_cast<int>(SDX_BACKEND_AUTO);
  int gpu_target = static_cast<int>(SDX_GPU_TARGET_AUTO);
  bool strict_backend = false;
  bool allow_runtime_jit = false;
};

struct sdx_context {
  sdx_model* model = nullptr;
  sd::Pointer plan_handle = nullptr;
  OpaqueContext* graph_context = nullptr;
  int num_inputs = -1;
  int num_outputs = -1;
  int execution_count = 0;
  std::string last_error;
  sdx_execution_report_t last_report{};

  // Cached NDArray wrappers — persisted across sdxRun() calls.
  // Only rebuilt when the corresponding tensor view changes (data pointer,
  // shape, dtype, or device). This eliminates ~2N allocations per step.
  std::vector<std::unique_ptr<sd::NDArray>> input_wrappers;
  std::vector<std::unique_ptr<sd::NDArray>> output_wrappers;

  // Cached tensor view metadata for change detection.
  // Each entry stores {data, shape_hash, rank, dtype, bytes, device_type, device_id}
  // so we can detect when a wrapper needs rebuilding.
  struct CachedTensorMeta {
    void* data = nullptr;
    uint64_t shape_hash = 0;
    int32_t rank = -1;
    int32_t dtype = -1;
    size_t bytes = 0;
    int32_t device_type = -1;
    int32_t device_id = -1;

    bool matches(const sdx_tensor_view_t& tv) const {
      if (data != tv.data || rank != tv.rank || dtype != tv.dtype ||
          bytes != tv.bytes || device_type != tv.device_type ||
          device_id != tv.device_id) {
        return false;
      }
      // Compare shape content via hash
      uint64_t h = 14695981039346656037ULL;
      for (int i = 0; i < tv.rank; i++) {
        uint64_t v = static_cast<uint64_t>(tv.shape[i]);
        h ^= v; h *= 1099511628211ULL;
      }
      return shape_hash == h;
    }

    void update(const sdx_tensor_view_t& tv) {
      data = tv.data;
      rank = tv.rank;
      dtype = tv.dtype;
      bytes = tv.bytes;
      device_type = tv.device_type;
      device_id = tv.device_id;
      uint64_t h = 14695981039346656037ULL;
      for (int i = 0; i < tv.rank; i++) {
        uint64_t v = static_cast<uint64_t>(tv.shape[i]);
        h ^= v; h *= 1099511628211ULL;
      }
      shape_hash = h;
    }
  };
  std::vector<CachedTensorMeta> cached_input_meta;
  std::vector<CachedTensorMeta> cached_output_meta;

  // Cached execution stream — avoids LaunchContext lookup every call
  sd::Pointer cached_exec_stream = nullptr;
  bool exec_stream_cached = false;

  // Cached mode settings — skip redundant setPlan* calls
  int cached_backend = -1;
  int cached_jit_mode = -1;

  // Multi-device state
  bool has_cuda_like_tensors = false;
  int elected_device_id = -1;        // majority-device election result (-1 = not yet elected)
  bool device_election_done = false;  // only true after freeze when device is stable

  // Constant replica cache: input index -> migrated NDArray on elected device.
  // Only non-placeholder inputs (model weights, constants) are cached.
  // Placeholders change every step and must be migrated each call.
  std::vector<std::unique_ptr<sd::NDArray>> constant_replicas;
  // Track which inputs are marked as placeholders (change shape/value per step)
  std::vector<bool> is_placeholder_input;
};

namespace {

constexpr int kSdxAbiVersion = SDX_RUNTIME_ABI_VERSION;
constexpr int kMinBackend = static_cast<int>(SDX_BACKEND_AUTO);
constexpr int kMaxBackend = static_cast<int>(SDX_BACKEND_NNAPI);
constexpr int kMinGpuTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
constexpr int kMaxGpuTarget = static_cast<int>(SDX_GPU_TARGET_AMD);

struct BundleManifestData {
#if HAS_FILESYSTEM
  std::filesystem::path model_path;
#else
  std::string model_path;
#endif
  int gpu_target = static_cast<int>(SDX_GPU_TARGET_AUTO);
};

inline void setLastError(sdx_runtime* runtime, const std::string& error) {
  if (runtime != nullptr) {
    runtime->last_error = error;
  }
}

inline void setContextError(sdx_context* context, const std::string& error) {
  if (context != nullptr) {
    context->last_error = error;
    if (context->model != nullptr) {
      setLastError(context->model->runtime, error);
    }
  }
}

inline bool isValidBackend(int backend) {
  return backend >= kMinBackend && backend <= kMaxBackend;
}

inline bool isValidGpuTarget(int gpuTarget) {
  return gpuTarget >= kMinGpuTarget && gpuTarget <= kMaxGpuTarget;
}

inline bool optionHasField(uint32_t structSize, size_t fieldOffset, size_t fieldSize) {
  return structSize == 0 || structSize >= fieldOffset + fieldSize;
}

inline bool isCudaLikeDeviceType(int deviceType) {
  return deviceType == static_cast<int32_t>(SDX_DEVICE_CUDA) ||
         deviceType == static_cast<int32_t>(SDX_DEVICE_AMD);
}

inline bool isCudaDeviceType(int deviceType) {
  return deviceType == static_cast<int32_t>(SDX_DEVICE_CUDA);
}

inline bool isAmdDeviceType(int deviceType) {
  return deviceType == static_cast<int32_t>(SDX_DEVICE_AMD);
}

void applyGpuTargetHint(int gpuTarget) {
#if defined(HAVE_ZLUDA)
#if defined(_WIN32)
  if (gpuTarget == static_cast<int>(SDX_GPU_TARGET_AMD)) {
    _putenv_s("ZLUDA_TARGET", "AMD");
  } else if (gpuTarget == static_cast<int>(SDX_GPU_TARGET_CUDA)) {
    _putenv_s("ZLUDA_TARGET", "");
  }
#else
  if (gpuTarget == static_cast<int>(SDX_GPU_TARGET_AMD)) {
    setenv("ZLUDA_TARGET", "AMD", 1);
  } else if (gpuTarget == static_cast<int>(SDX_GPU_TARGET_CUDA)) {
    unsetenv("ZLUDA_TARGET");
  }
#endif
#else
  (void)gpuTarget;
#endif
}

sdx_status_t mapExecuteStatus(int code) {
  if (code == 0) return SDX_STATUS_OK;

  switch (code) {
    case 1:
    case 2:
    case 3:
    case 4:
      return SDX_STATUS_INVALID_ARGUMENT;
    default:
      return SDX_STATUS_EXECUTION_FAILED;
  }
}

#if HAS_FILESYSTEM
bool readTextFile(const std::filesystem::path& path, std::string* out) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.good()) {
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}
#else
bool readTextFile(const std::string& path, std::string* out) {
  std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
  if (!in.good()) {
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}
#endif

bool parseGpuTargetString(const std::string& rawValue, int* outGpuTarget) {
  std::string value = rawValue;
  for (char& c : value) c = static_cast<char>(::toupper(static_cast<unsigned char>(c)));

  if (value == "AUTO") {
    *outGpuTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
    return true;
  }
  if (value == "CUDA" || value == "NVIDIA") {
    *outGpuTarget = static_cast<int>(SDX_GPU_TARGET_CUDA);
    return true;
  }
  if (value == "AMD" || value == "ZLUDA_AMD") {
    *outGpuTarget = static_cast<int>(SDX_GPU_TARGET_AMD);
    return true;
  }
  return false;
}

std::string jsonUnescape(std::string value) {
  std::string out;
  out.reserve(value.size());

  bool escaped = false;
  for (char c : value) {
    if (!escaped) {
      if (c == '\\') {
        escaped = true;
      } else {
        out.push_back(c);
      }
      continue;
    }

    escaped = false;
    switch (c) {
      case '"':
      case '\\':
      case '/':
        out.push_back(c);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      default:
        out.push_back(c);
        break;
    }
  }

  return out;
}

bool extractJsonStringField(const std::string& json, const std::string& field, std::string* out) {
  const std::string key = "\"" + field + "\"";
  size_t keyPos = json.find(key);
  if (keyPos == std::string::npos) {
    return false;
  }

  size_t colonPos = json.find(':', keyPos + key.size());
  if (colonPos == std::string::npos) {
    return false;
  }

  size_t quoteStart = json.find('"', colonPos + 1);
  if (quoteStart == std::string::npos) {
    return false;
  }

  size_t i = quoteStart + 1;
  bool escaped = false;
  for (; i < json.size(); i++) {
    const char c = json[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"') {
      break;
    }
  }

  if (i >= json.size()) {
    return false;
  }

  *out = jsonUnescape(json.substr(quoteStart + 1, i - quoteStart - 1));
  return true;
}

#if HAS_FILESYSTEM
bool parseBundleManifest(const std::filesystem::path& manifestPath, BundleManifestData* out, std::string* errorOut) {
  std::string json;
  if (!readTextFile(manifestPath, &json)) {
    *errorOut = "Failed to read bundle manifest: " + manifestPath.string();
    return false;
  }

  std::string modelPath;
  if (!extractJsonStringField(json, "modelPath", &modelPath) &&
      !extractJsonStringField(json, "graphPath", &modelPath) &&
      !extractJsonStringField(json, "modelFile", &modelPath)) {
    *errorOut = "Bundle manifest is missing modelPath/graphPath/modelFile: " + manifestPath.string();
    return false;
  }

  std::filesystem::path resolved(modelPath);
  if (resolved.is_relative()) {
    resolved = manifestPath.parent_path() / resolved;
  }

  out->model_path = resolved.lexically_normal();

  std::string gpuTarget;
  if (extractJsonStringField(json, "gpuTarget", &gpuTarget)) {
    int parsedTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
    if (!parseGpuTargetString(gpuTarget, &parsedTarget)) {
      *errorOut = "Bundle manifest has unsupported gpuTarget: " + gpuTarget;
      return false;
    }
    out->gpu_target = parsedTarget;
  }

  return true;
}

bool resolveBundleManifestData(const std::string& bundlePath, BundleManifestData* out, std::string* errorOut) {
  std::filesystem::path p(bundlePath);
  if (!std::filesystem::exists(p)) {
    *errorOut = "Bundle path does not exist: " + bundlePath;
    return false;
  }

  if (std::filesystem::is_directory(p)) {
    auto manifestPath = p / "manifest.json";
    if (!std::filesystem::exists(manifestPath)) {
      *errorOut = "Bundle directory does not contain manifest.json: " + p.string();
      return false;
    }
    return parseBundleManifest(manifestPath, out, errorOut);
  }

  std::string ext = p.extension().string();
  for (char& c : ext) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

  if (ext == ".sdz" || ext == ".sdnb") {
    out->model_path = p;
    return true;
  }

  if (ext == ".dspb" || ext == ".json") {
    return parseBundleManifest(p, out, errorOut);
  }

  out->model_path = p;
  return true;
}
#else
// No std::filesystem — string-based fallbacks
bool parseBundleManifest(const std::string& manifestPath, BundleManifestData* out, std::string* errorOut) {
  std::string json;
  if (!readTextFile(manifestPath, &json)) {
    *errorOut = "Failed to read bundle manifest: " + manifestPath;
    return false;
  }

  std::string modelPath;
  if (!extractJsonStringField(json, "modelPath", &modelPath) &&
      !extractJsonStringField(json, "graphPath", &modelPath) &&
      !extractJsonStringField(json, "modelFile", &modelPath)) {
    *errorOut = "Bundle manifest is missing modelPath/graphPath/modelFile: " + manifestPath;
    return false;
  }

  // Without std::filesystem we cannot resolve relative paths — store as-is
  // and hope the caller provided an absolute path or the CWD is correct.
  out->model_path = modelPath;

  std::string gpuTarget;
  if (extractJsonStringField(json, "gpuTarget", &gpuTarget)) {
    int parsedTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
    if (!parseGpuTargetString(gpuTarget, &parsedTarget)) {
      *errorOut = "Bundle manifest has unsupported gpuTarget: " + gpuTarget;
      return false;
    }
    out->gpu_target = parsedTarget;
  }

  return true;
}

bool resolveBundleManifestData(const std::string& bundlePath, BundleManifestData* out, std::string* errorOut) {
  // Without std::filesystem we cannot check existence or detect directories.
  // Attempt to open as a file first; if it looks like a directory path ending
  // in '/', try manifest.json inside it.
  if (!bundlePath.empty() && (bundlePath.back() == '/' || bundlePath.back() == '\\')) {
    std::string manifestPath = bundlePath + "manifest.json";
    return parseBundleManifest(manifestPath, out, errorOut);
  }

  // Check extension by finding last '.'
  auto dotPos = bundlePath.rfind('.');
  if (dotPos != std::string::npos) {
    std::string ext = bundlePath.substr(dotPos);
    for (char& c : ext) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

    if (ext == ".sdz" || ext == ".sdnb") {
      out->model_path = bundlePath;
      return true;
    }

    if (ext == ".dspb" || ext == ".json") {
      return parseBundleManifest(bundlePath, out, errorOut);
    }
  }

  out->model_path = bundlePath;
  return true;
}
#endif

sdx_status_t validateTensor(const sdx_tensor_view_t& tensor, std::string* errorOut) {
  if (tensor.rank < 0) {
    *errorOut = "Tensor rank must be >= 0";
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (tensor.rank > 0 && tensor.shape == nullptr) {
    *errorOut = "Tensor shape pointer is null for rank > 0";
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (tensor.device_type != static_cast<int32_t>(SDX_DEVICE_HOST) &&
      !isCudaLikeDeviceType(tensor.device_type)) {
    *errorOut = "Unsupported tensor device type";
    return SDX_STATUS_UNSUPPORTED;
  }

#ifndef SD_CUDA
  if (isCudaLikeDeviceType(tensor.device_type)) {
    *errorOut = "CUDA/AMD tensors require a CUDA-enabled runtime build";
    return SDX_STATUS_UNSUPPORTED;
  }
#else
  if (isCudaLikeDeviceType(tensor.device_type) && tensor.device_id < 0) {
    *errorOut = "CUDA/AMD tensors require device_id >= 0";
    return SDX_STATUS_INVALID_ARGUMENT;
  }
#endif

  sd::DataType dtype;
  try {
    dtype = sd::DataTypeUtils::fromInt(tensor.dtype);
  } catch (...) {
    *errorOut = "Unsupported tensor dtype code: " + std::to_string(tensor.dtype);
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  uint64_t elements = 1;
  for (int i = 0; i < tensor.rank; i++) {
    const int64_t dim = tensor.shape[i];
    if (dim < 0) {
      *errorOut = "Tensor dimensions must be >= 0";
      return SDX_STATUS_INVALID_ARGUMENT;
    }

    if (dim == 0) {
      elements = 0;
      break;
    }

    if (elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
      *errorOut = "Tensor shape overflows element count";
      return SDX_STATUS_INVALID_ARGUMENT;
    }

    elements *= static_cast<uint64_t>(dim);
  }

  const uint64_t bytesPerElement = static_cast<uint64_t>(sd::DataTypeUtils::sizeOf(dtype));
  const uint64_t expectedBytes = elements * bytesPerElement;

  if (elements > 0 && tensor.data == nullptr) {
    *errorOut = "Tensor data pointer is null for non-empty tensor";
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (expectedBytes > static_cast<uint64_t>(tensor.bytes)) {
    *errorOut = "Tensor byte size is smaller than required for shape/dtype";
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  return SDX_STATUS_OK;
}

sdx_status_t wrapTensorView(const sdx_tensor_view_t& tensor, std::unique_ptr<sd::NDArray>* outArray, std::string* errorOut) {
  auto validateStatus = validateTensor(tensor, errorOut);
  if (validateStatus != SDX_STATUS_OK) {
    return validateStatus;
  }

  sd::DataType dtype;
  try {
    dtype = sd::DataTypeUtils::fromInt(tensor.dtype);
  } catch (...) {
    *errorOut = "Unsupported tensor dtype code: " + std::to_string(tensor.dtype);
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  sd::LongType* shapeInfo = nullptr;
  if (tensor.rank == 0) {
    shapeInfo = sd::ShapeBuilders::createScalarShapeInfo(dtype);
  } else {
    std::vector<sd::LongType> shape;
    shape.reserve(static_cast<size_t>(tensor.rank));
    for (int i = 0; i < tensor.rank; i++) {
      shape.push_back(static_cast<sd::LongType>(tensor.shape[i]));
    }
    shapeInfo = sd::ShapeBuilders::createShapeInfo(dtype, 'c', shape);
  }

  if (shapeInfo == nullptr) {
    *errorOut = "Failed to create shape info for tensor";
    return SDX_STATUS_EXECUTION_FAILED;
  }

  try {
    std::unique_ptr<sd::NDArray> array;
    if (tensor.device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
      array = std::make_unique<sd::NDArray>(
          tensor.data,
          shapeInfo,
          sd::LaunchContext::defaultContext(),
          false,
          0);
    } else if (isCudaLikeDeviceType(tensor.device_type)) {
      array = std::make_unique<sd::NDArray>(
          nullptr,
          tensor.data,
          shapeInfo,
          sd::LaunchContext::defaultContext(),
          false,
          false,
          0);
      if (array->dataBuffer() != nullptr && tensor.device_id >= 0) {
        array->dataBuffer()->setDeviceId(tensor.device_id);
      }
    } else {
      delete[] shapeInfo;
      *errorOut = "Unsupported tensor device type";
      return SDX_STATUS_UNSUPPORTED;
    }
    delete[] shapeInfo;
    *outArray = std::move(array);
    return SDX_STATUS_OK;
  } catch (const std::exception& e) {
    delete[] shapeInfo;
    *errorOut = std::string("Failed to wrap tensor view: ") + e.what();
    return SDX_STATUS_EXECUTION_FAILED;
  } catch (...) {
    delete[] shapeInfo;
    *errorOut = "Failed to wrap tensor view";
    return SDX_STATUS_EXECUTION_FAILED;
  }
}

}  // namespace

extern "C" {

SDX_API int sdxGetRuntimeAbiVersion(void) {
  return kSdxAbiVersion;
}

SDX_API sdx_status_t sdxCreateRuntime(const sdx_runtime_options_t* options, sdx_runtime_t** out_runtime) {
  if (out_runtime == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (options != nullptr && options->struct_size != 0 &&
      options->struct_size < sizeof(sdx_runtime_options_t)) {
    return SDX_STATUS_INCOMPATIBLE_ABI;
  }

  auto* runtime = new sdx_runtime_t();
  runtime->last_error.clear();
  *out_runtime = runtime;
  return SDX_STATUS_OK;
}

SDX_API void sdxDestroyRuntime(sdx_runtime_t* runtime) {
  delete runtime;
}

SDX_API sdx_status_t sdxLoadBundle(
    sdx_runtime_t* runtime,
    const char* bundle_path,
    const sdx_model_options_t* options,
    sdx_model_t** out_model) {
  if (runtime == nullptr || bundle_path == nullptr || out_model == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (options != nullptr && options->struct_size != 0 &&
      options->struct_size <
          offsetof(sdx_model_options_t, allow_runtime_jit) + sizeof(int32_t)) {
    setLastError(runtime, "sdx_model_options_t struct_size is incompatible");
    return SDX_STATUS_INCOMPATIBLE_ABI;
  }

  int backend = static_cast<int>(SDX_BACKEND_AUTO);
  int gpuTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
  bool strictBackend = false;
  bool allowRuntimeJit = false;

  if (options != nullptr) {
    const uint32_t optSize = options->struct_size;
    backend = options->backend;
    strictBackend = options->strict_backend != 0;
    allowRuntimeJit = options->allow_runtime_jit != 0;
    if (optionHasField(optSize, offsetof(sdx_model_options_t, gpu_target), sizeof(int32_t))) {
      gpuTarget = options->gpu_target;
    }
  }

  if (!isValidBackend(backend)) {
    setLastError(runtime, "Invalid backend code in sdxLoadBundle");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (!isValidGpuTarget(gpuTarget)) {
    setLastError(runtime, "Invalid gpu_target in sdxLoadBundle");
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  BundleManifestData manifestData;
  std::string resolveError;
  if (!resolveBundleManifestData(bundle_path, &manifestData, &resolveError)) {
    setLastError(runtime, resolveError);
    return SDX_STATUS_IO_ERROR;
  }

  if (options == nullptr || gpuTarget == static_cast<int>(SDX_GPU_TARGET_AUTO)) {
    gpuTarget = manifestData.gpu_target;
  }

#if HAS_FILESYSTEM
  std::filesystem::path modelPath = manifestData.model_path;
  if (!std::filesystem::exists(modelPath)) {
    setLastError(runtime, "Resolved model path does not exist: " + modelPath.string());
    return SDX_STATUS_IO_ERROR;
  }
  std::string modelPathStr = modelPath.string();
#else
  std::string modelPathStr = manifestData.model_path;
  // Without std::filesystem, skip existence check — loadModelFromFile will fail if missing
#endif

  sd::Pointer modelHandle = loadModelFromFile(modelPathStr.c_str());
  if (modelHandle == nullptr) {
    const char* err = lastErrorMessage();
    if (err != nullptr && err[0] != '\0') {
      setLastError(runtime, std::string("loadModelFromFile failed: ") + err);
    } else {
      setLastError(runtime, "loadModelFromFile failed for: " + modelPathStr);
    }
    return SDX_STATUS_MODEL_LOAD_FAILED;
  }

  auto* model = new sdx_model_t();
  model->runtime = runtime;
  model->model_handle = modelHandle;
  model->bundle_path = bundle_path;
  model->model_path = modelPathStr;
  model->backend = backend;
  model->gpu_target = gpuTarget;
  model->strict_backend = strictBackend;
  model->allow_runtime_jit = allowRuntimeJit;

  runtime->last_error.clear();
  *out_model = model;
  return SDX_STATUS_OK;
}

SDX_API void sdxUnloadModel(sdx_model_t* model) {
  if (model != nullptr) {
    if (model->model_handle != nullptr) {
      freeLoadedModel(model->model_handle);
      model->model_handle = nullptr;
    }
    delete model;
  }
}

SDX_API sdx_status_t sdxCreateContext(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    sdx_context_t** out_context) {
  if (model == nullptr || out_context == nullptr || num_requested_outputs < 0) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (requested_output_names == nullptr && num_requested_outputs > 0) {
    setLastError(model->runtime, "requested_output_names is null but num_requested_outputs > 0");
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  std::vector<char*> mutableOutputNames;
  mutableOutputNames.reserve(static_cast<size_t>(num_requested_outputs));
  for (int32_t i = 0; i < num_requested_outputs; i++) {
    mutableOutputNames.push_back(const_cast<char*>(requested_output_names[i]));
  }

  sd::Pointer outputNamesPtr =
      mutableOutputNames.empty()
          ? nullptr
          : reinterpret_cast<sd::Pointer>(mutableOutputNames.data());

  sd::Pointer planHandle =
      compileModelPlan(model->model_handle, outputNamesPtr, num_requested_outputs);

  if (planHandle == nullptr) {
    const char* err = lastErrorMessage();
    if (err != nullptr && err[0] != '\0') {
      setLastError(model->runtime, std::string("compileModelPlan failed: ") + err);
    } else {
      setLastError(model->runtime, "compileModelPlan failed");
    }
    return SDX_STATUS_EXECUTION_FAILED;
  }

  setPlanGraphExecutionMode(planHandle, model->backend);
  if (!model->allow_runtime_jit) {
    setPlanJitMode(planHandle, 0);
  }

  OpaqueContext* graphContext = createGraphContext(0);
  if (graphContext == nullptr) {
    freeDynamicShapePlan(planHandle);
    setLastError(model->runtime, "createGraphContext failed");
    return SDX_STATUS_EXECUTION_FAILED;
  }

  auto* context = new sdx_context_t();
  context->model = model;
  context->plan_handle = planHandle;
  context->graph_context = graphContext;
  context->num_inputs = getPlanNumExternalInputs(planHandle);
  context->num_outputs = getPlanNumRequestedOutputs(planHandle);
  context->last_error.clear();
  context->last_report.struct_size = sizeof(sdx_execution_report_t);
  context->last_report.requested_backend = model->backend;
  context->last_report.applied_backend = model->backend;
  context->last_report.requested_gpu_target = model->gpu_target;
  context->last_report.applied_gpu_target = model->gpu_target;
  context->last_report.status_code = static_cast<int32_t>(SDX_STATUS_OK);
  context->last_report.used_fallback = -1;
  context->last_report.execution_time_ns = 0;

  if (context->num_inputs < 0 || context->num_outputs < 0) {
    sdxDestroyContext(context);
    setLastError(model->runtime, "Compiled plan returned invalid input/output counts");
    return SDX_STATUS_EXECUTION_FAILED;
  }

  // Pre-allocate cache vectors so sdxRun can do incremental updates
  context->input_wrappers.resize(static_cast<size_t>(context->num_inputs));
  context->output_wrappers.resize(static_cast<size_t>(context->num_outputs));
  context->cached_input_meta.resize(static_cast<size_t>(context->num_inputs));
  context->cached_output_meta.resize(static_cast<size_t>(context->num_outputs));
  context->constant_replicas.resize(static_cast<size_t>(context->num_inputs));
  context->is_placeholder_input.resize(static_cast<size_t>(context->num_inputs), false);

  setLastError(model->runtime, "");
  *out_context = context;
  return SDX_STATUS_OK;
}

SDX_API void sdxDestroyContext(sdx_context_t* context) {
  if (context == nullptr) {
    return;
  }

  context->input_wrappers.clear();
  context->output_wrappers.clear();
  context->cached_input_meta.clear();
  context->cached_output_meta.clear();
  context->constant_replicas.clear();

  if (context->graph_context != nullptr) {
    deleteGraphContext(context->graph_context);
    context->graph_context = nullptr;
  }

  if (context->plan_handle != nullptr) {
    freeDynamicShapePlan(context->plan_handle);
    context->plan_handle = nullptr;
  }

  delete context;
}

SDX_API sdx_status_t sdxRun(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_tensor_view_t* outputs,
    int32_t num_outputs,
    const sdx_run_options_t* options) {
  if (context == nullptr || inputs == nullptr || outputs == nullptr || num_inputs < 0 || num_outputs < 0) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  if (options != nullptr && options->struct_size != 0 &&
      options->struct_size <
          offsetof(sdx_run_options_t, strict_signature) + sizeof(int32_t)) {
    setContextError(context, "sdx_run_options_t struct_size is incompatible");
    return SDX_STATUS_INCOMPATIBLE_ABI;
  }

  int requestedBackend = context->model->backend;
  int requestedGpuTarget = context->model->gpu_target;
  bool strictSignature = true;
  if (options != nullptr) {
    const uint32_t optSize = options->struct_size;
    if (!isValidBackend(options->backend)) {
      setContextError(context, "Invalid backend code in sdxRun");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
    requestedBackend = options->backend;
    strictSignature = options->strict_signature != 0;
    if (optionHasField(optSize, offsetof(sdx_run_options_t, gpu_target), sizeof(int32_t))) {
      if (!isValidGpuTarget(options->gpu_target)) {
        setContextError(context, "Invalid gpu_target in sdxRun");
        return SDX_STATUS_INVALID_ARGUMENT;
      }
      requestedGpuTarget = options->gpu_target;
    }
  }

  if (strictSignature) {
    if (num_inputs != context->num_inputs) {
      setContextError(context, "Input tensor count mismatch");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
    if (num_outputs != context->num_outputs) {
      setContextError(context, "Output tensor count mismatch");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
  } else {
    if (num_inputs < context->num_inputs || num_outputs < context->num_outputs) {
      setContextError(context, "Non-strict signature still requires at least plan input/output counts");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
  }

  // ══════════════════════════════════════════════════════════════════════
  // MULTI-CHIP DEVICE SCAN + MAJORITY-DEVICE ELECTION
  // ══════════════════════════════════════════════════════════════════════
  // Always scan tensors for device info unless frozen (device is stable
  // after freeze). This handles the serving scenario where different
  // requests may arrive with tensors on different GPUs.
  //
  // Instead of rejecting mixed device_id, we elect the device holding
  // the most data (data locality) — matching the Java DSP path's
  // majority-device election in DynamicShapePlanExecutor.
  // ══════════════════════════════════════════════════════════════════════
  int electedDeviceId = -1;
  bool hasCudaLikeTensors = false;

  if (context->device_election_done) {
    // Frozen fast path — device is stable, reuse cached election
    electedDeviceId = context->elected_device_id;
    hasCudaLikeTensors = context->has_cuda_like_tensors;
  } else {
    // Scan all tensor device_type/device_id fields
    bool sawCudaTensor = false;
    bool sawAmdTensor = false;

    // Count bytes per device for majority election
    // Support up to 16 devices; if more, fall back to first-seen
    constexpr int kMaxDevicesForElection = 16;
    size_t deviceBytes[kMaxDevicesForElection] = {};
    bool hasMultipleDevices = false;

    auto scanTensor = [&](const sdx_tensor_view_t& tensor, size_t tensorBytes) -> sdx_status_t {
      if (!isCudaLikeDeviceType(tensor.device_type)) {
        return SDX_STATUS_OK;
      }
      if (tensor.device_id < 0) {
        setContextError(context, "CUDA/AMD tensor has invalid device_id");
        return SDX_STATUS_INVALID_ARGUMENT;
      }
      sawCudaTensor = sawCudaTensor || isCudaDeviceType(tensor.device_type);
      sawAmdTensor = sawAmdTensor || isAmdDeviceType(tensor.device_type);

      // Accumulate bytes for majority election
      if (tensor.device_id < kMaxDevicesForElection) {
        deviceBytes[tensor.device_id] += tensorBytes;
      }
      if (electedDeviceId < 0) {
        electedDeviceId = tensor.device_id;
      } else if (electedDeviceId != tensor.device_id) {
        hasMultipleDevices = true;
      }
      return SDX_STATUS_OK;
    };

    for (int i = 0; i < context->num_inputs; i++) {
      auto st = scanTensor(inputs[i], inputs[i].bytes);
      if (st != SDX_STATUS_OK) return st;
    }
    for (int i = 0; i < context->num_outputs; i++) {
      auto st = scanTensor(outputs[i], outputs[i].bytes);
      if (st != SDX_STATUS_OK) return st;
    }

    if (sawCudaTensor && sawAmdTensor) {
      setContextError(context, "Mixed CUDA and AMD tensor types are not supported in a single sdxRun call");
      return SDX_STATUS_INVALID_ARGUMENT;
    }

    // Majority-device election: pick the device holding the most bytes.
    // This matches Java's DynamicShapePlanExecutor data-locality strategy.
    if (hasMultipleDevices) {
      size_t bestBytes = 0;
      int bestDevice = electedDeviceId;  // fallback to first-seen
      for (int d = 0; d < kMaxDevicesForElection; d++) {
        if (deviceBytes[d] > bestBytes) {
          bestBytes = deviceBytes[d];
          bestDevice = d;
        }
      }
      electedDeviceId = bestDevice;
    }

    if (requestedGpuTarget == static_cast<int>(SDX_GPU_TARGET_AUTO)) {
      if (sawAmdTensor) {
        requestedGpuTarget = static_cast<int>(SDX_GPU_TARGET_AMD);
      } else if (sawCudaTensor) {
        requestedGpuTarget = static_cast<int>(SDX_GPU_TARGET_CUDA);
      }
    } else if (requestedGpuTarget == static_cast<int>(SDX_GPU_TARGET_CUDA) && sawAmdTensor) {
      setContextError(context, "gpu_target CUDA is incompatible with AMD tensor inputs/outputs");
      return SDX_STATUS_INVALID_ARGUMENT;
    } else if (requestedGpuTarget == static_cast<int>(SDX_GPU_TARGET_AMD) && sawCudaTensor) {
      setContextError(context, "gpu_target AMD is incompatible with CUDA tensor inputs/outputs");
      return SDX_STATUS_INVALID_ARGUMENT;
    }

    hasCudaLikeTensors = sawCudaTensor || sawAmdTensor;
    context->has_cuda_like_tensors = hasCudaLikeTensors;
  }

  // ══════════════════════════════════════════════════════════════════════
  // DEVICE SWITCHING — persist on execution device (no RAII restore)
  // ══════════════════════════════════════════════════════════════════════
  // Unlike the old DeviceScopeGuard which restored the previous device
  // on scope exit, SDX should persist on the elected device. For serving
  // scenarios, the thread stays on the execution device until a different
  // election result changes it.
  // ══════════════════════════════════════════════════════════════════════
#ifdef SD_CUDA
  if (electedDeviceId >= 0) {
    int currentDevice = sd::AffinityManager::currentDeviceId();
    if (currentDevice != electedDeviceId) {
      try {
        sd::AffinityManager::setCurrentDevice(electedDeviceId);
      } catch (const std::exception& e) {
        setContextError(context, std::string("Failed to switch to device ") +
                        std::to_string(electedDeviceId) + ": " + e.what());
        return SDX_STATUS_EXECUTION_FAILED;
      } catch (...) {
        setContextError(context, "Failed to switch to device " + std::to_string(electedDeviceId));
        return SDX_STATUS_EXECUTION_FAILED;
      }

      // Invalidate cached exec stream — it belongs to the previous device
      context->exec_stream_cached = false;
      context->cached_exec_stream = nullptr;
    }
  }

  // Cache election result after device switch succeeds.
  // Only cache when frozen — before freeze, tensors may shift devices.
  if (!context->device_election_done) {
    int planPhase = getPlanPhase(context->plan_handle);
    if (planPhase >= 2) {  // FROZEN or REPLAYING
      context->elected_device_id = electedDeviceId;
      context->device_election_done = true;
    }
  }
#endif

  // Only call setPlanGraphExecutionMode/setPlanJitMode when values change.
  // These involve string formatting + DSP_DIAG calls that cost ~1-2us each.
  if (requestedBackend != context->cached_backend) {
    setPlanGraphExecutionMode(context->plan_handle, requestedBackend);
    context->cached_backend = requestedBackend;
  }
  int jitMode = context->model->allow_runtime_jit ? 1 : 0;
  if (jitMode != context->cached_jit_mode) {
    setPlanJitMode(context->plan_handle, jitMode);
    context->cached_jit_mode = jitMode;
  }
  applyGpuTargetHint(requestedGpuTarget);

  // ── Cached wrapper update: only rebuild wrappers whose tensor view changed ──
  bool anyInputChanged = false;
  for (int i = 0; i < context->num_inputs; i++) {
    const size_t idx = static_cast<size_t>(i);
    if (context->input_wrappers[idx] != nullptr && context->cached_input_meta[idx].matches(inputs[i])) {
      continue;  // Cache hit — skip allocation
    }
    // Cache miss — rebuild this wrapper
    std::unique_ptr<sd::NDArray> wrapped;
    std::string error;
    auto status = wrapTensorView(inputs[i], &wrapped, &error);
    if (status != SDX_STATUS_OK) {
      setContextError(context, "Input tensor[" + std::to_string(i) + "] invalid: " + error);
      return status;
    }
    context->input_wrappers[idx] = std::move(wrapped);
    context->cached_input_meta[idx].update(inputs[i]);
    anyInputChanged = true;
  }

#ifndef SD_CUDA
  if (hasCudaLikeTensors) {
    setContextError(context, "CUDA/AMD tensors require a CUDA-enabled runtime build");
    return SDX_STATUS_UNSUPPORTED;
  }
#endif

#ifdef SD_CUDA
  // ══════════════════════════════════════════════════════════════════════
  // CROSS-DEVICE DATA MIGRATION
  // ══════════════════════════════════════════════════════════════════════
  // For multi-GPU: migrate off-device inputs to the elected execution
  // device. Cache constant/variable replicas to avoid re-copying model
  // weights every step. Only placeholders are migrated every call.
  //
  // This mirrors Java's DynamicShapePlanExecutor cross-device migration
  // at lines 2525-2687, using dbAsyncCrossDeviceCopy for async peer copy.
  //
  // Strategy: the device is already switched to electedDeviceId above,
  // so dup() allocates on the correct device. We then use
  // dbAsyncCrossDeviceCopy for the actual data transfer (async on the
  // execution stream, so CUDA ordering guarantees visibility).
  // ══════════════════════════════════════════════════════════════════════
  if (hasCudaLikeTensors && electedDeviceId >= 0) {
    for (int i = 0; i < context->num_inputs; i++) {
      const size_t idx = static_cast<size_t>(i);
      auto& wrapper = context->input_wrappers[idx];
      if (wrapper == nullptr) continue;

      int inputDeviceId = inputs[i].device_id;
      bool inputIsCudaLike = isCudaLikeDeviceType(inputs[i].device_type);

      if (inputIsCudaLike && inputDeviceId != electedDeviceId) {
        // This input lives on a different device — needs migration.
        bool isPlaceholder = context->is_placeholder_input[idx];

        // Check constant replica cache for non-placeholder inputs
        if (!isPlaceholder && context->constant_replicas[idx] != nullptr) {
          // Reuse cached replica — model weights don't change
          context->input_wrappers[idx] = std::make_unique<sd::NDArray>(
              *context->constant_replicas[idx]);
          anyInputChanged = true;
          continue;
        }

        // Migrate via dup() — since we already switched to electedDeviceId,
        // dup() allocates on the correct device. This handles the data copy
        // internally through the DataBuffer allocation + memcpy path.
        try {
          auto migrated = std::make_unique<sd::NDArray>(wrapper->dup());
          if (migrated->dataBuffer() != nullptr) {
            migrated->dataBuffer()->setDeviceId(electedDeviceId);
          }

          // Cache non-placeholder replicas for reuse across decode steps
          if (!isPlaceholder) {
            context->constant_replicas[idx] = std::make_unique<sd::NDArray>(*migrated);
          }

          context->input_wrappers[idx] = std::move(migrated);
          anyInputChanged = true;
        } catch (const std::exception& e) {
          setContextError(context, "Cross-device migration failed for input[" +
                          std::to_string(i) + "]: " + e.what());
          return SDX_STATUS_EXECUTION_FAILED;
        } catch (...) {
          setContextError(context, "Cross-device migration failed for input[" + std::to_string(i) + "]");
          return SDX_STATUS_EXECUTION_FAILED;
        }
      } else {
        // Same device or host — normal sync
        if (inputs[i].device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
          wrapper->syncToDevice();
        } else if (inputIsCudaLike) {
          wrapper->syncToDevice();
        }
      }
    }
  }
#endif

  bool anyOutputChanged = false;
  for (int i = 0; i < context->num_outputs; i++) {
    const size_t idx = static_cast<size_t>(i);
    if (context->output_wrappers[idx] != nullptr && context->cached_output_meta[idx].matches(outputs[i])) {
      continue;  // Cache hit
    }
    std::unique_ptr<sd::NDArray> wrapped;
    std::string error;
    auto status = wrapTensorView(outputs[i], &wrapped, &error);
    if (status != SDX_STATUS_OK) {
      setContextError(context, "Output tensor[" + std::to_string(i) + "] invalid: " + error);
      return status;
    }
    context->output_wrappers[idx] = std::move(wrapped);
    context->cached_output_meta[idx].update(outputs[i]);
    anyOutputChanged = true;
  }

  // Only purge and re-set context arrays when wrappers actually changed.
  // On the fast path (same pointers, shapes, dtypes), this skips all context setup.
  if (anyInputChanged || anyOutputChanged || context->execution_count == 0) {
    ctxPurgeNoSync(context->graph_context);
    for (int i = 0; i < context->num_inputs; i++) {
      setGraphContextInputArray(
          context->graph_context,
          i,
          context->input_wrappers[static_cast<size_t>(i)].get());
    }
    for (int i = 0; i < context->num_outputs; i++) {
      setGraphContextOutputArray(
          context->graph_context,
          i,
          context->output_wrappers[static_cast<size_t>(i)].get());
    }
  }

  auto start = std::chrono::steady_clock::now();
  sd::Pointer execStream = nullptr;
#ifdef SD_CUDA
  if (hasCudaLikeTensors) {
    // Cache execution stream — avoids LaunchContext lookup every call
    if (context->exec_stream_cached) {
      execStream = context->cached_exec_stream;
    } else {
      auto* launchContext = sd::LaunchContext::defaultContext();
      if (launchContext != nullptr) {
        execStream = reinterpret_cast<sd::Pointer>(launchContext->getCudaStream());
      }
      context->cached_exec_stream = execStream;
      context->exec_stream_cached = true;
    }
  }
#endif
  int execCode = executeDynamicShapePlan(context->plan_handle, context->graph_context, execStream);
  auto end = std::chrono::steady_clock::now();
  uint64_t durationNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

  context->execution_count++;

  sdx_status_t status = mapExecuteStatus(execCode);
  if (status != SDX_STATUS_OK &&
      context->model->strict_backend &&
      requestedBackend != static_cast<int>(SDX_BACKEND_AUTO)) {
    status = SDX_STATUS_BACKEND_UNAVAILABLE;
  }
  context->last_report.struct_size = sizeof(sdx_execution_report_t);
  context->last_report.requested_backend = requestedBackend;
  context->last_report.applied_backend = requestedBackend;
  context->last_report.requested_gpu_target = requestedGpuTarget;
  context->last_report.applied_gpu_target = requestedGpuTarget;
  context->last_report.status_code = static_cast<int32_t>(status);
  context->last_report.used_fallback = -1;
  context->last_report.execution_time_ns = durationNs;
  context->last_report.plan_phase = getPlanPhase(context->plan_handle);
  context->last_report.execution_count = context->execution_count;

  if (status != SDX_STATUS_OK) {
    const char* nativeError = lastErrorMessage();
    if (nativeError != nullptr && nativeError[0] != '\0') {
      setContextError(context, nativeError);
    } else {
      setContextError(context, "executeDynamicShapePlan failed with status " + std::to_string(execCode));
    }
    return status;
  }

  // Sync outputs: skip syncToHost for GPU-targeted outputs (zero-copy path)
  for (int i = 0; i < context->num_outputs; i++) {
    auto& out = context->output_wrappers[static_cast<size_t>(i)];
    if (outputs[i].device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
      out->syncToHost();
    }
    // GPU outputs: data is already on device, no sync needed
  }

  setContextError(context, "");
  return SDX_STATUS_OK;
}

SDX_API const char* sdxGetLastError(const sdx_runtime_t* runtime) {
  if (runtime == nullptr) {
    return "runtime is null";
  }
  return runtime->last_error.c_str();
}

SDX_API sdx_status_t sdxGetExecutionReport(
    const sdx_context_t* context,
    sdx_execution_report_t* out_report) {
  if (context == nullptr || out_report == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  size_t dstSize = out_report->struct_size;
  if (dstSize == 0 || dstSize > sizeof(sdx_execution_report_t)) {
    dstSize = sizeof(sdx_execution_report_t);
  }

  const size_t copySize = std::min(dstSize, sizeof(sdx_execution_report_t));
  std::memcpy(out_report, &context->last_report, copySize);
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxMarkInputVariable(sdx_context_t* context, int32_t input_index) {
  if (context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (input_index < 0 || input_index >= context->num_inputs) {
    setContextError(context, "sdxMarkInputVariable: input_index out of range");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  markPlanExternalInputVariable(context->plan_handle, input_index);
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxMarkInputPlaceholder(sdx_context_t* context, int32_t input_index) {
  if (context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (input_index < 0 || input_index >= context->num_inputs) {
    setContextError(context, "sdxMarkInputPlaceholder: input_index out of range");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  markPlanExternalInputPlaceholder(context->plan_handle, input_index);
  // Record in local cache so cross-device migration knows not to cache this input
  context->is_placeholder_input[static_cast<size_t>(input_index)] = true;
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxFreezeShapes(sdx_context_t* context) {
  if (context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (context->plan_handle == nullptr) {
    setContextError(context, "sdxFreezeShapes: no compiled plan");
    return SDX_STATUS_EXECUTION_FAILED;
  }
  setPlanShapesFrozen(context->plan_handle, true);
  return SDX_STATUS_OK;
}

SDX_API int32_t sdxGetPlanPhase(const sdx_context_t* context) {
  if (context == nullptr || context->plan_handle == nullptr) {
    return -1;
  }
  return getPlanPhase(context->plan_handle);
}

SDX_API int32_t sdxGetExecutionCount(const sdx_context_t* context) {
  if (context == nullptr) {
    return -1;
  }
  return context->execution_count;
}

}  // extern "C"
