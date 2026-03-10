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
#include <legacy/NativeOps.h>

#include <array/DataTypeUtils.h>
#include <array/NDArray.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <helpers/ShapeBuilders.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  std::string last_error;
  sdx_execution_report_t last_report{};
  std::vector<std::unique_ptr<sd::NDArray>> input_wrappers;
  std::vector<std::unique_ptr<sd::NDArray>> output_wrappers;
};

namespace {

constexpr int kSdxAbiVersion = SDX_RUNTIME_ABI_VERSION;
constexpr int kMinBackend = static_cast<int>(SDX_BACKEND_AUTO);
constexpr int kMaxBackend = static_cast<int>(SDX_BACKEND_NNAPI);
constexpr int kMinGpuTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
constexpr int kMaxGpuTarget = static_cast<int>(SDX_GPU_TARGET_AMD);

struct BundleManifestData {
  std::filesystem::path model_path;
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

bool readTextFile(const std::filesystem::path& path, std::string* out) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.good()) {
    return false;
  }
  out->assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

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

  std::filesystem::path modelPath = manifestData.model_path;
  if (!std::filesystem::exists(modelPath)) {
    setLastError(runtime, "Resolved model path does not exist: " + modelPath.string());
    return SDX_STATUS_IO_ERROR;
  }

  sd::Pointer modelHandle = loadModelFromFile(modelPath.string().c_str());
  if (modelHandle == nullptr) {
    const char* err = lastErrorMessage();
    if (err != nullptr && err[0] != '\0') {
      setLastError(runtime, std::string("loadModelFromFile failed: ") + err);
    } else {
      setLastError(runtime, "loadModelFromFile failed for: " + modelPath.string());
    }
    return SDX_STATUS_MODEL_LOAD_FAILED;
  }

  auto* model = new sdx_model_t();
  model->runtime = runtime;
  model->model_handle = modelHandle;
  model->bundle_path = bundle_path;
  model->model_path = modelPath.string();
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

  int requestedDeviceId = -1;
  bool sawCudaTensor = false;
  bool sawAmdTensor = false;
  auto collectDeviceInfo = [&](const sdx_tensor_view_t& tensor) -> sdx_status_t {
    if (!isCudaLikeDeviceType(tensor.device_type)) {
      return SDX_STATUS_OK;
    }
    if (tensor.device_id < 0) {
      setContextError(context, "CUDA/AMD tensor has invalid device_id");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
    if (requestedDeviceId < 0) {
      requestedDeviceId = tensor.device_id;
    } else if (requestedDeviceId != tensor.device_id) {
      setContextError(context, "Mixed CUDA/AMD device_id values are not supported in a single sdxRun call");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
    sawCudaTensor = sawCudaTensor || isCudaDeviceType(tensor.device_type);
    sawAmdTensor = sawAmdTensor || isAmdDeviceType(tensor.device_type);
    return SDX_STATUS_OK;
  };

  for (int i = 0; i < context->num_inputs; i++) {
    auto infoStatus = collectDeviceInfo(inputs[i]);
    if (infoStatus != SDX_STATUS_OK) {
      return infoStatus;
    }
  }
  for (int i = 0; i < context->num_outputs; i++) {
    auto infoStatus = collectDeviceInfo(outputs[i]);
    if (infoStatus != SDX_STATUS_OK) {
      return infoStatus;
    }
  }

  if (sawCudaTensor && sawAmdTensor) {
    setContextError(context, "Mixed CUDA and AMD tensor types are not supported in a single sdxRun call");
    return SDX_STATUS_INVALID_ARGUMENT;
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

#ifdef SD_CUDA
  struct DeviceScopeGuard {
    int previous = -1;
    bool switched = false;
    std::string switch_error;
    explicit DeviceScopeGuard(int targetDevice) {
      if (targetDevice < 0) return;
      previous = sd::AffinityManager::currentDeviceId();
      if (previous != targetDevice) {
        try {
          sd::AffinityManager::setCurrentDevice(targetDevice);
          switched = true;
        } catch (const std::exception& e) {
          switch_error = e.what();
        } catch (...) {
          switch_error = "unknown error";
        }
      }
    }
    ~DeviceScopeGuard() noexcept {
      if (switched && previous >= 0) {
        try {
          sd::AffinityManager::setCurrentDevice(previous);
        } catch (...) {
        }
      }
    }
    bool ok() const { return switch_error.empty(); }
  };
  DeviceScopeGuard deviceScope(requestedDeviceId);
  if (!deviceScope.ok()) {
    setContextError(context, "Failed to set CUDA/AMD device for sdxRun: " + deviceScope.switch_error);
    return SDX_STATUS_EXECUTION_FAILED;
  }
#endif

  setPlanGraphExecutionMode(context->plan_handle, requestedBackend);
  if (!context->model->allow_runtime_jit) {
    setPlanJitMode(context->plan_handle, 0);
  }
  applyGpuTargetHint(requestedGpuTarget);

  context->input_wrappers.clear();
  context->output_wrappers.clear();
  context->input_wrappers.reserve(static_cast<size_t>(context->num_inputs));
  context->output_wrappers.reserve(static_cast<size_t>(context->num_outputs));

  for (int i = 0; i < context->num_inputs; i++) {
    std::unique_ptr<sd::NDArray> wrapped;
    std::string error;
    auto status = wrapTensorView(inputs[i], &wrapped, &error);
    if (status != SDX_STATUS_OK) {
      setContextError(context, "Input tensor[" + std::to_string(i) + "] invalid: " + error);
      return status;
    }
    context->input_wrappers.emplace_back(std::move(wrapped));
  }

  bool hasCudaLikeTensors = false;
  for (int i = 0; i < context->num_inputs; i++) {
    if (isCudaLikeDeviceType(inputs[i].device_type)) {
      hasCudaLikeTensors = true;
      break;
    }
  }
  if (!hasCudaLikeTensors) {
    for (int i = 0; i < context->num_outputs; i++) {
      if (isCudaLikeDeviceType(outputs[i].device_type)) {
        hasCudaLikeTensors = true;
        break;
      }
    }
  }

#ifndef SD_CUDA
  if (hasCudaLikeTensors) {
    setContextError(context, "CUDA/AMD tensors require a CUDA-enabled runtime build");
    return SDX_STATUS_UNSUPPORTED;
  }
#endif

  if (hasCudaLikeTensors) {
    for (int i = 0; i < context->num_inputs; i++) {
      auto& in = context->input_wrappers[static_cast<size_t>(i)];
      if (inputs[i].device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
        in->syncToDevice();
      } else if (isCudaLikeDeviceType(inputs[i].device_type)) {
        in->syncToDevice();
      }
    }
  }

  for (int i = 0; i < context->num_outputs; i++) {
    std::unique_ptr<sd::NDArray> wrapped;
    std::string error;
    auto status = wrapTensorView(outputs[i], &wrapped, &error);
    if (status != SDX_STATUS_OK) {
      setContextError(context, "Output tensor[" + std::to_string(i) + "] invalid: " + error);
      return status;
    }
    context->output_wrappers.emplace_back(std::move(wrapped));
  }

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

  auto start = std::chrono::steady_clock::now();
  sd::Pointer execStream = nullptr;
#ifdef SD_CUDA
  if (hasCudaLikeTensors) {
    auto* launchContext = sd::LaunchContext::defaultContext();
    if (launchContext != nullptr) {
      execStream = reinterpret_cast<sd::Pointer>(launchContext->getCudaStream());
    }
  }
#endif
  int execCode = executeDynamicShapePlan(context->plan_handle, context->graph_context, execStream);
  auto end = std::chrono::steady_clock::now();
  uint64_t durationNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  ctxPurgeNoSync(context->graph_context);

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

  if (status != SDX_STATUS_OK) {
    const char* nativeError = lastErrorMessage();
    if (nativeError != nullptr && nativeError[0] != '\0') {
      setContextError(context, nativeError);
    } else {
      setContextError(context, "executeDynamicShapePlan failed with status " + std::to_string(execCode));
    }
    return status;
  }

  for (int i = 0; i < context->num_outputs; i++) {
    auto& out = context->output_wrappers[static_cast<size_t>(i)];
    if (outputs[i].device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
      out->syncToHost();
    } else if (isCudaLikeDeviceType(outputs[i].device_type)) {
      out->syncToDevice();
    }
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

}  // extern "C"
