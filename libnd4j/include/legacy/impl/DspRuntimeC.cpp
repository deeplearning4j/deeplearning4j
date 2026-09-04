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
#include <dsp/runtime/detail/DspRuntimeInternal.h>
#include <dsp/NativeOpsDsp.h>
#include <graph/SdzReader.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/DspDiagnostics.h>
#include <graph/NativeDynamicShapePlan.h>

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
#include <atomic>
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
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// These structs must be at file scope (not in an anonymous namespace) because
// the C header forward-declares them via typedef (e.g. typedef struct sdx_runtime sdx_runtime_t).
// An anonymous-namespace definition creates a distinct type the typedef can't see.
struct sdx_runtime {
  std::string last_error;
  // Guards last_error across threads. sdxGetLastError copies into
  // error_snapshot (fixed storage owned by the runtime) so the returned
  // pointer stays valid even if another thread updates the error afterwards;
  // only the content may change on a subsequent error.
  mutable std::mutex error_mutex;
  char error_snapshot[4096] = {0};
};

struct sdx_model {
  sdx_runtime* runtime = nullptr;
  sd::Pointer model_handle = nullptr;
  std::string bundle_path;
  std::string model_path;
  // Temp dir a packed .dspb archive was extracted into; removed at unload.
  std::string extracted_dir;
  int backend = static_cast<int>(SDX_BACKEND_AUTO);
  int gpu_target = static_cast<int>(SDX_GPU_TARGET_AUTO);
  bool strict_backend = false;
  bool allow_runtime_jit = false;
  std::string runtime_artifact_directory;
  std::string device_compilation_cache_directory;
  std::string device_compilation_cache_model_key;
  std::string tokenizer_path;
  std::string text_generation_config_path;
};

struct sdx_context {
  sdx_model* model = nullptr;
  sd::Pointer plan_handle = nullptr;
  OpaqueContext* graph_context = nullptr;
  // num_inputs is the public C ABI input count. plan_num_inputs is the full
  // executor width, including model-owned constants/variables.
  int num_inputs = -1;
  int plan_num_inputs = -1;
  int num_outputs = -1;
  bool binds_model_parameters = false;
  // Public context input index -> full plan input index and inverse mapping.
  std::vector<int> public_to_plan_input;
  std::vector<int> plan_to_public_input;
  // Borrowed model-owned NDArrays, indexed in full plan input order. The model
  // is required to outlive this context.
  std::vector<sd::NDArray*> bound_model_inputs;
  int execution_count = 0;
  std::string last_error;
  sdx_execution_report_t last_report{};

  // Cached NDArray wrappers — persisted across sdxRun() calls.
  // Only rebuilt when the corresponding tensor view changes (data pointer,
  // shape, dtype, or device). This eliminates ~2N allocations per step.
  std::vector<std::unique_ptr<sd::NDArray>> input_wrappers;
  std::vector<std::unique_ptr<sd::NDArray>> output_wrappers;
  // Explicit output names copied at context creation. Runtime-owned contiguous
  // copies are populated lazily only when a produced output is a strided/view
  // array that the C tensor-view ABI cannot represent.
  std::vector<std::string> output_names;
  std::vector<std::unique_ptr<sd::NDArray>> borrowed_output_copies;

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

  // Serializes sdxRun / report reads / plan-state mutations on this context.
  // A context is a single execution stream; concurrent callers are serialized
  // rather than corrupting the cached wrappers and graph context.
  mutable std::mutex exec_mutex;
};

namespace {

constexpr int kSdxAbiVersion = SDX_RUNTIME_ABI_VERSION;
constexpr int kMinBackend = static_cast<int>(SDX_BACKEND_AUTO);
constexpr int kMaxBackend = static_cast<int>(SDX_BACKEND_OPENVINO);
constexpr int kMinGpuTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
constexpr int kMaxGpuTarget = static_cast<int>(SDX_GPU_TARGET_METAL);

struct BundleManifestData {
#if HAS_FILESYSTEM
  std::filesystem::path model_path;
#else
  std::string model_path;
#endif
  // Set when a packed .dspb archive was extracted; the model owns the dir.
  std::string extracted_dir;
  int gpu_target = static_cast<int>(SDX_GPU_TARGET_AUTO);
  std::string vulkan_spirv_directory;
  std::string hexagon_kernel_directory;
  std::string tokenizer_path;
  std::string text_generation_config_path;
  std::string device_compilation_cache_model_key;
  std::vector<std::string> targets;
};

// Sniff the 4-byte local-file-header magic that identifies a packed (ZIP)
// .dspb archive, as opposed to a manifest-JSON .dspb file.
inline bool fileHasZipMagic(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return false;
  char magic[4] = {0, 0, 0, 0};
  f.read(magic, 4);
  return f.gcount() == 4 && magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 &&
         magic[3] == 0x04;
}

inline void setLastError(sdx_runtime* runtime, const std::string& error) {
  if (runtime != nullptr) {
    std::lock_guard<std::mutex> lock(runtime->error_mutex);
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

std::string describePublicInput(
    const sdx_context* context,
    int publicIndex) {
  std::string description = "public_input=" + std::to_string(publicIndex);
  if (context == nullptr || publicIndex < 0 ||
      publicIndex >= context->num_inputs ||
      static_cast<size_t>(publicIndex) >= context->public_to_plan_input.size()) {
    return description;
  }

  const int planIndex =
      context->public_to_plan_input[static_cast<size_t>(publicIndex)];
  description += " plan_input=" + std::to_string(planIndex);
  const char* inputName =
      getPlanExternalInputName(context->plan_handle, planIndex);
  description += " name=";
  description += inputName == nullptr ? "<unnamed>" : inputName;

  auto* plan = reinterpret_cast<sd::graph::NativeDynamicShapePlan*>(
      context->plan_handle);
  if (plan == nullptr) return description;
  const auto* slots = plan->getSlots();
  const int numSlots = plan->getNumSlots();
  const int externalSource = -(planIndex + 1);
  int consumers = 0;
  for (int slotIndex = 0; slotIndex < numSlots; ++slotIndex) {
    const auto& slot = slots[slotIndex];
    for (int opInput = 0; opInput < slot.wiring.numInputs; ++opInput) {
      if (slot.wiring.inputSourceIndices[opInput] != externalSource) continue;
      description += consumers++ == 0 ? " consumers=[" : ",";
      description += "slot=" + std::to_string(slotIndex);
      description += ":op=";
      description += slot.ident.opName.empty() ? "<unnamed>" : slot.ident.opName;
      description += ":input=" + std::to_string(opInput);
    }
  }
  description += consumers == 0 ? " consumers=[]" : "]";
  return description;
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
  if (value == "VULKAN") {
    *outGpuTarget = static_cast<int>(SDX_GPU_TARGET_VULKAN);
    return true;
  }
  if (value == "METAL" || value == "MPS") {
    *outGpuTarget = static_cast<int>(SDX_GPU_TARGET_METAL);
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

bool extractJsonStringArrayField(const std::string& json,
                                 const std::string& field,
                                 std::vector<std::string>* out,
                                 std::string* errorOut) {
  out->clear();
  const std::string key = "\"" + field + "\"";
  const size_t keyPos = json.find(key);
  if (keyPos == std::string::npos) return true;

  const size_t colonPos = json.find(':', keyPos + key.size());
  const size_t arrayStart =
      colonPos == std::string::npos ? std::string::npos
                                    : json.find('[', colonPos + 1);
  if (arrayStart == std::string::npos) {
    *errorOut = "Bundle manifest field " + field + " must be a string array";
    return false;
  }

  size_t cursor = arrayStart + 1;
  while (cursor < json.size()) {
    while (cursor < json.size() &&
           std::isspace(static_cast<unsigned char>(json[cursor]))) {
      cursor++;
    }
    if (cursor >= json.size()) break;
    if (json[cursor] == ']') return true;
    if (json[cursor] != '"') {
      *errorOut = "Bundle manifest field " + field +
                  " contains a non-string value";
      return false;
    }

    const size_t quoteStart = ++cursor;
    bool escaped = false;
    for (; cursor < json.size(); cursor++) {
      const char c = json[cursor];
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        break;
      }
    }
    if (cursor >= json.size()) {
      *errorOut = "Bundle manifest field " + field +
                  " has an unterminated string";
      return false;
    }
    out->push_back(jsonUnescape(
        json.substr(quoteStart, cursor - quoteStart)));
    cursor++;
    while (cursor < json.size() &&
           std::isspace(static_cast<unsigned char>(json[cursor]))) {
      cursor++;
    }
    if (cursor < json.size() && json[cursor] == ',') {
      cursor++;
      continue;
    }
    if (cursor < json.size() && json[cursor] == ']') return true;
    *errorOut = "Bundle manifest field " + field +
                " must separate target strings with commas";
    return false;
  }

  *errorOut = "Bundle manifest field " + field + " is unterminated";
  return false;
}

std::string requiredBundleTarget(int backend, int gpuTarget) {
  if (backend == static_cast<int>(SDX_BACKEND_VULKAN) ||
      gpuTarget == static_cast<int>(SDX_GPU_TARGET_VULKAN)) {
    return "android-arm64-vulkan";
  }
  if (backend == static_cast<int>(SDX_BACKEND_HEXAGON)) {
    return "android-arm64-hexagon-htp";
  }
  if (backend == static_cast<int>(SDX_BACKEND_NNAPI) ||
      backend == static_cast<int>(SDX_BACKEND_ARM_HYBRID)) {
    return "android-arm64-nnapi-accelerator";
  }
  if (backend == static_cast<int>(SDX_BACKEND_METAL) ||
      backend == static_cast<int>(SDX_BACKEND_MLX) ||
      gpuTarget == static_cast<int>(SDX_GPU_TARGET_METAL)) {
    return "ios-arm64-metal";
  }
  return std::string();
}

std::string resolveManifestRelativePath(const std::string& manifestPath,
                                        const std::string& assetPath) {
  if (assetPath.empty() || assetPath.front() == '/' ||
      assetPath.front() == '\\' ||
      (assetPath.size() >= 2 &&
       std::isalpha(static_cast<unsigned char>(assetPath[0])) &&
       assetPath[1] == ':')) {
    return assetPath;
  }
  const size_t separator = manifestPath.find_last_of("/\\");
  return separator == std::string::npos
             ? assetPath
             : manifestPath.substr(0, separator + 1) + assetPath;
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

  auto resolveManifestDirectory =
      [&](const char* field, std::string* destination) -> bool {
    std::string relativePath;
    if (!extractJsonStringField(json, field, &relativePath) ||
        relativePath.empty()) {
      return true;
    }
    std::filesystem::path resolvedArtifacts(relativePath);
    if (resolvedArtifacts.is_relative()) {
      resolvedArtifacts = manifestPath.parent_path() / resolvedArtifacts;
    }
    resolvedArtifacts = resolvedArtifacts.lexically_normal();
    if (!std::filesystem::exists(resolvedArtifacts) ||
        !std::filesystem::is_directory(resolvedArtifacts)) {
      *errorOut = std::string("Bundle manifest ") + field +
                  " directory does not exist: " + resolvedArtifacts.string();
      return false;
    }
    *destination = resolvedArtifacts.string();
    return true;
  };
  if (!resolveManifestDirectory("vulkanSpirv",
                                &out->vulkan_spirv_directory) ||
      !resolveManifestDirectory("hexagonKernels",
                                &out->hexagon_kernel_directory)) {
    return false;
  }

  auto resolveManifestAsset =
      [&](const char* field, std::string* destination) -> bool {
    std::string relativePath;
    if (!extractJsonStringField(json, field, &relativePath) ||
        relativePath.empty()) {
      return true;
    }
    std::filesystem::path resolvedAsset(relativePath);
    if (resolvedAsset.is_relative()) {
      resolvedAsset = manifestPath.parent_path() / resolvedAsset;
    }
    resolvedAsset = resolvedAsset.lexically_normal();
    if (!std::filesystem::exists(resolvedAsset) ||
        !std::filesystem::is_regular_file(resolvedAsset)) {
      *errorOut = std::string("Bundle manifest ") + field +
                  " file does not exist: " + resolvedAsset.string();
      return false;
    }
    *destination = resolvedAsset.string();
    return true;
  };
  if (!resolveManifestAsset("tokenizerPath", &out->tokenizer_path) ||
      !resolveManifestAsset("configPath",
                            &out->text_generation_config_path)) {
    return false;
  }

  std::string gpuTarget;
  if (extractJsonStringField(json, "gpuTarget", &gpuTarget)) {
    int parsedTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
    if (!parseGpuTargetString(gpuTarget, &parsedTarget)) {
      *errorOut = "Bundle manifest has unsupported gpuTarget: " + gpuTarget;
      return false;
    }
    out->gpu_target = parsedTarget;
  }

  extractJsonStringField(json, "compileKey",
                         &out->device_compilation_cache_model_key);
  if (!extractJsonStringArrayField(json, "targets", &out->targets,
                                   errorOut)) {
    return false;
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
    // Packed .dspb: a ZIP archive of an unpacked bundle directory
    // (manifest.json + graph/weights/segments, per sdx-compile.sh layout).
    // Extract to a temp dir owned by the model, then resolve as a directory.
    if (ext == ".dspb" && fileHasZipMagic(p.string())) {
      static std::atomic<uint64_t> extractCounter{0};
      std::error_code ec;
      const auto tempBase = std::filesystem::temp_directory_path(ec);
      if (ec) {
        *errorOut = "Cannot resolve temp directory for packed .dspb extraction";
        return false;
      }
      std::filesystem::path extractDir;
      for (int attempt = 0; attempt < 1024 && extractDir.empty(); attempt++) {
        auto candidate =
            tempBase / ("sdx-dspb-" + std::to_string(extractCounter.fetch_add(1)));
        std::error_code createEc;
        if (std::filesystem::create_directory(candidate, createEc)) {
          extractDir = candidate;
        }
      }
      if (extractDir.empty()) {
        *errorOut = "Could not allocate temp extraction dir for packed .dspb bundle";
        return false;
      }

      std::string zipError;
      if (!sd::graph::SdzReader::extractArchive(p.string().c_str(),
                                                extractDir.string().c_str(), &zipError)) {
        std::error_code cleanupEc;
        std::filesystem::remove_all(extractDir, cleanupEc);
        *errorOut = "Failed to extract packed .dspb bundle: " + zipError;
        return false;
      }

      auto manifestPath = extractDir / "manifest.json";
      if (!std::filesystem::exists(manifestPath)) {
        std::error_code cleanupEc;
        std::filesystem::remove_all(extractDir, cleanupEc);
        *errorOut = "Packed .dspb archive has no manifest.json: " + p.string();
        return false;
      }
      if (!parseBundleManifest(manifestPath, out, errorOut)) {
        std::error_code cleanupEc;
        std::filesystem::remove_all(extractDir, cleanupEc);
        return false;
      }
      out->extracted_dir = extractDir.string();
      return true;
    }
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

  // iOS toolchains may intentionally compile without std::filesystem. Resolve
  // bundle-relative paths lexically so unpacked mobile bundles still work.
  out->model_path = resolveManifestRelativePath(manifestPath, modelPath);

  std::string vulkanSpirvPath;
  if (extractJsonStringField(json, "vulkanSpirv", &vulkanSpirvPath)) {
    out->vulkan_spirv_directory =
        resolveManifestRelativePath(manifestPath, vulkanSpirvPath);
  }
  std::string hexagonKernelPath;
  if (extractJsonStringField(json, "hexagonKernels", &hexagonKernelPath)) {
    out->hexagon_kernel_directory =
        resolveManifestRelativePath(manifestPath, hexagonKernelPath);
  }
  std::string tokenizerPath;
  if (extractJsonStringField(json, "tokenizerPath", &tokenizerPath)) {
    out->tokenizer_path =
        resolveManifestRelativePath(manifestPath, tokenizerPath);
  }
  std::string configPath;
  if (extractJsonStringField(json, "configPath", &configPath)) {
    out->text_generation_config_path =
        resolveManifestRelativePath(manifestPath, configPath);
  }

  std::string gpuTarget;
  if (extractJsonStringField(json, "gpuTarget", &gpuTarget)) {
    int parsedTarget = static_cast<int>(SDX_GPU_TARGET_AUTO);
    if (!parseGpuTargetString(gpuTarget, &parsedTarget)) {
      *errorOut = "Bundle manifest has unsupported gpuTarget: " + gpuTarget;
      return false;
    }
    out->gpu_target = parsedTarget;
  }

  extractJsonStringField(json, "compileKey",
                         &out->device_compilation_cache_model_key);
  if (!extractJsonStringArrayField(json, "targets", &out->targets,
                                   errorOut)) {
    return false;
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
      if (ext == ".dspb" && fileHasZipMagic(bundlePath)) {
        *errorOut = "Packed .dspb archives require std::filesystem support in this build";
        return false;
      }
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

  if (!sd::graph::dspIsCudaBuild()) {
    if (isCudaLikeDeviceType(tensor.device_type)) {
      *errorOut = "CUDA/AMD tensors require a CUDA-enabled runtime build";
      return SDX_STATUS_UNSUPPORTED;
    }
  } else {
    if (isCudaLikeDeviceType(tensor.device_type) && tensor.device_id < 0) {
      *errorOut = "CUDA/AMD tensors require device_id >= 0";
      return SDX_STATUS_INVALID_ARGUMENT;
    }
  }

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

SDX_API sdx_status_t sdxConfigureDiagnostics(
    sdx_runtime_t* runtime,
    uint32_t category_mask,
    int32_t level,
    const char* json_path) {
  if (runtime == nullptr) return SDX_STATUS_INVALID_ARGUMENT;
  if ((category_mask & ~sd::graph::DSP_DIAG_ALL) != 0) {
    setLastError(runtime, "diagnostic category mask contains unsupported bits");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (level < static_cast<int32_t>(sd::graph::DSP_LEVEL_SUMMARY) ||
      level > static_cast<int32_t>(sd::graph::DSP_LEVEL_FULL)) {
    setLastError(runtime, "diagnostic level must be summary, detailed, or full");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (json_path == nullptr) {
    setLastError(runtime, "diagnostic JSON path is null");
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  auto& diagnostics = sd::graph::DspDiagnostics::getInstance();
  diagnostics.setCategories(category_mask);
  diagnostics.setLevel(static_cast<sd::graph::DspDiagLevel>(level));
  diagnostics.setJsonPath(json_path);
  setLastError(runtime, "");
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxRecordDiagnosticEvent(
    sdx_runtime_t* runtime,
    uint32_t category,
    const char* message) {
  if (runtime == nullptr) return SDX_STATUS_INVALID_ARGUMENT;
  if (category == 0 || (category & ~sd::graph::DSP_DIAG_ALL) != 0 ||
      (category & (category - 1)) != 0) {
    setLastError(runtime, "diagnostic event requires one supported category");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  if (message == nullptr) {
    setLastError(runtime, "diagnostic event message is null");
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  auto& diagnostics = sd::graph::DspDiagnostics::getInstance();
  if (diagnostics.isEnabled(category)) {
    diagnostics.recordEvent(
        category, -1, -1, -1, nullptr, 0, "%s", message);
  }
  setLastError(runtime, "");
  return SDX_STATUS_OK;
}

SDX_API void sdxClearDiagnostics(void) {
  sd::graph::DspDiagnostics::getInstance().clear();
}

SDX_API void sdxFlushDiagnostics(void) {
  sd::graph::DspDiagnostics::getInstance().flushJsonReport();
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
  std::string deviceCompilationCacheDirectory;

  if (options != nullptr) {
    const uint32_t optSize = options->struct_size;
    backend = options->backend;
    strictBackend = options->strict_backend != 0;
    allowRuntimeJit = options->allow_runtime_jit != 0;
    if (optionHasField(optSize, offsetof(sdx_model_options_t, gpu_target), sizeof(int32_t))) {
      gpuTarget = options->gpu_target;
    }
    // A zero struct_size is accepted for the original fixed-width ABI only. Do
    // not read an appended pointer from a legacy caller whose allocation may be
    // smaller than the current structure.
    if (optSize != 0 &&
        optionHasField(optSize,
                       offsetof(sdx_model_options_t, device_compilation_cache_directory),
                       sizeof(const char*)) &&
        options->device_compilation_cache_directory != nullptr) {
      deviceCompilationCacheDirectory = options->device_compilation_cache_directory;
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

  const std::string requiredTarget = requiredBundleTarget(backend, gpuTarget);
  if (strictBackend && !requiredTarget.empty() && !manifestData.targets.empty() &&
      std::find(manifestData.targets.begin(), manifestData.targets.end(),
                requiredTarget) == manifestData.targets.end()) {
#if HAS_FILESYSTEM
    if (!manifestData.extracted_dir.empty()) {
      std::error_code cleanupEc;
      std::filesystem::remove_all(manifestData.extracted_dir, cleanupEc);
    }
#endif
    std::string declaredTargets;
    for (size_t i = 0; i < manifestData.targets.size(); i++) {
      if (i > 0) declaredTargets += ", ";
      declaredTargets += manifestData.targets[i];
    }
    setLastError(runtime, "Bundle target mismatch: strict backend requires " +
                              requiredTarget + "; manifest declares [" +
                              declaredTargets + "]");
    return SDX_STATUS_MODEL_LOAD_FAILED;
  }

  const bool precompiledOnlyVulkan =
      !allowRuntimeJit &&
      (gpuTarget == static_cast<int>(SDX_GPU_TARGET_VULKAN) ||
       backend == static_cast<int>(SDX_BACKEND_VULKAN));
  if (precompiledOnlyVulkan && manifestData.vulkan_spirv_directory.empty()) {
#if HAS_FILESYSTEM
    if (!manifestData.extracted_dir.empty()) {
      std::error_code cleanupEc;
      std::filesystem::remove_all(manifestData.extracted_dir, cleanupEc);
    }
#endif
    setLastError(
        runtime,
        "Vulkan bundle forbids runtime compilation but "
        "compiledArtifacts.vulkanSpirv is missing");
    return SDX_STATUS_MODEL_LOAD_FAILED;
  }

  const bool precompiledOnlyHexagon =
      !allowRuntimeJit && backend == static_cast<int>(SDX_BACKEND_HEXAGON);
  if (precompiledOnlyHexagon && manifestData.hexagon_kernel_directory.empty()) {
#if HAS_FILESYSTEM
    if (!manifestData.extracted_dir.empty()) {
      std::error_code cleanupEc;
      std::filesystem::remove_all(manifestData.extracted_dir, cleanupEc);
    }
#endif
    setLastError(
        runtime,
        "Hexagon bundle forbids runtime compilation but "
        "compiledArtifacts.hexagonKernels is missing");
    return SDX_STATUS_MODEL_LOAD_FAILED;
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

  const bool requireFileBacked =
      strictBackend ||
      backend == static_cast<int>(SDX_BACKEND_NNAPI) ||
      backend == static_cast<int>(SDX_BACKEND_ARM_HYBRID) ||
      backend == static_cast<int>(SDX_BACKEND_HEXAGON) ||
      backend == static_cast<int>(SDX_BACKEND_VULKAN);
  sd::Pointer modelHandle =
      loadModelFromFileWithOptions(modelPathStr.c_str(), requireFileBacked);
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
  model->extracted_dir = manifestData.extracted_dir;
  model->backend = backend;
  model->gpu_target = gpuTarget;
  model->strict_backend = strictBackend;
  model->allow_runtime_jit = allowRuntimeJit;
  model->runtime_artifact_directory =
      backend == static_cast<int>(SDX_BACKEND_HEXAGON)
          ? manifestData.hexagon_kernel_directory
          : manifestData.vulkan_spirv_directory;
  model->device_compilation_cache_directory =
      std::move(deviceCompilationCacheDirectory);
  model->device_compilation_cache_model_key =
      manifestData.device_compilation_cache_model_key;
  model->tokenizer_path = manifestData.tokenizer_path;
  model->text_generation_config_path =
      manifestData.text_generation_config_path;

  setLastError(runtime, "");
  *out_model = model;
  return SDX_STATUS_OK;
}

SDX_API void sdxUnloadModel(sdx_model_t* model) {
  if (model != nullptr) {
    if (model->model_handle != nullptr) {
      freeLoadedModel(model->model_handle);
      model->model_handle = nullptr;
    }
#if HAS_FILESYSTEM
    if (!model->extracted_dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(model->extracted_dir, ec);
    }
#endif
    delete model;
  }
}

SDX_API const char* sdxGetTokenizerPath(const sdx_model_t* model) {
  return model == nullptr || model->tokenizer_path.empty()
             ? nullptr
             : model->tokenizer_path.c_str();
}

SDX_API const char* sdxGetTextGenerationConfigPath(
    const sdx_model_t* model) {
  return model == nullptr || model->text_generation_config_path.empty()
             ? nullptr
             : model->text_generation_config_path.c_str();
}

static sdx_status_t createContextInternal(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    bool bind_model_parameters,
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
    if (requested_output_names[i] == nullptr) {
      setLastError(model->runtime,
                   "requested_output_names contains a null entry");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
    mutableOutputNames.push_back(const_cast<char*>(requested_output_names[i]));
  }

  sd::Pointer outputNamesPtr =
      mutableOutputNames.empty()
          ? nullptr
          : reinterpret_cast<sd::Pointer>(mutableOutputNames.data());

  sd::Pointer planHandle = compileModelPlanWithRuntimeOptions(
      model->model_handle, outputNamesPtr, num_requested_outputs,
      model->backend, model->allow_runtime_jit,
      model->runtime_artifact_directory.c_str(),
      model->device_compilation_cache_directory.c_str(),
      model->device_compilation_cache_model_key.c_str());

  if (planHandle == nullptr) {
    const char* err = lastErrorMessage();
    if (err != nullptr && err[0] != '\0') {
      setLastError(model->runtime, std::string("compileModelPlan failed: ") + err);
    } else {
      setLastError(model->runtime, "compileModelPlan failed");
    }
    return SDX_STATUS_EXECUTION_FAILED;
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
  context->plan_num_inputs = getPlanNumExternalInputs(planHandle);
  context->num_outputs = getPlanNumRequestedOutputs(planHandle);
  context->binds_model_parameters = bind_model_parameters;
  context->last_error.clear();
  context->last_report.struct_size = sizeof(sdx_execution_report_t);
  context->last_report.requested_backend = model->backend;
  context->last_report.applied_backend = model->backend;
  context->last_report.requested_gpu_target = model->gpu_target;
  context->last_report.applied_gpu_target = model->gpu_target;
  context->last_report.status_code = static_cast<int32_t>(SDX_STATUS_OK);
  context->last_report.used_fallback = -1;
  context->last_report.execution_time_ns = 0;

  if (context->plan_num_inputs < 0 || context->num_outputs < 0) {
    sdxDestroyContext(context);
    setLastError(model->runtime, "Compiled plan returned invalid input/output counts");
    return SDX_STATUS_EXECUTION_FAILED;
  }

  const size_t planInputCount = static_cast<size_t>(context->plan_num_inputs);
  context->bound_model_inputs.resize(planInputCount, nullptr);
  context->plan_to_public_input.resize(planInputCount, -1);
  context->public_to_plan_input.reserve(planInputCount);

  for (int planIndex = 0; planIndex < context->plan_num_inputs; ++planIndex) {
    sd::NDArray* bound = nullptr;
    const char* inputName = getPlanExternalInputName(planHandle, planIndex);
    if (bind_model_parameters && inputName != nullptr) {
      bound = getLoadedModelVariable(model->model_handle, inputName);
    }
    if (bound != nullptr) {
      context->bound_model_inputs[static_cast<size_t>(planIndex)] = bound;
    } else {
      const int publicIndex = static_cast<int>(context->public_to_plan_input.size());
      context->public_to_plan_input.push_back(planIndex);
      context->plan_to_public_input[static_cast<size_t>(planIndex)] = publicIndex;
    }
  }
  context->num_inputs = static_cast<int>(context->public_to_plan_input.size());

  // Pre-allocate cache vectors so sdxRun can do incremental updates. These are
  // public-input sized; the graph context itself is bound at plan width.
  context->input_wrappers.resize(static_cast<size_t>(context->num_inputs));
  context->output_wrappers.resize(static_cast<size_t>(context->num_outputs));
  context->borrowed_output_copies.resize(
      static_cast<size_t>(context->num_outputs));
  context->output_names.resize(static_cast<size_t>(context->num_outputs));
  for (int32_t i = 0;
       i < num_requested_outputs && i < context->num_outputs; ++i) {
    context->output_names[static_cast<size_t>(i)] =
        requested_output_names[i];
  }
  context->cached_input_meta.resize(static_cast<size_t>(context->num_inputs));
  context->cached_output_meta.resize(static_cast<size_t>(context->num_outputs));
  context->constant_replicas.resize(static_cast<size_t>(context->num_inputs));
  context->is_placeholder_input.resize(static_cast<size_t>(context->num_inputs), false);
  for (int publicIndex = 0; publicIndex < context->num_inputs; ++publicIndex) {
    const int planIndex = context->public_to_plan_input[static_cast<size_t>(publicIndex)];
    context->is_placeholder_input[static_cast<size_t>(publicIndex)] =
        getPlanIsExternalInputPlaceholder(planHandle, planIndex);
  }

  setLastError(model->runtime, "");
  *out_context = context;
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxCreateContext(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    sdx_context_t** out_context) {
  return createContextInternal(model, requested_output_names,
                               num_requested_outputs, false, out_context);
}

SDX_API sdx_status_t sdxCreateContextWithOptions(
    sdx_model_t* model,
    const char* const* requested_output_names,
    int32_t num_requested_outputs,
    const sdx_context_options_t* options,
    sdx_context_t** out_context) {
  if (options != nullptr && options->struct_size != 0 &&
      options->struct_size <
          offsetof(sdx_context_options_t, bind_model_parameters) + sizeof(int32_t)) {
    if (model != nullptr) {
      setLastError(model->runtime, "sdx_context_options_t struct_size is incompatible");
    }
    return SDX_STATUS_INCOMPATIBLE_ABI;
  }
  const bool bindModelParameters =
      options != nullptr &&
      optionHasField(options->struct_size,
                     offsetof(sdx_context_options_t, bind_model_parameters),
                     sizeof(int32_t)) &&
      options->bind_model_parameters != 0;
  return createContextInternal(model, requested_output_names,
                               num_requested_outputs, bindModelParameters,
                               out_context);
}

SDX_API void sdxDestroyContext(sdx_context_t* context) {
  if (context == nullptr) {
    return;
  }

  // Wait for any in-flight run to finish before tearing down.
  { std::lock_guard<std::mutex> execLock(context->exec_mutex); }

  context->input_wrappers.clear();
  context->output_wrappers.clear();
  context->borrowed_output_copies.clear();
  context->output_names.clear();
  context->cached_input_meta.clear();
  context->cached_output_meta.clear();
  context->constant_replicas.clear();
  context->public_to_plan_input.clear();
  context->plan_to_public_input.clear();
  context->bound_model_inputs.clear();

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

static sdx_status_t runInternal(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_tensor_view_t* outputs,
    int32_t num_outputs,
    const sdx_run_options_t* options,
    bool copy_to_caller_outputs) {
  if (context == nullptr || num_inputs < 0 ||
      (num_inputs > 0 && inputs == nullptr) ||
      (copy_to_caller_outputs &&
       (num_outputs < 0 || (num_outputs > 0 && outputs == nullptr)))) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> execLock(context->exec_mutex);

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
    if (copy_to_caller_outputs && num_outputs != context->num_outputs) {
      setContextError(context, "Output tensor count mismatch");
      return SDX_STATUS_INVALID_ARGUMENT;
    }
  } else {
    if (num_inputs < context->num_inputs ||
        (copy_to_caller_outputs && num_outputs < context->num_outputs)) {
      setContextError(
          context,
          "Non-strict signature still requires at least plan input/output counts");
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
    if (copy_to_caller_outputs) {
      for (int i = 0; i < context->num_outputs; i++) {
        auto st = scanTensor(outputs[i], outputs[i].bytes);
        if (st != SDX_STATUS_OK) return st;
      }
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
  if (sd::graph::dspHasDeviceMemory() && electedDeviceId >= 0) {
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
  if (sd::graph::dspHasDeviceMemory() && !context->device_election_done) {
    int planPhase = getPlanPhase(context->plan_handle);
    if (planPhase >= 2) {  // FROZEN or REPLAYING
      context->elected_device_id = electedDeviceId;
      context->device_election_done = true;
    }
  }

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
      const std::string inputDescription = describePublicInput(context, i);
      const std::string message =
          "Input tensor[" + std::to_string(i) + "] " + inputDescription +
          " invalid: " + error;
      DSP_DIAG(EXECUTE, "SDX_INPUT_BIND_FAILURE %s", message.c_str());
      setContextError(context, message);
      return status;
    }
    context->input_wrappers[idx] = std::move(wrapped);
    context->cached_input_meta[idx].update(inputs[i]);
    anyInputChanged = true;
  }

  if (!sd::graph::dspIsCudaBuild()) {
    if (hasCudaLikeTensors) {
      setContextError(context, "CUDA/AMD tensors require a CUDA-enabled runtime build");
      return SDX_STATUS_UNSUPPORTED;
    }
  }

  if (sd::graph::dspIsCudaBuild()) {
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
  }

  bool anyOutputChanged = false;
  if (copy_to_caller_outputs) {
    for (int i = 0; i < context->num_outputs; i++) {
      const size_t idx = static_cast<size_t>(i);
      if (context->output_wrappers[idx] != nullptr &&
          context->cached_output_meta[idx].matches(outputs[i])) {
        continue;  // Cache hit
      }
      std::unique_ptr<sd::NDArray> wrapped;
      std::string error;
      auto status = wrapTensorView(outputs[i], &wrapped, &error);
      if (status != SDX_STATUS_OK) {
        setContextError(context,
                        "Output tensor[" + std::to_string(i) +
                            "] invalid: " + error);
        return status;
      }
      context->output_wrappers[idx] = std::move(wrapped);
      context->cached_output_meta[idx].update(outputs[i]);
      anyOutputChanged = true;
    }
  }

  // Only purge and re-set context arrays when wrappers actually changed.
  // On the fast path (same pointers, shapes, dtypes), this skips all context setup.
  if (anyInputChanged || anyOutputChanged || context->execution_count == 0) {
    ctxPurgeNoSync(context->graph_context);
    for (int planIndex = 0; planIndex < context->plan_num_inputs; ++planIndex) {
      sd::NDArray* input =
          context->bound_model_inputs[static_cast<size_t>(planIndex)];
      if (input == nullptr) {
        const int publicIndex =
            context->plan_to_public_input[static_cast<size_t>(planIndex)];
        if (publicIndex < 0 || publicIndex >= context->num_inputs) {
          setContextError(context,
                          "sdxRun: missing public-to-plan input mapping at plan index " +
                              std::to_string(planIndex));
          return SDX_STATUS_EXECUTION_FAILED;
        }
        input = context->input_wrappers[static_cast<size_t>(publicIndex)].get();
      }
      setGraphContextInputArray(context->graph_context, planIndex, input);
    }
    if (copy_to_caller_outputs) {
      for (int i = 0; i < context->num_outputs; i++) {
        setGraphContextOutputArray(
            context->graph_context,
            i,
            context->output_wrappers[static_cast<size_t>(i)].get());
      }
    }
  }

  // Any borrowed output view from the prior execution expires at this point.
  for (auto& copy : context->borrowed_output_copies) {
    copy.reset();
  }

  auto start = std::chrono::steady_clock::now();
  sd::Pointer execStream = nullptr;
  if (sd::graph::dspHasDeviceMemory()) {
    // Cache the active device backend stream — avoids LaunchContext lookup every call
    if (context->exec_stream_cached) {
      execStream = context->cached_exec_stream;
    } else {
      execStream = sd::graph::dspGetExecutionStream();
      context->cached_exec_stream = execStream;
      context->exec_stream_cached = true;
    }
  }
  int execCode = executeDynamicShapePlan(context->plan_handle, context->graph_context, execStream);
  auto end = std::chrono::steady_clock::now();
  uint64_t durationNs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

  context->execution_count++;

  sdx_status_t status = mapExecuteStatus(execCode);
  const bool explicitAcceleratorBackend =
      requestedBackend != static_cast<int>(SDX_BACKEND_AUTO) &&
      requestedBackend != static_cast<int>(SDX_BACKEND_SLOT_BY_SLOT);
  if (status != SDX_STATUS_OK &&
      context->model->strict_backend &&
      explicitAcceleratorBackend) {
    status = SDX_STATUS_BACKEND_UNAVAILABLE;
  }
  context->last_report.struct_size = sizeof(sdx_execution_report_t);
  context->last_report.requested_backend = requestedBackend;
  // Applied backend is read back from the plan: reflects clamping and any
  // mode changes the plan made, rather than echoing the request.
  {
    int appliedMode = getPlanGraphExecutionMode(context->plan_handle);
    context->last_report.applied_backend =
        appliedMode >= 0 ? appliedMode : requestedBackend;
  }
  context->last_report.requested_gpu_target = requestedGpuTarget;
  context->last_report.applied_gpu_target = requestedGpuTarget;
  context->last_report.execution_time_ns = durationNs;
  context->last_report.plan_phase = getPlanPhase(context->plan_handle);
  context->last_report.execution_count = context->execution_count;
  // Fallback telemetry covers capture/replay failure and, for an explicitly
  // requested accelerator, any non-constant segment resolved to slot-by-slot
  // execution. The latter is host execution and must never be reported as a
  // successful strict accelerator run.
  {
    int32_t usedFallback = 0;
    int numSegments = getPlanNumSegments(context->plan_handle);
    for (int s = 0; s < numSegments; s++) {
      if (isPlanSegmentCaptureFailed(context->plan_handle, s)) {
        usedFallback = 1;
        break;
      }
    }
    if (usedFallback == 0 && context->last_report.plan_phase == 3) {
      usedFallback = 1;
    }
    if (usedFallback == 0 && explicitAcceleratorBackend) {
      const auto* plan =
          reinterpret_cast<const sd::graph::NativeDynamicShapePlan*>(
              context->plan_handle);
      if (plan != nullptr) {
        for (const auto& segment : plan->getSegments()) {
          if (!segment.def.allFrozenConstants &&
              segment.def.selectedBackend ==
                  sd::graph::SelectedBackend::SLOT_BY_SLOT) {
            usedFallback = 1;
            break;
          }
        }
      }
    }
    context->last_report.used_fallback = usedFallback;
  }

  const bool strictFallbackViolation =
      status == SDX_STATUS_OK &&
      context->model->strict_backend &&
      explicitAcceleratorBackend &&
      context->last_report.used_fallback != 0;
  if (strictFallbackViolation) {
    status = SDX_STATUS_BACKEND_UNAVAILABLE;
  }
  context->last_report.status_code = static_cast<int32_t>(status);

  if (status != SDX_STATUS_OK) {
    if (strictFallbackViolation) {
      setContextError(
          context,
          "Strict accelerator execution rejected slot-by-slot host fallback");
    } else {
      const char* nativeError = lastErrorMessage();
      if (nativeError != nullptr && nativeError[0] != '\0') {
        setContextError(context, nativeError);
      } else {
        if (execCode == static_cast<int>(sd::Status::KERNEL_FAILURE)) {
          setContextError(
              context,
              "executeDynamicShapePlan returned KERNEL_FAILURE (50) without "
              "native failure detail; the originating plan path did not set "
              "LaunchContext::errorReference");
        } else {
          setContextError(
              context,
              "executeDynamicShapePlan failed with status " +
                  std::to_string(execCode));
        }
      }
    }
    return status;
  }

  // Copy results into the caller-provided output buffers.
  //
  // executeDynamicShapePlan does NOT write into the context's bound output
  // arrays: the plan produces its own arrays and the context's output slots
  // are REPLACED with pointers to them post-execute (see NativeOps_dsp.cpp).
  // The caller-buffer wrappers therefore act as destinations we copy into.
  if (copy_to_caller_outputs) {
    for (int i = 0; i < context->num_outputs; i++) {
      auto& out = context->output_wrappers[static_cast<size_t>(i)];
      sd::NDArray* produced = context->graph_context->outputArray(i);
      if (produced == nullptr) {
        setContextError(
            context,
            "sdxRun: plan produced no output at index " + std::to_string(i));
        return SDX_STATUS_EXECUTION_FAILED;
      }
      if (produced != out.get()) {
        if (produced->dataType() != out->dataType()) {
          setContextError(
              context,
              "sdxRun: output dtype mismatch at index " + std::to_string(i) +
                  " (produced=" +
                  std::to_string(static_cast<int>(produced->dataType())) +
                  ", caller=" +
                  std::to_string(static_cast<int>(out->dataType())) + ")");
          return SDX_STATUS_EXECUTION_FAILED;
        }
        if (produced->lengthOf() != out->lengthOf()) {
          setContextError(
              context,
              "sdxRun: output length mismatch at index " + std::to_string(i) +
                  " (produced=" + std::to_string(produced->lengthOf()) +
                  ", caller=" + std::to_string(out->lengthOf()) + ")");
          return SDX_STATUS_EXECUTION_FAILED;
        }
        out->assign(produced);
      }
      if (outputs[i].device_type == static_cast<int32_t>(SDX_DEVICE_HOST)) {
        out->syncToHost();
      }
      // GPU outputs: data stays on device, no host sync needed.
    }
  } else {
    // Fail here rather than returning success with an unusable dynamic output.
    for (int i = 0; i < context->num_outputs; ++i) {
      if (context->graph_context->outputArray(i) == nullptr) {
        setContextError(
            context,
            "sdxRunAllocating: plan produced no output at index " +
                std::to_string(i));
        return SDX_STATUS_EXECUTION_FAILED;
      }
    }
  }

  setContextError(context, "");
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxRun(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_tensor_view_t* outputs,
    int32_t num_outputs,
    const sdx_run_options_t* options) {
  return runInternal(context, inputs, num_inputs, outputs, num_outputs,
                     options, true);
}

SDX_API sdx_status_t sdxRunAllocating(
    sdx_context_t* context,
    const sdx_tensor_view_t* inputs,
    int32_t num_inputs,
    const sdx_run_options_t* options) {
  return runInternal(context, inputs, num_inputs, nullptr, 0, options, false);
}

SDX_API const char* sdxGetLastError(const sdx_runtime_t* runtime) {
  if (runtime == nullptr) {
    return "runtime is null";
  }
  std::lock_guard<std::mutex> lock(runtime->error_mutex);
  auto* snapshot = const_cast<char*>(runtime->error_snapshot);
  std::strncpy(snapshot, runtime->last_error.c_str(), sizeof(runtime->error_snapshot) - 1);
  snapshot[sizeof(runtime->error_snapshot) - 1] = '\0';
  return snapshot;
}

SDX_API sdx_status_t sdxGetExecutionReport(
    const sdx_context_t* context,
    sdx_execution_report_t* out_report) {
  if (context == nullptr || out_report == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> execLock(context->exec_mutex);

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
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  if (input_index < 0 || input_index >= context->num_inputs) {
    setContextError(context, "sdxMarkInputVariable: input_index out of range");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  const int planIndex =
      context->public_to_plan_input[static_cast<size_t>(input_index)];
  markPlanExternalInputVariable(context->plan_handle, planIndex);
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxMarkInputPlaceholder(sdx_context_t* context, int32_t input_index) {
  if (context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  if (input_index < 0 || input_index >= context->num_inputs) {
    setContextError(context, "sdxMarkInputPlaceholder: input_index out of range");
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  const int planIndex =
      context->public_to_plan_input[static_cast<size_t>(input_index)];
  markPlanExternalInputPlaceholder(context->plan_handle, planIndex);
  // Record in local cache so cross-device migration knows not to cache this input
  context->is_placeholder_input[static_cast<size_t>(input_index)] = true;
  return SDX_STATUS_OK;
}

SDX_API sdx_status_t sdxFreezeShapes(sdx_context_t* context) {
  if (context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
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
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  return getPlanPhase(context->plan_handle);
}

SDX_API const char* sdxGetPlanSegmentsSummaryJson(
    const sdx_context_t* context) {
  if (context == nullptr || context->plan_handle == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  return getPlanSegmentsSummaryJson(context->plan_handle);
}

SDX_API int32_t sdxGetExecutionCount(const sdx_context_t* context) {
  if (context == nullptr) {
    return -1;
  }
  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  return context->execution_count;
}

SDX_API int32_t sdxGetNumInputs(const sdx_context_t* context) {
  if (context == nullptr) {
    return -1;
  }
  return context->num_inputs;
}

SDX_API int32_t sdxGetNumOutputs(const sdx_context_t* context) {
  if (context == nullptr) {
    return -1;
  }
  return context->num_outputs;
}

SDX_API const char* sdxGetInputName(const sdx_context_t* context,
                                             int32_t input_index) {
  if (context == nullptr || context->plan_handle == nullptr ||
      input_index < 0 || input_index >= context->num_inputs) {
    return nullptr;
  }
  const int planIndex =
      context->public_to_plan_input[static_cast<size_t>(input_index)];
  return getPlanExternalInputName(context->plan_handle, planIndex);
}

SDX_API const char* sdxGetOutputName(const sdx_context_t* context,
                                    int32_t output_index) {
  if (context == nullptr || output_index < 0 ||
      output_index >= context->num_outputs) {
    return nullptr;
  }
  const auto& name =
      context->output_names[static_cast<size_t>(output_index)];
  return name.empty() ? nullptr : name.c_str();
}

SDX_API sdx_status_t sdxGetOutputTensor(
    sdx_context_t* context,
    int32_t output_index,
    sdx_tensor_view_t* out_tensor) {
  if (context == nullptr || out_tensor == nullptr || output_index < 0 ||
      output_index >= context->num_outputs) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  std::lock_guard<std::mutex> execLock(context->exec_mutex);
  if (context->execution_count <= 0 ||
      context->last_report.status_code !=
          static_cast<int32_t>(SDX_STATUS_OK) ||
      !context->last_error.empty()) {
    setContextError(context,
                    "sdxGetOutputTensor requires a successful prior run");
    return SDX_STATUS_EXECUTION_FAILED;
  }

  sd::NDArray* produced =
      context->graph_context->outputArray(output_index);
  if (produced == nullptr) {
    setContextError(context,
                    "sdxGetOutputTensor: output is unavailable at index " +
                        std::to_string(output_index));
    return SDX_STATUS_EXECUTION_FAILED;
  }

  try {
    const sd::LongType length = produced->lengthOf();
    if (length < 0) {
      setContextError(context,
                      "sdxGetOutputTensor: output length is invalid");
      return SDX_STATUS_EXECUTION_FAILED;
    }

    auto& contiguousCopy =
        context->borrowed_output_copies[static_cast<size_t>(output_index)];
    if (length > 1 &&
        (produced->ordering() != 'c' || produced->ews() != 1)) {
      contiguousCopy.reset(produced->dup('c'));
      if (contiguousCopy == nullptr) {
        setContextError(
            context,
            "sdxGetOutputTensor: failed to materialize contiguous output");
        return SDX_STATUS_EXECUTION_FAILED;
      }
      produced = contiguousCopy.get();
    }

    produced->syncToHost();
    const int64_t rank = static_cast<int64_t>(produced->rankOf());
    if (rank < 0 || rank > std::numeric_limits<int32_t>::max()) {
      setContextError(context,
                      "sdxGetOutputTensor: output rank exceeds the C ABI");
      return SDX_STATUS_EXECUTION_FAILED;
    }
    const size_t elementSize =
        sd::DataTypeUtils::sizeOfElement(produced->dataType());
    const uint64_t unsignedLength =
        static_cast<uint64_t>(produced->lengthOf());
    if (elementSize != 0 &&
        unsignedLength >
            std::numeric_limits<size_t>::max() / elementSize) {
      setContextError(context,
                      "sdxGetOutputTensor: output byte size overflow");
      return SDX_STATUS_EXECUTION_FAILED;
    }

    out_tensor->data = produced->buffer();
    out_tensor->shape =
        rank == 0
            ? nullptr
            : reinterpret_cast<const int64_t*>(produced->shapeOf());
    out_tensor->rank = static_cast<int32_t>(rank);
    out_tensor->dtype = static_cast<int32_t>(produced->dataType());
    out_tensor->bytes =
        static_cast<size_t>(unsignedLength) * elementSize;
    out_tensor->device_type = static_cast<int32_t>(SDX_DEVICE_HOST);
    out_tensor->device_id = -1;
  } catch (const std::exception& e) {
    setContextError(
        context,
        std::string("sdxGetOutputTensor failed: ") + e.what());
    return SDX_STATUS_EXECUTION_FAILED;
  } catch (...) {
    setContextError(context, "sdxGetOutputTensor failed");
    return SDX_STATUS_EXECUTION_FAILED;
  }

  setContextError(context, "");
  return SDX_STATUS_OK;
}

}  // extern "C"

namespace sd {
namespace dsp {
namespace runtime {
namespace detail {

void setModelError(sdx_model_t* model, const std::string& error) {
  if (model != nullptr) {
    setLastError(model->runtime, error);
  }
}

bool modelVariableShape(
    sdx_model_t* model,
    const std::string& variableName,
    std::vector<int64_t>* shape) {
  if (model == nullptr || model->model_handle == nullptr || shape == nullptr ||
      variableName.empty()) {
    return false;
  }
  const int rank = getLoadedModelVariableShape(
      model->model_handle, variableName.c_str(), nullptr, 0);
  if (rank <= 0) return false;
  std::vector<LongType> dimensions(static_cast<size_t>(rank));
  if (getLoadedModelVariableShape(
          model->model_handle, variableName.c_str(), dimensions.data(), rank) !=
      rank) {
    return false;
  }
  shape->assign(dimensions.begin(), dimensions.end());
  return true;
}

std::string contextError(const sdx_context_t* context) {
  return context == nullptr ? std::string() : context->last_error;
}

sdx_status_t runOwnedArrays(
    sdx_context_t* context,
    const std::vector<NDArray*>& publicInputs) {
  if (context == nullptr ||
      publicInputs.size() != static_cast<size_t>(context->num_inputs)) {
    if (context != nullptr) {
      setContextError(context, "runOwnedArrays input count mismatch");
    }
    return SDX_STATUS_INVALID_ARGUMENT;
  }

  std::vector<sdx_tensor_view_t> views(publicInputs.size());
  for (size_t i = 0; i < publicInputs.size(); ++i) {
    NDArray* array = publicInputs[i];
    if (array == nullptr) {
      setContextError(
          context, "runOwnedArrays received a null input at index " +
                       std::to_string(i));
      return SDX_STATUS_INVALID_ARGUMENT;
    }

    try {
      array->syncToHost();
      const LongType length = array->lengthOf();
      const size_t elementSize = DataTypeUtils::sizeOfElement(array->dataType());
      if (length < 0 ||
          (elementSize != 0 &&
           static_cast<uint64_t>(length) >
               std::numeric_limits<size_t>::max() / elementSize)) {
        setContextError(context, "runOwnedArrays input byte size overflow");
        return SDX_STATUS_INVALID_ARGUMENT;
      }

      auto& view = views[i];
      view.data = array->buffer();
      view.shape = array->rankOf() == 0
                       ? nullptr
                       : reinterpret_cast<const int64_t*>(array->shapeOf());
      view.rank = static_cast<int32_t>(array->rankOf());
      view.dtype = static_cast<int32_t>(array->dataType());
      view.bytes = static_cast<size_t>(length) * elementSize;
      if (view.bytes > 0 && view.data == nullptr) {
        setContextError(
            context, "runOwnedArrays input has no host buffer at index " +
                         std::to_string(i));
        return SDX_STATUS_INVALID_ARGUMENT;
      }
      view.device_type = static_cast<int32_t>(SDX_DEVICE_HOST);
      view.device_id = -1;
    } catch (const std::exception& e) {
      setContextError(
          context, std::string("runOwnedArrays failed to expose input: ") +
                       e.what());
      return SDX_STATUS_EXECUTION_FAILED;
    }
  }

  return sdxRunAllocating(
      context,
      views.empty() ? nullptr : views.data(),
      static_cast<int32_t>(views.size()),
      nullptr);
}

sdx_status_t precompileBoundContext(sdx_context_t* context) {
  if (context == nullptr || context->plan_handle == nullptr ||
      context->graph_context == nullptr) {
    return SDX_STATUS_INVALID_ARGUMENT;
  }
  auto* plan = reinterpret_cast<graph::NativeDynamicShapePlan*>(
      context->plan_handle);
  std::vector<NDArray*> inputs(static_cast<size_t>(context->plan_num_inputs));
  for (int32_t index = 0; index < context->plan_num_inputs; ++index) {
    inputs[static_cast<size_t>(index)] = context->graph_context->array(index);
    if (inputs[static_cast<size_t>(index)] == nullptr) {
      setContextError(
          context,
          "precompileBoundContext has no bound array at plan input " +
              std::to_string(index));
      return SDX_STATUS_EXECUTION_FAILED;
    }
  }
  void* stream = sd::graph::dspHasDeviceMemory()
                     ? sd::graph::dspGetExecutionStream()
                     : nullptr;
  const Status status = plan->precompilePlan(
      inputs.empty() ? nullptr : inputs.data(),
      static_cast<int>(inputs.size()), stream);
  if (status != Status::OK) {
    const char* nativeError = lastErrorMessage();
    setContextError(
        context,
        nativeError != nullptr && nativeError[0] != '\0'
            ? std::string(nativeError)
            : "precompileBoundContext failed");
    return SDX_STATUS_EXECUTION_FAILED;
  }
  setContextError(context, "");
  return SDX_STATUS_OK;
}

NDArray* contextOutputArray(sdx_context_t* context, int32_t outputIndex) {
  if (context == nullptr || context->graph_context == nullptr ||
      outputIndex < 0 || outputIndex >= context->num_outputs) {
    return nullptr;
  }
  return context->graph_context->outputArray(outputIndex);
}

graph::NativeDynamicShapePlan* contextPlan(sdx_context_t* context) {
  return context == nullptr
             ? nullptr
             : reinterpret_cast<graph::NativeDynamicShapePlan*>(
                   context->plan_handle);
}

graph::Context* contextGraph(sdx_context_t* context) {
  return context == nullptr ? nullptr : context->graph_context;
}

int32_t contextPlanInputCount(const sdx_context_t* context) {
  return context == nullptr ? -1 : context->plan_num_inputs;
}

int32_t contextOutputCount(const sdx_context_t* context) {
  return context == nullptr ? -1 : context->num_outputs;
}

int32_t contextPlanInputIndex(
    const sdx_context_t* context,
    const std::string& inputName) {
  if (context == nullptr || context->plan_handle == nullptr ||
      inputName.empty()) {
    return -1;
  }
  for (int32_t i = 0; i < context->plan_num_inputs; ++i) {
    const char* candidate = getPlanExternalInputName(context->plan_handle, i);
    if (candidate != nullptr && inputName == candidate) return i;
  }
  return -1;
}

NDArray* contextPlanInputArray(
    sdx_context_t* context,
    int32_t planInputIndex) {
  if (context == nullptr || context->graph_context == nullptr ||
      planInputIndex < 0 || planInputIndex >= context->plan_num_inputs) {
    return nullptr;
  }
  return context->graph_context->array(planInputIndex);
}

}  // namespace detail
}  // namespace runtime
}  // namespace dsp
}  // namespace sd
