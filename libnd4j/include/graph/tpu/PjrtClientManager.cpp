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

#ifdef SD_TPU

#include <graph/tpu/PjrtClientManager.h>

#include <array/DataTypeUtils.h>
#include <external/pjrt/pjrt_c_api.h>
#include <graph/DspDiagnostics.h>
#include <helpers/shape.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace sd {
namespace graph {
namespace {

static const char* kKnownPjrtPluginNames[] = {
    "libtpu.so", "xla_rocm_plugin.so", "pjrt_c_api_gpu_plugin.so",
    "libpjrt_c_api_cpu_dynamic.so", nullptr};

using GetPjrtApiFn = const PJRT_Api* (*)();

bool pathIsRegularFile(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool pathIsDir(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

void collectPluginCandidatesFromEnv(std::vector<std::string>& result) {
  static const char* kEnvVars[] = {"PJRT_PLUGIN_LIBRARY_PATH", "ROCM_PJRT_PATH",
                                    "PJRT_PATH", "TPU_LIBRARY_PATH", nullptr};
  for (int i = 0; kEnvVars[i] != nullptr; ++i) {
    const char* raw = std::getenv(kEnvVars[i]);
    if (raw == nullptr || raw[0] == '\0') continue;
    std::string value(raw);
    if (value.find("${") != std::string::npos) continue;
    if (pathIsRegularFile(value)) {
      result.push_back(value);
    } else if (pathIsDir(value)) {
      for (int k = 0; kKnownPjrtPluginNames[k] != nullptr; ++k) {
        const std::string candidate = value + "/" + kKnownPjrtPluginNames[k];
        if (pathIsRegularFile(candidate)) result.push_back(candidate);
      }
    }
  }
}

PJRT_Buffer_Type toPjrtType(DataType dataType) {
  switch (dataType) {
    case BOOL: return PJRT_Buffer_Type_PRED;
    case INT8: return PJRT_Buffer_Type_S8;
    case INT16: return PJRT_Buffer_Type_S16;
    case INT32: return PJRT_Buffer_Type_S32;
    case INT64: return PJRT_Buffer_Type_S64;
    case UINT8: return PJRT_Buffer_Type_U8;
    case UINT16: return PJRT_Buffer_Type_U16;
    case UINT32: return PJRT_Buffer_Type_U32;
    case UINT64: return PJRT_Buffer_Type_U64;
    case HALF: return PJRT_Buffer_Type_F16;
    case FLOAT32: return PJRT_Buffer_Type_F32;
    case DOUBLE: return PJRT_Buffer_Type_F64;
    case BFLOAT16: return PJRT_Buffer_Type_BF16;
    default: return PJRT_Buffer_Type_INVALID;
  }
}

std::string lowerCopy(const std::string& value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

}  // namespace

PjrtClientManager* PjrtClientManager::instance_ = nullptr;
std::once_flag PjrtClientManager::instanceOnce_;
thread_local int PjrtClientManager::currentDevice_ = 0;

PjrtClientManager& PjrtClientManager::getInstance() {
  std::call_once(instanceOnce_, []() {
    instance_ = new PjrtClientManager();
  });
  return *instance_;
}

PjrtClientManager::PjrtClientManager() = default;

PjrtClientManager::~PjrtClientManager() {
  std::lock_guard<std::mutex> lock(initMutex_);
  shutdownUnlocked();
}

void PjrtClientManager::setLastError(const std::string& message) {
  std::lock_guard<std::mutex> lock(errorMutex_);
  lastError_ = message;
}

std::string PjrtClientManager::getLastError() const {
  std::lock_guard<std::mutex> lock(errorMutex_);
  return lastError_;
}

bool PjrtClientManager::consumeError(void* opaqueError, const char* operation) {
  auto* error = static_cast<PJRT_Error*>(opaqueError);
  if (error == nullptr) return true;

  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  std::string message = operation == nullptr ? "PJRT operation failed" : operation;
  if (api != nullptr && api->PJRT_Error_Message != nullptr) {
    PJRT_Error_Message_Args messageArgs{};
    messageArgs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    messageArgs.error = error;
    api->PJRT_Error_Message(&messageArgs);
    if (messageArgs.message != nullptr) {
      message += ": ";
      message.append(messageArgs.message, messageArgs.message_size);
    }
  }
  setLastError(message);
  DSP_DIAG(BACKEND, "PjrtClientManager: %s", message.c_str());

  if (api != nullptr && api->PJRT_Error_Destroy != nullptr) {
    PJRT_Error_Destroy_Args destroyArgs{};
    destroyArgs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroyArgs.error = error;
    api->PJRT_Error_Destroy(&destroyArgs);
  }
  return false;
}

bool PjrtClientManager::awaitAndDestroyEvent(void* opaqueEvent,
                                             const char* operation) {
  auto* event = static_cast<PJRT_Event*>(opaqueEvent);
  if (event == nullptr) return true;
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  if (api == nullptr || api->PJRT_Event_Await == nullptr ||
      api->PJRT_Event_Destroy == nullptr) {
    setLastError("PJRT plugin does not provide event lifecycle functions");
    return false;
  }

  PJRT_Event_Await_Args awaitArgs{};
  awaitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  awaitArgs.event = event;
  const bool awaited = consumeError(api->PJRT_Event_Await(&awaitArgs), operation);

  PJRT_Event_Destroy_Args destroyArgs{};
  destroyArgs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroyArgs.event = event;
  const bool destroyed = consumeError(api->PJRT_Event_Destroy(&destroyArgs),
                                      "PJRT_Event_Destroy");
  return awaited && destroyed;
}

bool PjrtClientManager::loadLibrary() {
  std::vector<std::string> candidates;
  collectPluginCandidatesFromEnv(candidates);
  candidates.push_back("libtpu.so");
  candidates.push_back("/usr/lib/libtpu.so");
  candidates.push_back("/usr/local/lib/libtpu.so");

  for (const auto& candidate : candidates) {
    libHandle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (libHandle_ != nullptr) {
      DSP_DIAG(BACKEND, "PjrtClientManager: loaded PJRT plugin %s",
               candidate.c_str());
      break;
    }
  }
  if (libHandle_ == nullptr) {
    const char* error = dlerror();
    setLastError(std::string("Failed to load a PJRT plugin. Set "
                             "PJRT_PLUGIN_LIBRARY_PATH to a plugin .so: ") +
                 (error == nullptr ? "unknown dlopen error" : error));
    return false;
  }

  dlerror();
  auto getPjrtApi = reinterpret_cast<GetPjrtApiFn>(dlsym(libHandle_, "GetPjrtApi"));
  const char* symbolError = dlerror();
  if (getPjrtApi == nullptr || symbolError != nullptr) {
    setLastError(std::string("Failed to resolve GetPjrtApi: ") +
                 (symbolError == nullptr ? "symbol missing" : symbolError));
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }

  const PJRT_Api* api = getPjrtApi();
  if (api == nullptr) {
    setLastError("GetPjrtApi returned nullptr");
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }
  const size_t versionFieldSize = PJRT_STRUCT_SIZE(PJRT_Api, pjrt_api_version);
  if (api->struct_size < versionFieldSize ||
      api->pjrt_api_version.struct_size < PJRT_Api_Version_STRUCT_SIZE) {
    setLastError("PJRT plugin API table is too small to read its ABI version");
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }
  if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
    setLastError("PJRT ABI major version mismatch: plugin=" +
                 std::to_string(api->pjrt_api_version.major_version) +
                 " headers=" + std::to_string(PJRT_API_MAJOR));
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }
  const size_t requiredApiSize =
      PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Buffer_ToHostBuffer);
  if (api->struct_size < requiredApiSize || api->PJRT_Plugin_Initialize == nullptr ||
      api->PJRT_Client_Create == nullptr || api->PJRT_Client_Destroy == nullptr ||
      api->PJRT_Client_PlatformName == nullptr ||
      api->PJRT_Client_AddressableDevices == nullptr ||
      api->PJRT_Client_BufferFromHostBuffer == nullptr ||
      api->PJRT_Client_Compile == nullptr ||
      api->PJRT_LoadedExecutable_GetExecutable == nullptr ||
      api->PJRT_LoadedExecutable_AddressableDevices == nullptr ||
      api->PJRT_Executable_NumOutputs == nullptr ||
      api->PJRT_Executable_Destroy == nullptr ||
      api->PJRT_LoadedExecutable_Execute == nullptr ||
      api->PJRT_LoadedExecutable_Destroy == nullptr ||
      api->PJRT_Buffer_ElementType == nullptr ||
      api->PJRT_Buffer_Dimensions == nullptr ||
      api->PJRT_Buffer_ToHostBuffer == nullptr || api->PJRT_Buffer_Destroy == nullptr ||
      api->PJRT_Error_Destroy == nullptr || api->PJRT_Error_Message == nullptr ||
      api->PJRT_Event_Await == nullptr || api->PJRT_Event_Destroy == nullptr) {
    setLastError("PJRT plugin API is too old or omits required functions");
    dlclose(libHandle_);
    libHandle_ = nullptr;
    return false;
  }

  pjrtApi_ = const_cast<PJRT_Api*>(api);
  return true;
}

bool PjrtClientManager::initClient() {
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  if (api == nullptr) return false;

  PJRT_Plugin_Initialize_Args pluginArgs{};
  pluginArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  if (!consumeError(api->PJRT_Plugin_Initialize(&pluginArgs),
                    "PJRT_Plugin_Initialize")) {
    return false;
  }

  PJRT_Client_Create_Args clientArgs{};
  clientArgs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (!consumeError(api->PJRT_Client_Create(&clientArgs), "PJRT_Client_Create") ||
      clientArgs.client == nullptr) {
    if (clientArgs.client == nullptr && getLastError().empty()) {
      setLastError("PJRT_Client_Create returned a null client");
    }
    return false;
  }
  client_ = clientArgs.client;

  PJRT_Client_PlatformName_Args platformArgs{};
  platformArgs.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  platformArgs.client = clientArgs.client;
  if (!consumeError(api->PJRT_Client_PlatformName(&platformArgs),
                    "PJRT_Client_PlatformName")) {
    return false;
  }
  platformName_.assign(platformArgs.platform_name == nullptr ? "" :
                       platformArgs.platform_name,
                       platformArgs.platform_name == nullptr ? 0 :
                       platformArgs.platform_name_size);

  PJRT_Client_AddressableDevices_Args deviceArgs{};
  deviceArgs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  deviceArgs.client = clientArgs.client;
  if (!consumeError(api->PJRT_Client_AddressableDevices(&deviceArgs),
                    "PJRT_Client_AddressableDevices")) {
    return false;
  }
  devices_.clear();
  if (deviceArgs.num_addressable_devices > 0 &&
      deviceArgs.addressable_devices != nullptr) {
    devices_.assign(deviceArgs.addressable_devices,
                    deviceArgs.addressable_devices + deviceArgs.num_addressable_devices);
  }
  if (devices_.empty()) {
    setLastError("PJRT client has no addressable devices");
    return false;
  }

  deviceNames_.clear();
  deviceNames_.reserve(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    std::string name = platformName_ + " device " + std::to_string(i);
    if (api->PJRT_Device_GetDescription != nullptr &&
        api->PJRT_DeviceDescription_DebugString != nullptr) {
      PJRT_Device_GetDescription_Args descriptionArgs{};
      descriptionArgs.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
      descriptionArgs.device = static_cast<PJRT_Device*>(devices_[i]);
      PJRT_Error* descriptionError = api->PJRT_Device_GetDescription(&descriptionArgs);
      if (descriptionError == nullptr && descriptionArgs.device_description != nullptr) {
        PJRT_DeviceDescription_DebugString_Args stringArgs{};
        stringArgs.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
        stringArgs.device_description = descriptionArgs.device_description;
        PJRT_Error* stringError = api->PJRT_DeviceDescription_DebugString(&stringArgs);
        if (stringError == nullptr && stringArgs.debug_string != nullptr) {
          name.assign(stringArgs.debug_string, stringArgs.debug_string_size);
        } else if (stringError != nullptr) {
          consumeError(stringError, "PJRT_DeviceDescription_DebugString");
        }
      } else if (descriptionError != nullptr) {
        consumeError(descriptionError, "PJRT_Device_GetDescription");
      }
    }
    deviceNames_.push_back(name);
  }

  DSP_DIAG(BACKEND, "PjrtClientManager: initialized platform=%s devices=%d API=%d.%d",
           platformName_.c_str(), static_cast<int>(devices_.size()),
           api->pjrt_api_version.major_version, api->pjrt_api_version.minor_version);
  return true;
}

bool PjrtClientManager::initialize() {
  std::lock_guard<std::mutex> lock(initMutex_);
  if (initialized_) return true;

  setLastError("");
  shutdownUnlocked();
  if (!loadLibrary() || !initClient()) {
    shutdownUnlocked();
    return false;
  }
  initialized_ = true;
  return true;
}

void PjrtClientManager::shutdownUnlocked() {
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  devices_.clear();
  deviceNames_.clear();
  platformName_.clear();

  if (client_ != nullptr && api != nullptr && api->PJRT_Client_Destroy != nullptr) {
    PJRT_Client_Destroy_Args args{};
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.client = static_cast<PJRT_Client*>(client_);
    consumeError(api->PJRT_Client_Destroy(&args), "PJRT_Client_Destroy");
  }
  client_ = nullptr;
  pjrtApi_ = nullptr;
  initialized_ = false;

  if (libHandle_ != nullptr) {
    dlclose(libHandle_);
    libHandle_ = nullptr;
  }
}

bool PjrtClientManager::isAvailable() const {
  return const_cast<PjrtClientManager*>(this)->initialize();
}

bool PjrtClientManager::isTpuPlatform() const {
  if (!isAvailable()) return false;
  const std::string platform = lowerCopy(getPlatformName());
  return platform.find("tpu") != std::string::npos;
}

std::string PjrtClientManager::getPlatformName() const {
  if (!isAvailable()) return "";
  std::lock_guard<std::mutex> lock(initMutex_);
  return platformName_;
}

int PjrtClientManager::getDeviceCount() const {
  if (!isAvailable()) return 0;
  std::lock_guard<std::mutex> lock(initMutex_);
  return static_cast<int>(devices_.size());
}

std::vector<void*> PjrtClientManager::getDevices() const {
  if (!isAvailable()) return {};
  std::lock_guard<std::mutex> lock(initMutex_);
  return devices_;
}

std::string PjrtClientManager::getDeviceName(int deviceIdx) const {
  if (!isAvailable()) return "";
  std::lock_guard<std::mutex> lock(initMutex_);
  if (deviceIdx < 0 || deviceIdx >= static_cast<int>(deviceNames_.size())) return "";
  return deviceNames_[static_cast<size_t>(deviceIdx)];
}

bool PjrtClientManager::setCurrentDevice(int deviceIdx) {
  if (!isTpuPlatform() || deviceIdx < 0 || deviceIdx >= getDeviceCount()) {
    setLastError("TPU device index is out of range");
    return false;
  }
  currentDevice_ = deviceIdx;
  return true;
}

int PjrtClientManager::getCurrentDevice() const {
  if (!isTpuPlatform()) return 0;
  const int count = getDeviceCount();
  return currentDevice_ >= 0 && currentDevice_ < count ? currentDevice_ : 0;
}

bool PjrtClientManager::validDeviceIndex(int deviceIdx) const {
  return deviceIdx >= 0 && deviceIdx < static_cast<int>(devices_.size());
}

LongType PjrtClientManager::getDeviceTotalMemory(int deviceIdx) const {
  if (!isAvailable()) return 0;
  std::lock_guard<std::mutex> callLock(executionMutex_);
  std::lock_guard<std::mutex> stateLock(initMutex_);
  if (!validDeviceIndex(deviceIdx)) return 0;
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  if (api->PJRT_Device_MemoryStats == nullptr) return 0;
  PJRT_Device_MemoryStats_Args args{};
  args.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
  args.device = static_cast<PJRT_Device*>(devices_[static_cast<size_t>(deviceIdx)]);
  if (!const_cast<PjrtClientManager*>(this)->consumeError(
          api->PJRT_Device_MemoryStats(&args), "PJRT_Device_MemoryStats")) {
    return 0;
  }
  return args.bytes_limit_is_set ? static_cast<LongType>(args.bytes_limit) : 0;
}

LongType PjrtClientManager::getDeviceFreeMemory(int deviceIdx) const {
  if (!isAvailable()) return 0;
  std::lock_guard<std::mutex> callLock(executionMutex_);
  std::lock_guard<std::mutex> stateLock(initMutex_);
  if (!validDeviceIndex(deviceIdx)) return 0;
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  if (api->PJRT_Device_MemoryStats == nullptr) return 0;
  PJRT_Device_MemoryStats_Args args{};
  args.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
  args.device = static_cast<PJRT_Device*>(devices_[static_cast<size_t>(deviceIdx)]);
  if (!const_cast<PjrtClientManager*>(this)->consumeError(
          api->PJRT_Device_MemoryStats(&args), "PJRT_Device_MemoryStats")) {
    return 0;
  }
  if (!args.bytes_limit_is_set) return 0;
  return static_cast<LongType>(std::max<int64_t>(0, args.bytes_limit - args.bytes_in_use));
}

void* PjrtClientManager::createBuffer(NDArray* array, int deviceIdx) {
  if (array == nullptr || !initialize()) return nullptr;
  setLastError("");
  std::lock_guard<std::mutex> callLock(executionMutex_);
  if (!validDeviceIndex(deviceIdx)) {
    setLastError("PJRT device index is out of range");
    return nullptr;
  }

  const PJRT_Buffer_Type type = toPjrtType(array->dataType());
  if (type == PJRT_Buffer_Type_INVALID) {
    setLastError("NDArray dtype is not representable by PJRT");
    return nullptr;
  }

  array->syncToHost();
  void* data = array->buffer();
  if (array->lengthOf() > 0 && data == nullptr) {
    setLastError("NDArray has no host buffer for PJRT upload");
    return nullptr;
  }

  std::vector<int64_t> dimensions(static_cast<size_t>(array->rankOf()));
  std::vector<int64_t> byteStrides(static_cast<size_t>(array->rankOf()));
  const int64_t elementSize =
      static_cast<int64_t>(DataTypeUtils::sizeOfElement(array->dataType()));
  for (int i = 0; i < array->rankOf(); ++i) {
    dimensions[static_cast<size_t>(i)] = array->sizeAt(i);
    byteStrides[static_cast<size_t>(i)] = array->strideAt(i) * elementSize;
  }

  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.client = static_cast<PJRT_Client*>(client_);
  args.data = data;
  args.type = type;
  args.dims = dimensions.empty() ? nullptr : dimensions.data();
  args.num_dims = dimensions.size();
  args.byte_strides = byteStrides.empty() ? nullptr : byteStrides.data();
  args.num_byte_strides = byteStrides.size();
  args.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  args.device = static_cast<PJRT_Device*>(devices_[static_cast<size_t>(deviceIdx)]);

  if (!consumeError(api->PJRT_Client_BufferFromHostBuffer(&args),
                    "PJRT_Client_BufferFromHostBuffer") || args.buffer == nullptr) {
    if (args.done_with_host_buffer != nullptr) {
      awaitAndDestroyEvent(args.done_with_host_buffer,
                           "PJRT host-to-device transfer");
    }
    if (args.buffer != nullptr) {
      PJRT_Buffer_Destroy_Args destroyArgs{};
      destroyArgs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      destroyArgs.buffer = args.buffer;
      consumeError(api->PJRT_Buffer_Destroy(&destroyArgs), "PJRT_Buffer_Destroy");
    }
    return nullptr;
  }
  if (!awaitAndDestroyEvent(args.done_with_host_buffer,
                            "PJRT host-to-device transfer")) {
    PJRT_Buffer_Destroy_Args destroyArgs{};
    destroyArgs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    destroyArgs.buffer = args.buffer;
    consumeError(api->PJRT_Buffer_Destroy(&destroyArgs), "PJRT_Buffer_Destroy");
    return nullptr;
  }
  return args.buffer;
}

void PjrtClientManager::destroyBuffer(void* buffer) {
  if (buffer == nullptr || !initialize()) return;
  std::lock_guard<std::mutex> callLock(executionMutex_);
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = static_cast<PJRT_Buffer*>(buffer);
  consumeError(api->PJRT_Buffer_Destroy(&args), "PJRT_Buffer_Destroy");
}

bool PjrtClientManager::bufferToArray(void* buffer, NDArray* destination) {
  if (buffer == nullptr || destination == nullptr || !initialize()) return false;
  setLastError("");
  if (destination->ordering() != 'c' ||
      !shape::strideDescendingCAscendingF(destination->shapeInfo())) {
    setLastError("PJRT output destination must be dense C-order");
    return false;
  }

  destination->syncToHost();
  void* hostBuffer = destination->buffer();
  const size_t bytes = static_cast<size_t>(destination->lengthOf()) *
                       DataTypeUtils::sizeOfElement(destination->dataType());
  if (bytes > 0 && hostBuffer == nullptr) {
    setLastError("PJRT output destination has no host buffer");
    return false;
  }

  std::lock_guard<std::mutex> callLock(executionMutex_);
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  auto* pjrtBuffer = static_cast<PJRT_Buffer*>(buffer);

  PJRT_Buffer_ElementType_Args typeArgs{};
  typeArgs.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  typeArgs.buffer = pjrtBuffer;
  if (!consumeError(api->PJRT_Buffer_ElementType(&typeArgs),
                    "PJRT_Buffer_ElementType") ||
      typeArgs.type != toPjrtType(destination->dataType())) {
    setLastError("PJRT output dtype does not match its NDArray destination");
    return false;
  }

  PJRT_Buffer_Dimensions_Args dimensionsArgs{};
  dimensionsArgs.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  dimensionsArgs.buffer = pjrtBuffer;
  if (!consumeError(api->PJRT_Buffer_Dimensions(&dimensionsArgs),
                    "PJRT_Buffer_Dimensions") ||
      dimensionsArgs.num_dims != static_cast<size_t>(destination->rankOf())) {
    setLastError("PJRT output rank does not match its NDArray destination");
    return false;
  }
  for (size_t i = 0; i < dimensionsArgs.num_dims; ++i) {
    if (dimensionsArgs.dims[i] != destination->sizeAt(static_cast<int>(i))) {
      setLastError("PJRT output dimensions do not match its NDArray destination");
      return false;
    }
  }

  std::vector<int64_t> byteStrides(static_cast<size_t>(destination->rankOf()));
  const int64_t elementSize = static_cast<int64_t>(
      DataTypeUtils::sizeOfElement(destination->dataType()));
  for (int i = 0; i < destination->rankOf(); ++i) {
    byteStrides[static_cast<size_t>(i)] = destination->strideAt(i) * elementSize;
  }
  PJRT_Buffer_MemoryLayout hostLayout{};
  hostLayout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  hostLayout.type = PJRT_Buffer_MemoryLayout_Type_Strides;
  hostLayout.strides.struct_size = PJRT_Buffer_MemoryLayout_Strides_STRUCT_SIZE;
  hostLayout.strides.byte_strides = byteStrides.empty() ? nullptr : byteStrides.data();
  hostLayout.strides.num_byte_strides = byteStrides.size();

  PJRT_Buffer_ToHostBuffer_Args args{};
  args.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  args.src = pjrtBuffer;
  args.host_layout = &hostLayout;
  args.dst = hostBuffer;
  args.dst_size = bytes;
  if (!consumeError(api->PJRT_Buffer_ToHostBuffer(&args),
                    "PJRT_Buffer_ToHostBuffer")) {
    return false;
  }
  if (!awaitAndDestroyEvent(args.event, "PJRT device-to-host transfer")) {
    return false;
  }
  destination->tickWriteHost();
  return true;
}

void* PjrtClientManager::compile(const void* programBytes, size_t programSize,
                                 const char* programFormat, int deviceIdx) {
  if (programBytes == nullptr || programSize == 0 || programFormat == nullptr ||
      programFormat[0] == '\0' || !initialize()) {
    setLastError("Cannot compile an empty PJRT program");
    return nullptr;
  }
  setLastError("");
  std::lock_guard<std::mutex> callLock(executionMutex_);
  if (!validDeviceIndex(deviceIdx)) {
    setLastError("PJRT compilation device index is out of range");
    return nullptr;
  }

  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = const_cast<char*>(static_cast<const char*>(programBytes));
  program.code_size = programSize;
  program.format = programFormat;
  program.format_size = std::strlen(programFormat);

  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  PJRT_Client_Compile_Args args{};
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  args.client = static_cast<PJRT_Client*>(client_);
  args.program = &program;
  if (!consumeError(api->PJRT_Client_Compile(&args), "PJRT_Client_Compile") ||
      args.executable == nullptr) {
    return nullptr;
  }

  PJRT_LoadedExecutable_AddressableDevices_Args deviceArgs{};
  deviceArgs.struct_size =
      PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
  deviceArgs.executable = args.executable;
  bool deviceSupported = consumeError(
      api->PJRT_LoadedExecutable_AddressableDevices(&deviceArgs),
      "PJRT_LoadedExecutable_AddressableDevices");
  if (deviceSupported) {
    deviceSupported = false;
    PJRT_Device* requestedDevice =
        static_cast<PJRT_Device*>(devices_[static_cast<size_t>(deviceIdx)]);
    for (size_t i = 0; i < deviceArgs.num_addressable_devices; ++i) {
      if (deviceArgs.addressable_devices[i] == requestedDevice) {
        deviceSupported = true;
        break;
      }
    }
  }
  if (!deviceSupported) {
    setLastError("Compiled PJRT executable is not addressable on the requested TPU device");
    PJRT_LoadedExecutable_Destroy_Args destroyArgs{};
    destroyArgs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    destroyArgs.executable = args.executable;
    consumeError(api->PJRT_LoadedExecutable_Destroy(&destroyArgs),
                 "PJRT_LoadedExecutable_Destroy");
    return nullptr;
  }
  DSP_DIAG(COMPILE, "PjrtClientManager: compiled %zu-byte %s program for device %d",
           programSize, programFormat, deviceIdx);
  return args.executable;
}

bool PjrtClientManager::execute(void* executable, void** inputBuffers,
                                int numInputs, int deviceIdx,
                                std::vector<void*>& outputBuffers) {
  outputBuffers.clear();
  if (executable == nullptr || numInputs < 0 ||
      (numInputs > 0 && inputBuffers == nullptr) || !initialize()) {
    setLastError("Invalid PJRT execute arguments");
    return false;
  }
  setLastError("");
  std::lock_guard<std::mutex> callLock(executionMutex_);
  if (!validDeviceIndex(deviceIdx)) {
    setLastError("PJRT execution device index is out of range");
    return false;
  }

  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  auto* loaded = static_cast<PJRT_LoadedExecutable*>(executable);

  PJRT_LoadedExecutable_GetExecutable_Args executableArgs{};
  executableArgs.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  executableArgs.loaded_executable = loaded;
  if (!consumeError(api->PJRT_LoadedExecutable_GetExecutable(&executableArgs),
                    "PJRT_LoadedExecutable_GetExecutable") ||
      executableArgs.executable == nullptr) {
    return false;
  }

  PJRT_Executable_NumOutputs_Args countArgs{};
  countArgs.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  countArgs.executable = executableArgs.executable;
  const bool counted = consumeError(api->PJRT_Executable_NumOutputs(&countArgs),
                                    "PJRT_Executable_NumOutputs");

  PJRT_Executable_Destroy_Args executableDestroyArgs{};
  executableDestroyArgs.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  executableDestroyArgs.executable = executableArgs.executable;
  const bool metadataDestroyed = consumeError(
      api->PJRT_Executable_Destroy(&executableDestroyArgs),
      "PJRT_Executable_Destroy");
  if (!counted || !metadataDestroyed) return false;

  std::vector<PJRT_Buffer*> typedInputs(static_cast<size_t>(numInputs));
  for (int i = 0; i < numInputs; ++i) {
    typedInputs[static_cast<size_t>(i)] = static_cast<PJRT_Buffer*>(inputBuffers[i]);
  }
  std::vector<PJRT_Buffer*> typedOutputs(countArgs.num_outputs, nullptr);
  PJRT_Buffer* const* inputRow = typedInputs.empty() ? nullptr : typedInputs.data();
  PJRT_Buffer* const* argumentLists[1] = {inputRow};
  PJRT_Buffer** outputLists[1] = {typedOutputs.empty() ? nullptr : typedOutputs.data()};
  PJRT_Event* completionEvents[1] = {nullptr};

  PJRT_ExecuteOptions options{};
  options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

  PJRT_LoadedExecutable_Execute_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  args.executable = loaded;
  args.options = &options;
  args.argument_lists = argumentLists;
  args.num_devices = 1;
  args.num_args = typedInputs.size();
  args.output_lists = outputLists;
  args.device_complete_events = completionEvents;
  args.execute_device =
      static_cast<PJRT_Device*>(devices_[static_cast<size_t>(deviceIdx)]);

  if (!consumeError(api->PJRT_LoadedExecutable_Execute(&args),
                    "PJRT_LoadedExecutable_Execute") ||
      !awaitAndDestroyEvent(completionEvents[0], "PJRT execution")) {
    for (auto* output : typedOutputs) {
      if (output != nullptr) {
        PJRT_Buffer_Destroy_Args destroyArgs{};
        destroyArgs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroyArgs.buffer = output;
        consumeError(api->PJRT_Buffer_Destroy(&destroyArgs), "PJRT_Buffer_Destroy");
      }
    }
    return false;
  }

  outputBuffers.reserve(typedOutputs.size());
  for (auto* output : typedOutputs) outputBuffers.push_back(output);
  return true;
}

void PjrtClientManager::destroyExecutable(void* executable) {
  if (executable == nullptr || !initialize()) return;
  std::lock_guard<std::mutex> callLock(executionMutex_);
  const auto* api = static_cast<const PJRT_Api*>(pjrtApi_);
  PJRT_LoadedExecutable_Destroy_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = static_cast<PJRT_LoadedExecutable*>(executable);
  consumeError(api->PJRT_LoadedExecutable_Destroy(&args),
               "PJRT_LoadedExecutable_Destroy");
}

void PjrtClientManager::invalidateCompilationCache() {
  std::lock_guard<std::mutex> lock(initMutex_);
  ++compilationGeneration_;
}

uint64_t PjrtClientManager::compilationGeneration() const {
  std::lock_guard<std::mutex> lock(initMutex_);
  return compilationGeneration_;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
