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

#include "vlmBackend.h"
#include "vlmUtils.h"

#if HAVE_VLM

#include <ggml-backend.h>
#include <system/env_functions.h>

#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#endif

#ifdef GGML_USE_METAL
#include <ggml-metal.h>
#endif

#ifdef GGML_USE_VULKAN
#include <ggml-vulkan.h>
#endif

namespace sd {
namespace vlmUtils {

// ============================================================================
// VlmBackendManager Implementation
// ============================================================================

VlmBackendManager::VlmBackendManager()
    : _initialized(false)
    , _activeBackendType(VlmBackendType::CPU)
    , _activeBackend(nullptr)
    , _cpuBackend(nullptr)
    , _gpuBackend(nullptr) {
}

VlmBackendManager::~VlmBackendManager() {
    shutdown();
}

VlmBackendManager& VlmBackendManager::getInstance() {
    static VlmBackendManager instance;
    return instance;
}

bool VlmBackendManager::initialize(const VlmBackendConfig& config) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_initialized) {
        return true;
    }

    _config = config;

    // Always initialize CPU backend first (fallback)
    if (!initializeCpuBackend()) {
        return false;
    }

    // Enumerate available devices
    _deviceInfo.clear();

    // Add CPU device
    VlmDeviceInfo cpuInfo;
    cpuInfo.type = VlmBackendType::CPU;
    cpuInfo.deviceIndex = 0;
    cpuInfo.name = "CPU";
    cpuInfo.totalMemory = 0;  // System memory
    cpuInfo.freeMemory = 0;
    cpuInfo.available = true;
    _deviceInfo.push_back(cpuInfo);

#ifdef GGML_USE_CUDA
    // Enumerate CUDA devices
    int cudaDeviceCount = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < cudaDeviceCount; i++) {
        VlmDeviceInfo cudaInfo;
        cudaInfo.type = VlmBackendType::CUDA;
        cudaInfo.deviceIndex = i;

        char deviceDesc[256];
        ggml_backend_cuda_get_device_description(i, deviceDesc, sizeof(deviceDesc));
        cudaInfo.name = deviceDesc;

        size_t free, total;
        ggml_backend_cuda_get_device_memory(i, &free, &total);
        cudaInfo.totalMemory = total;
        cudaInfo.freeMemory = free;
        cudaInfo.available = true;
        _deviceInfo.push_back(cudaInfo);
    }
#endif

#ifdef GGML_USE_METAL
    // Metal is available on Apple Silicon
    VlmDeviceInfo metalInfo;
    metalInfo.type = VlmBackendType::METAL;
    metalInfo.deviceIndex = 0;
    metalInfo.name = "Metal";
    metalInfo.totalMemory = 0;
    metalInfo.freeMemory = 0;
    metalInfo.available = true;
    _deviceInfo.push_back(metalInfo);
#endif

#ifdef GGML_USE_VULKAN
    // Enumerate Vulkan devices
    int vulkanDeviceCount = ggml_backend_vk_get_device_count();
    for (int i = 0; i < vulkanDeviceCount; i++) {
        VlmDeviceInfo vkInfo;
        vkInfo.type = VlmBackendType::VULKAN;
        vkInfo.deviceIndex = i;

        char deviceDesc[256];
        ggml_backend_vk_get_device_description(i, deviceDesc, sizeof(deviceDesc));
        vkInfo.name = deviceDesc;

        size_t free, total;
        ggml_backend_vk_get_device_memory(i, &free, &total);
        vkInfo.totalMemory = total;
        vkInfo.freeMemory = free;
        vkInfo.available = true;
        _deviceInfo.push_back(vkInfo);
    }
#endif

    // Select backend based on configuration
    VlmBackendType targetBackend = config.preferredBackend;
    if (targetBackend == VlmBackendType::AUTO) {
        targetBackend = selectBestBackend();
    }

    // Initialize the selected backend
    if (!selectBackend(targetBackend, config.deviceIndex)) {
        // Fallback to CPU
        _activeBackend = _cpuBackend;
        _activeBackendType = VlmBackendType::CPU;
    }

    _initialized = true;
    return true;
}

void VlmBackendManager::shutdown() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_initialized) return;

    // Free all backends
    for (auto backend : _allBackends) {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
    _allBackends.clear();

    _activeBackend = nullptr;
    _cpuBackend = nullptr;
    _gpuBackend = nullptr;
    _deviceInfo.clear();
    _initialized = false;
}

void VlmBackendManager::setConfig(const VlmBackendConfig& config) {
    std::lock_guard<std::mutex> lock(_mutex);
    _config = config;
}

std::vector<VlmDeviceInfo> VlmBackendManager::getAvailableDevices() const {
    std::lock_guard<std::mutex> lock(_mutex);
    return _deviceInfo;
}

bool VlmBackendManager::isBackendAvailable(VlmBackendType type) const {
    std::lock_guard<std::mutex> lock(_mutex);
    for (const auto& info : _deviceInfo) {
        if (info.type == type && info.available) {
            return true;
        }
    }
    return false;
}

bool VlmBackendManager::selectBackend(VlmBackendType type, int deviceIndex) {
    std::lock_guard<std::mutex> lock(_mutex);

    switch (type) {
        case VlmBackendType::CPU:
            if (_cpuBackend != nullptr) {
                _activeBackend = _cpuBackend;
                _activeBackendType = VlmBackendType::CPU;
                return true;
            }
            return initializeCpuBackend();

        case VlmBackendType::CUDA:
            return initializeCudaBackend(deviceIndex);

        case VlmBackendType::METAL:
            return initializeMetalBackend();

        case VlmBackendType::VULKAN:
            return initializeVulkanBackend(deviceIndex);

        case VlmBackendType::AUTO:
            return selectBackend(selectBestBackend(), deviceIndex);

        default:
            return false;
    }
}

std::string VlmBackendManager::backendTypeToString(VlmBackendType type) {
    switch (type) {
        case VlmBackendType::CPU: return "CPU";
        case VlmBackendType::CUDA: return "CUDA";
        case VlmBackendType::METAL: return "Metal";
        case VlmBackendType::VULKAN: return "Vulkan";
        case VlmBackendType::SYCL: return "SYCL";
        case VlmBackendType::OPENCL: return "OpenCL";
        case VlmBackendType::AUTO: return "Auto";
        default: return "Unknown";
    }
}

VlmBackendType VlmBackendManager::stringToBackendType(const std::string& name) {
    if (name == "CPU" || name == "cpu") return VlmBackendType::CPU;
    if (name == "CUDA" || name == "cuda") return VlmBackendType::CUDA;
    if (name == "Metal" || name == "metal") return VlmBackendType::METAL;
    if (name == "Vulkan" || name == "vulkan") return VlmBackendType::VULKAN;
    if (name == "SYCL" || name == "sycl") return VlmBackendType::SYCL;
    if (name == "OpenCL" || name == "opencl") return VlmBackendType::OPENCL;
    return VlmBackendType::AUTO;
}

bool VlmBackendManager::initializeCpuBackend() {
    if (_cpuBackend != nullptr) return true;

    _cpuBackend = ggml_backend_cpu_init();
    if (_cpuBackend == nullptr) return false;

    // Set number of threads
    int numThreads = _config.numThreads > 0 ?
        _config.numThreads : sd::env_maxMasterThreads();
    ggml_backend_cpu_set_n_threads(_cpuBackend, numThreads);

    _allBackends.push_back(_cpuBackend);

    if (_activeBackend == nullptr) {
        _activeBackend = _cpuBackend;
        _activeBackendType = VlmBackendType::CPU;
    }

    return true;
}

bool VlmBackendManager::initializeCudaBackend(int deviceIndex) {
#ifdef GGML_USE_CUDA
    // Check if device is valid
    int deviceCount = ggml_backend_cuda_get_device_count();
    if (deviceIndex >= deviceCount) return false;

    // Free existing GPU backend if different device
    if (_gpuBackend != nullptr && _activeBackendType == VlmBackendType::CUDA) {
        // Already using CUDA, check if same device
        // For now, just return true if already initialized
        _activeBackend = _gpuBackend;
        return true;
    }

    ggml_backend_t cudaBackend = ggml_backend_cuda_init(deviceIndex);
    if (cudaBackend == nullptr) return false;

    _gpuBackend = cudaBackend;
    _activeBackend = cudaBackend;
    _activeBackendType = VlmBackendType::CUDA;
    _allBackends.push_back(cudaBackend);

    return true;
#else
    return false;
#endif
}

bool VlmBackendManager::initializeMetalBackend() {
#ifdef GGML_USE_METAL
    if (_gpuBackend != nullptr && _activeBackendType == VlmBackendType::METAL) {
        _activeBackend = _gpuBackend;
        return true;
    }

    ggml_backend_t metalBackend = ggml_backend_metal_init();
    if (metalBackend == nullptr) return false;

    _gpuBackend = metalBackend;
    _activeBackend = metalBackend;
    _activeBackendType = VlmBackendType::METAL;
    _allBackends.push_back(metalBackend);

    return true;
#else
    return false;
#endif
}

bool VlmBackendManager::initializeVulkanBackend(int deviceIndex) {
#ifdef GGML_USE_VULKAN
    int deviceCount = ggml_backend_vk_get_device_count();
    if (deviceIndex >= deviceCount) return false;

    if (_gpuBackend != nullptr && _activeBackendType == VlmBackendType::VULKAN) {
        _activeBackend = _gpuBackend;
        return true;
    }

    ggml_backend_t vkBackend = ggml_backend_vk_init(deviceIndex);
    if (vkBackend == nullptr) return false;

    _gpuBackend = vkBackend;
    _activeBackend = vkBackend;
    _activeBackendType = VlmBackendType::VULKAN;
    _allBackends.push_back(vkBackend);

    return true;
#else
    return false;
#endif
}

VlmBackendType VlmBackendManager::selectBestBackend() const {
    // Priority: CUDA > Metal > Vulkan > CPU
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        return VlmBackendType::CUDA;
    }
#endif

#ifdef GGML_USE_METAL
    return VlmBackendType::METAL;
#endif

#ifdef GGML_USE_VULKAN
    if (ggml_backend_vk_get_device_count() > 0) {
        return VlmBackendType::VULKAN;
    }
#endif

    return VlmBackendType::CPU;
}

ggml_context* VlmBackendManager::createContext(size_t memSize) {
    if (memSize == 0) {
        memSize = _config.contextMemory;
    }

    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = nullptr,
        .no_alloc   = true  // We'll use backend allocation
    };

    return ggml_init(params);
}

void VlmBackendManager::freeContext(ggml_context* ctx) {
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

ggml_backend_buffer_t VlmBackendManager::allocateBuffer(size_t size) {
    if (_activeBackend == nullptr) return nullptr;
    return ggml_backend_alloc_buffer(_activeBackend, size);
}

void VlmBackendManager::freeBuffer(ggml_backend_buffer_t buffer) {
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
}

ggml_backend_buffer_t VlmBackendManager::allocContextTensors(ggml_context* ctx) {
    if (_activeBackend == nullptr || ctx == nullptr) return nullptr;
    return ggml_backend_alloc_ctx_tensors(ctx, _activeBackend);
}

ggml_tensor* VlmBackendManager::createTensorFromNDArray(ggml_context* ctx, const NDArray* array,
                                                         const char* name) {
    if (ctx == nullptr || array == nullptr) return nullptr;

    const auto rank = array->rankOf();
    const auto shape = array->shapeOf();
    const auto type = dataTypeToGgml(array->dataType());

    ggml_tensor* tensor = nullptr;

    // GGML uses column-major order, so we reverse dimensions
    switch (rank) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx, type, shape[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, type, shape[1], shape[0]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(ctx, type, shape[2], shape[1], shape[0]);
            break;
        case 4:
        default:
            tensor = ggml_new_tensor_4d(ctx, type,
                rank >= 4 ? shape[3] : 1,
                rank >= 3 ? shape[2] : 1,
                rank >= 2 ? shape[1] : 1,
                shape[0]);
            break;
    }

    if (tensor != nullptr && name != nullptr) {
        ggml_set_name(tensor, name);
    }

    return tensor;
}

void VlmBackendManager::copyTensorToNDArray(const ggml_tensor* tensor, NDArray* array) {
    if (tensor == nullptr || array == nullptr || _activeBackend == nullptr) return;

    size_t nbytes = ggml_nbytes(tensor);
    ggml_backend_tensor_get(tensor, array->buffer(), 0, nbytes);
}

void VlmBackendManager::setTensorData(ggml_tensor* tensor, const NDArray* array) {
    if (tensor == nullptr || array == nullptr || _activeBackend == nullptr) return;

    size_t nbytes = std::min(ggml_nbytes(tensor), array->lengthOf() * array->sizeOfT());
    ggml_backend_tensor_set(tensor, array->buffer(), 0, nbytes);
}

void VlmBackendManager::getTensorData(const ggml_tensor* tensor, NDArray* array) {
    copyTensorToNDArray(tensor, array);
}

bool VlmBackendManager::executeGraph(ggml_cgraph* graph) {
    return executeGraph(graph, _activeBackend);
}

bool VlmBackendManager::executeGraph(ggml_cgraph* graph, ggml_backend_t backend) {
    if (graph == nullptr || backend == nullptr) return false;

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    return status == GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t VlmBackendManager::createScheduler(ggml_cgraph* graph) {
    if (_activeBackend == nullptr) return nullptr;

    // Create array of backends for scheduler
    ggml_backend_t backends[] = { _activeBackend };
    int nBackends = 1;

    // Add CPU backend as fallback if different from active
    ggml_backend_t backendList[2];
    backendList[0] = _activeBackend;
    if (_cpuBackend != nullptr && _cpuBackend != _activeBackend) {
        backendList[1] = _cpuBackend;
        nBackends = 2;
    }

    return ggml_backend_sched_new(backendList, nullptr, nBackends, graph->n_nodes, false);
}

void VlmBackendManager::freeScheduler(ggml_backend_sched_t sched) {
    if (sched != nullptr) {
        ggml_backend_sched_free(sched);
    }
}

size_t VlmBackendManager::getTotalMemory() const {
    // Return memory for active device
    for (const auto& info : _deviceInfo) {
        if (info.type == _activeBackendType) {
            return info.totalMemory;
        }
    }
    return 0;
}

size_t VlmBackendManager::getFreeMemory() const {
    std::lock_guard<std::mutex> lock(_mutex);

#ifdef GGML_USE_CUDA
    if (_activeBackendType == VlmBackendType::CUDA) {
        size_t free, total;
        ggml_backend_cuda_get_device_memory(_config.deviceIndex, &free, &total);
        return free;
    }
#endif

#ifdef GGML_USE_VULKAN
    if (_activeBackendType == VlmBackendType::VULKAN) {
        size_t free, total;
        ggml_backend_vk_get_device_memory(_config.deviceIndex, &free, &total);
        return free;
    }
#endif

    return 0;
}

void VlmBackendManager::synchronize() {
    if (_activeBackend != nullptr) {
        ggml_backend_synchronize(_activeBackend);
    }
}

ggml_type VlmBackendManager::dataTypeToGgml(DataType dt) {
    return toGgmlType(dt);
}

DataType VlmBackendManager::ggmlToDataType(ggml_type gt) {
    return fromGgmlType(gt);
}

bool VlmBackendManager::isTypeSupported(DataType dt) {
    return isSupportedType(dt);
}

// ============================================================================
// VlmBackendContext Implementation
// ============================================================================

VlmBackendContext::VlmBackendContext(size_t memSize)
    : _ctx(nullptr), _buffer(nullptr) {

    auto& manager = VlmBackendManager::getInstance();
    if (!manager.isInitialized()) {
        manager.initialize();
    }

    // Allocate memory buffer for context metadata
    _memBuffer.resize(memSize);

    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = _memBuffer.data(),
        .no_alloc   = true  // We'll allocate tensors to backend
    };

    _ctx = ggml_init(params);
}

VlmBackendContext::~VlmBackendContext() {
    if (_buffer != nullptr) {
        ggml_backend_buffer_free(_buffer);
        _buffer = nullptr;
    }
    if (_ctx != nullptr) {
        ggml_free(_ctx);
        _ctx = nullptr;
    }
}

VlmBackendContext::VlmBackendContext(VlmBackendContext&& other) noexcept
    : _ctx(other._ctx), _buffer(other._buffer), _memBuffer(std::move(other._memBuffer)) {
    other._ctx = nullptr;
    other._buffer = nullptr;
}

VlmBackendContext& VlmBackendContext::operator=(VlmBackendContext&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (_buffer != nullptr) ggml_backend_buffer_free(_buffer);
        if (_ctx != nullptr) ggml_free(_ctx);

        // Move resources
        _ctx = other._ctx;
        _buffer = other._buffer;
        _memBuffer = std::move(other._memBuffer);

        other._ctx = nullptr;
        other._buffer = nullptr;
    }
    return *this;
}

ggml_tensor* VlmBackendContext::createTensor(const NDArray* array, const char* name) {
    auto& manager = VlmBackendManager::getInstance();
    return manager.createTensorFromNDArray(_ctx, array, name);
}

void VlmBackendContext::copyToNDArray(const ggml_tensor* tensor, NDArray* array) {
    auto& manager = VlmBackendManager::getInstance();
    manager.copyTensorToNDArray(tensor, array);
}

bool VlmBackendContext::allocate() {
    if (_ctx == nullptr) return false;

    auto& manager = VlmBackendManager::getInstance();
    _buffer = manager.allocContextTensors(_ctx);
    return _buffer != nullptr;
}

bool VlmBackendContext::execute(ggml_cgraph* graph) {
    auto& manager = VlmBackendManager::getInstance();
    return manager.executeGraph(graph);
}

}  // namespace vlmUtils
}  // namespace sd

#endif  // HAVE_VLM
