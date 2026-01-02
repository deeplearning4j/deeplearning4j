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
// Backend capability tracking and transparency for LlamaCpp operations
// Integrates with ND4J's device affinity and memory management systems
//

#include "llamacppUtils.h"
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <array/DataBuffer.h>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>

#if HAVE_LLAMACPP

namespace sd {
namespace llamacppUtils {

//////////////////////////////////////////////////////////////////////////
// Backend Capability Registry
//////////////////////////////////////////////////////////////////////////

/**
 * Detailed capability information for a GGML backend
 */
struct BackendCapability {
    GgmlBackend type;
    std::string name;
    std::string version;
    bool available;
    bool supportsQuantization;
    bool supportsFlashAttention;
    bool supportsModelParallel;
    bool supportsP2P;
    bool supportsAsyncCopy;
    bool supportsUnifiedMemory;
    int maxTensorDims;
    size_t maxMemory;
    size_t availableMemory;
    std::vector<ggml_type> supportedQuantTypes;
    std::vector<int> p2pConnectedDevices;

    // Performance hints
    float relativeMatmulSpeed;    // Relative to CPU (1.0)
    float relativeMemoryBandwidth;
    bool preferredForLargeModels;
    bool preferredForBatchInference;
};

/**
 * Device-specific backend state
 */
struct DeviceBackendState {
    int deviceId;
    GgmlBackend activeBackend;
    struct ggml_backend* backendHandle;
    struct ggml_backend_buffer_type* bufferType;
    size_t allocatedMemory;
    size_t peakMemory;
    bool initialized;
};

// Global registry
static std::mutex g_capabilityMutex;
static std::unordered_map<GgmlBackend, BackendCapability> g_backendCapabilities;
static std::unordered_map<int, DeviceBackendState> g_deviceStates;
static std::atomic<bool> g_capabilitiesInitialized{false};

//////////////////////////////////////////////////////////////////////////
// Backend Discovery and Initialization
//////////////////////////////////////////////////////////////////////////

static void initializeBackendCapabilities() {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    if (g_capabilitiesInitialized) return;

    // CPU Backend - Always available
    BackendCapability cpuCap;
    cpuCap.type = GGML_BACKEND_CPU;
    cpuCap.name = "CPU";
    cpuCap.version = "native";
    cpuCap.available = true;
    cpuCap.supportsQuantization = true;
    cpuCap.supportsFlashAttention = false;  // CPU flash attention is slower
    cpuCap.supportsModelParallel = false;
    cpuCap.supportsP2P = false;
    cpuCap.supportsAsyncCopy = false;
    cpuCap.supportsUnifiedMemory = true;
    cpuCap.maxTensorDims = 4;
    cpuCap.maxMemory = 0;  // No fixed limit
    cpuCap.availableMemory = 0;
    cpuCap.supportedQuantTypes = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ4_NL
    };
    cpuCap.relativeMatmulSpeed = 1.0f;
    cpuCap.relativeMemoryBandwidth = 1.0f;
    cpuCap.preferredForLargeModels = false;
    cpuCap.preferredForBatchInference = false;
    g_backendCapabilities[GGML_BACKEND_CPU] = cpuCap;

#if defined(GGML_USE_CUDA) || defined(__CUDACC__)
    // CUDA Backend
    BackendCapability cudaCap;
    cudaCap.type = GGML_BACKEND_CUDA;
    cudaCap.name = "CUDA";
    cudaCap.available = hasCudaBackend();
    cudaCap.supportsQuantization = true;
    cudaCap.supportsFlashAttention = true;
    cudaCap.supportsModelParallel = getCudaDeviceCount() > 1;
    cudaCap.supportsP2P = getCudaDeviceCount() > 1;  // Needs P2P check
    cudaCap.supportsAsyncCopy = true;
    cudaCap.supportsUnifiedMemory = true;  // CUDA unified memory
    cudaCap.maxTensorDims = 4;
    cudaCap.supportedQuantTypes = cpuCap.supportedQuantTypes;  // CUDA supports all
    cudaCap.relativeMatmulSpeed = 50.0f;  // Rough estimate
    cudaCap.relativeMemoryBandwidth = 10.0f;
    cudaCap.preferredForLargeModels = true;
    cudaCap.preferredForBatchInference = true;

    // Query actual CUDA device properties
    if (cudaCap.available) {
        // Would query actual device memory here
        cudaCap.version = "CUDA";
    }
    g_backendCapabilities[GGML_BACKEND_CUDA] = cudaCap;
#endif

#if defined(GGML_USE_METAL)
    // Metal Backend (Apple Silicon)
    BackendCapability metalCap;
    metalCap.type = GGML_BACKEND_METAL;
    metalCap.name = "Metal";
    metalCap.available = true;  // Would need actual check
    metalCap.supportsQuantization = true;
    metalCap.supportsFlashAttention = true;
    metalCap.supportsModelParallel = false;  // Single GPU on Apple Silicon
    metalCap.supportsP2P = false;
    metalCap.supportsAsyncCopy = true;
    metalCap.supportsUnifiedMemory = true;  // Apple Unified Memory
    metalCap.maxTensorDims = 4;
    metalCap.relativeMatmulSpeed = 30.0f;
    metalCap.relativeMemoryBandwidth = 8.0f;
    metalCap.preferredForLargeModels = true;
    metalCap.preferredForBatchInference = true;
    g_backendCapabilities[GGML_BACKEND_METAL] = metalCap;
#endif

#if defined(GGML_USE_VULKAN)
    // Vulkan Backend
    BackendCapability vulkanCap;
    vulkanCap.type = GGML_BACKEND_VULKAN;
    vulkanCap.name = "Vulkan";
    vulkanCap.available = true;  // Would need actual check
    vulkanCap.supportsQuantization = true;
    vulkanCap.supportsFlashAttention = false;
    vulkanCap.supportsModelParallel = false;
    vulkanCap.supportsP2P = false;
    vulkanCap.supportsAsyncCopy = true;
    vulkanCap.supportsUnifiedMemory = false;
    vulkanCap.maxTensorDims = 4;
    vulkanCap.relativeMatmulSpeed = 20.0f;
    vulkanCap.relativeMemoryBandwidth = 5.0f;
    vulkanCap.preferredForLargeModels = false;
    vulkanCap.preferredForBatchInference = false;
    g_backendCapabilities[GGML_BACKEND_VULKAN] = vulkanCap;
#endif

#if defined(GGML_USE_SYCL)
    // SYCL Backend (Intel)
    BackendCapability syclCap;
    syclCap.type = GGML_BACKEND_SYCL;
    syclCap.name = "SYCL";
    syclCap.available = true;
    syclCap.supportsQuantization = true;
    syclCap.supportsFlashAttention = true;
    syclCap.supportsModelParallel = true;
    syclCap.supportsP2P = true;
    syclCap.supportsAsyncCopy = true;
    syclCap.supportsUnifiedMemory = true;
    syclCap.maxTensorDims = 4;
    syclCap.relativeMatmulSpeed = 25.0f;
    syclCap.relativeMemoryBandwidth = 6.0f;
    syclCap.preferredForLargeModels = true;
    syclCap.preferredForBatchInference = true;
    g_backendCapabilities[GGML_BACKEND_SYCL] = syclCap;
#endif

    g_capabilitiesInitialized = true;
}

//////////////////////////////////////////////////////////////////////////
// Public Backend Capability API
//////////////////////////////////////////////////////////////////////////

/**
 * Get list of all available backends
 */
std::vector<GgmlBackend> getAvailableBackendsList() {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    std::vector<GgmlBackend> result;

    for (const auto& pair : g_backendCapabilities) {
        if (pair.second.available) {
            result.push_back(pair.first);
        }
    }
    return result;
}

/**
 * Check if a specific backend is available
 */
bool isBackendAvailable(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    return it != g_backendCapabilities.end() && it->second.available;
}

/**
 * Get backend name as string
 */
const char* getBackendName(GgmlBackend backend) {
    switch (backend) {
        case GGML_BACKEND_CPU: return "CPU";
        case GGML_BACKEND_CUDA: return "CUDA";
        case GGML_BACKEND_METAL: return "Metal";
        case GGML_BACKEND_VULKAN: return "Vulkan";
        case GGML_BACKEND_SYCL: return "SYCL";
        case GGML_BACKEND_OPENCL: return "OpenCL";
        default: return "Unknown";
    }
}

/**
 * Discover and initialize all available backends
 */
void discoverBackends() {
    initializeBackendCapabilities();
}

/**
 * Get the full capability struct for a backend
 */
BackendCapability getBackendCapability(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    if (it != g_backendCapabilities.end()) {
        // Convert internal struct to public struct
        BackendCapability result;
        result.type = it->second.type;
        result.name = it->second.name;
        result.available = it->second.available;
        result.supportsQuantization = it->second.supportsQuantization;
        result.supportsFlashAttention = it->second.supportsFlashAttention;
        result.supportsModelParallel = it->second.supportsModelParallel;
        result.supportsP2P = it->second.supportsP2P;
        result.supportsAsyncCopy = it->second.supportsAsyncCopy;
        result.supportsUnifiedMemory = it->second.supportsUnifiedMemory;
        result.supportedQuantTypes = it->second.supportedQuantTypes;
        result.relativeMatmulSpeed = it->second.relativeMatmulSpeed;
        result.maxMemoryBytes = it->second.maxMemory;
        result.computeCapability = 0;  // Would query from device
        return result;
    }

    // Return empty capability for unknown backend
    BackendCapability empty;
    empty.type = backend;
    empty.name = "Unknown";
    empty.available = false;
    empty.supportsQuantization = false;
    empty.supportsFlashAttention = false;
    empty.supportsModelParallel = false;
    empty.supportsP2P = false;
    empty.supportsAsyncCopy = false;
    empty.supportsUnifiedMemory = false;
    empty.relativeMatmulSpeed = 0.0f;
    empty.maxMemoryBytes = 0;
    empty.computeCapability = 0;
    return empty;
}

/**
 * Check if backend supports model parallelism
 */
bool backendSupportsModelParallel(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    return it != g_backendCapabilities.end() && it->second.supportsModelParallel;
}

/**
 * Check if backend supports flash attention
 */
bool backendSupportsFlashAttention(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    return it != g_backendCapabilities.end() && it->second.supportsFlashAttention;
}

/**
 * Check if backend supports quantization
 */
bool backendSupportsQuantization(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    return it != g_backendCapabilities.end() && it->second.supportsQuantization;
}

/**
 * Check if backend supports P2P transfers
 */
bool backendSupportsP2P(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    return it != g_backendCapabilities.end() && it->second.supportsP2P;
}

/**
 * Get backend capability summary as string
 */
std::string getBackendCapabilitySummary(GgmlBackend backend) {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);
    auto it = g_backendCapabilities.find(backend);
    if (it == g_backendCapabilities.end()) {
        return "Backend not found";
    }

    const auto& cap = it->second;
    std::string summary = cap.name + " Backend:\n";
    summary += "  Available: " + std::string(cap.available ? "Yes" : "No") + "\n";
    summary += "  Model Parallel: " + std::string(cap.supportsModelParallel ? "Yes" : "No") + "\n";
    summary += "  Flash Attention: " + std::string(cap.supportsFlashAttention ? "Yes" : "No") + "\n";
    summary += "  Quantization: " + std::string(cap.supportsQuantization ? "Yes" : "No") + "\n";
    summary += "  P2P Transfers: " + std::string(cap.supportsP2P ? "Yes" : "No") + "\n";
    summary += "  Async Copy: " + std::string(cap.supportsAsyncCopy ? "Yes" : "No") + "\n";
    summary += "  Unified Memory: " + std::string(cap.supportsUnifiedMemory ? "Yes" : "No") + "\n";
    summary += "  Relative MatMul Speed: " + std::to_string(cap.relativeMatmulSpeed) + "x\n";
    return summary;
}

/**
 * Print all backend capabilities
 */
void printAllBackendCapabilities() {
    initializeBackendCapabilities();

    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    printf("=== GGML Backend Capabilities ===\n\n");

    for (const auto& pair : g_backendCapabilities) {
        const auto& cap = pair.second;
        printf("%s Backend:\n", cap.name.c_str());
        printf("  Status:          %s\n", cap.available ? "AVAILABLE" : "NOT AVAILABLE");

        if (cap.available) {
            printf("  Model Parallel:  %s\n", cap.supportsModelParallel ? "YES" : "NO");
            printf("  Flash Attention: %s\n", cap.supportsFlashAttention ? "YES" : "NO");
            printf("  Quantization:    %s\n", cap.supportsQuantization ? "YES" : "NO");
            printf("  P2P Transfers:   %s\n", cap.supportsP2P ? "YES" : "NO");
            printf("  Async Copy:      %s\n", cap.supportsAsyncCopy ? "YES" : "NO");
            printf("  Unified Memory:  %s\n", cap.supportsUnifiedMemory ? "YES" : "NO");
            printf("  Matmul Speed:    %.1fx (relative to CPU)\n", cap.relativeMatmulSpeed);
            printf("  Preferred for:\n");
            if (cap.preferredForLargeModels) printf("    - Large models\n");
            if (cap.preferredForBatchInference) printf("    - Batch inference\n");
        }
        printf("\n");
    }
}

//////////////////////////////////////////////////////////////////////////
// Backend Selection Strategy
//////////////////////////////////////////////////////////////////////////

/**
 * Select best backend for an operation based on data locality and capabilities
 */
GgmlBackend selectBestBackend(const NDArray* input, bool requiresQuantization,
                               bool requiresFlashAttention, bool requiresModelParallel) {
    initializeBackendCapabilities();

    // Get device where input data resides
    int dataDeviceId = input->getDeviceId();

    // Check if we're on a GPU device
    bool dataOnGpu = (dataDeviceId >= 0);

    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    // Priority order: data locality > capability match > performance

    // If data is on GPU and CUDA is available, prefer CUDA
#if defined(GGML_USE_CUDA)
    if (dataOnGpu) {
        auto it = g_backendCapabilities.find(GGML_BACKEND_CUDA);
        if (it != g_backendCapabilities.end() && it->second.available) {
            const auto& cap = it->second;
            if ((!requiresQuantization || cap.supportsQuantization) &&
                (!requiresFlashAttention || cap.supportsFlashAttention) &&
                (!requiresModelParallel || cap.supportsModelParallel)) {
                return GGML_BACKEND_CUDA;
            }
        }
    }
#endif

#if defined(GGML_USE_METAL)
    // Apple Silicon - data is always unified memory
    auto metalIt = g_backendCapabilities.find(GGML_BACKEND_METAL);
    if (metalIt != g_backendCapabilities.end() && metalIt->second.available) {
        const auto& cap = metalIt->second;
        if ((!requiresQuantization || cap.supportsQuantization) &&
            (!requiresFlashAttention || cap.supportsFlashAttention)) {
            return GGML_BACKEND_METAL;
        }
    }
#endif

    // Fallback to CPU
    return GGML_BACKEND_CPU;
}

/**
 * Get the preferred backend for the current thread/device context
 */
GgmlBackend getPreferredBackendForContext() {
    int currentDevice = AffinityManager::currentDeviceId();

    // Device ID >= 0 typically indicates GPU
    if (currentDevice >= 0) {
#if defined(GGML_USE_CUDA)
        if (isBackendAvailable(GGML_BACKEND_CUDA)) {
            return GGML_BACKEND_CUDA;
        }
#endif
#if defined(GGML_USE_METAL)
        if (isBackendAvailable(GGML_BACKEND_METAL)) {
            return GGML_BACKEND_METAL;
        }
#endif
    }

    return GGML_BACKEND_CPU;
}

//////////////////////////////////////////////////////////////////////////
// Device State Management
//////////////////////////////////////////////////////////////////////////

/**
 * Initialize backend state for a device
 */
bool initializeDeviceBackend(int deviceId, GgmlBackend backend) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    if (g_deviceStates.find(deviceId) != g_deviceStates.end() &&
        g_deviceStates[deviceId].initialized) {
        return true;  // Already initialized
    }

    DeviceBackendState state;
    state.deviceId = deviceId;
    state.activeBackend = backend;
    state.backendHandle = nullptr;
    state.bufferType = nullptr;
    state.allocatedMemory = 0;
    state.peakMemory = 0;
    state.initialized = false;

    // Initialize backend-specific handles
    switch (backend) {
        case GGML_BACKEND_CPU:
            // CPU doesn't need special initialization
            state.initialized = true;
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            state.backendHandle = getCudaBackend(deviceId);
            state.initialized = (state.backendHandle != nullptr);
            break;
#endif

        default:
            // Unsupported backend
            return false;
    }

    g_deviceStates[deviceId] = state;
    return state.initialized;
}

/**
 * Get device backend state
 */
DeviceBackendState* getDeviceBackendState(int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    auto it = g_deviceStates.find(deviceId);
    if (it != g_deviceStates.end()) {
        return &it->second;
    }
    return nullptr;
}

/**
 * Update memory tracking for a device
 */
void updateDeviceMemoryUsage(int deviceId, size_t allocated) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    auto it = g_deviceStates.find(deviceId);
    if (it != g_deviceStates.end()) {
        it->second.allocatedMemory = allocated;
        if (allocated > it->second.peakMemory) {
            it->second.peakMemory = allocated;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Per-Device Backend State Management (Public API)
//////////////////////////////////////////////////////////////////////////

// Storage for per-device, per-backend states
static std::unordered_map<uint64_t, ::sd::llamacppUtils::DeviceBackendState> g_deviceBackendStates;

// Create unique key for backend+device combination
static uint64_t makeBackendDeviceKey(GgmlBackend backend, int deviceId) {
    return (static_cast<uint64_t>(backend) << 32) | static_cast<uint64_t>(deviceId);
}

/**
 * Get or create backend state for a device
 */
::sd::llamacppUtils::DeviceBackendState& getDeviceBackendState(GgmlBackend backend, int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    uint64_t key = makeBackendDeviceKey(backend, deviceId);
    auto it = g_deviceBackendStates.find(key);

    if (it == g_deviceBackendStates.end()) {
        // Create new state
        ::sd::llamacppUtils::DeviceBackendState state;
        state.backend = backend;
        state.deviceId = deviceId;
        state.initialized = false;
        state.memoryAllocated = 0;
        state.memoryUsed = 0;
        state.peakMemoryUsed = 0;
        state.activeComputeGraphs = 0;
        state.busy = false;
        g_deviceBackendStates[key] = state;
        return g_deviceBackendStates[key];
    }

    return it->second;
}

/**
 * Initialize a backend for a specific device
 */
bool initializeBackendForDevice(GgmlBackend backend, int deviceId) {
    auto& state = getDeviceBackendState(backend, deviceId);

    if (state.initialized) {
        return true;  // Already initialized
    }

    bool success = false;
    switch (backend) {
        case GGML_BACKEND_CPU:
            // CPU always succeeds
            success = true;
            break;

#if defined(GGML_USE_CUDA)
        case GGML_BACKEND_CUDA:
            if (deviceId >= 0 && deviceId < getCudaDeviceCount()) {
                setCudaDevice(deviceId);
                success = true;
            }
            break;
#endif

        default:
            // Other backends may need specific initialization
            success = false;
            break;
    }

    if (success) {
        std::lock_guard<std::mutex> lock(g_capabilityMutex);
        uint64_t key = makeBackendDeviceKey(backend, deviceId);
        g_deviceBackendStates[key].initialized = true;
    }

    return success;
}

/**
 * Shutdown backend on a device and release resources
 */
void shutdownBackendForDevice(GgmlBackend backend, int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    uint64_t key = makeBackendDeviceKey(backend, deviceId);
    auto it = g_deviceBackendStates.find(key);

    if (it != g_deviceBackendStates.end()) {
        // Release any backend-specific resources
        switch (backend) {
            case GGML_BACKEND_CPU:
                // Nothing to release
                break;

#if defined(GGML_USE_CUDA)
            case GGML_BACKEND_CUDA:
                // Would release CUDA resources here
                break;
#endif

            default:
                break;
        }

        it->second.initialized = false;
        it->second.memoryAllocated = 0;
        it->second.memoryUsed = 0;
        it->second.activeComputeGraphs = 0;
        it->second.busy = false;
    }
}

/**
 * Get memory usage for a backend on a device
 */
size_t getBackendMemoryUsage(GgmlBackend backend, int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    uint64_t key = makeBackendDeviceKey(backend, deviceId);
    auto it = g_deviceBackendStates.find(key);

    if (it != g_deviceBackendStates.end()) {
        return it->second.memoryUsed;
    }
    return 0;
}

/**
 * Get peak memory usage for a backend on a device
 */
size_t getBackendPeakMemoryUsage(GgmlBackend backend, int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    uint64_t key = makeBackendDeviceKey(backend, deviceId);
    auto it = g_deviceBackendStates.find(key);

    if (it != g_deviceBackendStates.end()) {
        return it->second.peakMemoryUsed;
    }
    return 0;
}

/**
 * Reset peak memory tracking for a backend
 */
void resetBackendPeakMemory(GgmlBackend backend, int deviceId) {
    std::lock_guard<std::mutex> lock(g_capabilityMutex);

    uint64_t key = makeBackendDeviceKey(backend, deviceId);
    auto it = g_deviceBackendStates.find(key);

    if (it != g_deviceBackendStates.end()) {
        it->second.peakMemoryUsed = it->second.memoryUsed;
    }
}

}  // namespace llamacppUtils
}  // namespace sd

#endif  // HAVE_LLAMACPP
