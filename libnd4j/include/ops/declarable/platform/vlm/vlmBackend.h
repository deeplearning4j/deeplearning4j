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
// VLM Backend Manager - Centralized management of GGML backends for VLM operations
//

#ifndef DEV_TESTS_VLM_BACKEND_H
#define DEV_TESTS_VLM_BACKEND_H

#include <array/NDArray.h>
#include <system/Environment.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <functional>

#if HAVE_VLM
#include <ggml.h>
#include <ggml-backend.h>
#endif

#include <system/BackendNamespace.h>

namespace sd {
SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
namespace vlmUtils {

#if HAVE_VLM

/**
 * Backend type enumeration for VLM operations
 */
enum class VlmBackendType {
    CPU = 0,
    CUDA = 1,
    METAL = 2,
    VULKAN = 3,
    SYCL = 4,
    OPENCL = 5,
    AUTO = 99  // Automatically select best available
};

/**
 * Backend device information
 */
struct VlmDeviceInfo {
    VlmBackendType type;
    int deviceIndex;
    std::string name;
    size_t totalMemory;
    size_t freeMemory;
    bool available;
};

/**
 * Backend configuration for VLM operations
 */
struct VlmBackendConfig {
    VlmBackendType preferredBackend = VlmBackendType::AUTO;
    int deviceIndex = 0;
    size_t contextMemory = 512 * 1024 * 1024;  // 512MB default
    int numThreads = 0;  // 0 = auto
    bool enableTensorOffloading = true;
    bool enableGraphOptimization = true;
    float gpuMemoryFraction = 0.9f;  // Use up to 90% of GPU memory
};

/**
 * VLM Backend Manager - Singleton class for managing GGML backends
 *
 * This class provides centralized management of GGML backends for VLM operations,
 * including automatic backend selection, device enumeration, and resource management.
 */
class VlmBackendManager {
public:
    /**
     * Get the singleton instance
     */
    static VlmBackendManager& getInstance();

    /**
     * Initialize the backend manager with given configuration
     */
    bool initialize(const VlmBackendConfig& config = VlmBackendConfig());

    /**
     * Shutdown and release all backend resources
     */
    void shutdown();

    /**
     * Check if the manager is initialized
     */
    bool isInitialized() const { return _initialized; }

    /**
     * Get the current configuration
     */
    const VlmBackendConfig& getConfig() const { return _config; }

    /**
     * Update configuration (requires re-initialization)
     */
    void setConfig(const VlmBackendConfig& config);

    // =========================================================================
    // Backend Selection and Information
    // =========================================================================

    /**
     * Get list of available devices
     */
    std::vector<VlmDeviceInfo> getAvailableDevices() const;

    /**
     * Get the currently active backend type
     */
    VlmBackendType getActiveBackendType() const { return _activeBackendType; }

    /**
     * Get the active backend handle
     */
    ggml_backend_t getActiveBackend() const { return _activeBackend; }

    /**
     * Get the CPU backend (always available as fallback)
     */
    ggml_backend_t getCpuBackend() const { return _cpuBackend; }

    /**
     * Check if a specific backend type is available
     */
    bool isBackendAvailable(VlmBackendType type) const;

    /**
     * Select a specific backend
     */
    bool selectBackend(VlmBackendType type, int deviceIndex = 0);

    /**
     * Get backend name as string
     */
    static std::string backendTypeToString(VlmBackendType type);

    /**
     * Parse backend type from string
     */
    static VlmBackendType stringToBackendType(const std::string& name);

    // =========================================================================
    // GGML Context and Buffer Management
    // =========================================================================

    /**
     * Create a GGML context with the active backend
     */
    ggml_context* createContext(size_t memSize = 0);

    /**
     * Free a GGML context
     */
    void freeContext(ggml_context* ctx);

    /**
     * Allocate a backend buffer
     */
    ggml_backend_buffer_t allocateBuffer(size_t size);

    /**
     * Free a backend buffer
     */
    void freeBuffer(ggml_backend_buffer_t buffer);

    /**
     * Allocate tensors in context to active backend
     */
    ggml_backend_buffer_t allocContextTensors(ggml_context* ctx);

    // =========================================================================
    // Tensor Operations
    // =========================================================================

    /**
     * Create a GGML tensor from NDArray
     */
    ggml_tensor* createTensorFromNDArray(ggml_context* ctx, const NDArray* array,
                                          const char* name = nullptr);

    /**
     * Copy GGML tensor data back to NDArray
     */
    void copyTensorToNDArray(const ggml_tensor* tensor, NDArray* array);

    /**
     * Set tensor data from NDArray
     */
    void setTensorData(ggml_tensor* tensor, const NDArray* array);

    /**
     * Get tensor data to NDArray
     */
    void getTensorData(const ggml_tensor* tensor, NDArray* array);

    // =========================================================================
    // Graph Execution
    // =========================================================================

    /**
     * Execute a computation graph
     */
    bool executeGraph(ggml_cgraph* graph);

    /**
     * Execute a computation graph with specific backend
     */
    bool executeGraph(ggml_cgraph* graph, ggml_backend_t backend);

    /**
     * Schedule graph for execution (allows backend to optimize)
     */
    ggml_backend_sched_t createScheduler(ggml_cgraph* graph);

    /**
     * Free a scheduler
     */
    void freeScheduler(ggml_backend_sched_t sched);

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Get total memory available on active backend
     */
    size_t getTotalMemory() const;

    /**
     * Get free memory on active backend
     */
    size_t getFreeMemory() const;

    /**
     * Synchronize backend (wait for all operations to complete)
     */
    void synchronize();

    // =========================================================================
    // Utility Functions
    // =========================================================================

    /**
     * Get GGML type from NDArray data type
     */
    static ggml_type dataTypeToGgml(DataType dt);

    /**
     * Get NDArray data type from GGML type
     */
    static DataType ggmlToDataType(ggml_type gt);

    /**
     * Check if data type is supported
     */
    static bool isTypeSupported(DataType dt);

private:
    VlmBackendManager();
    ~VlmBackendManager();

    // Prevent copying
    VlmBackendManager(const VlmBackendManager&) = delete;
    VlmBackendManager& operator=(const VlmBackendManager&) = delete;

    // Initialize specific backend types
    bool initializeCpuBackend();
    bool initializeCudaBackend(int deviceIndex);
    bool initializeMetalBackend();
    bool initializeVulkanBackend(int deviceIndex);

    // Select best available backend automatically
    VlmBackendType selectBestBackend() const;

    // Member variables
    VlmBackendConfig _config;
    bool _initialized;
    VlmBackendType _activeBackendType;

    ggml_backend_t _activeBackend;
    ggml_backend_t _cpuBackend;
    ggml_backend_t _gpuBackend;  // CUDA/Metal/Vulkan

    std::vector<ggml_backend_t> _allBackends;
    std::vector<VlmDeviceInfo> _deviceInfo;

    mutable std::mutex _mutex;
};

/**
 * RAII wrapper for VLM backend context
 *
 * Usage:
 *   VlmBackendContext ctx(memSize);
 *   auto tensor = ctx.createTensor(array, "input");
 *   // ... build computation graph ...
 *   ctx.execute(graph);
 *   ctx.copyToNDArray(outputTensor, outputArray);
 */
class VlmBackendContext {
public:
    explicit VlmBackendContext(size_t memSize = 256 * 1024 * 1024);
    ~VlmBackendContext();

    // Prevent copying
    VlmBackendContext(const VlmBackendContext&) = delete;
    VlmBackendContext& operator=(const VlmBackendContext&) = delete;

    // Allow moving
    VlmBackendContext(VlmBackendContext&& other) noexcept;
    VlmBackendContext& operator=(VlmBackendContext&& other) noexcept;

    /**
     * Get the underlying GGML context
     */
    ggml_context* get() const { return _ctx; }
    operator ggml_context*() const { return _ctx; }

    /**
     * Check if context is valid
     */
    bool isValid() const { return _ctx != nullptr; }

    /**
     * Create a tensor from NDArray
     */
    ggml_tensor* createTensor(const NDArray* array, const char* name = nullptr);

    /**
     * Copy tensor data to NDArray
     */
    void copyToNDArray(const ggml_tensor* tensor, NDArray* array);

    /**
     * Allocate context tensors to backend
     */
    bool allocate();

    /**
     * Execute a computation graph
     */
    bool execute(ggml_cgraph* graph);

    /**
     * Get the backend buffer (after allocation)
     */
    ggml_backend_buffer_t getBuffer() const { return _buffer; }

private:
    ggml_context* _ctx;
    ggml_backend_buffer_t _buffer;
    std::vector<uint8_t> _memBuffer;
};

/**
 * Helper function to get the global backend manager
 */
inline VlmBackendManager& getBackendManager() {
    return VlmBackendManager::getInstance();
}

/**
 * Helper function to execute a VLM operation with automatic backend selection
 */
template<typename OpFunc>
bool executeVlmOp(OpFunc&& op, size_t memSize = 256 * 1024 * 1024) {
    VlmBackendContext ctx(memSize);
    if (!ctx.isValid()) return false;
    return op(ctx);
}

// =========================================================================
// Model Parallelism Support
// =========================================================================

/**
 * Model parallelism configuration for VLM operations
 */
struct VlmParallelConfig {
    // Basic parallelism settings
    int tensorParallelSize = 1;       // Number of devices for tensor parallelism
    int pipelineParallelSize = 1;     // Number of pipeline stages
    std::vector<int> deviceIds;       // Device IDs to use

    // Vision encoder settings
    int visionEncoderDevice = 0;      // Device for vision encoder
    bool splitVisionEncoder = false;  // Whether to split vision encoder across devices

    // Language model settings
    int lmStartDevice = 0;            // Starting device for LM layers
    int lmLayersPerDevice = -1;       // Layers per device (-1 = auto)

    // Cross-attention settings
    bool parallelCrossAttention = true; // Enable parallel cross-attention

    // Transfer settings
    bool enableP2P = true;            // Enable P2P transfers
    bool enableAsyncTransfer = true;  // Enable async data movement
    bool prefetchActivations = true;  // Prefetch activations for next stage

    // Memory settings
    float gpuMemoryFraction = 0.9f;   // Fraction of GPU memory to use
    size_t imageBufferSize = 0;       // Image buffer per device (0 = auto)
    size_t kvCachePerDevice = 0;      // KV cache size per device (0 = auto)
};

/**
 * VLM Model Parallel Manager - Extends VlmBackendManager with parallelism
 *
 * This class provides model parallelism support for vision-language models:
 * - Vision encoder on separate device(s)
 * - Language model distributed across devices
 * - Efficient cross-attention across device boundaries
 * - Image-text parallel processing
 */
class VlmModelParallelManager {
public:
    /**
     * Get the singleton instance
     */
    static VlmModelParallelManager& getInstance();

    /**
     * Initialize with parallelism configuration
     */
    bool initialize(const VlmParallelConfig& config);

    /**
     * Shutdown and release resources
     */
    void shutdown();

    /**
     * Check if initialized
     */
    bool isInitialized() const { return _initialized; }

    /**
     * Get current configuration
     */
    const VlmParallelConfig& getConfig() const { return _config; }

    // =========================================================================
    // Device Management
    // =========================================================================

    /**
     * Get number of devices in use
     */
    int getDeviceCount() const;

    /**
     * Get device for vision encoder
     */
    int getVisionDevice() const { return _config.visionEncoderDevice; }

    /**
     * Get device for a specific LM layer
     */
    int getLMLayerDevice(int layerIndex) const;

    /**
     * Get all device IDs in use
     */
    std::vector<int> getAllDevices() const;

    // =========================================================================
    // Vision Encoder Operations
    // =========================================================================

    /**
     * Process image through vision encoder
     * @param image Input image tensor [B, C, H, W]
     * @param output Output embeddings
     * @return true if successful
     */
    bool encodeImage(const NDArray* image, NDArray* output);

    /**
     * Process multiple images in parallel
     * @param images Input image tensors
     * @param outputs Output embedding tensors (one per image)
     * @return true if all successful
     */
    bool encodeImagesBatched(
        const std::vector<NDArray*>& images,
        std::vector<NDArray*>& outputs
    );

    // =========================================================================
    // Cross-Modal Operations
    // =========================================================================

    /**
     * Prepare image embeddings for cross-attention
     * @param imageEmbeddings Vision encoder output
     * @param targetDevice Device for language model
     * @return Prepared embeddings on target device
     */
    NDArray* prepareImageForCrossAttention(
        const NDArray* imageEmbeddings,
        int targetDevice
    );

    /**
     * Execute cross-attention between image and text
     * @param query Text query tensor
     * @param imageKV Image key-value tensors
     * @param output Cross-attention output
     * @return true if successful
     */
    bool crossAttention(
        const NDArray* query,
        const NDArray* imageKV,
        NDArray* output
    );

    // =========================================================================
    // Tensor Parallelism
    // =========================================================================

    /**
     * Partition a weight for tensor parallelism
     */
    NDArray* partitionWeight(
        const NDArray* weight,
        int dim,
        int deviceId,
        int partitionIndex
    );

    /**
     * Gather outputs from tensor parallel execution
     */
    NDArray* gatherTensorParallel(
        const std::vector<NDArray*>& partitions,
        int dim,
        int targetDevice
    );

    /**
     * All-reduce across tensor parallel devices
     */
    void allReduce(NDArray* tensor);

    // =========================================================================
    // Pipeline Parallelism
    // =========================================================================

    /**
     * Transfer activations between stages
     */
    void transferActivations(
        NDArray* activations,
        int srcStage,
        int dstStage,
        bool async = true
    );

    /**
     * Synchronize pipeline stage
     */
    void syncStage(int stageId);

    /**
     * Synchronize all stages
     */
    void syncAll();

    // =========================================================================
    // Memory Management
    // =========================================================================

    /**
     * Get memory usage on a device
     */
    size_t getDeviceMemoryUsage(int deviceId) const;

    /**
     * Get total memory usage across all devices
     */
    size_t getTotalMemoryUsage() const;

    /**
     * Allocate image buffer on device
     */
    void* allocateImageBuffer(int deviceId, size_t size);

    /**
     * Free image buffer
     */
    void freeImageBuffer(void* buffer);

private:
    VlmModelParallelManager() = default;
    ~VlmModelParallelManager();

    VlmModelParallelManager(const VlmModelParallelManager&) = delete;
    VlmModelParallelManager& operator=(const VlmModelParallelManager&) = delete;

    bool _initialized = false;
    VlmParallelConfig _config;
    std::vector<void*> _imageBuffers;
    std::mutex _mutex;
};

/**
 * Helper to get the VLM model parallel manager
 */
inline VlmModelParallelManager& getVlmParallelManager() {
    return VlmModelParallelManager::getInstance();
}

/**
 * Check if VLM model parallelism is supported
 */
bool supportsVlmModelParallel();

/**
 * Get the number of available devices for VLM model parallelism
 */
int getVlmParallelDeviceCount();

#endif  // HAVE_VLM

}  // namespace vlmUtils
SD_BACKEND_ROOT_INLINE_NAMESPACE_END
}  // namespace sd

#endif  // DEV_TESTS_VLM_BACKEND_H
