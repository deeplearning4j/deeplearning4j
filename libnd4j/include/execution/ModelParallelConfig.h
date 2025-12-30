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
// Model Parallelism Configuration - Defines strategies for distributing models across devices
//

#ifndef SD_MODEL_PARALLEL_CONFIG_H
#define SD_MODEL_PARALLEL_CONFIG_H

#include <system/common.h>
#include <execution/Engine.h>

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace sd {
namespace modelparallel {

/**
 * Parallelism strategy for model distribution
 */
enum class ParallelStrategy {
    NONE = 0,              // No parallelism, single device execution
    DATA_PARALLEL,         // Same model on all devices, different data batches
    TENSOR_PARALLEL,       // Split tensors across devices (column/row splitting)
    PIPELINE_PARALLEL,     // Different layers on different devices
    HYBRID_PARALLEL,       // Combination of tensor and pipeline parallelism
    EXPERT_PARALLEL,       // For MoE models - different experts on different devices
    SEQUENCE_PARALLEL      // Split sequence dimension across devices
};

/**
 * Tensor partition dimension for tensor parallelism
 */
enum class TensorPartitionDim {
    ROW = 0,               // Split along row dimension (output features)
    COLUMN = 1,            // Split along column dimension (input features)
    HEAD = 2,              // Split attention heads (for multi-head attention)
    EXPERT = 3,            // Split expert networks (for MoE models)
    SEQUENCE = 4,          // Split sequence length
    BATCH = 5,             // Split batch dimension
    CUSTOM = 99            // Custom dimension specified by user
};

/**
 * Communication pattern between devices
 */
enum class CommunicationPattern {
    ALL_REDUCE = 0,        // Sum gradients/outputs across all devices
    ALL_GATHER,            // Gather outputs from all devices
    REDUCE_SCATTER,        // Reduce and scatter
    BROADCAST,             // Send from one to all
    POINT_TO_POINT,        // Direct device-to-device transfer
    RING,                  // Ring-based communication
    TREE                   // Tree-based reduction
};

/**
 * Device type for heterogeneous execution
 */
enum class DeviceType {
    CPU = 0,
    CUDA_GPU,
    METAL_GPU,
    VULKAN_GPU,
    OPENCL_GPU,
    TPU,
    ACCELERATOR,          // Generic accelerator (ARM Compute, MKL-DNN)
    ANY                   // Let system choose
};

/**
 * Device specification for a single compute device
 */
struct SD_LIB_EXPORT DeviceSpec {
    DeviceType type = DeviceType::ANY;
    int deviceIndex = 0;
    size_t memoryLimit = 0;           // 0 = no limit
    float computeFraction = 1.0f;     // Fraction of compute to use (0.0-1.0)
    int priority = 0;                 // Higher = prefer this device
    std::string name;                 // Optional device name

    bool operator==(const DeviceSpec& other) const {
        return type == other.type && deviceIndex == other.deviceIndex;
    }
};

/**
 * Layer assignment for pipeline parallelism
 */
struct SD_LIB_EXPORT LayerAssignment {
    int startLayer = 0;               // First layer index (inclusive)
    int endLayer = -1;                // Last layer index (inclusive), -1 = to end
    DeviceSpec device;                // Device to run these layers on
    size_t activationMemory = 0;      // Estimated activation memory for these layers
};

/**
 * Tensor partition specification
 */
struct SD_LIB_EXPORT TensorPartitionSpec {
    std::string tensorName;           // Name or pattern to match
    TensorPartitionDim dimension = TensorPartitionDim::COLUMN;
    int numPartitions = 0;            // 0 = auto (based on device count)
    std::vector<int> partitionSizes;  // Optional: custom partition sizes
    bool requiresReduce = true;       // Whether outputs need reduction
    CommunicationPattern reducePattern = CommunicationPattern::ALL_REDUCE;
};

/**
 * Model parallelism configuration
 *
 * This structure defines how a model should be distributed across multiple devices.
 * It supports various parallelism strategies including data parallel, tensor parallel,
 * pipeline parallel, and hybrid approaches.
 */
struct SD_LIB_EXPORT ModelParallelConfig {
    // =========================================================================
    // Basic Configuration
    // =========================================================================

    ParallelStrategy strategy = ParallelStrategy::NONE;
    std::vector<DeviceSpec> devices;  // Devices to use for parallelism
    int worldSize = 1;                // Total number of parallel workers
    int localRank = 0;                // This worker's rank

    // =========================================================================
    // Tensor Parallelism Configuration
    // =========================================================================

    int tensorParallelSize = 1;       // Number of devices for tensor parallelism
    std::vector<TensorPartitionSpec> tensorPartitions;
    bool enableColumnParallel = true; // Column-parallel for input projection
    bool enableRowParallel = true;    // Row-parallel for output projection

    // =========================================================================
    // Pipeline Parallelism Configuration
    // =========================================================================

    int pipelineStages = 1;           // Number of pipeline stages
    int microBatchSize = 1;           // Micro-batch size for pipelining
    int numMicroBatches = 1;          // Number of micro-batches per batch
    std::vector<LayerAssignment> layerAssignments;
    bool enableInterleaved = false;   // Use interleaved pipeline schedule

    // =========================================================================
    // Data Parallelism Configuration
    // =========================================================================

    int dataParallelSize = 1;         // Number of data parallel replicas
    bool enableGradientAccumulation = true;
    int gradientAccumulationSteps = 1;
    bool enableZeroOptimizer = false; // ZeRO-style optimizer state partitioning
    int zeroStage = 0;                // ZeRO stage (0=off, 1=optimizer, 2=+gradients, 3=+params)

    // =========================================================================
    // Communication Configuration
    // =========================================================================

    CommunicationPattern defaultComm = CommunicationPattern::ALL_REDUCE;
    bool enableP2P = true;            // Enable peer-to-peer transfers
    bool enableNVLink = true;         // Prefer NVLink when available
    bool asyncCommunication = true;   // Overlap communication with compute
    int commBufferSize = 64 * 1024 * 1024;  // Communication buffer size (64MB)

    // =========================================================================
    // Memory Configuration
    // =========================================================================

    bool enableActivationCheckpointing = false;
    std::vector<int> checkpointLayers;     // Layers to checkpoint
    bool enableCPUOffload = false;         // Offload to CPU when GPU memory is low
    bool enableDiskOffload = false;        // Offload to disk (extreme memory saving)
    float memoryFraction = 0.9f;           // Fraction of device memory to use
    size_t reservedMemory = 0;             // Memory to reserve for other uses

    // =========================================================================
    // Performance Tuning
    // =========================================================================

    bool enableOverlapCompute = true;      // Overlap compute across micro-batches
    bool enablePrefetch = true;            // Prefetch data/weights
    int prefetchDepth = 2;                 // Number of micro-batches to prefetch
    bool enableFusedKernels = true;        // Use fused kernels when available
    bool enableFlashAttention = true;      // Use flash attention when available

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /**
     * Check if tensor parallelism is enabled
     */
    bool isTensorParallel() const {
        return strategy == ParallelStrategy::TENSOR_PARALLEL ||
               strategy == ParallelStrategy::HYBRID_PARALLEL ||
               tensorParallelSize > 1;
    }

    /**
     * Check if pipeline parallelism is enabled
     */
    bool isPipelineParallel() const {
        return strategy == ParallelStrategy::PIPELINE_PARALLEL ||
               strategy == ParallelStrategy::HYBRID_PARALLEL ||
               pipelineStages > 1;
    }

    /**
     * Check if data parallelism is enabled
     */
    bool isDataParallel() const {
        return strategy == ParallelStrategy::DATA_PARALLEL ||
               dataParallelSize > 1;
    }

    /**
     * Check if any parallelism is enabled
     */
    bool isParallel() const {
        return strategy != ParallelStrategy::NONE ||
               worldSize > 1 ||
               tensorParallelSize > 1 ||
               pipelineStages > 1 ||
               dataParallelSize > 1;
    }

    /**
     * Get the total number of devices required
     */
    int totalDevices() const {
        return tensorParallelSize * pipelineStages * dataParallelSize;
    }

    /**
     * Get tensor parallel rank for this worker
     */
    int getTensorParallelRank() const {
        return localRank % tensorParallelSize;
    }

    /**
     * Get pipeline stage for this worker
     */
    int getPipelineStage() const {
        return (localRank / tensorParallelSize) % pipelineStages;
    }

    /**
     * Get data parallel rank for this worker
     */
    int getDataParallelRank() const {
        return localRank / (tensorParallelSize * pipelineStages);
    }

    /**
     * Create a simple tensor parallel config
     */
    static ModelParallelConfig tensorParallel(int numGpus) {
        ModelParallelConfig config;
        config.strategy = ParallelStrategy::TENSOR_PARALLEL;
        config.tensorParallelSize = numGpus;
        config.worldSize = numGpus;
        for (int i = 0; i < numGpus; ++i) {
            DeviceSpec spec;
            spec.type = DeviceType::CUDA_GPU;
            spec.deviceIndex = i;
            config.devices.push_back(spec);
        }
        return config;
    }

    /**
     * Create a simple pipeline parallel config
     */
    static ModelParallelConfig pipelineParallel(int numStages, int microBatches = 4) {
        ModelParallelConfig config;
        config.strategy = ParallelStrategy::PIPELINE_PARALLEL;
        config.pipelineStages = numStages;
        config.numMicroBatches = microBatches;
        config.worldSize = numStages;
        for (int i = 0; i < numStages; ++i) {
            DeviceSpec spec;
            spec.type = DeviceType::CUDA_GPU;
            spec.deviceIndex = i;
            config.devices.push_back(spec);
        }
        return config;
    }

    /**
     * Create a hybrid tensor + pipeline parallel config
     */
    static ModelParallelConfig hybrid(int tensorParallel, int pipelineStages) {
        ModelParallelConfig config;
        config.strategy = ParallelStrategy::HYBRID_PARALLEL;
        config.tensorParallelSize = tensorParallel;
        config.pipelineStages = pipelineStages;
        config.worldSize = tensorParallel * pipelineStages;
        for (int i = 0; i < config.worldSize; ++i) {
            DeviceSpec spec;
            spec.type = DeviceType::CUDA_GPU;
            spec.deviceIndex = i;
            config.devices.push_back(spec);
        }
        return config;
    }

    /**
     * Create a config for CPU+GPU heterogeneous execution
     */
    static ModelParallelConfig heterogeneous(int numGpus, bool useCpu = true) {
        ModelParallelConfig config;
        config.strategy = ParallelStrategy::PIPELINE_PARALLEL;
        config.enableCPUOffload = useCpu;

        if (useCpu) {
            DeviceSpec cpuSpec;
            cpuSpec.type = DeviceType::CPU;
            cpuSpec.deviceIndex = 0;
            config.devices.push_back(cpuSpec);
        }

        for (int i = 0; i < numGpus; ++i) {
            DeviceSpec gpuSpec;
            gpuSpec.type = DeviceType::CUDA_GPU;
            gpuSpec.deviceIndex = i;
            config.devices.push_back(gpuSpec);
        }

        config.worldSize = static_cast<int>(config.devices.size());
        config.pipelineStages = config.worldSize;
        return config;
    }
};

/**
 * Parallelism capabilities of a backend
 */
struct SD_LIB_EXPORT ParallelCapabilities {
    bool supportsTensorParallel = false;
    bool supportsPipelineParallel = false;
    bool supportsDataParallel = false;
    bool supportsP2P = false;
    bool supportsAsyncTransfer = false;
    bool supportsUnifiedMemory = false;
    bool supportsMixedPrecision = false;
    bool supportsQuantization = false;

    int maxTensorParallelSize = 1;
    int maxPipelineStages = 1;
    int maxDataParallelSize = 1;

    std::vector<CommunicationPattern> supportedCommPatterns;
    std::vector<DeviceType> supportedDeviceTypes;
};

/**
 * Convert ParallelStrategy to string
 */
inline std::string parallelStrategyToString(ParallelStrategy strategy) {
    switch (strategy) {
        case ParallelStrategy::NONE: return "NONE";
        case ParallelStrategy::DATA_PARALLEL: return "DATA_PARALLEL";
        case ParallelStrategy::TENSOR_PARALLEL: return "TENSOR_PARALLEL";
        case ParallelStrategy::PIPELINE_PARALLEL: return "PIPELINE_PARALLEL";
        case ParallelStrategy::HYBRID_PARALLEL: return "HYBRID_PARALLEL";
        case ParallelStrategy::EXPERT_PARALLEL: return "EXPERT_PARALLEL";
        case ParallelStrategy::SEQUENCE_PARALLEL: return "SEQUENCE_PARALLEL";
        default: return "UNKNOWN";
    }
}

/**
 * Convert DeviceType to string
 */
inline std::string deviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "CPU";
        case DeviceType::CUDA_GPU: return "CUDA_GPU";
        case DeviceType::METAL_GPU: return "METAL_GPU";
        case DeviceType::VULKAN_GPU: return "VULKAN_GPU";
        case DeviceType::OPENCL_GPU: return "OPENCL_GPU";
        case DeviceType::TPU: return "TPU";
        case DeviceType::ACCELERATOR: return "ACCELERATOR";
        case DeviceType::ANY: return "ANY";
        default: return "UNKNOWN";
    }
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_MODEL_PARALLEL_CONFIG_H
