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
// Model Parallelism - Main header for model parallelism infrastructure
//
// This header provides comprehensive infrastructure for distributing models
// across multiple GPUs and CPUs, supporting:
//
// - Tensor Parallelism: Split tensors across devices for parallel computation
// - Pipeline Parallelism: Distribute layers across devices in a pipeline
// - Data Parallelism: Same model on multiple devices with different data
// - Hybrid Parallelism: Combinations of the above strategies
//
// Usage:
//   #include <execution/ModelParallel.h>
//
//   // Initialize device manager
//   auto& dm = sd::modelparallel::getDeviceManager();
//   dm.initialize();
//
//   // Configure model parallelism
//   sd::modelparallel::ModelParallelConfig config;
//   config.strategy = sd::modelparallel::ParallelStrategy::TENSOR_PARALLEL;
//   config.tensorParallelSize = 4;
//
//   // Partition tensors for parallel execution
//   auto& partitioner = sd::modelparallel::getTensorPartitioner();
//   auto result = partitioner.partition(weightTensor, 1, 4);  // Split dim 1 across 4 devices
//
//   // Execute with pipeline parallelism
//   sd::modelparallel::PipelineScheduler scheduler(config);
//   scheduler.initialize(stages);
//   auto outputs = scheduler.forward(inputs);
//

#ifndef SD_MODEL_PARALLEL_H
#define SD_MODEL_PARALLEL_H

// Core model parallelism components
#include <execution/ModelParallelConfig.h>
#include <execution/DeviceManager.h>
#include <execution/TensorPartition.h>
#include <execution/DataTransferManager.h>
#include <execution/PipelineScheduler.h>
#include <execution/Engine.h>

namespace sd {
namespace modelparallel {

// =========================================================================
// Convenience Functions
// =========================================================================

/**
 * Initialize the model parallelism infrastructure
 * Discovers all devices and sets up P2P connections
 * @return true if initialization succeeded
 */
inline bool initialize() {
    auto& dm = DeviceManager::getInstance();
    if (!dm.initialize()) {
        return false;
    }

    auto& tm = DataTransferManager::getInstance();
    if (!tm.initialize()) {
        return false;
    }

    // Enable P2P for all compatible devices
    dm.enableAllP2P();

    return true;
}

/**
 * Shutdown the model parallelism infrastructure
 */
inline void shutdown() {
    DataTransferManager::getInstance().shutdown();
    DeviceManager::getInstance().shutdown();
}

/**
 * Check if model parallelism is available
 * @return true if at least 2 compute devices are available
 */
inline bool isAvailable() {
    auto& dm = DeviceManager::getInstance();
    if (!dm.isInitialized()) {
        dm.initialize();
    }
    return dm.getTotalDeviceCount() > 1;
}

/**
 * Get the maximum tensor parallel size
 */
inline int maxTensorParallelSize() {
    auto& dm = DeviceManager::getInstance();
    return dm.getCudaGpuCount();
}

/**
 * Get the maximum pipeline stages
 */
inline int maxPipelineStages() {
    auto& dm = DeviceManager::getInstance();
    return dm.getCudaGpuCount() + 1;  // GPUs + CPU
}

/**
 * Create a simple tensor parallel configuration
 * @param numGpus Number of GPUs to use
 * @return Configured ModelParallelConfig
 */
inline ModelParallelConfig createTensorParallelConfig(int numGpus) {
    return ModelParallelConfig::tensorParallel(numGpus);
}

/**
 * Create a simple pipeline parallel configuration
 * @param numStages Number of pipeline stages
 * @param microBatches Number of micro-batches for pipelining
 * @return Configured ModelParallelConfig
 */
inline ModelParallelConfig createPipelineParallelConfig(int numStages, int microBatches = 4) {
    return ModelParallelConfig::pipelineParallel(numStages, microBatches);
}

/**
 * Create a hybrid tensor+pipeline configuration
 * @param tensorParallel Tensor parallel size
 * @param pipelineStages Pipeline parallel size
 * @return Configured ModelParallelConfig
 */
inline ModelParallelConfig createHybridConfig(int tensorParallel, int pipelineStages) {
    return ModelParallelConfig::hybrid(tensorParallel, pipelineStages);
}

/**
 * Create a CPU+GPU heterogeneous configuration
 * @param numGpus Number of GPUs
 * @param useCpu Whether to include CPU
 * @return Configured ModelParallelConfig
 */
inline ModelParallelConfig createHeterogeneousConfig(int numGpus, bool useCpu = true) {
    return ModelParallelConfig::heterogeneous(numGpus, useCpu);
}

/**
 * Suggest optimal parallelism configuration for a model
 * @param numLayers Number of layers in the model
 * @param parameterCount Total number of parameters
 * @param maxBatchSize Maximum batch size
 * @return Suggested ModelParallelConfig
 */
inline ModelParallelConfig suggestConfig(
    int numLayers,
    size_t parameterCount,
    int maxBatchSize
) {
    auto& dm = DeviceManager::getInstance();
    int numGpus = dm.getCudaGpuCount();

    if (numGpus <= 1) {
        return ModelParallelConfig();  // No parallelism
    }

    // Estimate memory per layer
    size_t bytesPerParam = 4;  // FP32
    size_t paramsPerLayer = parameterCount / numLayers;
    size_t memPerLayer = paramsPerLayer * bytesPerParam;

    // Get GPU memory
    size_t gpuMemory = dm.getFreeMemory(DeviceType::CUDA_GPU) / numGpus;

    // If model fits on one GPU, use tensor parallelism for speed
    if (parameterCount * bytesPerParam < gpuMemory * 0.8) {
        if (numGpus >= 2) {
            return ModelParallelConfig::tensorParallel(std::min(numGpus, 8));
        }
    }

    // If model doesn't fit, use pipeline parallelism
    int layersPerGpu = static_cast<int>(gpuMemory * 0.7 / memPerLayer);
    int pipelineStages = (numLayers + layersPerGpu - 1) / layersPerGpu;
    pipelineStages = std::min(pipelineStages, numGpus);

    if (pipelineStages > 1) {
        return ModelParallelConfig::pipelineParallel(pipelineStages, numGpus * 2);
    }

    // Fallback to data parallelism
    ModelParallelConfig config;
    config.strategy = ParallelStrategy::DATA_PARALLEL;
    config.dataParallelSize = numGpus;
    return config;
}

/**
 * Get parallel capabilities for the current system
 */
inline ParallelCapabilities getCapabilities() {
    return DeviceManager::getInstance().getParallelCapabilities();
}

/**
 * Print system information for model parallelism
 */
inline void printSystemInfo() {
    auto& dm = DeviceManager::getInstance();
    dm.printDeviceInfo();

    auto caps = dm.getParallelCapabilities();
    std::cout << "\nParallel Capabilities:" << std::endl;
    std::cout << "  Tensor Parallel: " << (caps.supportsTensorParallel ? "Yes" : "No")
              << " (max " << caps.maxTensorParallelSize << ")" << std::endl;
    std::cout << "  Pipeline Parallel: " << (caps.supportsPipelineParallel ? "Yes" : "No")
              << " (max " << caps.maxPipelineStages << " stages)" << std::endl;
    std::cout << "  Data Parallel: " << (caps.supportsDataParallel ? "Yes" : "No")
              << " (max " << caps.maxDataParallelSize << ")" << std::endl;
    std::cout << "  P2P: " << (caps.supportsP2P ? "Yes" : "No") << std::endl;
    std::cout << "  Async Transfer: " << (caps.supportsAsyncTransfer ? "Yes" : "No") << std::endl;
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_MODEL_PARALLEL_H
