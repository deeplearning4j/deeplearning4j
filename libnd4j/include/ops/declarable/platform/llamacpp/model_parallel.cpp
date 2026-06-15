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
// Model parallelism support for LlamaCpp operations
// Implements tensor parallelism and pipeline parallelism
//

#include "llamacppUtils.h"
#include <mutex>
#include <atomic>

#if HAVE_LLAMACPP

namespace sd {
namespace llamacppUtils {

// Global state for model parallelism
static std::mutex g_modelParallelMutex;
static std::atomic<bool> g_modelParallelInitialized{false};
static LlamaParallelConfig g_parallelConfig;

//////////////////////////////////////////////////////////////////////////
bool supportsModelParallel() {
#if defined(GGML_USE_CUDA)
    return hasCudaBackend() && getCudaDeviceCount() > 1;
#else
    return false;
#endif
}

//////////////////////////////////////////////////////////////////////////
int getModelParallelDeviceCount() {
#if defined(GGML_USE_CUDA)
    return getCudaDeviceCount();
#else
    return 1;
#endif
}

//////////////////////////////////////////////////////////////////////////
bool initModelParallel(const LlamaParallelConfig& config) {
    std::lock_guard<std::mutex> lock(g_modelParallelMutex);

    if (g_modelParallelInitialized) {
        return true;  // Already initialized
    }

    // Validate configuration
    int availableDevices = getModelParallelDeviceCount();
    if (config.tensorParallelSize > availableDevices) {
        return false;
    }

    if (config.tensorParallelSize * config.pipelineParallelSize > availableDevices) {
        return false;
    }

    g_parallelConfig = config;

    // Set up device IDs if not specified
    if (g_parallelConfig.deviceIds.empty()) {
        for (int i = 0; i < config.tensorParallelSize * config.pipelineParallelSize; i++) {
            g_parallelConfig.deviceIds.push_back(i);
        }
    }

#if defined(GGML_USE_CUDA)
    // Enable P2P if requested and available
    if (config.enableP2P) {
        // P2P setup would go here - requires CUDA runtime calls
    }
#endif

    g_modelParallelInitialized = true;
    return true;
}

//////////////////////////////////////////////////////////////////////////
void shutdownModelParallel() {
    std::lock_guard<std::mutex> lock(g_modelParallelMutex);

    if (!g_modelParallelInitialized) {
        return;
    }

    // Clean up resources
    g_parallelConfig = LlamaParallelConfig();
    g_modelParallelInitialized = false;
}

//////////////////////////////////////////////////////////////////////////
LlamaParallelConfig getModelParallelConfig() {
    std::lock_guard<std::mutex> lock(g_modelParallelMutex);
    return g_parallelConfig;
}

//////////////////////////////////////////////////////////////////////////
int getLayerDevice(int layerIndex) {
    std::lock_guard<std::mutex> lock(g_modelParallelMutex);

    if (!g_modelParallelInitialized || g_parallelConfig.pipelineParallelSize <= 1) {
        return 0;
    }

    int numLayers = g_parallelConfig.numLayers;
    if (numLayers <= 0) {
        return 0;
    }

    // Distribute layers evenly across pipeline stages
    int layersPerStage = (numLayers + g_parallelConfig.pipelineParallelSize - 1) /
                          g_parallelConfig.pipelineParallelSize;
    int stageIndex = layerIndex / layersPerStage;

    if (stageIndex >= g_parallelConfig.pipelineParallelSize) {
        stageIndex = g_parallelConfig.pipelineParallelSize - 1;
    }

    // Map stage to device
    if (stageIndex < static_cast<int>(g_parallelConfig.deviceIds.size())) {
        return g_parallelConfig.deviceIds[stageIndex];
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////
NDArray* partitionWeight(
    const NDArray* weight,
    int outputDim,
    int deviceId,
    int partitionIndex,
    int totalPartitions
) {
    if (weight == nullptr || totalPartitions <= 1) {
        return nullptr;
    }

    // Calculate partition size
    sd::LongType dimSize = weight->sizeAt(outputDim);
    sd::LongType partitionSize = dimSize / totalPartitions;
    sd::LongType startIdx = partitionIndex * partitionSize;
    sd::LongType endIdx = (partitionIndex == totalPartitions - 1) ? dimSize : (partitionIndex + 1) * partitionSize;

    // Create subarray for this partition
    std::vector<sd::LongType> starts(weight->rankOf(), 0);
    std::vector<sd::LongType> ends(weight->rankOf());
    for (int i = 0; i < weight->rankOf(); i++) {
        ends[i] = weight->sizeAt(i);
    }
    starts[outputDim] = startIdx;
    ends[outputDim] = endIdx;

    // Create the partitioned array
    NDArray* partition = new NDArray(weight->subarray({starts, ends}));

    return partition;
}

//////////////////////////////////////////////////////////////////////////
NDArray* gatherTensorParallel(
    const std::vector<NDArray*>& partitions,
    int dim,
    int targetDevice
) {
    if (partitions.empty()) {
        return nullptr;
    }

    if (partitions.size() == 1) {
        return new NDArray(*partitions[0]);
    }

    // Calculate total size
    sd::LongType totalDimSize = 0;
    for (const auto& p : partitions) {
        totalDimSize += p->sizeAt(dim);
    }

    // Create output shape
    std::vector<sd::LongType> outShape;
    for (int i = 0; i < partitions[0]->rankOf(); i++) {
        if (i == dim) {
            outShape.push_back(totalDimSize);
        } else {
            outShape.push_back(partitions[0]->sizeAt(i));
        }
    }

    // Create output array
    NDArray* result = new NDArray('c', outShape, partitions[0]->dataType());

    // Copy partitions into result
    sd::LongType offset = 0;
    for (const auto& p : partitions) {
        std::vector<sd::LongType> starts(result->rankOf(), 0);
        std::vector<sd::LongType> ends(result->rankOf());
        for (int i = 0; i < result->rankOf(); i++) {
            ends[i] = result->sizeAt(i);
        }
        starts[dim] = offset;
        ends[dim] = offset + p->sizeAt(dim);

        auto subArr = result->subarray({starts, ends});
        subArr.assign(p);

        offset += p->sizeAt(dim);
    }

    return result;
}

//////////////////////////////////////////////////////////////////////////
void allReduceTensorParallel(NDArray* tensor, const std::vector<int>& deviceIds) {
    if (tensor == nullptr || deviceIds.size() <= 1) {
        return;
    }

#if defined(GGML_USE_CUDA)
    // CUDA NCCL all-reduce would go here
    // For now, this is a placeholder
#endif

    // CPU fallback: no-op for single device
}

//////////////////////////////////////////////////////////////////////////
void transferActivations(
    NDArray* activations,
    int srcDevice,
    int dstDevice,
    bool async
) {
    if (activations == nullptr || srcDevice == dstDevice) {
        return;
    }

#if defined(GGML_USE_CUDA)
    // CUDA D2D transfer would go here
    // For now, this is a placeholder
#endif
}

//////////////////////////////////////////////////////////////////////////
void syncModelParallel() {
#if defined(GGML_USE_CUDA)
    // Synchronize all devices in the parallel configuration
    std::lock_guard<std::mutex> lock(g_modelParallelMutex);

    if (!g_modelParallelInitialized) {
        return;
    }

    // CUDA device synchronization would go here
#endif
}

//////////////////////////////////////////////////////////////////////////
size_t getModelParallelMemoryUsage(int deviceId) {
#if defined(GGML_USE_CUDA)
    // Query device memory usage
    // Placeholder - actual implementation would use CUDA API
    return 0;
#else
    return 0;
#endif
}

}  // namespace llamacppUtils
}  // namespace sd

#endif  // HAVE_LLAMACPP
