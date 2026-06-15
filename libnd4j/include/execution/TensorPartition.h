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
// Tensor Partition - Tensor splitting and distribution across devices
//

#ifndef SD_TENSOR_PARTITION_H
#define SD_TENSOR_PARTITION_H

#include <system/common.h>
#include <array/NDArray.h>
#include <execution/ModelParallelConfig.h>
#include <execution/DeviceManager.h>

#include <vector>
#include <memory>
#include <mutex>
#include <functional>

namespace sd {
namespace modelparallel {

/**
 * Partition descriptor for a single tensor partition
 */
struct SD_LIB_EXPORT PartitionDescriptor {
    int partitionIndex = 0;       // Index of this partition
    int totalPartitions = 1;      // Total number of partitions
    int deviceIndex = 0;          // Device this partition resides on

    // Source tensor info
    std::vector<LongType> sourceShape;
    std::vector<LongType> sourceStrides;
    DataType dataType;

    // Partition location within source
    int splitDimension = 0;       // Which dimension was split
    LongType splitOffset = 0;     // Start offset in split dimension
    LongType splitSize = 0;       // Size in split dimension

    // Resulting partition shape
    std::vector<LongType> partitionShape;
    std::vector<LongType> partitionStrides;

    // Memory allocation
    size_t bytesRequired = 0;
    bool isContiguous = true;

    /**
     * Check if this partition covers the entire tensor
     */
    bool isFullTensor() const {
        return totalPartitions == 1 && partitionIndex == 0;
    }

    /**
     * Get element count for this partition
     */
    LongType numElements() const {
        LongType count = 1;
        for (auto dim : partitionShape) {
            count *= dim;
        }
        return count;
    }
};

/**
 * Result of a tensor partitioning operation
 */
struct SD_LIB_EXPORT PartitionResult {
    bool success = false;
    std::string errorMessage;

    std::vector<PartitionDescriptor> descriptors;
    std::vector<std::shared_ptr<NDArray>> partitions;

    int numPartitions() const { return static_cast<int>(partitions.size()); }

    std::shared_ptr<NDArray> getPartition(int index) const {
        if (index >= 0 && index < static_cast<int>(partitions.size())) {
            return partitions[index];
        }
        return nullptr;
    }
};

/**
 * Gather result - combining partitions back to a full tensor
 */
struct SD_LIB_EXPORT GatherResult {
    bool success = false;
    std::string errorMessage;
    std::shared_ptr<NDArray> result;
};

/**
 * Tensor Partitioner - Handles splitting and gathering tensors across devices
 *
 * This class provides:
 * - Tensor splitting along any dimension
 * - Balanced partitioning across devices
 * - Gather/reduce operations to combine partitions
 * - Support for various partition patterns (row, column, head, etc.)
 */
class SD_LIB_EXPORT TensorPartitioner {
public:
    /**
     * Get the singleton instance
     */
    static TensorPartitioner& getInstance();

    // =========================================================================
    // Basic Partitioning
    // =========================================================================

    /**
     * Partition a tensor along a dimension
     * @param tensor Source tensor to partition
     * @param dimension Dimension to split along
     * @param numPartitions Number of partitions to create
     * @param devices Target devices for each partition
     */
    PartitionResult partition(
        const NDArray* tensor,
        int dimension,
        int numPartitions,
        const std::vector<int>& devices = {}
    );

    /**
     * Partition a tensor according to specification
     */
    PartitionResult partition(
        const NDArray* tensor,
        const TensorPartitionSpec& spec,
        const std::vector<int>& devices = {}
    );

    /**
     * Partition for tensor parallelism (column-parallel linear layer)
     */
    PartitionResult partitionColumnParallel(
        const NDArray* weight,
        const NDArray* bias,
        int numPartitions
    );

    /**
     * Partition for tensor parallelism (row-parallel linear layer)
     */
    PartitionResult partitionRowParallel(
        const NDArray* weight,
        int numPartitions
    );

    /**
     * Partition attention heads across devices
     */
    PartitionResult partitionAttentionHeads(
        const NDArray* qkv,
        int numHeads,
        int numPartitions
    );

    /**
     * Partition along batch dimension
     */
    PartitionResult partitionBatch(
        const NDArray* tensor,
        int numPartitions
    );

    /**
     * Partition along sequence dimension
     */
    PartitionResult partitionSequence(
        const NDArray* tensor,
        int numPartitions
    );

    // =========================================================================
    // Gathering/Reducing
    // =========================================================================

    /**
     * Gather partitions back into a single tensor
     * @param partitions Vector of partition tensors
     * @param dimension Dimension to gather along
     * @param targetDevice Device for the result
     */
    GatherResult gather(
        const std::vector<NDArray*>& partitions,
        int dimension,
        int targetDevice = 0
    );

    /**
     * Gather partitions using descriptors
     */
    GatherResult gather(
        const PartitionResult& partitionResult,
        int targetDevice = 0
    );

    /**
     * All-gather across devices (each device gets full result)
     */
    std::vector<std::shared_ptr<NDArray>> allGather(
        const std::vector<NDArray*>& partitions,
        int dimension
    );

    /**
     * Reduce partitions (sum, mean, etc.)
     */
    GatherResult reduce(
        const std::vector<NDArray*>& partitions,
        CommunicationPattern pattern,
        int targetDevice = 0
    );

    /**
     * All-reduce across devices
     */
    std::vector<std::shared_ptr<NDArray>> allReduce(
        const std::vector<NDArray*>& tensors
    );

    /**
     * Reduce-scatter operation
     */
    PartitionResult reduceScatter(
        const std::vector<NDArray*>& tensors,
        int dimension
    );

    // =========================================================================
    // Partition Calculation
    // =========================================================================

    /**
     * Calculate partition descriptors without allocating
     */
    std::vector<PartitionDescriptor> calculatePartitions(
        const std::vector<LongType>& shape,
        DataType dtype,
        int dimension,
        int numPartitions
    );

    /**
     * Calculate balanced partition sizes
     */
    std::vector<LongType> calculatePartitionSizes(
        LongType totalSize,
        int numPartitions,
        LongType minSize = 1
    );

    /**
     * Check if a tensor can be evenly partitioned
     */
    bool canPartitionEvenly(
        const NDArray* tensor,
        int dimension,
        int numPartitions
    );

    /**
     * Get optimal partition count for a tensor and device configuration
     */
    int getOptimalPartitionCount(
        const NDArray* tensor,
        int dimension,
        const std::vector<int>& devices
    );

    // =========================================================================
    // In-place Operations
    // =========================================================================

    /**
     * Scatter data to partitions in-place
     */
    bool scatterToPartitions(
        const NDArray* source,
        std::vector<NDArray*>& destinations,
        int dimension
    );

    /**
     * Gather from partitions in-place
     */
    bool gatherFromPartitions(
        const std::vector<NDArray*>& sources,
        NDArray* destination,
        int dimension
    );

    /**
     * All-to-all shuffle between partitions
     */
    bool allToAll(
        std::vector<NDArray*>& tensors,
        int sendDim,
        int recvDim
    );

    // =========================================================================
    // View-based Partitioning
    // =========================================================================

    /**
     * Create views into a tensor for each partition (no copy)
     * Note: Source tensor must remain valid while views are in use
     */
    std::vector<NDArray> createPartitionViews(
        NDArray* tensor,
        int dimension,
        int numPartitions
    );

    /**
     * Check if partition can be done as a view (no copy needed)
     */
    bool canPartitionAsView(
        const NDArray* tensor,
        int dimension
    );

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * Validate partition configuration
     */
    bool validatePartition(
        const NDArray* tensor,
        int dimension,
        int numPartitions,
        std::string& errorMessage
    );

    /**
     * Get memory required for partitioning
     */
    size_t getPartitionMemoryRequired(
        const NDArray* tensor,
        int dimension,
        int numPartitions,
        bool includeGather = false
    );

    /**
     * Convert TensorPartitionDim to actual dimension index
     */
    int resolveDimension(
        TensorPartitionDim dimType,
        const std::vector<LongType>& shape,
        int numHeads = -1
    );

private:
    TensorPartitioner() = default;
    ~TensorPartitioner() = default;

    // Prevent copying
    TensorPartitioner(const TensorPartitioner&) = delete;
    TensorPartitioner& operator=(const TensorPartitioner&) = delete;

    // Internal helpers
    std::shared_ptr<NDArray> createPartitionArray(
        const PartitionDescriptor& desc,
        int deviceIndex
    );

    void copyToPartition(
        const NDArray* source,
        NDArray* partition,
        const PartitionDescriptor& desc
    );

    void copyFromPartition(
        const NDArray* partition,
        NDArray* destination,
        const PartitionDescriptor& desc
    );

    mutable std::mutex _mutex;
};

/**
 * Partitioned tensor wrapper - manages a tensor split across devices
 */
class SD_LIB_EXPORT PartitionedTensor {
public:
    PartitionedTensor() = default;

    /**
     * Create from partition result
     */
    explicit PartitionedTensor(PartitionResult&& result);

    /**
     * Create by partitioning an existing tensor
     */
    PartitionedTensor(
        const NDArray* tensor,
        int dimension,
        int numPartitions,
        const std::vector<int>& devices = {}
    );

    /**
     * Check if valid
     */
    bool isValid() const { return _valid && !_partitions.empty(); }

    /**
     * Get number of partitions
     */
    int numPartitions() const { return static_cast<int>(_partitions.size()); }

    /**
     * Get a specific partition
     */
    NDArray* getPartition(int index);
    const NDArray* getPartition(int index) const;

    /**
     * Get partition on a specific device
     */
    NDArray* getPartitionOnDevice(int deviceIndex);

    /**
     * Get all partitions
     */
    std::vector<NDArray*> getAllPartitions();

    /**
     * Get the descriptor for a partition
     */
    const PartitionDescriptor& getDescriptor(int index) const;

    /**
     * Gather back to a single tensor
     */
    std::shared_ptr<NDArray> gather(int targetDevice = 0);

    /**
     * Get the original shape before partitioning
     */
    std::vector<LongType> getOriginalShape() const { return _originalShape; }

    /**
     * Get the split dimension
     */
    int getSplitDimension() const { return _splitDimension; }

    /**
     * Get devices holding partitions
     */
    std::vector<int> getDevices() const;

    /**
     * Apply a function to all partitions
     */
    template<typename Func>
    void forEach(Func&& func) {
        for (int i = 0; i < numPartitions(); ++i) {
            func(i, getPartition(i));
        }
    }

    /**
     * Apply a function to all partitions in parallel
     */
    template<typename Func>
    void forEachParallel(Func&& func);

private:
    bool _valid = false;
    int _splitDimension = 0;
    std::vector<LongType> _originalShape;
    std::vector<PartitionDescriptor> _descriptors;
    std::vector<std::shared_ptr<NDArray>> _partitions;
};

/**
 * Helper function to get the tensor partitioner
 */
inline TensorPartitioner& getTensorPartitioner() {
    return TensorPartitioner::getInstance();
}

/**
 * Convenience function to partition a tensor
 */
inline PartitionResult partitionTensor(
    const NDArray* tensor,
    int dimension,
    int numPartitions
) {
    return TensorPartitioner::getInstance().partition(tensor, dimension, numPartitions);
}

/**
 * Convenience function to gather partitions
 */
inline GatherResult gatherPartitions(
    const std::vector<NDArray*>& partitions,
    int dimension,
    int targetDevice = 0
) {
    return TensorPartitioner::getInstance().gather(partitions, dimension, targetDevice);
}

}  // namespace modelparallel
}  // namespace sd

#endif  // SD_TENSOR_PARTITION_H
