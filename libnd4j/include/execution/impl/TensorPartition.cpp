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

#include <execution/TensorPartition.h>
#include <helpers/ShapeUtils.h>
#include <array/NDArrayFactory.h>

#include <algorithm>
#include <numeric>

namespace sd {
namespace modelparallel {

TensorPartitioner& TensorPartitioner::getInstance() {
    static TensorPartitioner instance;
    return instance;
}

PartitionResult TensorPartitioner::partition(
    const NDArray* tensor,
    int dimension,
    int numPartitions,
    const std::vector<int>& devices
) {
    PartitionResult result;

    // Validate inputs
    std::string errorMsg;
    if (!validatePartition(tensor, dimension, numPartitions, errorMsg)) {
        result.success = false;
        result.errorMessage = errorMsg;
        return result;
    }

    auto tensorNonConst = const_cast<NDArray*>(tensor);
    auto shapePtr = tensorNonConst->getShapeAsVector();
    const auto shape = *shapePtr;
    delete shapePtr;
    LongType dimSize = shape[dimension];

    // Calculate partition sizes
    auto partitionSizes = calculatePartitionSizes(dimSize, numPartitions);

    // Generate descriptors
    LongType offset = 0;
    for (int i = 0; i < numPartitions; ++i) {
        PartitionDescriptor desc;
        desc.partitionIndex = i;
        desc.totalPartitions = numPartitions;
        desc.deviceIndex = devices.empty() ? i : devices[i % devices.size()];

        desc.sourceShape = shape;
        desc.dataType = tensorNonConst->dataType();
        desc.splitDimension = dimension;
        desc.splitOffset = offset;
        desc.splitSize = partitionSizes[i];

        // Calculate partition shape
        desc.partitionShape = shape;
        desc.partitionShape[dimension] = partitionSizes[i];

        // Calculate bytes required
        LongType elements = 1;
        for (auto dim : desc.partitionShape) {
            elements *= dim;
        }
        desc.bytesRequired = elements * DataTypeUtils::sizeOf(tensorNonConst->dataType());

        result.descriptors.push_back(desc);
        offset += partitionSizes[i];
    }

    // Allocate and copy data to partitions
    for (int i = 0; i < numPartitions; ++i) {
        auto& desc = result.descriptors[i];

        // Create partition array
        auto partition = createPartitionArray(desc, desc.deviceIndex);
        if (!partition) {
            result.success = false;
            result.errorMessage = "Failed to allocate partition " + std::to_string(i);
            return result;
        }

        // Copy data from source to partition
        copyToPartition(tensor, partition.get(), desc);

        result.partitions.push_back(partition);
    }

    result.success = true;
    return result;
}

PartitionResult TensorPartitioner::partition(
    const NDArray* tensor,
    const TensorPartitionSpec& spec,
    const std::vector<int>& devices
) {
    auto tensorNonConst = const_cast<NDArray*>(tensor);
    auto shapePtr = tensorNonConst->getShapeAsVector();
    auto shapeVec = *shapePtr;
    delete shapePtr;
    int dimension = resolveDimension(spec.dimension, shapeVec);
    int numPartitions = spec.numPartitions > 0 ? spec.numPartitions :
                        (devices.empty() ? 2 : static_cast<int>(devices.size()));

    return partition(tensor, dimension, numPartitions, devices);
}

PartitionResult TensorPartitioner::partitionColumnParallel(
    const NDArray* weight,
    const NDArray* bias,
    int numPartitions
) {
    PartitionResult result;
    auto weightNonConst = const_cast<NDArray*>(weight);

    if (weightNonConst->rankOf() < 2) {
        result.success = false;
        result.errorMessage = "Weight must have at least 2 dimensions for column-parallel";
        return result;
    }

    // Column-parallel: split along the output dimension (last dim for linear layers)
    int outDim = weightNonConst->rankOf() - 1;
    result = partition(weight, outDim, numPartitions);

    if (!result.success) {
        return result;
    }

    // Also partition bias if provided
    auto biasNonConst = const_cast<NDArray*>(bias);
    if (bias != nullptr && biasNonConst->lengthOf() > 0) {
        auto biasResult = partition(bias, 0, numPartitions);
        if (biasResult.success) {
            // Merge bias partitions into result
            for (size_t i = 0; i < biasResult.partitions.size(); ++i) {
                result.partitions.push_back(biasResult.partitions[i]);
                result.descriptors.push_back(biasResult.descriptors[i]);
            }
        }
    }

    return result;
}

PartitionResult TensorPartitioner::partitionRowParallel(
    const NDArray* weight,
    int numPartitions
) {
    auto weightNonConst = const_cast<NDArray*>(weight);
    if (weightNonConst->rankOf() < 2) {
        PartitionResult result;
        result.success = false;
        result.errorMessage = "Weight must have at least 2 dimensions for row-parallel";
        return result;
    }

    // Row-parallel: split along the input dimension (second-to-last dim)
    int inDim = weightNonConst->rankOf() - 2;
    return partition(weight, inDim, numPartitions);
}

PartitionResult TensorPartitioner::partitionAttentionHeads(
    const NDArray* qkv,
    int numHeads,
    int numPartitions
) {
    PartitionResult result;
    auto qkvNonConst = const_cast<NDArray*>(qkv);

    // Attention heads are typically in dimension 1 or 2
    // Assuming shape: [batch, seq, num_heads, head_dim] or [batch, num_heads, seq, head_dim]
    if (qkvNonConst->rankOf() < 3) {
        result.success = false;
        result.errorMessage = "QKV tensor must have at least 3 dimensions";
        return result;
    }

    // Find the head dimension (looking for dimension that equals numHeads)
    int headDim = -1;
    auto shapePtr = qkvNonConst->getShapeAsVector();
    auto shape = *shapePtr;
    delete shapePtr;
    for (int i = 1; i < static_cast<int>(shape.size()) - 1; ++i) {
        if (shape[i] == numHeads || shape[i] % numHeads == 0) {
            headDim = i;
            break;
        }
    }

    if (headDim < 0) {
        headDim = 1;  // Default to dimension 1
    }

    return partition(qkv, headDim, numPartitions);
}

PartitionResult TensorPartitioner::partitionBatch(
    const NDArray* tensor,
    int numPartitions
) {
    return partition(tensor, 0, numPartitions);
}

PartitionResult TensorPartitioner::partitionSequence(
    const NDArray* tensor,
    int numPartitions
) {
    // Sequence is typically dimension 1
    auto tensorNonConst = const_cast<NDArray*>(tensor);
    int seqDim = (tensorNonConst->rankOf() > 1) ? 1 : 0;
    return partition(tensor, seqDim, numPartitions);
}

GatherResult TensorPartitioner::gather(
    const std::vector<NDArray*>& partitions,
    int dimension,
    int targetDevice
) {
    GatherResult result;

    if (partitions.empty()) {
        result.success = false;
        result.errorMessage = "No partitions to gather";
        return result;
    }

    // Calculate output shape
    auto shapePtr = partitions[0]->getShapeAsVector();
    std::vector<LongType> outputShape = *shapePtr;
    delete shapePtr;
    LongType totalDimSize = 0;
    for (auto* partition : partitions) {
        totalDimSize += partition->sizeAt(dimension);
    }
    outputShape[dimension] = totalDimSize;

    // Allocate output tensor
    result.result = std::shared_ptr<NDArray>(new NDArray('c', outputShape, partitions[0]->dataType()));

    // Copy from each partition
    LongType offset = 0;
    for (size_t i = 0; i < partitions.size(); ++i) {
        auto* partition = partitions[i];
        LongType partSize = partition->sizeAt(dimension);

        // Create index range for operator()
        // Format: {dim0Start, dim0End, dim1Start, dim1End, ...}
        std::vector<LongType> indices;
        for (size_t d = 0; d < outputShape.size(); ++d) {
            if (static_cast<int>(d) == dimension) {
                indices.push_back(offset);
                indices.push_back(offset + partSize);
            } else {
                indices.push_back(0);
                indices.push_back(outputShape[d]);
            }
        }

        NDArray* view = (*result.result)(indices);
        view->assign(partition);
        delete view;

        offset += partSize;
    }

    result.success = true;
    return result;
}

GatherResult TensorPartitioner::gather(
    const PartitionResult& partitionResult,
    int targetDevice
) {
    if (!partitionResult.success || partitionResult.partitions.empty()) {
        GatherResult result;
        result.success = false;
        result.errorMessage = "Invalid partition result";
        return result;
    }

    std::vector<NDArray*> rawPartitions;
    for (const auto& p : partitionResult.partitions) {
        rawPartitions.push_back(p.get());
    }

    int dimension = partitionResult.descriptors[0].splitDimension;
    return gather(rawPartitions, dimension, targetDevice);
}

GatherResult TensorPartitioner::reduce(
    const std::vector<NDArray*>& partitions,
    CommunicationPattern pattern,
    int targetDevice
) {
    GatherResult result;

    if (partitions.empty()) {
        result.success = false;
        result.errorMessage = "No partitions to reduce";
        return result;
    }

    // Create output with same shape as first partition
    result.result = std::shared_ptr<NDArray>(new NDArray(partitions[0]->shapeInfo(), false, LaunchContext::defaultContext(), true));
    result.result->assign(partitions[0]);

    // Sum all partitions
    for (size_t i = 1; i < partitions.size(); ++i) {
        *result.result += *partitions[i];
    }

    result.success = true;
    return result;
}

std::vector<PartitionDescriptor> TensorPartitioner::calculatePartitions(
    const std::vector<LongType>& shape,
    DataType dtype,
    int dimension,
    int numPartitions
) {
    std::vector<PartitionDescriptor> descriptors;

    if (dimension < 0 || dimension >= static_cast<int>(shape.size())) {
        return descriptors;
    }

    LongType dimSize = shape[dimension];
    auto partitionSizes = calculatePartitionSizes(dimSize, numPartitions);

    LongType offset = 0;
    for (int i = 0; i < numPartitions; ++i) {
        PartitionDescriptor desc;
        desc.partitionIndex = i;
        desc.totalPartitions = numPartitions;
        desc.sourceShape = shape;
        desc.dataType = dtype;
        desc.splitDimension = dimension;
        desc.splitOffset = offset;
        desc.splitSize = partitionSizes[i];

        desc.partitionShape = shape;
        desc.partitionShape[dimension] = partitionSizes[i];

        LongType elements = 1;
        for (auto dim : desc.partitionShape) {
            elements *= dim;
        }
        desc.bytesRequired = elements * DataTypeUtils::sizeOf(dtype);

        descriptors.push_back(desc);
        offset += partitionSizes[i];
    }

    return descriptors;
}

std::vector<LongType> TensorPartitioner::calculatePartitionSizes(
    LongType totalSize,
    int numPartitions,
    LongType minSize
) {
    std::vector<LongType> sizes(numPartitions);

    LongType baseSize = totalSize / numPartitions;
    LongType remainder = totalSize % numPartitions;

    for (int i = 0; i < numPartitions; ++i) {
        sizes[i] = baseSize + (i < remainder ? 1 : 0);
        if (sizes[i] < minSize) {
            sizes[i] = minSize;
        }
    }

    return sizes;
}

bool TensorPartitioner::canPartitionEvenly(
    const NDArray* tensor,
    int dimension,
    int numPartitions
) {
    auto tensorNonConst = const_cast<NDArray*>(tensor);
    if (dimension < 0 || dimension >= tensorNonConst->rankOf()) {
        return false;
    }
    return tensorNonConst->sizeAt(dimension) % numPartitions == 0;
}

int TensorPartitioner::getOptimalPartitionCount(
    const NDArray* tensor,
    int dimension,
    const std::vector<int>& devices
) {
    if (devices.empty()) {
        return 1;
    }

    auto tensorNonConst = const_cast<NDArray*>(tensor);
    LongType dimSize = tensorNonConst->sizeAt(dimension);
    int numDevices = static_cast<int>(devices.size());

    // Find largest divisor <= numDevices
    for (int n = numDevices; n > 0; --n) {
        if (dimSize % n == 0) {
            return n;
        }
    }

    return numDevices;  // Fall back to numDevices even if not even
}

bool TensorPartitioner::validatePartition(
    const NDArray* tensor,
    int dimension,
    int numPartitions,
    std::string& errorMessage
) {
    if (tensor == nullptr) {
        errorMessage = "Tensor is null";
        return false;
    }

    auto tensorNonConst = const_cast<NDArray*>(tensor);
    if (dimension < 0 || dimension >= tensorNonConst->rankOf()) {
        errorMessage = "Invalid dimension: " + std::to_string(dimension);
        return false;
    }

    if (numPartitions <= 0) {
        errorMessage = "Number of partitions must be positive";
        return false;
    }

    if (tensorNonConst->sizeAt(dimension) < numPartitions) {
        errorMessage = "Dimension size " + std::to_string(tensorNonConst->sizeAt(dimension)) +
                       " is smaller than number of partitions " + std::to_string(numPartitions);
        return false;
    }

    return true;
}

size_t TensorPartitioner::getPartitionMemoryRequired(
    const NDArray* tensor,
    int dimension,
    int numPartitions,
    bool includeGather
) {
    auto tensorNonConst = const_cast<NDArray*>(tensor);
    size_t perPartition = tensorNonConst->lengthOf() / numPartitions * DataTypeUtils::sizeOf(tensorNonConst->dataType());
    size_t total = perPartition * numPartitions;

    if (includeGather) {
        total += tensorNonConst->lengthOf() * DataTypeUtils::sizeOf(tensorNonConst->dataType());
    }

    return total;
}

int TensorPartitioner::resolveDimension(
    TensorPartitionDim dimType,
    const std::vector<LongType>& shape,
    int numHeads
) {
    int rank = static_cast<int>(shape.size());

    switch (dimType) {
        case TensorPartitionDim::ROW:
            return (rank >= 2) ? rank - 2 : 0;
        case TensorPartitionDim::COLUMN:
            return (rank >= 1) ? rank - 1 : 0;
        case TensorPartitionDim::HEAD:
            // For attention: typically dimension 1 or 2
            if (numHeads > 0) {
                for (int i = 1; i < rank - 1; ++i) {
                    if (shape[i] == numHeads) return i;
                }
            }
            return (rank >= 2) ? 1 : 0;
        case TensorPartitionDim::SEQUENCE:
            return (rank >= 2) ? 1 : 0;
        case TensorPartitionDim::BATCH:
            return 0;
        case TensorPartitionDim::EXPERT:
            return (rank >= 2) ? 1 : 0;
        case TensorPartitionDim::CUSTOM:
        default:
            return 0;
    }
}

std::shared_ptr<NDArray> TensorPartitioner::createPartitionArray(
    const PartitionDescriptor& desc,
    int deviceIndex
) {
    std::vector<LongType> shape = desc.partitionShape;
    return std::shared_ptr<NDArray>(new NDArray('c', shape, desc.dataType));
}

void TensorPartitioner::copyToPartition(
    const NDArray* source,
    NDArray* partition,
    const PartitionDescriptor& desc
) {
    // Create view into source for this partition's range
    // Format: {dim0Start, dim0End, dim1Start, dim1End, ...}
    std::vector<LongType> indices;
    for (size_t d = 0; d < desc.sourceShape.size(); ++d) {
        if (static_cast<int>(d) == desc.splitDimension) {
            indices.push_back(desc.splitOffset);
            indices.push_back(desc.splitOffset + desc.splitSize);
        } else {
            indices.push_back(0);
            indices.push_back(desc.sourceShape[d]);
        }
    }

    auto sourceNonConst = const_cast<NDArray*>(source);
    NDArray* view = (*sourceNonConst)(indices);
    partition->assign(view);
    delete view;
}

void TensorPartitioner::copyFromPartition(
    const NDArray* partition,
    NDArray* destination,
    const PartitionDescriptor& desc
) {
    // Format: {dim0Start, dim0End, dim1Start, dim1End, ...}
    std::vector<LongType> indices;
    for (size_t d = 0; d < desc.sourceShape.size(); ++d) {
        if (static_cast<int>(d) == desc.splitDimension) {
            indices.push_back(desc.splitOffset);
            indices.push_back(desc.splitOffset + desc.splitSize);
        } else {
            indices.push_back(0);
            indices.push_back(desc.sourceShape[d]);
        }
    }

    NDArray* view = (*destination)(indices);
    view->assign(const_cast<NDArray*>(partition));
    delete view;
}

// PartitionedTensor implementation
PartitionedTensor::PartitionedTensor(PartitionResult&& result)
    : _valid(result.success)
    , _descriptors(std::move(result.descriptors))
    , _partitions(std::move(result.partitions)) {

    if (_valid && !_descriptors.empty()) {
        _splitDimension = _descriptors[0].splitDimension;
        _originalShape = _descriptors[0].sourceShape;
    }
}

PartitionedTensor::PartitionedTensor(
    const NDArray* tensor,
    int dimension,
    int numPartitions,
    const std::vector<int>& devices
) {
    auto result = TensorPartitioner::getInstance().partition(tensor, dimension, numPartitions, devices);
    _valid = result.success;
    _descriptors = std::move(result.descriptors);
    _partitions = std::move(result.partitions);

    if (_valid && !_descriptors.empty()) {
        _splitDimension = dimension;
        auto tensorNonConst = const_cast<NDArray*>(tensor);
        auto shapePtr = tensorNonConst->getShapeAsVector();
        _originalShape = *shapePtr;
        delete shapePtr;
    }
}

NDArray* PartitionedTensor::getPartition(int index) {
    if (index >= 0 && index < static_cast<int>(_partitions.size())) {
        return _partitions[index].get();
    }
    return nullptr;
}

const NDArray* PartitionedTensor::getPartition(int index) const {
    if (index >= 0 && index < static_cast<int>(_partitions.size())) {
        return _partitions[index].get();
    }
    return nullptr;
}

NDArray* PartitionedTensor::getPartitionOnDevice(int deviceIndex) {
    for (size_t i = 0; i < _descriptors.size(); ++i) {
        if (_descriptors[i].deviceIndex == deviceIndex) {
            return _partitions[i].get();
        }
    }
    return nullptr;
}

std::vector<NDArray*> PartitionedTensor::getAllPartitions() {
    std::vector<NDArray*> result;
    for (auto& p : _partitions) {
        result.push_back(p.get());
    }
    return result;
}

const PartitionDescriptor& PartitionedTensor::getDescriptor(int index) const {
    static PartitionDescriptor empty;
    if (index >= 0 && index < static_cast<int>(_descriptors.size())) {
        return _descriptors[index];
    }
    return empty;
}

std::shared_ptr<NDArray> PartitionedTensor::gather(int targetDevice) {
    if (!_valid || _partitions.empty()) {
        return nullptr;
    }

    std::vector<NDArray*> rawPartitions;
    for (auto& p : _partitions) {
        rawPartitions.push_back(p.get());
    }

    auto result = TensorPartitioner::getInstance().gather(rawPartitions, _splitDimension, targetDevice);
    return result.success ? result.result : nullptr;
}

std::vector<int> PartitionedTensor::getDevices() const {
    std::vector<int> devices;
    for (const auto& desc : _descriptors) {
        devices.push_back(desc.deviceIndex);
    }
    return devices;
}

}  // namespace modelparallel
}  // namespace sd
