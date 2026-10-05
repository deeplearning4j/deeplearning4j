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
// Created by george on 05.04.18.
//
#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/dynamic.h>

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// An index array selects, for every one of its elements, a slice of the data: the data has the shape of the indices
// followed by the dimensions of a slice (dynamic_partition, dynamic_stitch). The loops below address the data in the
// logical (C) order of its elements: element t of the data belongs to index t / sliceLength and is element
// t % sliceLength of that slice. Every operand is read and written through its own shape and strides, so views of
// any layout work, and the elements of all slices are spread over the threads.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The partition of every element of the indices, and its position in the partition: the number of earlier elements
// (in the logical order of the indices) of the same partition. Elements of an index outside [0, numPartitions)
// belong to no partition (-1).
static void partitionPositions(NDArray* indices, LongType numPartitions, std::vector<LongType>& partitions,
                               std::vector<LongType>& positions) {
  const LongType length = indices->lengthOf();
  partitions.assign(length, -1);
  positions.assign(length, -1);
  std::vector<LongType> counters(numPartitions, 0);
  for (LongType e = 0; e < length; e++) {
    const LongType partition = indices->e<LongType>(e);
    if (partition < 0 || partition >= numPartitions) continue;
    partitions[e] = partition;
    positions[e] = counters[partition]++;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void _dynamicPartitionFunctor(NDArray* input, NDArray* indices, std::vector<NDArray*>& outputList) {
  const LongType numPartitions = outputList.size();
  const LongType length = indices->lengthOf();
  const LongType total = input->lengthOf();
  if (numPartitions == 0 || length == 0 || total == 0) return;

  // the dimensions of the input beyond the indices' own are moved with their slice
  const LongType sliceLength = total / length;

  std::vector<LongType> partitions, positions;
  partitionPositions(indices, numPartitions, partitions, positions);

  NDArray::preparePrimaryUse(outputList, {indices, input});

  const T* x = input->bufferAsT<T>();
  const LongType xRank = input->rankOf();
  const LongType* xShape = input->shapeOf();
  const LongType* xStrides = input->stridesOf();

  std::vector<T*> z(numPartitions);
  std::vector<LongType> zRanks(numPartitions);
  std::vector<const LongType*> zShapes(numPartitions);
  std::vector<const LongType*> zStrides(numPartitions);
  for (LongType p = 0; p < numPartitions; p++) {
    NDArray* output = outputList[p];
    z[p] = output->lengthOf() > 0 ? output->bufferAsT<T>() : nullptr;
    zRanks[p] = output->rankOf();
    zShapes[p] = output->shapeOf();
    zStrides[p] = output->stridesOf();
  }

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK] = {};
    for (auto t = start; t < stop; t++) {
      const LongType e = t / sliceLength;
      const LongType partition = partitions[e];
      if (partition < 0) continue;

      LongType xOffset = 0;
      INDEX2COORDS(t, xRank, xShape, coords);
      COORDS2INDEX(xRank, xStrides, coords, xOffset);

      const LongType zRank = zRanks[partition];
      const LongType* zShape = zShapes[partition];
      const LongType* zStride = zStrides[partition];
      LongType zOffset = 0;
      INDEX2COORDS(positions[e] * sliceLength + (t - e * sliceLength), zRank, zShape, coords);
      COORDS2INDEX(zRank, zStride, coords, zOffset);

      z[partition][zOffset] = x[xOffset];
    }
  };
  samediff::Threads::parallel_for(func, 0, total);

  NDArray::registerPrimaryUse(outputList, {indices, input});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_stitch: a slice of an input goes to the row of the output that its index names. When an index appears more
// than once the slice of the last input wins, and within an input the later slice. A row that no index names is zero.
template <typename T>
static Status _dynamicStitchFunctor(std::vector<NDArray*> const& inputs, std::vector<NDArray*> const& indices,
                                    NDArray* output) {
  const LongType numOfData = inputs.size();
  const LongType numRows = output->rankOf() == 0 ? 1 : output->sizeAt(0);
  if (output->lengthOf() == 0 || numRows == 0) return Status::OK;

  // the dimensions of an input beyond its indices' own are the dimensions of a row of the output
  const LongType sliceLength = output->lengthOf() / numRows;

  // the winning slice of every row: the input and the slice in it (-1 when no index names the row)
  std::vector<LongType> winnerInput(numRows, -1);
  std::vector<LongType> winnerSlice(numRows, -1);
  for (LongType e = 0; e < numOfData; e++) {
    NDArray* data = inputs[e];
    NDArray* index = indices[e];
    if (data->lengthOf() != index->lengthOf() * sliceLength) {
      sd_printf("dynamic_stitch: input %lld has %lld elements, but %lld indices of slices of %lld elements need %lld\n",
                (long long)e, (long long)data->lengthOf(), (long long)index->lengthOf(), (long long)sliceLength,
                (long long)(index->lengthOf() * sliceLength));
      return Status::VALIDATION;
    }
    for (LongType j = 0; j < index->lengthOf(); j++) {
      const LongType row = index->e<LongType>(j);
      if (row < 0) {
        sd_printf("dynamic_stitch: Index value should be non-negative. But %lld was given\n", (long long)row);
        return Status::VALIDATION;
      }
      if (row >= numRows) {
        sd_printf("dynamic_stitch: Index should be less than %lld. But %lld was given\n", (long long)numRows,
                  (long long)row);
        return Status::VALIDATION;
      }
      winnerInput[row] = e;
      winnerSlice[row] = j;
    }
  }

  std::vector<NDArray*> readArrays(inputs.begin(), inputs.end());
  NDArray::preparePrimaryUse({output}, readArrays);

  std::vector<const T*> x(numOfData);
  std::vector<LongType> xRanks(numOfData);
  std::vector<const LongType*> xShapes(numOfData);
  std::vector<const LongType*> xStrides(numOfData);
  for (LongType e = 0; e < numOfData; e++) {
    x[e] = inputs[e]->lengthOf() > 0 ? inputs[e]->bufferAsT<T>() : nullptr;
    xRanks[e] = inputs[e]->rankOf();
    xShapes[e] = inputs[e]->shapeOf();
    xStrides[e] = inputs[e]->stridesOf();
  }
  T* z = output->bufferAsT<T>();
  const LongType zRank = output->rankOf();
  const LongType* zShape = output->shapeOf();
  const LongType* zStrides = output->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK] = {};
    for (auto t = start; t < stop; t++) {
      const LongType row = t / sliceLength;
      const LongType e = winnerInput[row];

      LongType zOffset = 0;
      INDEX2COORDS(t, zRank, zShape, coords);
      COORDS2INDEX(zRank, zStrides, coords, zOffset);

      // a row that no index names is zero (the output is not initialized)
      if (e < 0) {
        z[zOffset] = static_cast<T>(0);
        continue;
      }

      LongType xOffset = 0;
      INDEX2COORDS(winnerSlice[row] * sliceLength + (t - row * sliceLength), xRanks[e], xShapes[e], coords);
      COORDS2INDEX(xRanks[e], xStrides[e], coords, xOffset);

      z[zOffset] = x[e][xOffset];
    }
  };
  samediff::Threads::parallel_for(func, 0, output->lengthOf());

  NDArray::registerPrimaryUse({output}, readArrays);
  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gradient of dynamic_partition: a slice of the input's gradient is the slice of its partition's gradient at the
// position of the slice in the partition. The partitions drop the slices of an index outside [0, numPartitions), so
// those slices get a zero gradient.
template <typename T>
static void _dynamicPartitionFunctorBP(NDArray* input, NDArray* indices,
                                       std::vector<NDArray*> const& inputGradientList,
                                       std::vector<NDArray*>& outputList) {
  // outputList[0] is the gradient of the input: it has the input's shape
  NDArray* gradInput = outputList.at(0);
  const LongType numPartitions = inputGradientList.size();
  const LongType length = indices->lengthOf();
  const LongType total = gradInput->lengthOf();
  if (length == 0 || total == 0) return;

  const LongType sliceLength = total / length;

  std::vector<LongType> partitions, positions;
  partitionPositions(indices, numPartitions, partitions, positions);

  std::vector<NDArray*> readArrays(inputGradientList.begin(), inputGradientList.end());
  readArrays.push_back(indices);
  NDArray::preparePrimaryUse(outputList, readArrays);

  T* gi = gradInput->bufferAsT<T>();
  const LongType giRank = gradInput->rankOf();
  const LongType* giShape = gradInput->shapeOf();
  const LongType* giStrides = gradInput->stridesOf();

  std::vector<const T*> go(numPartitions);
  std::vector<LongType> goRanks(numPartitions);
  std::vector<const LongType*> goShapes(numPartitions);
  std::vector<const LongType*> goStrides(numPartitions);
  for (LongType p = 0; p < numPartitions; p++) {
    NDArray* gradient = inputGradientList[p];
    go[p] = gradient->lengthOf() > 0 ? gradient->bufferAsT<T>() : nullptr;
    goRanks[p] = gradient->rankOf();
    goShapes[p] = gradient->shapeOf();
    goStrides[p] = gradient->stridesOf();
  }

  auto func = PRAGMA_THREADS_FOR {
    LongType coords[SD_MAX_RANK] = {};
    for (auto t = start; t < stop; t++) {
      const LongType e = t / sliceLength;
      const LongType partition = partitions[e];

      LongType giOffset = 0;
      INDEX2COORDS(t, giRank, giShape, coords);
      COORDS2INDEX(giRank, giStrides, coords, giOffset);

      if (partition < 0) {
        gi[giOffset] = static_cast<T>(0);
        continue;
      }

      const LongType goRank = goRanks[partition];
      const LongType* goShape = goShapes[partition];
      const LongType* goStride = goStrides[partition];
      LongType goOffset = 0;
      INDEX2COORDS(positions[e] * sliceLength + (t - e * sliceLength), goRank, goShape, coords);
      COORDS2INDEX(goRank, goStride, coords, goOffset);

      gi[giOffset] = go[partition][goOffset];
    }
  };
  samediff::Threads::parallel_for(func, 0, total);

  NDArray::registerPrimaryUse(outputList, readArrays);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dynamicPartitionFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                             std::vector<NDArray*>& outputList) {
  auto xType = input->dataType();

  BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctor, (input, indices, outputList), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status dynamicStitchFunctor(LaunchContext* context, std::vector<NDArray*> const& inputs,
                            std::vector<NDArray*> const& indices, NDArray* output) {
  auto xType = inputs.at(0)->dataType();

  BUILD_SINGLE_SELECTOR(xType, return _dynamicStitchFunctor, (inputs, indices, output), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dynamicPartitionFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                               std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {
  auto xType = outputList.at(0)->dataType();

  BUILD_SINGLE_SELECTOR(xType, _dynamicPartitionFunctorBP, (input, indices, inputGradientList, outputList),
                        SD_COMMON_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
