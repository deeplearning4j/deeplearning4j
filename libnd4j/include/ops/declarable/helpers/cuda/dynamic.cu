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
// @author raver119@gmail.com
//
#include <helpers/PointersManager.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/dynamic.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// An index array selects, for every one of its elements, a slice of the data: the data has the shape of the indices
// followed by the dimensions of a slice (dynamic_partition, dynamic_stitch). The kernels below address the data in the
// logical (C) order of its elements: element t of the data belongs to index t / sliceLength and is element
// t % sliceLength of that slice. Every operand is read and written through its own shape and strides, so views of any
// layout work, and every slice is moved by a grid-stride loop over all of its elements.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// blocks and threads of a launch with one thread per element: the named launch dimensions (x = blocks, y = threads)
// give the thread count and cap the grid, the kernels stride over whatever the grid does not cover
static void flatLaunch(const dim3& named, LongType elements, unsigned int& blocks, unsigned int& threads) {
  threads = named.y > 0 ? named.y : 1;
  if (threads > SD_MAX_NUM_THREADS) threads = SD_MAX_NUM_THREADS;
  const LongType needed = math::sd_max<LongType>(1, (elements + threads - 1) / threads);
  const LongType cap = math::sd_max<LongType>(1, named.x);
  blocks = static_cast<unsigned int>(math::sd_min<LongType>(needed, cap));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// launch errors of an asynchronous launch; inside a CUDA graph capture nothing has run yet
static SD_INLINE void checkLaunch(cudaStream_t* stream, const char* message) {
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode(message);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the integer stored at the element with the given logical index of an index array
template <typename Y>
static SD_INLINE SD_DEVICE LongType indexValueAt(const Y* indices, LongType iRank, const LongType* iShape,
                                                 const LongType* iStride, LongType linearIndex) {
  LongType coords[SD_MAX_RANK];
  LongType offset;
  INDEX2COORDS(linearIndex, iRank, iShape, coords);
  COORDS2INDEX(iRank, iStride, coords, offset);
  return static_cast<LongType>(indices[offset]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// position of every element inside its partition: the number of earlier elements (in the logical order of the
// indices) of the same partition. A block per partition (grid-stride over the partitions) walks the indices a block
// wide tile at a time and ranks the elements of the tile that belong to its partition with warp votes, so equal
// partitions keep the order of their elements. positions[e] is written for the elements of a valid partition only.
// The block size is a multiple of the warp size, at most 1024 threads.
template <typename Y>
static SD_KERNEL void dynamicPartitionPositionsKernel(const void* vindices, const LongType* iShapeInfo, LongType length,
                                                      LongType numPartitions, LongType* positions) {
  const auto indices = reinterpret_cast<const Y*>(vindices);
  const LongType iRank = shape::rank(iShapeInfo);
  const LongType* iShape = shape::shapeOf(iShapeInfo);
  const LongType* iStride = shape::stride(iShapeInfo);

  __shared__ int warpCounts[32];  // the elements of the tile in each warp of the block
  __shared__ LongType before;     // the elements of the partition ahead of the tile
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int numWarps = (blockDim.x + 31) >> 5;

  for (LongType partition = blockIdx.x; partition < numPartitions; partition += gridDim.x) {
    if (threadIdx.x == 0) before = 0;
    __syncthreads();

    for (LongType tile = 0; tile < length; tile += blockDim.x) {
      const LongType e = tile + threadIdx.x;
      bool mine = false;
      if (e < length) mine = indexValueAt<Y>(indices, iRank, iShape, iStride, e) == partition;

      const unsigned int votes = __ballot_sync(0xffffffffu, mine);
      if (lane == 0) warpCounts[warp] = __popc(votes);
      __syncthreads();

      if (mine) {
        LongType position = before + __popc(votes & ((1u << lane) - 1u));
        for (int w = 0; w < warp; w++) position += warpCounts[w];
        positions[e] = position;
      }
      __syncthreads();

      if (threadIdx.x == 0) {
        LongType inTile = 0;
        for (int w = 0; w < numWarps; w++) inTile += warpCounts[w];
        before += inTile;
      }
      __syncthreads();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_partition: a slice of the data goes to its partition's output at the position of the slice in the partition.
// Slices of an index outside [0, numPartitions) belong to no partition.
template <typename X, typename Y>
static SD_KERNEL void dynamicPartitionGatherKernel(const void* vx, const LongType* xShapeInfo, const void* vindices,
                                                   const LongType* iShapeInfo, const LongType* positions, void** vz,
                                                   LongType** zShapeInfos, LongType numPartitions,
                                                   LongType sliceLength, LongType total) {
  const auto x = reinterpret_cast<const X*>(vx);
  const auto indices = reinterpret_cast<const Y*>(vindices);
  const LongType xRank = shape::rank(xShapeInfo);
  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);
  const LongType iRank = shape::rank(iShapeInfo);
  const LongType* iShape = shape::shapeOf(iShapeInfo);
  const LongType* iStride = shape::stride(iShapeInfo);

  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType linearIndex = start; linearIndex < total; linearIndex += step) {
    const LongType e = linearIndex / sliceLength;
    const LongType partition = indexValueAt<Y>(indices, iRank, iShape, iStride, e);
    if (partition < 0 || partition >= numPartitions) continue;

    LongType coords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(linearIndex, xRank, xShape, coords);
    COORDS2INDEX(xRank, xStride, coords, xOffset);

    const LongType* zShapeInfo = zShapeInfos[partition];
    const LongType zRank = shape::rank(zShapeInfo);
    const LongType* zShape = shape::shapeOf(zShapeInfo);
    const LongType* zStride = shape::stride(zShapeInfo);
    LongType zOffset;
    const LongType zLinearIndex = positions[e] * sliceLength + (linearIndex - e * sliceLength);
    INDEX2COORDS(zLinearIndex, zRank, zShape, coords);
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    reinterpret_cast<X*>(vz[partition])[zOffset] = x[xOffset];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gradient of dynamic_partition: a slice of the input's gradient is the slice of its partition's gradient at the
// position of the slice in the partition. The partitions drop the slices of an index outside [0, numPartitions), so
// those slices get a zero gradient.
template <typename X, typename Y>
static SD_KERNEL void dynamicPartitionBpKernel(void* vgi, const LongType* giShapeInfo, const void* vindices,
                                               const LongType* iShapeInfo, const LongType* positions, void** vgo,
                                               LongType** goShapeInfos, LongType numPartitions, LongType sliceLength,
                                               LongType total) {
  const auto gradInput = reinterpret_cast<X*>(vgi);
  const auto indices = reinterpret_cast<const Y*>(vindices);
  const LongType giRank = shape::rank(giShapeInfo);
  const LongType* giShape = shape::shapeOf(giShapeInfo);
  const LongType* giStride = shape::stride(giShapeInfo);
  const LongType iRank = shape::rank(iShapeInfo);
  const LongType* iShape = shape::shapeOf(iShapeInfo);
  const LongType* iStride = shape::stride(iShapeInfo);

  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType linearIndex = start; linearIndex < total; linearIndex += step) {
    const LongType e = linearIndex / sliceLength;
    const LongType partition = indexValueAt<Y>(indices, iRank, iShape, iStride, e);

    LongType coords[SD_MAX_RANK];
    LongType giOffset;
    INDEX2COORDS(linearIndex, giRank, giShape, coords);
    COORDS2INDEX(giRank, giStride, coords, giOffset);

    if (partition < 0 || partition >= numPartitions) {
      gradInput[giOffset] = static_cast<X>(0);
      continue;
    }

    const LongType* goShapeInfo = goShapeInfos[partition];
    const LongType goRank = shape::rank(goShapeInfo);
    const LongType* goShape = shape::shapeOf(goShapeInfo);
    const LongType* goStride = shape::stride(goShapeInfo);
    LongType goOffset;
    const LongType goLinearIndex = positions[e] * sliceLength + (linearIndex - e * sliceLength);
    INDEX2COORDS(goLinearIndex, goRank, goShape, coords);
    COORDS2INDEX(goRank, goStride, coords, goOffset);

    gradInput[giOffset] = reinterpret_cast<const X*>(vgo[partition])[goOffset];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the position of every element of the indices in its partition (a device array of length elements), launched on the
// context's stream. The scan kernel needs whole warps: the named thread count is rounded up to a multiple of 32.
template <typename Y>
static void launchPartitionPositions(LaunchContext* context, NDArray* indices, LongType numPartitions,
                                     LongType* positions) {
  const LongType length = indices->lengthOf();
  const dim3 named = getLaunchDims("dynamic_partition_tad");
  unsigned int threads = named.y > 0 ? named.y : 32;
  if (threads > SD_MAX_NUM_THREADS) threads = SD_MAX_NUM_THREADS;
  threads = (threads + 31) / 32 * 32;
  const unsigned int blocks =
      static_cast<unsigned int>(math::sd_min<LongType>(numPartitions, math::sd_max<LongType>(1, named.x)));

  auto stream = context->getCudaStream();
  dynamicPartitionPositionsKernel<Y><<<blocks, threads, 0, *stream>>>(indices->specialBuffer(),
                                                                       indices->specialShapeInfo(), length,
                                                                       numPartitions, positions);
  checkLaunch(stream, "dynamicPartitionPositionsKernel failed: ");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void _dynamicPartitionFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                                     std::vector<NDArray*>& outputList) {
  const LongType numPartitions = outputList.size();
  const LongType length = indices->lengthOf();
  const LongType total = input->lengthOf();
  if (numPartitions == 0 || length == 0 || total == 0) return;

  // the dimensions of the input beyond the indices' own are moved with their slice
  const LongType sliceLength = total / length;

  PointersManager pm(context, "dynamicPartition");
  std::vector<void*> outBuffers(numPartitions);
  std::vector<const LongType*> outShapes(numPartitions);
  for (LongType i = 0; i < numPartitions; i++) {
    outBuffers[i] = outputList[i]->lengthOf() > 0 ? outputList[i]->specialBuffer() : nullptr;
    outShapes[i] = outputList[i]->specialShapeInfo();
  }
  auto dOutBuffers = reinterpret_cast<void**>(pm.replicatePointer(outBuffers.data(), numPartitions * sizeof(void*)));
  auto dOutShapes =
      reinterpret_cast<LongType**>(pm.replicatePointer(outShapes.data(), numPartitions * sizeof(LongType*)));
  auto dPositions = reinterpret_cast<LongType*>(pm.allocateDevMem(length * sizeof(LongType)));

  launchPartitionPositions<Y>(context, indices, numPartitions, dPositions);

  unsigned int blocks, threads;
  flatLaunch(getLaunchDims("dynamic_partition_tad"), total, blocks, threads);
  auto stream = context->getCudaStream();
  dynamicPartitionGatherKernel<X, Y><<<blocks, threads, 0, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), indices->specialBuffer(), indices->specialShapeInfo(),
      dPositions, dOutBuffers, dOutShapes, numPartitions, sliceLength, total);
  checkLaunch(stream, "dynamicPartitionGatherKernel failed: ");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void _dynamicPartitionFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                                       std::vector<NDArray*> const& inputGradientList,
                                       std::vector<NDArray*>& outputList) {
  // outputList[0] is the gradient of the input: it has the input's shape
  NDArray* gradInput = outputList.at(0);
  const LongType numPartitions = inputGradientList.size();
  const LongType length = indices->lengthOf();
  const LongType total = gradInput->lengthOf();
  if (length == 0 || total == 0) return;

  const LongType sliceLength = total / length;

  PointersManager pm(context, "dynamicPartitionBp");
  std::vector<void*> gradBuffers(numPartitions);
  std::vector<const LongType*> gradShapes(numPartitions);
  for (LongType i = 0; i < numPartitions; i++) {
    gradBuffers[i] = inputGradientList[i]->lengthOf() > 0 ? inputGradientList[i]->specialBuffer() : nullptr;
    gradShapes[i] = inputGradientList[i]->specialShapeInfo();
  }
  auto dGradBuffers =
      reinterpret_cast<void**>(pm.replicatePointer(gradBuffers.data(), numPartitions * sizeof(void*)));
  auto dGradShapes =
      reinterpret_cast<LongType**>(pm.replicatePointer(gradShapes.data(), numPartitions * sizeof(LongType*)));
  auto dPositions = reinterpret_cast<LongType*>(pm.allocateDevMem(length * sizeof(LongType)));

  // with no partition at all every slice has a zero gradient and there is nothing to rank
  if (numPartitions > 0) launchPartitionPositions<Y>(context, indices, numPartitions, dPositions);

  unsigned int blocks, threads;
  flatLaunch(getLaunchDims("dynamic_partition_tad"), total, blocks, threads);
  auto stream = context->getCudaStream();
  dynamicPartitionBpKernel<X, Y><<<blocks, threads, 0, *stream>>>(
      gradInput->specialBuffer(), gradInput->specialShapeInfo(), indices->specialBuffer(),
      indices->specialShapeInfo(), dPositions, dGradBuffers, dGradShapes, numPartitions, sliceLength, total);
  checkLaunch(stream, "dynamicPartitionBpKernel failed: ");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the output of dynamic_stitch starts as zeros: a row that no index names is zero (the output is not initialized)
template <typename X>
static SD_KERNEL void dynamicStitchZeroKernel(void* vz, const LongType* zShapeInfo, LongType total) {
  const auto z = reinterpret_cast<X*>(vz);
  const LongType zRank = shape::rank(zShapeInfo);
  const LongType* zShape = shape::shapeOf(zShapeInfo);
  const LongType* zStride = shape::stride(zShapeInfo);

  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType linearIndex = start; linearIndex < total; linearIndex += step) {
    LongType coords[SD_MAX_RANK];
    LongType zOffset;
    INDEX2COORDS(linearIndex, zRank, zShape, coords);
    COORDS2INDEX(zRank, zStride, coords, zOffset);
    z[zOffset] = static_cast<X>(0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_stitch: a slice of an input goes to the row of the output that its index names. The inputs are stitched
// in order, each by its own launch on the context's stream, so when an index appears in several inputs the last
// input wins; within one input the slices of equal indices are written in no particular order. Slices of an index
// outside [0, numRows) are dropped.
template <typename X, typename Y>
static SD_KERNEL void dynamicStitchKernel(const void* vx, const LongType* xShapeInfo, const void* vindices,
                                          const LongType* iShapeInfo, void* vz, const LongType* zShapeInfo,
                                          LongType numRows, LongType sliceLength, LongType total) {
  const auto x = reinterpret_cast<const X*>(vx);
  const auto indices = reinterpret_cast<const Y*>(vindices);
  const auto z = reinterpret_cast<X*>(vz);
  const LongType xRank = shape::rank(xShapeInfo);
  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);
  const LongType iRank = shape::rank(iShapeInfo);
  const LongType* iShape = shape::shapeOf(iShapeInfo);
  const LongType* iStride = shape::stride(iShapeInfo);
  const LongType zRank = shape::rank(zShapeInfo);
  const LongType* zShape = shape::shapeOf(zShapeInfo);
  const LongType* zStride = shape::stride(zShapeInfo);

  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType linearIndex = start; linearIndex < total; linearIndex += step) {
    const LongType e = linearIndex / sliceLength;
    const LongType row = indexValueAt<Y>(indices, iRank, iShape, iStride, e);
    if (row < 0 || row >= numRows) continue;

    LongType coords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(linearIndex, xRank, xShape, coords);
    COORDS2INDEX(xRank, xStride, coords, xOffset);

    LongType zOffset;
    INDEX2COORDS(row * sliceLength + (linearIndex - e * sliceLength), zRank, zShape, coords);
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    z[zOffset] = x[xOffset];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static Status _dynamicStitchFunctor(LaunchContext* context, std::vector<NDArray*> const& inputs,
                                    std::vector<NDArray*> const& indices, NDArray* output) {
  const LongType inputSize = inputs.size();
  const LongType numRows = output->rankOf() == 0 ? 1 : output->sizeAt(0);
  if (output->lengthOf() == 0 || numRows == 0) return Status::OK;

  // the output rows are stitched from slices of the inputs: the dimensions of an input beyond its indices' own
  // are the dimensions of a row
  const LongType sliceLength = output->lengthOf() / numRows;
  for (LongType e = 0; e < inputSize; e++) {
    if (inputs[e]->lengthOf() != indices[e]->lengthOf() * sliceLength) {
      sd_printf("dynamic_stitch: input %lld has %lld elements, but %lld indices of slices of %lld elements need %lld\n",
                (long long)e, (long long)inputs[e]->lengthOf(), (long long)indices[e]->lengthOf(),
                (long long)sliceLength, (long long)(indices[e]->lengthOf() * sliceLength));
      return Status::VALIDATION;
    }
  }

  auto stream = context->getCudaStream();
  const dim3 named = getLaunchDims("dynamic_stitch_tad");

  // the rows that no index names stay zero, the others are written by the inputs below
  {
    unsigned int blocks, threads;
    flatLaunch(named, output->lengthOf(), blocks, threads);
    dynamicStitchZeroKernel<X><<<blocks, threads, 0, *stream>>>(output->specialBuffer(), output->specialShapeInfo(),
                                                                output->lengthOf());
    checkLaunch(stream, "dynamicStitchZeroKernel failed: ");
  }

  for (LongType e = 0; e < inputSize; e++) {
    const LongType total = inputs[e]->lengthOf();
    if (total == 0) continue;

    unsigned int blocks, threads;
    flatLaunch(named, total, blocks, threads);
    dynamicStitchKernel<X, Y><<<blocks, threads, 0, *stream>>>(
        inputs[e]->specialBuffer(), inputs[e]->specialShapeInfo(), indices[e]->specialBuffer(),
        indices[e]->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), numRows, sliceLength,
        total);
    checkLaunch(stream, "dynamicStitchKernel failed: ");
  }

  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the arrays that hold data: a partition that got no slice has zero-length arrays, which have no device memory to
// prepare or register
static std::vector<NDArray*> withData(std::vector<NDArray*> const& arrays) {
  std::vector<NDArray*> result;
  for (auto array : arrays) {
    if (array != nullptr && array->lengthOf() > 0) result.push_back(array);
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dynamicPartitionFunctor(LaunchContext* context, NDArray* input, NDArray* indices,
                             std::vector<NDArray*>& outputList) {
  auto xType = input->dataType();
  auto yType = indices->dataType();

  const std::vector<NDArray*> writeArrays = withData(outputList);
  const std::vector<NDArray*> readArrays = withData({indices, input});

  NDArray::prepareSpecialUse(writeArrays, readArrays);

  BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicPartitionFunctor, (context, input, indices, outputList), SD_NUMERIC_TYPES,
                        SD_INDEXING_TYPES);

  NDArray::registerSpecialUse(writeArrays, readArrays);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status dynamicStitchFunctor(LaunchContext* context, std::vector<NDArray*> const& inputs,
                            std::vector<NDArray*> const& indices, NDArray* output) {
  auto xType = inputs.at(0)->dataType();
  auto yType = indices.at(0)->dataType();

  // Build combined read list: all inputs + all indices arrays
  std::vector<NDArray*> allArrays;
  allArrays.insert(allArrays.end(), inputs.begin(), inputs.end());
  allArrays.insert(allArrays.end(), indices.begin(), indices.end());

  const std::vector<NDArray*> readArrays = withData(allArrays);
  const std::vector<NDArray*> writeArrays = withData({output});

  NDArray::prepareSpecialUse(writeArrays, readArrays);

  Status result = Status::OK;
  BUILD_DOUBLE_SELECTOR(xType, yType, result = _dynamicStitchFunctor, (context, inputs, indices, output),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse(writeArrays, readArrays);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dynamicPartitionFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices,
                               std::vector<NDArray*> const& inputGradientList, std::vector<NDArray*>& outputList) {
  auto xType = outputList.at(0)->dataType();
  auto yType = indices->dataType();

  std::vector<NDArray*> allArrays(inputGradientList.begin(), inputGradientList.end());
  allArrays.push_back(indices);

  const std::vector<NDArray*> writeArrays = withData(outputList);
  const std::vector<NDArray*> readArrays = withData(allArrays);

  NDArray::prepareSpecialUse(writeArrays, readArrays);

  BUILD_DOUBLE_SELECTOR(xType, yType, _dynamicPartitionFunctorBP, (context, input, indices, inputGradientList, outputList),
                        SD_NUMERIC_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse(writeArrays, readArrays);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
