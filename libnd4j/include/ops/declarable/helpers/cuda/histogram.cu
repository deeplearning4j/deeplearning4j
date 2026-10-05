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
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_histogram)
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/histogram.h>

#include <cstdint>
#include <string>
#include <vector>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {
namespace helpers {

// Counts the elements of x into counts (one 64-bit counter per bin): every element goes to the bin
// histogramBin(x, low, width), low being *minValue and width (*maxValue - *minValue) / numBins in DOUBLE. x is read
// through its own shape and strides.
//
// With sharedBins the block counts into 32-bit counters in dynamic shared memory (numBins of them) and adds its
// non-zero counters to counts at the end: one global atomic per bin and block instead of one per element. Without,
// every element is added to counts directly (more bins than a block's shared memory holds). The grid-stride loop and
// the 64-bit indices make any launch cover any length.
template <typename X>
static SD_KERNEL void histogramKernel(const void *xBuffer, const LongType *xShapeInfo, LongType *counts,
                                      const LongType numBins, const X *minValue, const X *maxValue,
                                      const bool sharedBins) {
  extern __shared__ unsigned char sharedMemory[];
  int32_t *blockBins = reinterpret_cast<int32_t *>(sharedMemory);
  const X *x = reinterpret_cast<const X *>(xBuffer);

  __shared__ LongType length;
  __shared__ LongType rank;
  __shared__ const LongType *xShape;
  __shared__ const LongType *xStride;
  __shared__ double low;
  __shared__ double binWidth;

  if (threadIdx.x == 0) {
    length = shape::length(xShapeInfo);
    rank = shape::rank(xShapeInfo);
    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    low = static_cast<double>(*minValue);
    binWidth = histogramBinWidth(low, static_cast<double>(*maxValue), numBins);
  }

  if (sharedBins) {
    for (LongType e = threadIdx.x; e < numBins; e += blockDim.x) blockBins[e] = 0;
  }
  __syncthreads();

  LongType coords[SD_MAX_RANK];
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    LongType xOffset;
    INDEX2COORDS(i, rank, xShape, coords);
    COORDS2INDEX(rank, xStride, coords, xOffset);

    const LongType bin = histogramBin(static_cast<double>(x[xOffset]), low, binWidth, numBins);
    if (sharedBins)
      math::atomics::sd_atomicAdd<int32_t>(&blockBins[bin], 1);
    else
      math::atomics::sd_atomicAdd<LongType>(&counts[bin], 1);
  }

  if (sharedBins) {
    __syncthreads();
    for (LongType e = threadIdx.x; e < numBins; e += blockDim.x) {
      const int32_t blockCount = blockBins[e];
      if (blockCount != 0) math::atomics::sd_atomicAdd<LongType>(&counts[e], static_cast<LongType>(blockCount));
    }
  }
}

// launch dimensions of the "histogram" key: x blocks at most, y threads per block. The dynamic shared memory is the
// counters of the bins, which the launch computes, not a tuning size.
static dim3 histogramLaunchDims(LaunchContext *context, LongType length) {
  dim3 dims = getLaunchDims("histogram");
  int maxBlocks = 0;
  int maxThreads = 0;
  const int device = context->getDeviceID();
  if (cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMaxGridDimX, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device) != cudaSuccess)
    THROW_EXCEPTION("histogram: cannot query the launch limits of the device");
  if (dims.x == 0 || dims.x > static_cast<unsigned int>(maxBlocks) || dims.y == 0 ||
      dims.y > static_cast<unsigned int>(maxThreads)) {
    std::string message = "histogram: invalid launch dimensions (blocks " + std::to_string(dims.x) + ", threads " +
                          std::to_string(dims.y) + ")";
    THROW_EXCEPTION(message.c_str());
  }
  // no more blocks than threads' worth of elements
  const LongType blocksForLength = (length + dims.y - 1) / dims.y;
  if (blocksForLength < static_cast<LongType>(dims.x)) dims.x = static_cast<unsigned int>(blocksForLength);
  if (dims.x == 0) dims.x = 1;
  return dims;
}

template <typename X>
static void histogram_(LaunchContext *context, NDArray *input, NDArray *minValue, NDArray *maxValue, NDArray *counts) {
  const LongType length = input->lengthOf();
  const LongType numBins = counts->lengthOf();
  auto stream = context->getCudaStream();

  dim3 dims = histogramLaunchDims(context, length);

  // 32-bit counters per bin in shared memory when the bins fit (with room for the kernel's own shared variables) and
  // no block can see more elements than a counter holds; otherwise the elements are added to the global counts
  int maxShared = 0;
  if (cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlock, context->getDeviceID()) != cudaSuccess)
    THROW_EXCEPTION("histogram: cannot query the shared memory of the device");
  const LongType sharedBytes = numBins * static_cast<LongType>(sizeof(int32_t));
  const LongType staticSharedBytes = 256;
  const LongType elementsPerBlock = length / dims.x + dims.y;
  const bool sharedBins = sharedBytes + staticSharedBytes <= maxShared && elementsPerBlock < INT32_MAX;

  NDArray::prepareSpecialUse({counts}, {input, minValue, maxValue});
  histogramKernel<X><<<dims.x, dims.y, sharedBins ? static_cast<unsigned int>(sharedBytes) : 0, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), reinterpret_cast<LongType *>(counts->specialBuffer()), numBins,
      reinterpret_cast<const X *>(minValue->specialBuffer()), reinterpret_cast<const X *>(maxValue->specialBuffer()),
      sharedBins);
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode("histogramKernel failed");
  NDArray::registerSpecialUse({counts}, {input, minValue, maxValue});
}

// The counts go to the output through assign: it casts them to the output's integer type and stores each through the
// output's own strides.
void histogramHelper(LaunchContext *context, NDArray &input, NDArray &output) {
  const LongType numBins = output.lengthOf();
  if (numBins == 0) return;
  if (input.lengthOf() == 0) {
    output.nullify();
    return;
  }

  // counts has the output's shape, so assign pairs the two element by element
  std::vector<LongType> countsShape(shape::shapeOf(output.shapeInfo()),
                                    shape::shapeOf(output.shapeInfo()) + shape::rank(output.shapeInfo()));
  NDArray counts('c', countsShape, DataType::INT64, context);
  counts.nullify();

  NDArray *minValue = input.reduceNumber(reduce::SameOps::Min);
  NDArray *maxValue = input.reduceNumber(reduce::SameOps::Max);
  BUILD_SINGLE_SELECTOR(input.dataType(), histogram_, (context, &input, minValue, maxValue, &counts),
                        SD_NUMERIC_TYPES);
  output.assign(&counts);

  // counts, minValue and maxValue are released when this returns: wait for the kernel and the assign that read them
  // (a no-op while a CUDA graph is captured, where a release is deferred)
  PointersManager manager(context, "histogram");
  manager.synchronize();
  delete minValue;
  delete maxValue;
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
