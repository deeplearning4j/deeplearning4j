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
//  @author GS <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <system/selective_rendering.h>

#include "helpers/DebugHelper.h"
namespace sd {
namespace ops {
namespace helpers {

// -------------------------------------------------------------------------------------------------------------- //
// Sorted segments ops implementations

template <typename T, typename I>
static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& aoutput) {
  return true;
}

bool segmentIndicesValidate(LaunchContext* context, NDArray* indices, NDArray& expected, NDArray& output) {
  auto indicesDType = indices->dataType();
  auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(output.dataType(), indices->dataType(), return segmentIndicesValidate_,
                        (indices, expected, output), SD_NUMERIC_TYPES, SD_INDEXING_TYPES);
}

// -------------------------------------------------------------------------------------------------------------- //
// Unsorted segment ops functors implementation
// -------------------------------------------------------------------------------------------------------------- //
template <typename I>
static SD_KERNEL void unsortedSegmentIndexValidateKernel(const I* indices, const LongType* indicesShape, I expected,
                                                         I* found) {
  __shared__ bool onlyTrue;
  __shared__ LongType len;

  if (threadIdx.x == 0) {
    onlyTrue = true;
    len = shape::length(indicesShape);
  }
  __syncthreads();
  auto start = threadIdx.x + blockIdx.x * blockDim.x;
  auto step = gridDim.x * blockDim.x;
  for (LongType e = start; e < len && onlyTrue; e += step) {
    math::atomics::sd_atomicMax(found, indices[e]);
    if (expected < *found) onlyTrue = false;
  }
}

template <typename I>
static bool unsortedSegmentIndicesValidate_(LaunchContext* context, NDArray* indices, LongType expected,
                                            LongType& output) {
  output = expected;
  I found = output;
  I exp = expected;
  auto stream = context->getCudaStream();
  I* devFound;
  cudaMalloc(&devFound, sizeof(I));
  cudaMemcpy(devFound, &found, sizeof(I), cudaMemcpyHostToDevice);

  dim3 launchDims = segmentValidateIndices(indices->lengthOf());
  unsortedSegmentIndexValidateKernel<I><<<launchDims.y,launchDims.x, launchDims.z, *stream>>>(
      reinterpret_cast<I*>(indices->specialBuffer()), indices->specialShapeInfo(), exp, devFound);
  sd::DebugHelper::checkErrorCode(stream, "unsortedSegmentIndexValidateKernel failed");

  cudaMemcpy(&found, devFound, sizeof(I), cudaMemcpyDeviceToHost);
  cudaFree(devFound);
  output = found;
  return expected == output;
}

bool unsortedSegmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType expected, LongType& output) {
  BUILD_SINGLE_SELECTOR(indices->dataType(), return unsortedSegmentIndicesValidate_,
                        (context, indices, expected, output), SD_INDEXING_TYPES);
}

// -------------------------------------------------------------------------------------------------------------- //

// -------------------------------------------------------------------------------------------------------------- //
// fill up segments starts and ends - splitted ordered case
template <typename I>
static SD_KERNEL void fillUpSegmentsKernel(const void* indices, const LongType* indexShape, LongType numClasses,
                                           LongType* classesRangesStart, LongType* classesRangesLengths) {
  __shared__ const I* idxBuf;
  __shared__ LongType idxLen;
  __shared__ LongType* result;
  if (threadIdx.x == 0) {
    idxBuf = reinterpret_cast<const I*>(indices);
    idxLen = shape::length(indexShape);
  }
  __syncthreads();

  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto j = tid; j < idxLen; j += step) {
    auto pos = idxBuf[j];
    math::atomics::sd_atomicMin<LongType>(&classesRangesStart[pos], (LongType)j);
    math::atomics::sd_atomicAdd<LongType>(&classesRangesLengths[pos], 1);
  }
}

// -------------------------------------------------------------------------------------------------------------- //

template <typename I>
static void fillUpSegments_(NDArray* indices, LongType numClasses, NDArray& classesRangesBegs,
                            NDArray& classesRangesLens) {
  dim3 dims = getFillUpSegmentsDims(numClasses, indices->lengthOf());
  LongType* begins = reinterpret_cast<LongType*>(classesRangesBegs.specialBuffer());
  LongType* lengths = reinterpret_cast<LongType*>(classesRangesLens.specialBuffer());
  auto stream = classesRangesBegs.getContext()->getCudaStream();
  fillUpSegmentsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(indices->specialBuffer(), indices->specialShapeInfo(),
                                                               numClasses, begins, lengths);
  sd::DebugHelper::checkErrorCode(stream, "fillUpSegmentsKernel failed");

}
// -------------------------------------------------------------------------------------------------------------- //

void fillUpSegments(NDArray* indices, LongType numClasses, NDArray& classesRangesBegs, NDArray& classesRangesLens) {
  BUILD_SINGLE_SELECTOR(indices->dataType(), fillUpSegments_,
                        (indices, numClasses, classesRangesBegs, classesRangesLens), SD_INDEXING_TYPES);
}
// -------------------------------------------------------------------------------------------------------------- //

}  // namespace helpers
}  // namespace ops
}  // namespace sd
// -------------------------------------------------------------------------------------------------------------- //
// -------------------------------------------------------------------------------------------------------------- //
