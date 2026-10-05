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
//  Segment ids on the device: validation (with a stream-ordered readback of one small report), conversion of the ids
//  of any integer dtype / rank / strides to one dense int64 sequence, and the per-class boundaries and counts every
//  segment kernel needs. Every kernel strides over its elements with 64 bit indices, so the launch sizes of the
//  segment family only cap the grid.
//
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/segment.h>
#include <ops/declarable/helpers/segment_common.h>
#include <ops/declarable/helpers/segment_semantics.h>
#include <system/selective_rendering.h>

#include <string>

namespace sd {
namespace ops {
namespace helpers {

namespace {

SD_INLINE int segmentClampInt(LongType value) {
  if (value < 0) return 0;
  if (value > static_cast<LongType>(2147483647)) return 2147483647;
  return static_cast<int>(value);
}

// the launch errors are reported after every enqueue (no synchronization); inside a graph capture there is nothing to
// query
SD_INLINE void segmentCheckLaunch(cudaStream_t* stream, const char* what) {
  if (!DebugHelper::inGraphCapture(stream)) DebugHelper::checkGlobalErrorCode(what);
}

SD_INLINE void segmentCudaCheck(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::string message = std::string(what) + " failed: [" + std::to_string(static_cast<int>(status)) + "] " +
                          cudaGetErrorString(status);
    THROW_EXCEPTION(message.c_str());
  }
}

}  // namespace

// -------------------------------------------------------------------------------------------------------------- //
// ids -> dense int64
// -------------------------------------------------------------------------------------------------------------- //
template <typename I>
static SD_KERNEL void segmentReadIdsKernel(const I* ids, const LongType* idsShapeInfo, LongType n, LongType* dense) {
  const LongType rank = shape::rank(idsShapeInfo);
  const LongType* shp = shape::shapeOf(idsShapeInfo);
  const LongType* str = shape::stride(idsShapeInfo);
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    dense[i] = segment_sem::idToLong<I>(ids[segment_sem::logicalOffset(i, rank, shp, str)]);
  }
}

template <typename I>
static void segmentReadIdsLauncher(LaunchContext* context, NDArray* indices, LongType* dense) {
  const LongType n = indices->lengthOf();
  dim3 dims = segmentValidateIndices(segmentClampInt(n));
  auto stream = context->getCudaStream();
  segmentReadIdsKernel<I><<<dims.x, dims.y, dims.z, *stream>>>(reinterpret_cast<const I*>(indices->specialBuffer()),
                                                                indices->specialShapeInfo(), n, dense);
  segmentCheckLaunch(stream, "segmentReadIdsKernel failed");
}

void segmentReadIds(LaunchContext* context, NDArray* indices, LongType* dense) {
  if (indices->lengthOf() == 0) return;
  BUILD_SINGLE_SELECTOR(indices->dataType(), segmentReadIdsLauncher, (context, indices, dense), SD_INTEGER_TYPES);
}

// -------------------------------------------------------------------------------------------------------------- //
// per class boundaries and counts
// -------------------------------------------------------------------------------------------------------------- //
static SD_KERNEL void segmentBuildRangesKernel(const LongType* ids, LongType n, LongType numClasses, LongType* begin,
                                               LongType* end) {
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType s = ids[i];
    if (s < 0 || s >= numClasses) continue;
    // each boundary of a run is written by exactly one thread, no atomics (sorted ids)
    if (i == 0 || ids[i - 1] != s) begin[s] = i;
    if (i == n - 1 || ids[i + 1] != s) end[s] = i + 1;
  }
}

void segmentBuildRanges(LaunchContext* context, const LongType* ids, LongType n, LongType numClasses, LongType* begin,
                        LongType* end) {
  if (numClasses <= 0) return;
  auto stream = context->getCudaStream();
  segmentCudaCheck(cudaMemsetAsync(begin, 0, static_cast<size_t>(numClasses) * sizeof(LongType), *stream),
                   "segmentBuildRanges memset");
  segmentCudaCheck(cudaMemsetAsync(end, 0, static_cast<size_t>(numClasses) * sizeof(LongType), *stream),
                   "segmentBuildRanges memset");
  if (n <= 0) return;
  dim3 dims = getFillUpSegmentsDims(segmentClampInt(numClasses), segmentClampInt(n));
  segmentBuildRangesKernel<<<dims.x, dims.y, dims.z, *stream>>>(ids, n, numClasses, begin, end);
  segmentCheckLaunch(stream, "segmentBuildRangesKernel failed");
}

static SD_KERNEL void segmentCountIdsKernel(const LongType* ids, LongType n, LongType numClasses,
                                            unsigned long long* counts) {
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType s = ids[i];
    if (s >= 0 && s < numClasses) atomicAdd(counts + s, 1ULL);
  }
}

void segmentCountIds(LaunchContext* context, const LongType* ids, LongType n, LongType numClasses,
                     unsigned long long* counts) {
  if (numClasses <= 0) return;
  auto stream = context->getCudaStream();
  segmentCudaCheck(cudaMemsetAsync(counts, 0, static_cast<size_t>(numClasses) * sizeof(unsigned long long), *stream),
                   "segmentCountIds memset");
  if (n <= 0) return;
  dim3 dims = getFillUpSegmentsDims(segmentClampInt(numClasses), segmentClampInt(n));
  segmentCountIdsKernel<<<dims.x, dims.y, dims.z, *stream>>>(ids, n, numClasses, counts);
  segmentCheckLaunch(stream, "segmentCountIdsKernel failed");
}

// -------------------------------------------------------------------------------------------------------------- //
// validation
// -------------------------------------------------------------------------------------------------------------- //
static SD_KERNEL void segmentCheckSortedKernel(const LongType* ids, LongType n, unsigned long long* firstBad) {
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType v = ids[i];
    if (v < 0 || (i > 0 && v < ids[i - 1])) atomicMin(firstBad, static_cast<unsigned long long>(i));
  }
}

static SD_KERNEL void segmentCheckRangeKernel(const LongType* ids, LongType n, LongType numClasses,
                                              unsigned long long* firstBad) {
  for (LongType i = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<LongType>(gridDim.x) * blockDim.x) {
    const LongType v = ids[i];
    if (v < 0 || v >= numClasses) atomicMin(firstBad, static_cast<unsigned long long>(i));
  }
}

// report = {position of the first violation or -1, id before it, the offending id}
static SD_KERNEL void segmentCheckReportKernel(const LongType* ids, const unsigned long long* firstBad,
                                               LongType* report) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const unsigned long long bad = *firstBad;
  if (bad == ~0ULL) {
    report[0] = -1;
    report[1] = 0;
    report[2] = 0;
  } else {
    report[0] = static_cast<LongType>(bad);
    report[1] = bad > 0 ? ids[bad - 1] : ids[bad];
    report[2] = ids[bad];
  }
}

// Reads the ids and, outside a graph capture, returns the first violation of the check through a stream-ordered
// copy of one small report. Inside a capture the host cannot read anything back; every segment kernel then drops the
// ids that are out of range instead of indexing with them.
static bool segmentIdsCheck(LaunchContext* context, NDArray* indices, bool sorted, LongType numClasses,
                            LongType& previous, LongType& offending) {
  const LongType n = indices->lengthOf();
  if (n == 0) return true;
  auto stream = context->getCudaStream();
  if (DebugHelper::inGraphCapture(stream)) return true;

  PointersManager manager(context, "segmentIdsCheck");
  NDArray::prepareSpecialUse({}, {indices});
  auto* dense = reinterpret_cast<LongType*>(manager.allocateDevMem(static_cast<size_t>(n) * sizeof(LongType)));
  auto* firstBad = reinterpret_cast<unsigned long long*>(manager.allocateDevMem(sizeof(unsigned long long)));
  auto* report = reinterpret_cast<LongType*>(manager.allocateDevMem(3 * sizeof(LongType)));
  segmentCudaCheck(cudaMemsetAsync(firstBad, 0xFF, sizeof(unsigned long long), *stream), "segmentIdsCheck memset");

  segmentReadIds(context, indices, dense);
  dim3 dims = segmentValidateIndices(segmentClampInt(n));
  if (sorted) {
    segmentCheckSortedKernel<<<dims.x, dims.y, dims.z, *stream>>>(dense, n, firstBad);
  } else {
    segmentCheckRangeKernel<<<dims.x, dims.y, dims.z, *stream>>>(dense, n, numClasses, firstBad);
  }
  segmentCheckLaunch(stream, "segment ids check failed");
  segmentCheckReportKernel<<<1, 1, 0, *stream>>>(dense, firstBad, report);
  segmentCheckLaunch(stream, "segmentCheckReportKernel failed");

  LongType host[3] = {-1, 0, 0};
  segmentCudaCheck(cudaMemcpyAsync(host, report, sizeof(host), cudaMemcpyDeviceToHost, *stream),
                   "segmentIdsCheck readback");
  segmentCudaCheck(cudaStreamSynchronize(*stream), "segmentIdsCheck stream synchronization");
  NDArray::registerSpecialUse({}, {indices});
  if (host[0] < 0) return true;
  previous = host[1];
  offending = host[2];
  return false;
}

// Sorted ids: no negative id and no id smaller than the one before it. On failure previous holds the id before the
// offending one (the offending one itself for a negative first id) and offending that id.
bool segmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType& previous, LongType& offending) {
  return segmentIdsCheck(context, indices, true, 0, previous, offending);
}

// Unsorted ids: every id in [0, numOfClasses). On failure output holds the first offending id, on success
// numOfClasses.
bool unsortedSegmentIndicesValidate(LaunchContext* context, NDArray* indices, LongType expected, LongType& output) {
  LongType previous = 0;
  LongType offending = 0;
  if (segmentIdsCheck(context, indices, false, expected, previous, offending)) {
    output = expected;
    return true;
  }
  output = offending;
  return false;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
