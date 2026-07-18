/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Shared CUDA device-side reduction primitives.
//
// Historically each .cu kernel that needed a warp/block reduction defined its
// own copy (warpReduceSum / blockReduceSum / warpReduceMax ...), distinguished
// only by a name prefix (rms*, mha*, mla*, kvq* ...) to avoid symbol clashes.
// The bodies were byte-identical. This header provides one templated,
// SD_INLINE (so ODR-safe across translation units) implementation of each.
//
// CUDA-only: relies on warp-shuffle intrinsics; include only from .cu files.
//

#ifndef LIBND4J_DEVICE_PRIMITIVES_CUH
#define LIBND4J_DEVICE_PRIMITIVES_CUH

#include <array/DataTypeUtils.h>
#include <math/templatemath.h>
#include <system/common.h>

namespace sd {
namespace device {

// Warp width is 32 on all currently supported NVIDIA/AMD targets.
constexpr int WARP_SIZE = 32;

//////////////////////////////////////////////////////////////////////////////
// Warp-level reductions (full-warp mask). The reduced value is valid in lane 0.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename T>
SD_DEVICE SD_INLINE T warpReduceMax(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val = sd::math::sd_max<T>(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

template <typename T>
SD_DEVICE SD_INLINE T warpReduceMin(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    val = sd::math::sd_min<T>(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level sum reduction. Caller supplies shared-memory scratch of at least
// ceil(blockDim.x / WARP_SIZE) elements. The reduced value is valid in thread 0.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T blockReduceSum(T val, T* sharedMem) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warpReduceSum<T>(val);

  if (lane == 0) sharedMem[wid] = val;
  __syncthreads();

  val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : static_cast<T>(0);
  if (wid == 0) val = warpReduceSum<T>(val);
  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level max reduction. Missing lanes are filled with the type's lowest
// finite value so they never win the max. Result is valid in thread 0.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T blockReduceMax(T val, T* sharedMem) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warpReduceMax<T>(val);

  if (lane == 0) sharedMem[wid] = val;
  __syncthreads();

  val = (threadIdx.x < numWarps) ? sharedMem[threadIdx.x] : -sd::DataTypeUtils::max<T>();
  if (wid == 0) val = warpReduceMax<T>(val);
  return val;
}

//////////////////////////////////////////////////////////////////////////////
// Block-level reduce + broadcast: like blockReduce* but the reduced value is
// returned on EVERY thread (not just thread 0). Collapses the common idiom of
// warp-reduce -> shared-mem cross-warp reduce -> thread-0 store -> broadcast.
// Caller supplies scratch of at least ceil(blockDim.x / WARP_SIZE) elements;
// scratch[0] is reused for the broadcast. Ends with a barrier, so scratch is
// safe for the caller to reuse immediately.
//////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE SD_INLINE T blockAllReduceSum(T val, T* scratch) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warpReduceSum<T>(val);
  if (lane == 0) scratch[wid] = val;
  __syncthreads();

  T result = (threadIdx.x < numWarps) ? scratch[threadIdx.x] : static_cast<T>(0);
  if (wid == 0) result = warpReduceSum<T>(result);
  __syncthreads();

  if (threadIdx.x == 0) scratch[0] = result;
  __syncthreads();
  result = scratch[0];
  __syncthreads();
  return result;
}

template <typename T>
SD_DEVICE SD_INLINE T blockAllReduceMax(T val, T* scratch) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  val = warpReduceMax<T>(val);
  if (lane == 0) scratch[wid] = val;
  __syncthreads();

  T result = (threadIdx.x < numWarps) ? scratch[threadIdx.x] : -sd::DataTypeUtils::max<T>();
  if (wid == 0) result = warpReduceMax<T>(result);
  __syncthreads();

  if (threadIdx.x == 0) scratch[0] = result;
  __syncthreads();
  result = scratch[0];
  __syncthreads();
  return result;
}

}  // namespace device
}  // namespace sd

#endif  // LIBND4J_DEVICE_PRIMITIVES_CUH
