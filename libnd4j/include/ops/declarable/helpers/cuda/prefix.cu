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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.06.2019
//
#include <helpers/ConstantTadHelper.h>
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/prefix.h>
#include <ops/ops.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// Cumulative sums and products on the device: one block per scanned sequence (a TAD along the dimensions, or the
// whole array in C order), walking it in chunks of blockDim elements. Each chunk is scanned in shared memory (a
// barrier-separated Hillis-Steele inclusive scan) and combined with the carry of the chunks before it, so a sequence
// of any length takes ceil(length / blockDim) steps of O(log blockDim) each. `exclusive` gives each element the
// combination of the elements strictly before it in scan order, `reverse` scans from the end. Every element of a
// chunk is read before any is written, which keeps the in-place form (z == x) correct. Element offsets come from each
// operand's own shape and strides, so views and differing layouts pair by logical position.
template <typename T>
SD_DEVICE SD_INLINE T prefixCombine(const scalar::Ops op, const T a, const T b) {
  return op == scalar::Add ? static_cast<T>(a + b) : static_cast<T>(a * b);
}

template <typename T>
SD_KERNEL static void prefixScanCuda(const scalar::Ops op, const void* vx, const LongType* xTadShapeInfo,
                                     const LongType* xTadOffsets, void* vz, const LongType* zTadShapeInfo,
                                     const LongType* zTadOffsets, const LongType numTads, const LongType tadLen,
                                     const bool exclusive, const bool reverse) {
  // blockDim scan slots and, after them, the carry of the chunks done so far (dynamic shared memory: the 16-bit
  // float types are classes, which static __shared__ declarations do not take)
  extern __shared__ unsigned char prefixShared[];
  T* scan = reinterpret_cast<T*>(prefixShared);
  T& carry = scan[blockDim.x];

  const auto x = reinterpret_cast<const T*>(vx);
  auto z = reinterpret_cast<T*>(vz);
  const T identity = op == scalar::Add ? static_cast<T>(0) : static_cast<T>(1);

  const auto xRank = shape::rank(xTadShapeInfo);
  const auto xShape = shape::shapeOf(xTadShapeInfo);
  const auto xStride = shape::stride(xTadShapeInfo);
  const auto zRank = shape::rank(zTadShapeInfo);
  const auto zShape = shape::shapeOf(zTadShapeInfo);
  const auto zStride = shape::stride(zTadShapeInfo);

  const LongType lane = threadIdx.x;
  const LongType width = blockDim.x;

  for (LongType tad = blockIdx.x; tad < numTads; tad += gridDim.x) {
    const T* xTad = x + (xTadOffsets != nullptr ? xTadOffsets[tad] : 0);
    T* zTad = z + (zTadOffsets != nullptr ? zTadOffsets[tad] : 0);

    if (lane == 0) carry = identity;
    __syncthreads();

    for (LongType chunk = 0; chunk < tadLen; chunk += width) {
      // position in scan order, and the element it names
      const LongType position = chunk + lane;
      const bool valid = position < tadLen;
      const LongType element = reverse ? tadLen - 1 - position : position;

      T value = identity;
      if (valid) {
        LongType coords[SD_MAX_RANK];
        LongType xOffset;
        INDEX2COORDS(element, xRank, xShape, coords);
        COORDS2INDEX(xRank, xStride, coords, xOffset);
        value = xTad[xOffset];
      }
      scan[lane] = value;
      __syncthreads();

      // inclusive scan of the chunk: after the step with `offset`, scan[lane] combines the 2 * offset elements
      // ending at lane (fewer at the start)
      for (LongType offset = 1; offset < width; offset <<= 1) {
        const T before = lane >= offset ? scan[lane - offset] : identity;
        __syncthreads();
        scan[lane] = prefixCombine(op, before, scan[lane]);
        __syncthreads();
      }

      if (valid) {
        const T result = exclusive ? (lane == 0 ? carry : prefixCombine(op, carry, scan[lane - 1]))
                                   : prefixCombine(op, carry, scan[lane]);
        LongType coords[SD_MAX_RANK];
        LongType zOffset;
        INDEX2COORDS(element, zRank, zShape, coords);
        COORDS2INDEX(zRank, zStride, coords, zOffset);
        zTad[zOffset] = result;
      }
      __syncthreads();

      // the chunk's total joins the carry (lanes past the end held the identity)
      if (lane == 0) carry = prefixCombine(op, carry, scan[width - 1]);
      __syncthreads();
    }
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void prefixScanLauncher(LaunchContext* context, const scalar::Ops op, NDArray* x, const LongType* xTadShapeInfo,
                               const LongType* xTadOffsets, NDArray* z, const LongType* zTadShapeInfo,
                               const LongType* zTadOffsets, const LongType numTads, const LongType tadLen,
                               const bool exclusive, const bool reverse) {
  if (numTads <= 0 || tadLen <= 0) return;

  // One block per sequence; a short sequence gets no more lanes than it fills (rounded up to whole warps)
  dim3 launchDims = prefixDims(static_cast<int>(sd::math::sd_min<LongType>(numTads, 65535)), sizeof(T));
  LongType lanes = sd::math::sd_min<LongType>(static_cast<LongType>(launchDims.y), tadLen);
  lanes = sd::math::sd_max<LongType>(32, ((lanes + 31) / 32) * 32);
  lanes = sd::math::sd_min<LongType>(lanes, SD_MAX_NUM_THREADS);
  const int sharedBytes = static_cast<int>((lanes + 1) * sizeof(T));

  prefixScanCuda<T><<<launchDims.x, lanes, sharedBytes, *context->getCudaStream()>>>(
      op, x->specialBuffer(), xTadShapeInfo, xTadOffsets, z->specialBuffer(), zTadShapeInfo, zTadOffsets, numTads,
      tadLen, exclusive, reverse);
  sd::DebugHelper::checkGlobalErrorCode("prefix scan failed");
}

///////////////////////////////////////////////////////////////////
void prefix(LaunchContext* context, scalar::Ops op, NDArray* x, NDArray* z, bool exclusive, bool reverse) {
  // The whole array is one sequence, in C order over its shape
  NDArray::prepareSpecialUse({z}, {x});
  BUILD_SINGLE_SELECTOR(x->dataType(), prefixScanLauncher,
                        (context, op, x, x->specialShapeInfo(), nullptr, z, z->specialShapeInfo(), nullptr, 1,
                         x->lengthOf(), exclusive, reverse),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({z}, {x});
}

void prefix(LaunchContext* context, scalar::Ops op, NDArray* x, NDArray* z, const std::vector<LongType>& dims,
            bool exclusive, bool reverse) {
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x->shapeInfo(), const_cast<std::vector<LongType>*>(&dims));
  auto packZ = ConstantTadHelper::getInstance().tadForDimensions(z->shapeInfo(), const_cast<std::vector<LongType>*>(&dims));
  const LongType numTads = packX->numberOfTads();
  const LongType tadLen = numTads > 0 ? x->lengthOf() / numTads : 0;

  NDArray::prepareSpecialUse({z}, {x});
  BUILD_SINGLE_SELECTOR(x->dataType(), prefixScanLauncher,
                        (context, op, x, packX->platformShapeInfo(), packX->platformOffsets(), z,
                         packZ->platformShapeInfo(), packZ->platformOffsets(), numTads, tadLen, exclusive, reverse),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({z}, {x});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
