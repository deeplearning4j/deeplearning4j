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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
///
///

template <typename T>
SD_KERNEL static void concatCuda(void* pVx, void* pxShapeInfo, void* vz, const sd::LongType* zShapeInfo,
                                 const int axis) {
  T* z = reinterpret_cast<T*>(vz);

  __shared__ LongType zLen, totalThreads;
  __shared__ LongType zRank;
  __shared__ LongType* zShape;
  __shared__ LongType* zStride;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;

    // Cache shape information
    zRank = shape::rank(zShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType coords[SD_MAX_RANK];

  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, zRank, zShape, coords);

    LongType zOffset;
    COORDS2INDEX(zRank, zStride, coords, zOffset);

    int inArrIdx = 0;
    LongType* xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[inArrIdx];

    // Cache the input array's shape information for the current iteration
    LongType xRank = shape::rank(xShapeInfo);
    LongType* xStride = shape::stride(xShapeInfo);

    while (coords[axis] >= xShapeInfo[axis + 1]) {
      coords[axis] -= xShapeInfo[axis + 1];
      xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[++inArrIdx];
      // Update shape information for new input array
      xRank = shape::rank(xShapeInfo);
      xStride = shape::stride(xShapeInfo);
    }

    const auto* x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[inArrIdx]);
    LongType xOffset;
    COORDS2INDEX(xRank, xStride, coords, xOffset);

    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void concatCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, void* pVx, void* pxShapeInfo, void* vz,
                                       const LongType* zShapeInfo, const int axis) {
  concatCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(pVx, pxShapeInfo, vz, zShapeInfo, axis);
  DebugHelper::checkGlobalErrorCode("concat general case failed(...) failed");
}


//////////////////////////////////////////////////////////////////////////
void concat(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int axis) {
  const int numInArrs = inArrs.size();

  // Handle case where there are no input arrays
  if (numInArrs == 0) {
    return;
  }

  // Handle case where output is empty - check both isEmpty() flag AND length
  // Arrays with shape like [0,1] might not have ARRAY_EMPTY flag but still have no data
  if (output.isEmpty() || output.lengthOf() == 0) {
    return;
  }

  // Also check if output buffer is null (defensive check)
  if (output.getDataBuffer() == nullptr) {
    return;
  }

  NDArray::prepareSpecialUse({&output}, inArrs);

  // prepare arrays of pointers on buffers and shapes
  std::vector<const void*> hInBuffers(numInArrs);
  std::vector<const LongType*> hInShapeInfo(numInArrs);
  
  for (int i = 0; i < numInArrs; i++) {
    // Check for empty arrays before accessing specialBuffer to avoid null pointer exception
    // Check both isEmpty() flag AND length AND buffer pointer
    bool isEffectivelyEmpty = inArrs[i]->isEmpty() || inArrs[i]->lengthOf() == 0 || inArrs[i]->getDataBuffer() == nullptr;
    hInBuffers[i] = isEffectivelyEmpty ? nullptr : inArrs[i]->specialBuffer();
    hInShapeInfo[i] = inArrs[i]->specialShapeInfo();
  }

  PointersManager manager(context, "helpers::concat");

  void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));
  void* dInShapeInfo = manager.replicatePointer(hInShapeInfo.data(), hInShapeInfo.size() * sizeof(LongType*));

  dim3 dims = getConcat(output.lengthOf());

  BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher,
                        (dims.x, dims.y, dims.z, context->getCudaStream(), dInBuffers, dInShapeInfo,
                         output.specialBuffer(), output.specialShapeInfo(), axis),
                        SD_COMMON_TYPES);

  manager.synchronize();

  NDArray::registerSpecialUse({&output}, inArrs);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
