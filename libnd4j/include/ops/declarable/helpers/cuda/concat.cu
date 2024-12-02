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
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
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

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType coords[SD_MAX_RANK];

  for (LongType i = tid; i < zLen; i += totalThreads) {
    INDEX2COORDS(i, shape::rank(zShapeInfo), shape::shapeOf(zShapeInfo), coords);

    LongType zOffset;
    COORDS2INDEX(shape::rank(zShapeInfo), shape::stride(zShapeInfo), coords, zOffset);

    int inArrIdx = 0;
    LongType* xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[inArrIdx];

    while (coords[axis] >= xShapeInfo[axis + 1]) {
      coords[axis] -= xShapeInfo[axis + 1];
      xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[++inArrIdx];
    }

    const auto* x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[inArrIdx]);
    LongType xOffset;
    COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), coords, xOffset);

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

  NDArray::prepareSpecialUse({&output}, inArrs);

  bool luckCase1 =
      ((axis == 0 && output.ordering() == 'c') || (axis == output.rankOf() - 1 && output.ordering() == 'f')) &&
          output.ews() == 1 ||
      inArrs[0]->lengthOf() < 1;

  if (luckCase1) {
    for (LongType i = 0; i < numInArrs; ++i) {
      luckCase1 &= inArrs[i]->ordering() == output.ordering() && inArrs[i]->ews() == 1;
      if (!luckCase1) break;
    }
  }

  // prepare arrays of pointers on buffers and shapes
  std::vector<const void*> hInBuffers(numInArrs);
  std::vector<const LongType*> hInShapeInfo(numInArrs);
 std::vector <int> lenPerArray(numInArrs);
  for (int i = 0; i < numInArrs; i++) {
    hInBuffers[i] = inArrs[i]->specialBuffer();
    hInShapeInfo[i] = inArrs[i]->specialShapeInfo();
    lenPerArray[i] = inArrs[i]->isEmpty() ? 0 : inArrs[i]->isScalar() ? 1 : inArrs[i]->lengthOf();
  }

  PointersManager manager(context, "helpers::concat");

  void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));

  dim3 dims = getConcat(output.lengthOf());

  if (luckCase1) {  // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6}
    void* z = static_cast<int8_t*>(output.specialBuffer());

    for (sd::LongType i = 0; i < numInArrs; ++i) {
      const auto sizeofT = output.sizeOfT();
      const auto memAmountToCopy = inArrs[i]->lengthOf() * sizeofT;
      cudaMemcpyAsync(z, reinterpret_cast<const int8_t*>(inArrs[i]->specialBuffer()), memAmountToCopy,
                      cudaMemcpyDeviceToDevice, *context->getCudaStream());
      z = static_cast<int8_t*>(z) + memAmountToCopy;
    }

    if (cudaStreamSynchronize(*context->getCudaStream()) != 0)
      THROW_EXCEPTION("concat cuda: luckCase1 failed!");

    for (int i = 0; i < numInArrs; ++i) inArrs[i]->tickReadDevice();
    output.tickWriteDevice();
    manager.synchronize();
    output.syncToHost();
    return;
  }

  void* dInShapeInfo = manager.replicatePointer(hInShapeInfo.data(), hInShapeInfo.size() * sizeof(LongType*));

  BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher,
                        (dims.x, dims.y, dims.z, context->getCudaStream(), dInBuffers, dInShapeInfo,
                         output.specialBuffer(), output.specialShapeInfo(), axis),
                        SD_COMMON_TYPES);

  manager.synchronize();
  manager.synchronize();
  output.syncToHost();
  NDArray::registerSpecialUse({&output}, inArrs);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
