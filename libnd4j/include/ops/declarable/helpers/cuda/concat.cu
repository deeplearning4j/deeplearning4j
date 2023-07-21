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
template <typename T>
SD_KERNEL static void concatCuda(void* pVx, void* pxShapeInfo, void* vz, const sd::LongType* zShapeInfo,
                                 const int axis) {
  T* z = reinterpret_cast<T*>(vz);
  __shared__ sd::LongType zLen, totalThreads;
  __shared__ int rank;

  if (threadIdx.x == 0) {
    zLen = shape::length(zShapeInfo);
    rank = shape::rank(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  sd::LongType coords[SD_MAX_RANK];

  for (sd::LongType i = tid; i < zLen; i += totalThreads) {
    shape::index2coords(i, zShapeInfo, coords);

    const auto zOffset = shape::getOffset(zShapeInfo, coords);

    int inArrIdx = 0;
    sd::LongType* xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[inArrIdx];

    while (coords[axis] >= xShapeInfo[axis + 1]) {
      coords[axis] -= xShapeInfo[axis + 1];
      xShapeInfo = reinterpret_cast<sd::LongType**>(pxShapeInfo)[++inArrIdx];
    }

    const auto* x = reinterpret_cast<T*>(reinterpret_cast<void**>(pVx)[inArrIdx]);
    const auto xOffset = shape::getOffset(xShapeInfo, coords);

    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void concatCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                       const cudaStream_t* stream, void* pVx, void* pxShapeInfo, void* vz,
                                       const sd::LongType* zShapeInfo, const int axis) {
  concatCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(pVx, pxShapeInfo, vz, zShapeInfo, axis);
}

//////////////////////////////////////////////////////////////////////////
void concat(sd::LaunchContext* context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int axis) {
  const int numOfInArrs = inArrs.size();
  const auto sizeofT = output.sizeOfT();

  NDArray::prepareSpecialUse({&output}, inArrs);

  bool luckCase1 =
      ((axis == 0 && output.ordering() == 'c') || (axis == output.rankOf() - 1 && output.ordering() == 'f')) &&
      output.ews() == 1;

  if (luckCase1) {
    for (sd::LongType i = 0; i < numOfInArrs; ++i) {
      luckCase1 &= inArrs[i]->ordering() == output.ordering() && inArrs[i]->ews() == 1;
      if (!luckCase1) break;
    }
  }

  if (luckCase1) {  // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6}
                    // order f

    void* z = static_cast<int8_t*>(output.specialBuffer());

    for (sd::LongType i = 0; i < numOfInArrs; ++i) {
      const auto memAmountToCopy = inArrs[i]->lengthOf() * sizeofT;
      cudaMemcpyAsync(z, reinterpret_cast<const int8_t*>(inArrs[i]->specialBuffer()), memAmountToCopy,
                      cudaMemcpyDeviceToDevice, *context->getCudaStream());
      z = static_cast<int8_t*>(z) + memAmountToCopy;
    }

    if (cudaStreamSynchronize(*context->getCudaStream()) != 0)
      THROW_EXCEPTION("concat cuda: luckCase1 failed!");

    for (int i = 0; i < numOfInArrs; ++i) inArrs[i]->tickReadDevice();
    output.tickWriteDevice();

    return;
  }



  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (output.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
  const int sharedMem = 256;

  dim3 dims = getConcat(output.lengthOf());

  // prepare arrays of pointers on buffers and shapes
  std::vector<const void*> hInBuffers(numOfInArrs);
  std::vector<const sd::LongType*> hInShapeInfo(numOfInArrs);

  for (int i = 0; i < numOfInArrs; ++i) {
    hInBuffers[i] = inArrs[i]->specialBuffer();
    hInShapeInfo[i] = inArrs[i]->specialShapeInfo();
  }

  PointersManager manager(context, "helpers::concat");

  void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));
  void* dInShapeInfo = manager.replicatePointer(hInShapeInfo.data(), hInShapeInfo.size() * sizeof(sd::LongType*));

  BUILD_SINGLE_SELECTOR(inArrs[0]->dataType(), concatCudaLauncher,
                        (dims.y,dims.x, sharedMem, context->getCudaStream(), dInBuffers, dInShapeInfo,
                         output.specialBuffer(), output.specialShapeInfo(), axis),
                        SD_COMMON_TYPES);

  manager.synchronize();
  // }

  NDArray::registerSpecialUse({&output}, inArrs);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
