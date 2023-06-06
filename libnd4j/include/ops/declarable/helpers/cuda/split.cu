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
// @author Yurii Shyrma (iuriish@yahoo.com)
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

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void splitCuda(const void* vx, const sd::LongType* xShapeInfo, void* pVz,
                                const sd::LongType* zTadShapeInfo, const int axis) {
  const T* x = reinterpret_cast<const T*>(vx);

  __shared__ sd::LongType xLen, totalThreads;
  __shared__ int xRank, zDim;

  if (threadIdx.x == 0) {
    xLen = shape::length(xShapeInfo);
    xRank = shape::rank(xShapeInfo);
    zDim = shape::shapeOf(zTadShapeInfo)[axis];  // same for all input arrays
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  sd::LongType coords[SD_MAX_RANK];

  for (sd::LongType i = tid; i < xLen; i += totalThreads) {
    shape::index2coords(i, xShapeInfo, coords);

    const sd::LongType xOffset = shape::getOffset(xShapeInfo, coords);

    auto* z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[coords[axis] / zDim]);

    coords[axis] %= zDim;

    const sd::LongType zOffset = shape::getOffset(zTadShapeInfo, coords);

    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void splitCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream,
                                      const void* vx, const sd::LongType* xShapeInfo, void* pVz,
                                      const sd::LongType* zTadShapeInfo, const int axis) {
  splitCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz, zTadShapeInfo, axis);
}
BUILD_SINGLE_TEMPLATE(template void splitCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx,
                       const sd::LongType* xShapeInfo, void* pVz, const sd::LongType* zTadShapeInfo, const int axis),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void split(sd::LaunchContext* context, const NDArray& input, std::vector<NDArray*>& outArrs, const int axis) {
  const int numOfSubArrs = outArrs.size();
  const auto sizeofT = input.sizeOfT();

  for (int i = 0; i < numOfSubArrs; ++i) outArrs[i]->syncToDevice();
  input.syncToDevice();

  bool luckCase1 =
      ((axis == 0 && input.ordering() == 'c') || (axis == input.rankOf() - 1 && input.ordering() == 'f')) &&
      input.ews() == 1;

  if (luckCase1) {
    for (sd::LongType i = 0; i < numOfSubArrs; ++i) {
      luckCase1 &= outArrs[i]->ordering() == input.ordering() && outArrs[i]->ews() == 1;
      if (!luckCase1) break;
    }
  }

  if (luckCase1) {  // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6}
                    // order f

    auto x = static_cast<const int8_t*>(input.specialBuffer());

    for (sd::LongType i = 0; i < numOfSubArrs; ++i) {
      const auto memAmountToCopy = outArrs[i]->lengthOf() * sizeofT;
      cudaMemcpyAsync(static_cast<int8_t*>(outArrs[i]->specialBuffer()), x, memAmountToCopy, cudaMemcpyDeviceToDevice,
                      *context->getCudaStream());
      x = static_cast<const int8_t*>(x) + memAmountToCopy;
    }

    if (cudaStreamSynchronize(*context->getCudaStream()) != 0)
      THROW_EXCEPTION("split cuda: luckCase1 failed!");

    for (int i = 0; i < numOfSubArrs; ++i) outArrs[i]->tickWriteDevice();
    input.tickReadDevice();

    return;
  }



  const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
  const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

  // prepare arrays of pointers on buffers and shapes
  std::vector<void*> hOutBuffers(numOfSubArrs);

  for (int i = 0; i < numOfSubArrs; ++i) hOutBuffers[i] = outArrs[i]->specialBuffer();

  PointersManager manager(context, "helpers::split");

  void* dOutBuffers = manager.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void*));

  BUILD_SINGLE_SELECTOR(input.dataType(), splitCudaLauncher,
                        (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(),
                         input.specialShapeInfo(), dOutBuffers, outArrs[0]->specialShapeInfo(), axis),
                        SD_COMMON_TYPES);

  manager.synchronize();
  // }

  for (int i = 0; i < numOfSubArrs; ++i) outArrs[i]->tickWriteDevice();
  input.tickReadDevice();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
