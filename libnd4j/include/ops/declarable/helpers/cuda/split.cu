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

#include <ops/declarable/helpers/transforms.h>

#include <numeric>


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void splitCuda(const void* vx, const LongType* xShapeInfo, void* pVz,
                                 const LongType* zTadShapeInfo, const LongType axis) {
  const T* x = reinterpret_cast<const T*>(vx);

  // Shared memory for caching shape information and related variables
  extern __shared__ unsigned char shmem[];
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_xLen;
  __shared__ LongType shared_totalThreads;
  __shared__ int shared_xRank;
  __shared__ LongType shared_zDim;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_xShape;
  __shared__ const LongType* shared_xStride;
  __shared__ const LongType* shared_zTadShape;
  __shared__ const LongType* shared_zTadStride;
  __shared__ int shared_zTadRank;

  if (threadIdx.x == 0) {
    // Cache shape and stride information for xShapeInfo
    shared_xRank = shape::rank(xShapeInfo);
    shared_xShape = shape::shapeOf(xShapeInfo);
    shared_xStride = shape::stride(xShapeInfo);

    // Cache shape and stride information for zTadShapeInfo
    shared_zTadRank = shape::rank(zTadShapeInfo);
    shared_zTadShape = shape::shapeOf(zTadShapeInfo);
    shared_zTadStride = shape::stride(zTadShapeInfo);
    shared_zDim = shared_zTadShape[axis];  // Assuming zDim is constant across splits

    // Cache length and total threads
    shared_xLen = shape::length(xShapeInfo);
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates
  LongType* coords = sharedMem + threadIdx.x * shared_xRank;

  for (LongType i = tid; i < shared_xLen; i += shared_totalThreads) {
    // Convert linear index to multi-dimensional coordinates
    INDEX2COORDS(i, shared_xRank, shared_xShape, coords);

    LongType xOffset;
    // Convert coordinates to linear index for x
    COORDS2INDEX(shared_xRank, shared_xStride, coords, xOffset);

    // Determine the split index along the specified axis
    LongType splitIndex = coords[axis] / shared_zDim;

    // Retrieve the pointer to the target output tensor
    T* z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[splitIndex]);

    // Update the coordinate along the split axis
    coords[axis] %= shared_zDim;

    LongType zOffset;
    // Convert updated coordinates to linear index for z
    COORDS2INDEX(shared_zTadRank, shared_zTadStride, coords, zOffset);

    // Perform the split operation
    z[zOffset] = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void splitCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream,
                                      const void* vx, const LongType* xShapeInfo, void* pVz,
                                      const LongType* zTadShapeInfo, const LongType axis) {
  splitCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz, zTadShapeInfo, axis);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "splitCuda failed");

}
BUILD_SINGLE_TEMPLATE(template void splitCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx,
                       const sd::LongType* xShapeInfo, void* pVz, const sd::LongType* zTadShapeInfo, const sd::LongType axis),
                      SD_COMMON_TYPES);

//////////////////////////////////////////////////////////////////////////
void split(LaunchContext* context, NDArray& input, std::vector<NDArray*>& outArrs, const LongType axis) {
  const int numOfSubArrs = outArrs.size();
  const auto sizeofT = input.sizeOfT();

  for (int i = 0; i < numOfSubArrs; ++i) outArrs[i]->syncToDevice();
  input.syncToDevice();

  bool luckCase1 = false;

  if (luckCase1) {
    for (LongType i = 0; i < numOfSubArrs; ++i) {
      luckCase1 &= outArrs[i]->ordering() == input.ordering();
      if (!luckCase1) break;
    }
  }

  if (luckCase1) {  // for example {1,10} + {2,10} + {3,10} = {6, 10} order c; or {10,1} + {10,2} + {10,3} = {10, 6}
                    // order f

    auto x = static_cast<const int8_t*>(input.specialBuffer());

    for (LongType i = 0; i < numOfSubArrs; ++i) {
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
