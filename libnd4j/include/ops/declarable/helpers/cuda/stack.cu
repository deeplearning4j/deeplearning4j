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
// Created by Yurii Shyrma on 02.01.2018
//
#include <array/ResultSet.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/stack.h>

#include "execution/cuda/LaunchDims.h"
#include <legacy/NativeOpExecutioner.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void stackScalarsCuda(void* pVx, void* vz, const LongType* zShapeInfo) {
  T* z = reinterpret_cast<T*>(vz);

  // Shared memory for caching shape information of z
  __shared__ LongType shared_zRank;
  __shared__ const LongType* shared_zShape;
  __shared__ const LongType* shared_zStride;

  __shared__ LongType zLen;
  __shared__ LongType totalThreads;

  // Initialize shared memory with shape information and other parameters
  if (threadIdx.x == 0) {
    // Cache the rank of the output tensor
    shared_zRank = shape::rank(zShapeInfo);

    // Cache the shape and stride pointers of the output tensor
    shared_zShape = shape::shapeOf(zShapeInfo);
    shared_zStride = shape::stride(zShapeInfo);

    // Cache the total length of the output tensor
    zLen = shape::length(zShapeInfo);

    // Calculate the total number of threads across all blocks
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads(); // Ensure all threads have access to the cached values

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Temporary variables for coordinates and offset
  LongType zCoords[SD_MAX_RANK];
  LongType zOffset;

  // Iterate over the elements assigned to this thread
  for (LongType i = tid; i < zLen; i += totalThreads) {
    // Retrieve the pointer to the input scalar
    const T* x = reinterpret_cast<const T*>(reinterpret_cast<void**>(pVx)[i]);

    // Convert the linear index 'i' to multi-dimensional coordinates using cached shape
    INDEX2COORDS(i, shared_zRank, shared_zShape, zCoords);

    // Convert the multi-dimensional coordinates back to a linear index using cached stride
    COORDS2INDEX(shared_zRank, shared_zStride, zCoords, zOffset);

    // Assign the scalar value to the output tensor at the computed offset
    z[zOffset] = *x;
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void stackScalarsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                             const cudaStream_t* stream, void* pVx, void* vz,
                                             const LongType* zShapeInfo) {
  stackScalarsCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(pVx, vz, zShapeInfo);
  DebugHelper::checkGlobalErrorCode("stackScalar failed(...) failed");
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void stack_(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output,
                   const int dim) {
  const int numOfSubArrs = inArrs.size();

  NDArray::prepareSpecialUse({&output}, inArrs);

  if (inArrs[0]->rankOf() < 1 && !inArrs[0]->isEmpty()) {
    std::vector<void *> hInBuffers(numOfSubArrs);

    for (int i = 0; i < numOfSubArrs; ++i) hInBuffers[i] = inArrs[i]->specialBuffer();

    PointersManager manager(context, "helpers::stack cuda");

    void* dInBuffers = manager.replicatePointer(hInBuffers.data(), hInBuffers.size() * sizeof(void*));

    dim3 stackDims2 = stackDims(output.lengthOf());
    stackScalarsCudaLauncher<T>(stackDims2.y, stackDims2.x, stackDims2.z, context->getCudaStream(), dInBuffers,
                                output.specialBuffer(), output.specialShapeInfo());

    manager.synchronize();
  } else if (!inArrs[0]->isEmpty()) {
    std::vector<LongType> dims = {dim};
    auto zTadPack = ConstantTadHelper::getInstance().tadForDimensions(
        output.shapeInfo(), ShapeUtils::evalDimsToExclude(output.rankOf(),1, dims.data()));
    auto zTadShapeInfo = zTadPack->primaryShapeInfo();

    for (LongType i = 0; i < numOfSubArrs; ++i) {
      void* zBuff = const_cast<void*>(output.specialBufferWithOffset(zTadPack->primaryOffsets()[i]));

      NativeOpExecutioner::execTransformAny(context, transform::Assign, nullptr, inArrs[i]->shapeInfo(),
                                            inArrs[i]->specialBuffer(), inArrs[i]->specialShapeInfo(), nullptr,
                                            zTadShapeInfo, zBuff, zTadPack->specialShapeInfo(),
                                            nullptr,
                                            false);
    }
  }

  NDArray::registerSpecialUse({&output}, inArrs);
}

////////////////////////////////////////////////////////////////////////
void stack(LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output, const int dim) {
  BUILD_SINGLE_SELECTOR(output.dataType(), stack_, (context, inArrs, output, dim), SD_COMMON_TYPES);
}
BUILD_SINGLE_TEMPLATE( void stack_,
                      (LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& output,
                          const int dim),
                      SD_COMMON_TYPES);

///////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void unstackScalarsCuda(const void* vx, const LongType* xShapeInfo, void* pVz) {
  const T* x = reinterpret_cast<const T*>(vx);

  // Shared memory for caching shape information
  __shared__ LongType shared_xRank;
  __shared__ const LongType* shared_xShape;
  __shared__ const LongType* shared_xStride;

  __shared__ LongType xLen;
  __shared__ LongType totalThreads;

  // Initialize shared memory with shape information and other parameters
  if (threadIdx.x == 0) {
    // Cache the rank of the input tensor
    shared_xRank = shape::rank(xShapeInfo);

    // Cache the shape and stride pointers
    shared_xShape = shape::shapeOf(xShapeInfo);
    shared_xStride = shape::stride(xShapeInfo);

    // Cache the total length of the input tensor
    xLen = shape::length(xShapeInfo);

    // Calculate the total number of threads across all blocks
    totalThreads = gridDim.x * blockDim.x;
  }
  __syncthreads(); // Ensure all threads have access to the cached values

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Temporary variables for coordinates and offset
  LongType xCoords[SD_MAX_RANK];
  LongType xOffset;

  // Iterate over the elements assigned to this thread
  for (LongType i = tid; i < xLen; i += totalThreads) {
    // Retrieve the pointer to the output location
    T* z = reinterpret_cast<T*>(reinterpret_cast<void**>(pVz)[i]);

    // Convert the linear index to multi-dimensional coordinates using cached shape
    INDEX2COORDS(i, shared_xRank, shared_xShape, xCoords);

    // Convert the multi-dimensional coordinates back to a linear index using cached stride
    COORDS2INDEX(shared_xRank, shared_xStride, xCoords, xOffset);

    // Assign the value from the input tensor to the output location
    *z = x[xOffset];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void unstackScalarsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock,
                                               const cudaStream_t* stream, const void* vx,
                                               const LongType* xShapeInfo, void* pVz) {
  unstackScalarsCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, pVz);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "unstackScalarsCudaLauncher failed");

}

///////////////////////////////////////////////////////////////////
template <typename T>
static void unstack_(LaunchContext* context, NDArray& input, const std::vector<NDArray*>& outArrs,
                     const int dim) {
  const int numOfSubArrs = outArrs.size();

  input.syncToDevice();
  for (const auto a : outArrs) a->getDataBuffer()->allocateSpecial();

  if (outArrs[0]->rankOf() == 0) {
    std::vector<void*> hOutBuffers(numOfSubArrs);

    for (int i = 0; i < numOfSubArrs; ++i) hOutBuffers[i] = outArrs[i]->specialBuffer();

    PointersManager manager(context, "helpers::unstack cuda");

    void* dOutBuffers = manager.replicatePointer(hOutBuffers.data(), hOutBuffers.size() * sizeof(void*));

    const int threadsPerBlock = SD_MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    unstackScalarsCudaLauncher<T>(blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(),
                                  input.specialShapeInfo(), dOutBuffers);

    manager.synchronize();
  } else {
    std::vector<LongType> dims = {dim};
    auto xTadPack = ConstantTadHelper::getInstance().tadForDimensions(
        input.shapeInfo(), ShapeUtils::evalDimsToExclude(input.rankOf(), 1,dims.data()));
    auto xTadShapeInfo = xTadPack->primaryShapeInfo();

    for (LongType i = 0; i < numOfSubArrs; ++i) {
      auto xBuff = input.specialBufferWithOffset(xTadPack->primaryOffsets()[i]);

      NativeOpExecutioner::execTransformAny(input.getContext(), transform::Assign, nullptr, xTadShapeInfo, xBuff,
                                            xTadPack->specialShapeInfo(), nullptr, outArrs[i]->shapeInfo(),
                                            outArrs[i]->specialBuffer(), outArrs[i]->specialShapeInfo(), nullptr,
                                            false);
    }
  }

   NDArray::registerSpecialUse(outArrs, {&input});
  input.tickReadDevice();
  for (const auto p : outArrs) p->tickWriteDevice();
}

////////////////////////////////////////////////////////////////////////
void unstack(LaunchContext* context, NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {
  BUILD_SINGLE_SELECTOR(input.dataType(), unstack_, (context, input, outArrs, dim), SD_COMMON_TYPES);
}
BUILD_SINGLE_TEMPLATE( void unstack_,
                      (LaunchContext * context, NDArray& input, const std::vector<NDArray*>& outArrs,
                          const int dim),
                      SD_COMMON_TYPES);



}  // namespace helpers
}  // namespace ops
}  // namespace sd
