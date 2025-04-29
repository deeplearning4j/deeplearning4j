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
// Created by raver119 on 24/09/18.
//
#include <array/NDArrayList.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/where.h>
#include <exceptions/cuda_exception.h>
#include "helpers/DebugHelper.h"
namespace sd {
namespace ops {
namespace helpers {

// First pass: count true elements
template <typename T>
static SD_KERNEL void countTrueElements(const void *vx, const LongType *xShapeInfo, int *count) {
 const T *x = reinterpret_cast<const T *>(vx);

 __shared__ LongType xRank;
 __shared__ LongType length;
 __shared__ int counter;
 __shared__ const LongType *xShape;
 __shared__ const LongType *xStride;

 if (threadIdx.x == 0) {
   xRank = shape::rank(xShapeInfo);
   length = shape::length(xShapeInfo);
   xShape = shape::shapeOf(xShapeInfo);
   xStride = shape::stride(xShapeInfo);
   counter = 0;
 }
 __syncthreads();

 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
   LongType xCoords[SD_MAX_RANK];
   LongType xOffset;

   INDEX2COORDS(i, xRank, xShape, xCoords);
   COORDS2INDEX(xRank, xStride, xCoords, xOffset);

   if (x[xOffset] > static_cast<T>(0)) {
     atomicAdd(&counter, 1);
   }
 }

 __syncthreads();
 if (threadIdx.x == 0) {
   *count = counter;
 }
}

// Second pass: fill in the indices
template <typename T, typename Z>
static SD_KERNEL void fillIndices(const void *vx, const LongType *xShapeInfo, void *vz, const LongType *zShapeInfo) {
 const T *x = reinterpret_cast<const T *>(vx);
 Z *z = reinterpret_cast<Z *>(vz);

 __shared__ LongType xRank;
 __shared__ LongType length;
 __shared__ LongType zRank;
 __shared__ const LongType *xShape;
 __shared__ const LongType *xStride;
 __shared__ const LongType *zShape;
 __shared__ const LongType *zStride;
 __shared__ int counter;

 if (threadIdx.x == 0) {
   xRank = shape::rank(xShapeInfo);
   length = shape::length(xShapeInfo);
   zRank = shape::rank(zShapeInfo);
   xShape = shape::shapeOf(xShapeInfo);
   xStride = shape::stride(xShapeInfo);
   zShape = shape::shapeOf(zShapeInfo);
   zStride = shape::stride(zShapeInfo);
   counter = 0;
 }
 __syncthreads();

 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 for (LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
   LongType xCoords[SD_MAX_RANK];
   LongType xOffset;

   INDEX2COORDS(i, xRank, xShape, xCoords);
   COORDS2INDEX(xRank, xStride, xCoords, xOffset);

   if (x[xOffset] > static_cast<T>(0)) {
     LongType idx = atomicAdd(&counter, 1);

     // Write each coordinate to the output array
     for (int j = 0; j < xRank; j++) {
       // For output array at position (idx, j)
       LongType zCoords[SD_MAX_RANK];
       LongType zOffset;

       // Set coordinates for 2D output array [idx, j]
       zCoords[0] = idx;
       zCoords[1] = j;

       COORDS2INDEX(zRank, zStride, zCoords, zOffset);

       z[zOffset] = static_cast<Z>(xCoords[j]);
     }
   }
 }
}

template <typename T>
static void whereKernelLauncher(LaunchContext *context, NDArray &condition, NDArray &output) {
 int blockSize = SD_MAX_NUM_THREADS / 2;
 int gridSize = (condition.lengthOf() + blockSize - 1) / blockSize;

 PointersManager manager(context, "whereKernelLauncher");

 // First step: count true elements
 int *dCount = reinterpret_cast<int *>(manager.allocateDeviceMemory(sizeof(int)));
 int zero = 0;
 manager.memcpyToDevice(&zero, dCount, sizeof(int));

 countTrueElements<T><<<gridSize, blockSize, 512, *context->getCudaStream()>>>(
     condition.specialBuffer(), condition.specialShapeInfo(), dCount);

 // Wait for count to complete
 int hCount;
 manager.memcpyToHost(dCount, &hCount, sizeof(int));

 if (hCount == 0) {
   // No true elements, return empty array
   std::vector<LongType> emptyShape = {0, condition.rankOf()};
   auto emptyOutput = NDArrayFactory::create_('c', emptyShape, output.dataType(), output.getContext());
   output.assign(emptyOutput);
   delete emptyOutput;
 } else {
   // Allocate output array
   std::vector<LongType> outShape = {hCount, condition.rankOf()};
   auto temp = NDArrayFactory::create_('c', outShape, output.dataType(), output.getContext());

   // Now fill the indices
   switch (output.dataType()) {
     case sd::DataType::INT32: {
       fillIndices<T, int><<<gridSize, blockSize, 512, *context->getCudaStream()>>>(
           condition.specialBuffer(), condition.specialShapeInfo(),
           temp->specialBuffer(), temp->specialShapeInfo());
       break;
     }
     case sd::DataType::INT64: {
       fillIndices<T, LongType><<<gridSize, blockSize, 512, *context->getCudaStream()>>>(
           condition.specialBuffer(), condition.specialShapeInfo(),
           temp->specialBuffer(), temp->specialShapeInfo());
       break;
     }
     default: {
       fillIndices<T, float><<<gridSize, blockSize, 512, *context->getCudaStream()>>>(
           condition.specialBuffer(), condition.specialShapeInfo(),
           temp->specialBuffer(), temp->specialShapeInfo());
       break;
     }
   }

   output.assign(temp);
   delete temp;
 }

 manager.synchronize();
 sd::DebugHelper::checkErrorCode(context->getCudaStream(), "whereKernelLauncher failed");
}

void _where(LaunchContext *context, NDArray &condition, NDArray &output, memory::Workspace *workspace) {
 NDArray::prepareSpecialUse({&output}, {&condition});

 BUILD_SINGLE_SELECTOR(condition.dataType(), whereKernelLauncher, (context, condition, output), SD_NUMERIC_TYPES);

 NDArray::registerSpecialUse({&output}, {&condition});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd