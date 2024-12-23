/******************************************************************************
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
// @author raver119@gmail.com
//
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <loops/transform_same.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <execution/cuda/DeviceValidator.h>

using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
// Cached kernel that caches shape info in shared memory and uses cached variables
template <typename X, typename OpType>
__global__ void transformSameSimpleCached(
   const void* x,
   const sd::LongType* xShapeInfo,
   sd::LongType xRank,
   void* params,
   void* z,
   const sd::LongType* zShapeInfo,
   sd::LongType zRank,
   sd::LongType* allocationPointer,
   void* reductionPointer,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets)
{
 // Delegate the transform to the transformCuda method with cached shape info
 functions::transform::TransformSame<X>::template transformCuda<OpType>(
     x, xShapeInfo, params, z, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
}

namespace functions {
namespace transform {

////////////////////////////////////////////////////////////////////////////////
// Implementation of the "executeTransformShaped" that launches the cached kernel
template <typename X, typename Y>
SD_HOST void TransformSame<X, Y>::executeTransformShaped(
   dim3 launchDims,
   cudaStream_t* stream,
   const int opNum,
   const void* x,
   const sd::LongType* xShape,
   sd::LongType xRank,
   void* extraParams,
   void* z,
   const sd::LongType* zShape,
   sd::LongType zRank,
   sd::LongType* allocationPointer,
   void* reductionPointer,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets)
{
 DISPATCH_BY_OPNUM_T(
     intermediateShaped,
     PARAMS(launchDims, stream, x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer,
            reductionPointer, tadShapeInfo, tadOffsets),
     TRANSFORM_SAME_OPS);

 sd::DebugHelper::checkErrorCode(stream, "transformSame executeTransformShaped(...) failed");
}

////////////////////////////////////////////////////////////////////////////////
// Device function that caches shape info and uses cached variables for computations
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void TransformSame<X, Z>::transformCuda(
   const void* vx,
   const sd::LongType* xShapeInfo,
   void* vparams,
   void* vz,
   const sd::LongType* zShapeInfo,
   sd::LongType* allocationPointer,
   void* vreductionPointer,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets)
{
 // Cast pointers to appropriate types
 auto x = reinterpret_cast<const X*>(vx);
 auto z = reinterpret_cast<X*>(vz);
 auto params = reinterpret_cast<X*>(vparams);
 auto reductionPointer = reinterpret_cast<X*>(vreductionPointer);

 // Check for special operations
 if (OpType::requiresSpecial) {
   OpType::execSpecialCuda(x, xShapeInfo, z, zShapeInfo, params, allocationPointer, reductionPointer, tadShapeInfo,
                           tadOffsets);
   return;
 }

 // Shared memory for caching shape information
 __shared__ sd::LongType length;
 __shared__ int xRankCached;
 __shared__ const sd::LongType* xShapePtrCached;
 __shared__ const sd::LongType* xStridePtrCached;

 __shared__ int zRankCached;
 __shared__ const sd::LongType* zShapePtrCached;
 __shared__ const sd::LongType* zStridePtrCached;

 // Thread 0 caches the shape information
 if (threadIdx.x == 0) {
   length = shape::length(xShapeInfo);

   xRankCached = shape::rank(xShapeInfo);
   xShapePtrCached = shape::shapeOf(xShapeInfo);
   xStridePtrCached = shape::stride(xShapeInfo);

   zRankCached = shape::rank(zShapeInfo);
   zShapePtrCached = shape::shapeOf(zShapeInfo);
   zStridePtrCached = shape::stride(zShapeInfo);
 }
 __syncthreads();

 // Calculate thread ID and total threads
 auto tid = blockIdx.x * blockDim.x + threadIdx.x;
 int totalThreads = gridDim.x * blockDim.x;

 // Loop over all elements using cached shape info
 for (sd::LongType i = tid; i < length; i += totalThreads) {
   sd::LongType xCoords[SD_MAX_RANK];
   sd::LongType zCoords[SD_MAX_RANK];
   sd::LongType xOffset;
   sd::LongType zOffset;

   // Convert index to coordinates using cached shape info
   INDEX2COORDS(i, xRankCached, xShapePtrCached, xCoords);
   COORDS2INDEX(xRankCached, xStridePtrCached, xCoords, xOffset);

   INDEX2COORDS(i, zRankCached, zShapePtrCached, zCoords);
   COORDS2INDEX(zRankCached, zStridePtrCached, zCoords, zOffset);

   // Apply the operation using cached offsets
   z[zOffset] = OpType::op(x[xOffset], params);
 }
}

////////////////////////////////////////////////////////////////////////////////
// Host function that launches the cached kernel
template <typename X, typename Z>
template <typename OpType>
SD_HOST void TransformSame<X, Z>::intermediateShaped(
   dim3 launchDims,
   cudaStream_t* stream,
   const void* x,
   const sd::LongType* xShape,
   sd::LongType xRank,
   void* extraParams,
   void* z,
   const sd::LongType* zShape,
   sd::LongType zRank,
   sd::LongType* allocationPointer,
   void* reductionPointer,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets)
{
 // Launch the cached kernel
 transformSameSimpleCached<X, OpType>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x,
         xShape,
         xRank,
         extraParams,
         z,
         zShape,
         zRank,
         allocationPointer,
         reductionPointer,
         tadShapeInfo,
         tadOffsets
     );

 // Check for any errors during kernel execution
 sd::DebugHelper::checkErrorCode(stream, "transformSame(...) cached kernel failed");
}

BUILD_SINGLE_TEMPLATE(template class TransformSame, , SD_COMMON_TYPES);
}  // namespace transform
}  // namespace functions
