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
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
//  @author raver119@gmail.com
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/StringUtils.h>
#include <loops/broadcasting_int.h>
#include <loops/legacy_ops.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <loops/pairwise_instantiations.h>
#include <stdexcept>
#include <string>

using namespace simdOps;

//////////////////////////////////////////////////////////////////////////////
// Cached kernel that caches shape info in shared memory and performs the broadcast
template <typename X, typename OpClass>
__global__ void broadcastIntSimpleCached(
   void const* x,
   sd::LongType const* xShapeInfo,
   void const* y,
   sd::LongType const* yShapeInfo,
   void* z,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // Delegate the broadcast operation to the transformCuda method with cached shape info
 functions::broadcast::BroadcastInt<X>::template transformCuda<OpClass>(
     x,
     xShapeInfo,
     y,
     yShapeInfo,
     z,
     zShapeInfo,
     dimension,
     dimensionLength,
     tadOnlyShapeInfo,
     tadOffsets,
     tadOnlyShapeInfoZ,
     tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////////
// Cached kernel that caches shape info in shared memory and performs the inverse broadcast
template <typename X, typename OpClass>
__global__ void broadcastIntInverseSimpleCached(
   void const* x,
   sd::LongType const* xShapeInfo,
   void const* y,
   sd::LongType const* yShapeInfo,
   void* z,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // Delegate the inverse broadcast operation to the transformInverseCuda method with cached shape info
 functions::broadcast::BroadcastInt<X>::template transformInverseCuda<OpClass>(
     x,
     xShapeInfo,
     y,
     yShapeInfo,
     z,
     zShapeInfo,
     dimension,
     dimensionLength,
     tadOnlyShapeInfo,
     tadOffsets,
     tadOnlyShapeInfoZ,
     tadOffsetsZ);
}

namespace functions {
namespace broadcast {

//////////////////////////////////////////////////////////////////////////////
// Implementation of the intermediateBroadcast function that launches the cached kernel with dimensions
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateBroadcast(
   dim3 launchDims,
   cudaStream_t* stream,
   void const* x,
   sd::LongType const* xShapeInfo,
   void const* y,
   sd::LongType const* yShapeInfo,
   void* z,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // Launch the cached broadcastIntSimpleCached kernel with all parameters
 broadcastIntSimpleCached<X, OpClass>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x,
         xShapeInfo,
         y,
         yShapeInfo,
         z,
         zShapeInfo,
         dimension,
         dimensionLength,
         tadOnlyShapeInfo,
         tadOffsets,
         tadOnlyShapeInfoZ,
         tadOffsetsZ);

 // Check for any errors during kernel execution
 sd::DebugHelper::checkErrorCode(stream, "intermediateBroadcast(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of the intermediateBroadcast function that launches the cached kernel without dimensions
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateBroadcast(
   dim3 launchDims,
   cudaStream_t* stream,
   const void* x,
   const sd::LongType* xShapeInfo,
   const void* y,
   const sd::LongType* yShapeInfo,
   void* z,
   const sd::LongType* zShapeInfo)
{
 // Launch the cached broadcastIntSimpleCached kernel without dimensions
 broadcastIntSimpleCached<X, OpClass>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x,
         xShapeInfo,
         y,
         yShapeInfo,
         z,
         zShapeInfo,
         nullptr, // dimension
         0,       // dimensionLength
         nullptr, // tadOnlyShapeInfo
         nullptr, // tadOffsets
         nullptr, // tadOnlyShapeInfoZ
         nullptr  // tadOffsetsZ
     );

 // Check for any errors during kernel execution
 sd::DebugHelper::checkGlobalErrorCode("broadcastIntSimpleCached(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of the intermediateInverseBroadcast function that launches the cached inverse kernel with dimensions
template <typename X>
template <typename OpClass>
SD_HOST void BroadcastInt<X>::intermediateInverseBroadcast(
   dim3 launchDims,
   cudaStream_t* stream,
   void const* x,
   sd::LongType const* xShapeInfo,
   void const* y,
   sd::LongType const* yShapeInfo,
   void* z,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // Launch the cached broadcastIntInverseSimpleCached kernel with all parameters
 broadcastIntInverseSimpleCached<X, OpClass>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x,
         xShapeInfo,
         y,
         yShapeInfo,
         z,
         zShapeInfo,
         dimension,
         dimensionLength,
         tadOnlyShapeInfo,
         tadOffsets,
         tadOnlyShapeInfoZ,
         tadOffsetsZ);

 // Check for any errors during kernel execution
 sd::DebugHelper::checkGlobalErrorCode("broadcastIntInverseSimpleCached(...) failed");
}



//////////////////////////////////////////////////////////////////////////////
// Implementation of the transformCuda device function for BroadcastInt with cached shape info
template <typename X>
template <typename OpClass>
SD_DEVICE void BroadcastInt<X>::transformCuda(
   void const* vx,
   sd::LongType const* xShapeInfo,
   void const* vy,
   sd::LongType const* yShapeInfo,
   void* vz,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // If tadOnlyShapeInfoZ is null, set it to tadOnlyShapeInfo and tadOffsetsZ to tadOffsets
 if (tadOnlyShapeInfoZ == nullptr) {
   tadOnlyShapeInfoZ = tadOnlyShapeInfo;
   tadOffsetsZ       = tadOffsets;
 }

 // Cast pointers to appropriate types
 auto x = reinterpret_cast<const X*>(vx);
 auto y = reinterpret_cast<const X*>(vy);
 auto z = reinterpret_cast<X*>(vz);

 // Shared memory for caching shape information
 __shared__ sd::LongType tadLength;
 __shared__ int numTads;
 __shared__ int xRank;
 __shared__ int yRank;
 __shared__ int zRank;

 __shared__ const sd::LongType* tadShape;
 __shared__ const sd::LongType* tadStride;
 __shared__ const sd::LongType* tadShapeZ;
 __shared__ const sd::LongType* tadStrideZ;

 if (threadIdx.x == 0) {
   // Cache essential shape information
   tadLength = shape::length(tadOnlyShapeInfo);
   numTads   = shape::length(xShapeInfo) / tadLength;

   xRank = shape::rank(xShapeInfo);
   yRank = shape::rank(yShapeInfo);
   zRank = shape::rank(zShapeInfo);

   tadShape  = shape::shapeOf(tadOnlyShapeInfo);
   tadStride = shape::stride(tadOnlyShapeInfo);

   tadShapeZ  = shape::shapeOf(tadOnlyShapeInfoZ);
   tadStrideZ = shape::stride(tadOnlyShapeInfoZ);
 }
 __syncthreads();

 // Each block handles a subset of TADs
 for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
   auto xTad = x + tadOffsets[r];
   auto zTad = z + tadOffsetsZ[r];

   // Loop over TAD elements
   for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
     sd::LongType coords[SD_MAX_RANK];
     sd::LongType xOffset, yOffset, zOffset;

     // Convert index to coordinates using cached shape info
     INDEX2COORDS(i, xRank, tadShape, coords);
     COORDS2INDEX(xRank, tadStride, coords, xOffset);

     COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);

     INDEX2COORDS(i, zRank, tadShapeZ, coords);
     COORDS2INDEX(zRank, tadStrideZ, coords, zOffset);

     // Apply the operation
     zTad[zOffset] = OpClass::op(x[xOffset], y[yOffset]);
   }
 }
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of the transformInverseCuda device function for BroadcastInt with cached shape info
template <typename X>
template <typename OpClass>
SD_DEVICE void BroadcastInt<X>::transformInverseCuda(
   void const* vx,
   sd::LongType const* xShapeInfo,
   void const* vy,
   sd::LongType const* yShapeInfo,
   void* vz,
   sd::LongType const* zShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadOnlyShapeInfo,
   sd::LongType const* tadOffsets,
   sd::LongType const* tadOnlyShapeInfoZ,
   sd::LongType const* tadOffsetsZ)
{
 // If tadOnlyShapeInfoZ is null, set it to tadOnlyShapeInfo and tadOffsetsZ to tadOffsets
 if (tadOnlyShapeInfoZ == nullptr) {
   tadOnlyShapeInfoZ = tadOnlyShapeInfo;
   tadOffsetsZ       = tadOffsets;
 }

 // Cast pointers to appropriate types
 auto x = reinterpret_cast<const X*>(vx);
 auto y = reinterpret_cast<const X*>(vy);
 auto z = reinterpret_cast<X*>(vz);

 // Shared memory for caching shape information
 __shared__ sd::LongType tadLength;
 __shared__ int numTads;
 __shared__ int xRank;
 __shared__ int yRank;
 __shared__ int zRank;

 __shared__ const sd::LongType* tadShape;
 __shared__ const sd::LongType* tadStride;
 __shared__ const sd::LongType* tadShapeZ;
 __shared__ const sd::LongType* tadStrideZ;

 if (threadIdx.x == 0) {
   // Cache essential shape information
   tadLength = shape::length(tadOnlyShapeInfo);
   numTads   = shape::length(yShapeInfo) / tadLength;

   xRank = shape::rank(xShapeInfo);
   yRank = shape::rank(yShapeInfo);
   zRank = shape::rank(zShapeInfo);

   tadShape  = shape::shapeOf(tadOnlyShapeInfo);
   tadStride = shape::stride(tadOnlyShapeInfo);

   tadShapeZ  = shape::shapeOf(tadOnlyShapeInfoZ);
   tadStrideZ = shape::stride(tadOnlyShapeInfoZ);
 }
 __syncthreads();

 // Each block handles a subset of TADs
 for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
   auto zTad = z + tadOffsetsZ[r];
   auto yTad = y + tadOffsets[r];

   // Loop over TAD elements
   for (sd::LongType i = threadIdx.x; i < tadLength; i += blockDim.x) {
     // Derive coordinates and offsets
     sd::LongType coords[SD_MAX_RANK];
     sd::LongType xOffset, yOffset, zOffset;

     // Convert index to coordinates using cached shape info
     INDEX2COORDS(i, xRank, tadShape, coords);
     COORDS2INDEX(xRank, tadStride, coords, xOffset);

     COORDS2INDEX(yRank, shape::stride(yShapeInfo), coords, yOffset);

     INDEX2COORDS(i, zRank, tadShapeZ, coords);
     COORDS2INDEX(zRank, tadStrideZ, coords, zOffset);

     // Apply the inverse operation
     zTad[zOffset] = OpClass::op(x[xOffset], yTad[yOffset]);
   }
 }
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of the transformCuda device function for BroadcastInt without dimensions
template <typename X>
template <typename OpClass>
SD_DEVICE void BroadcastInt<X>::transformCuda(
   const void* vx,
   const sd::LongType* xShapeInfo,
   const void* vy,
   const sd::LongType* yShapeInfo,
   void* vz,
   const sd::LongType* zShapeInfo)
{
 const X* x = reinterpret_cast<const X*>(vx);
 const X* y = reinterpret_cast<const X*>(vy);
 X* z       = reinterpret_cast<X*>(vz);

 // Shared memory for caching shape information
 __shared__ sd::LongType zLen;
 __shared__ int rank;
 __shared__ bool xzSameOffsets, yzSameOffsets;

 __shared__ const sd::LongType* xShapeCached;
 __shared__ const sd::LongType* yShapeCached;
 __shared__ const sd::LongType* zShapeCached;

 __shared__ const sd::LongType* xStrideCached;
 __shared__ const sd::LongType* yStrideCached;
 __shared__ const sd::LongType* zStrideCached;

 if (threadIdx.x == 0) {
   // Cache essential shape information
   zLen           = shape::length(zShapeInfo);
   rank           = shape::rank(zShapeInfo);

   xzSameOffsets  = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
   yzSameOffsets  = shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo);

   xShapeCached   = shape::shapeOf(xShapeInfo);
   yShapeCached   = shape::shapeOf(yShapeInfo);
   zShapeCached   = shape::shapeOf(zShapeInfo);

   xStrideCached  = shape::stride(xShapeInfo);
   yStrideCached  = shape::stride(yShapeInfo);
   zStrideCached  = shape::stride(zShapeInfo);
 }
 __syncthreads();

 const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
 const auto totalThreads = blockDim.x * gridDim.x;

 sd::LongType coords[SD_MAX_RANK];

 for (sd::LongType i = tid; i < zLen; i += totalThreads) {
   // Quick coordinate transform
   INDEX2COORDS(i, rank, zShapeCached, coords);

   sd::LongType zOffset, xOffset, yOffset;
   COORDS2INDEX(rank, zStrideCached, coords, zOffset);

   if (xzSameOffsets) {
     xOffset = zOffset;
   } else {
     COORDS2INDEX(rank, xStrideCached, coords, xOffset);
   }

   if (yzSameOffsets) {
     yOffset = zOffset;
   } else {
     COORDS2INDEX(rank, yStrideCached, coords, yOffset);
   }

   z[zOffset] = OpClass::op(x[xOffset], y[yOffset]);
 }
}


//////////////////////////////////////////////////////////////////////////////
// Instantiate templates for common integer types
BUILD_SINGLE_TEMPLATE(
   template class BroadcastInt, ,
   SD_INTEGER_TYPES);

}  // namespace broadcast
}  // namespace functions




