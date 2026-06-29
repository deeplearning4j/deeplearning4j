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
#include <system/env_functions.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <loops/pairwise_instantiations.h>
#include <stdexcept>
#include <string>

using namespace simdOps;

//////////////////////////////////////////////////////////////////////////////
// Cached kernel that caches shape info in shared memory and performs the broadcast
template <typename X, typename OpClass>
SD_KERNEL SD_INLINE void broadcastIntSimpleCached(
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
SD_KERNEL SD_INLINE void broadcastIntInverseSimpleCached(
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

 // All shared variables declared at function scope (CUDA requirement).
 // The first set is used for the element-wise fallback path (tadOnlyShapeInfo==null).
 // The second set is used for the TAD path.
 __shared__ sd::LongType sharedLen;         // zLen (EW) or tadLength (TAD)
 __shared__ int sharedNumTadsOrXRank;        // numTads (TAD) or xRank (EW)
 __shared__ int sharedYRank;
 __shared__ int sharedZRank;
 __shared__ const sd::LongType* sharedPtr0; // tadShape (TAD) or xShape (EW)
 __shared__ const sd::LongType* sharedPtr1; // tadStride (TAD) or yShape (EW)
 __shared__ const sd::LongType* sharedPtr2; // tadShapeZ (TAD) or zShape (EW)
 __shared__ const sd::LongType* sharedPtr3; // tadStrideZ (TAD) or xStride (EW)
 __shared__ const sd::LongType* sharedPtr4; // yStride (EW only)
 __shared__ const sd::LongType* sharedPtr5; // zStride (EW only)

 // When tadOnlyShapeInfo is null (no-dimension element-wise broadcast path),
 // fall back to a simple element-wise loop over z using full shape info.
 // This avoids a NULL-pointer dereference in the TAD path.
 if (tadOnlyShapeInfo == nullptr) {
   if (threadIdx.x == 0) {
     sharedLen             = shape::length(zShapeInfo);
     sharedNumTadsOrXRank  = shape::rank(xShapeInfo);
     sharedYRank           = shape::rank(yShapeInfo);
     sharedZRank           = shape::rank(zShapeInfo);
     sharedPtr0            = shape::shapeOf(xShapeInfo);
     sharedPtr1            = shape::shapeOf(yShapeInfo);
     sharedPtr2            = shape::shapeOf(zShapeInfo);
     sharedPtr3            = shape::stride(xShapeInfo);
     sharedPtr4            = shape::stride(yShapeInfo);
     sharedPtr5            = shape::stride(zShapeInfo);
   }
   __syncthreads();

   const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
   const auto totalThreads = gridDim.x * blockDim.x;

   for (sd::LongType i = tid; i < sharedLen; i += totalThreads) {
     sd::LongType coords[SD_MAX_RANK];
     INDEX2COORDS(i, sharedZRank, sharedPtr2, coords);

     sd::LongType zOffset;
     COORDS2INDEX(sharedZRank, sharedPtr5, coords, zOffset);

     sd::LongType xOffset;
     COORDS2INDEX(sharedNumTadsOrXRank, sharedPtr3, coords, xOffset);

     sd::LongType yOffset;
     COORDS2INDEX(sharedYRank, sharedPtr4, coords, yOffset);

     z[zOffset] = OpClass::op(x[xOffset], y[yOffset]);
   }
   return;
 }

 if (threadIdx.x == 0) {
   // Cache essential shape information for TAD path
   sharedLen            = shape::length(tadOnlyShapeInfo);
   sharedNumTadsOrXRank = (int)(shape::length(xShapeInfo) / sharedLen);
   sharedYRank          = shape::rank(yShapeInfo);
   sharedZRank          = shape::rank(zShapeInfo);
   sharedPtr0           = shape::shapeOf(tadOnlyShapeInfo);
   sharedPtr1           = shape::stride(tadOnlyShapeInfo);
   sharedPtr2           = shape::shapeOf(tadOnlyShapeInfoZ);
   sharedPtr3           = shape::stride(tadOnlyShapeInfoZ);
 }
 __syncthreads();

 const int xRank = shape::rank(xShapeInfo);

 // Each block handles a subset of TADs
 for (sd::LongType r = blockIdx.x; r < sharedNumTadsOrXRank; r += gridDim.x) {
   auto xTad = x + tadOffsets[r];
   auto zTad = z + tadOffsetsZ[r];

   // Loop over TAD elements
   for (sd::LongType i = threadIdx.x; i < sharedLen; i += blockDim.x) {
     sd::LongType coords[SD_MAX_RANK];
     sd::LongType xOffset, yOffset, zOffset;

     // Convert index to coordinates using cached shape info
     INDEX2COORDS(i, xRank, sharedPtr0, coords);
     COORDS2INDEX(xRank, sharedPtr1, coords, xOffset);

     COORDS2INDEX(sharedYRank, shape::stride(yShapeInfo), coords, yOffset);

     INDEX2COORDS(i, sharedZRank, sharedPtr2, coords);
     COORDS2INDEX(sharedZRank, sharedPtr3, coords, zOffset);

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

 // All shared variables at function scope (CUDA requirement).
 __shared__ sd::LongType sharedLen;
 __shared__ int sharedNumTadsOrXRank;
 __shared__ int sharedYRank;
 __shared__ int sharedZRank;
 __shared__ const sd::LongType* sharedPtr0;
 __shared__ const sd::LongType* sharedPtr1;
 __shared__ const sd::LongType* sharedPtr2;
 __shared__ const sd::LongType* sharedPtr3;
 __shared__ const sd::LongType* sharedPtr4;
 __shared__ const sd::LongType* sharedPtr5;

 // When tadOnlyShapeInfo is null, use element-wise fallback to avoid NULL dereference.
 if (tadOnlyShapeInfo == nullptr) {
   if (threadIdx.x == 0) {
     sharedLen             = shape::length(zShapeInfo);
     sharedNumTadsOrXRank  = shape::rank(xShapeInfo);
     sharedYRank           = shape::rank(yShapeInfo);
     sharedZRank           = shape::rank(zShapeInfo);
     sharedPtr0            = shape::shapeOf(xShapeInfo);
     sharedPtr1            = shape::shapeOf(yShapeInfo);
     sharedPtr2            = shape::shapeOf(zShapeInfo);
     sharedPtr3            = shape::stride(xShapeInfo);
     sharedPtr4            = shape::stride(yShapeInfo);
     sharedPtr5            = shape::stride(zShapeInfo);
   }
   __syncthreads();

   const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
   const auto totalThreads = gridDim.x * blockDim.x;

   for (sd::LongType i = tid; i < sharedLen; i += totalThreads) {
     sd::LongType coords[SD_MAX_RANK];
     INDEX2COORDS(i, sharedZRank, sharedPtr2, coords);

     sd::LongType zOffset;
     COORDS2INDEX(sharedZRank, sharedPtr5, coords, zOffset);

     sd::LongType xOffset;
     COORDS2INDEX(sharedNumTadsOrXRank, sharedPtr3, coords, xOffset);

     sd::LongType yOffset;
     COORDS2INDEX(sharedYRank, sharedPtr4, coords, yOffset);

     // Inverse op: x is the "small" arg, y is the "large" arg
     z[zOffset] = OpClass::op(x[xOffset], y[yOffset]);
   }
   return;
 }

 if (threadIdx.x == 0) {
   // Cache essential shape information for TAD path
   sharedLen            = shape::length(tadOnlyShapeInfo);
   sharedNumTadsOrXRank = (int)(shape::length(yShapeInfo) / sharedLen);
   sharedYRank          = shape::rank(yShapeInfo);
   sharedZRank          = shape::rank(zShapeInfo);
   sharedPtr0           = shape::shapeOf(tadOnlyShapeInfo);
   sharedPtr1           = shape::stride(tadOnlyShapeInfo);
   sharedPtr2           = shape::shapeOf(tadOnlyShapeInfoZ);
   sharedPtr3           = shape::stride(tadOnlyShapeInfoZ);
 }
 __syncthreads();

 const int xRank = shape::rank(xShapeInfo);

 // Each block handles a subset of TADs
 for (int r = blockIdx.x; r < sharedNumTadsOrXRank; r += gridDim.x) {
   auto zTad = z + tadOffsetsZ[r];
   auto yTad = y + tadOffsets[r];

   // Loop over TAD elements
   for (sd::LongType i = threadIdx.x; i < sharedLen; i += blockDim.x) {
     // Derive coordinates and offsets
     sd::LongType coords[SD_MAX_RANK];
     sd::LongType xOffset, yOffset, zOffset;

     // Convert index to coordinates using cached shape info
     INDEX2COORDS(i, xRank, sharedPtr0, coords);
     COORDS2INDEX(xRank, sharedPtr1, coords, xOffset);

     COORDS2INDEX(sharedYRank, shape::stride(yShapeInfo), coords, yOffset);

     INDEX2COORDS(i, sharedZRank, sharedPtr2, coords);
     COORDS2INDEX(sharedZRank, sharedPtr3, coords, zOffset);

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
// Implementation of execBroadcast with dimensions
template <typename X>
SD_HOST void BroadcastInt<X>::execBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    int opNum,
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    const sd::LongType* tadOnlyShapeInfo,
    const sd::LongType* tadOffsets,
    const sd::LongType* tadOnlyShapeInfoZ,
    const sd::LongType* tadOffsetsZ) {

  DISPATCH_BY_OPNUM_T(
      intermediateBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,
             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
             tadOnlyShapeInfoZ, tadOffsetsZ),
      BROADCAST_INT_OPS);

  sd::DebugHelper::checkErrorCode(stream, "execBroadcast(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of execBroadcast without dimensions
template <typename X>
SD_HOST void BroadcastInt<X>::execBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    const int opNum,
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo) {

  DISPATCH_BY_OPNUM_T(
      intermediateBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo),
      BROADCAST_INT_OPS);

  DEBUG_KERNEL(stream, opNum);
}

//////////////////////////////////////////////////////////////////////////////
// Implementation of execInverseBroadcast with dimensions
template <typename X>
SD_HOST void BroadcastInt<X>::execInverseBroadcast(
    dim3 launchDims,
    cudaStream_t* stream,
    int opNum,
    const void* x,
    const sd::LongType* xShapeInfo,
    const void* y,
    const sd::LongType* yShapeInfo,
    void* z,
    const sd::LongType* zShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    const sd::LongType* tadOnlyShapeInfo,
    const sd::LongType* tadOffsets,
    const sd::LongType* tadOnlyShapeInfoZ,
    const sd::LongType* tadOffsetsZ) {

  DISPATCH_BY_OPNUM_T(
      intermediateInverseBroadcast,
      PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,
             dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets,
             tadOnlyShapeInfoZ, tadOffsetsZ),
      BROADCAST_INT_OPS);

  sd::DebugHelper::checkErrorCode(stream, "execInverseBroadcast(...) failed");
}

//////////////////////////////////////////////////////////////////////////////
// Instantiate templates for common integer types
BUILD_SINGLE_TEMPLATE(class BroadcastInt, , SD_INTEGER_TYPES);

}  // namespace broadcast
}  // namespace functions




