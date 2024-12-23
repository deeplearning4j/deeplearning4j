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
// @author raver119@gmail.com
//

#include <ops/specials_cuda.h>
#include <helpers/shape.h>
#include <helpers/DebugHelper.h>
#include <types/types.h>
#include <system/op_boilerplate.h>

using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
// Cached kernel for execOesTadKernelKey that caches shape info in shared memory
template <typename X, typename Y>
__global__ void execOesTadKernelKeyCached(
   void* vx,
   sd::LongType const* xShapeInfo,
   void* vy,
   sd::LongType const* yShapeInfo,
   long long int* dimension,
   long long int dimensionLength,
   sd::LongType const* tadShapeInfo,
   sd::LongType const* tadOffsets,
   bool descending) {

 auto x = static_cast<X*>(vx);
 auto y = static_cast<Y*>(vy);

 // Shared memory for caching shape information
 __shared__ int xLength;
 __shared__ int xTadLength;
 __shared__ int numTads;

 __shared__ int tadRank;
 __shared__ const sd::LongType* tadShape;
 __shared__ const sd::LongType* tadStride;

 __shared__ int yRankCached;
 __shared__ const sd::LongType* yShapePtrCached;
 __shared__ const sd::LongType* yStridePtrCached;

 // Thread 0 caches the shape information
 if (threadIdx.x == 0) {
   xLength     = shape::length(xShapeInfo);
   xTadLength  = shape::length(tadShapeInfo);
   numTads     = xLength / xTadLength;

   tadRank     = shape::rank(tadShapeInfo);
   tadShape    = shape::shapeOf(tadShapeInfo);
   tadStride   = shape::stride(tadShapeInfo);

   yRankCached = shape::rank(yShapeInfo);
   yShapePtrCached = shape::shapeOf(yShapeInfo);
   yStridePtrCached = shape::stride(yShapeInfo);
 }
 __syncthreads();

 // Each block handles a subset of TADs
 for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
   auto dx = x + tadOffsets[r];
   auto dy = y + tadOffsets[r];

   // Naive odd-even sort for the TAD
   const int iterations = xTadLength;

   for (int i = 0; i < iterations; i++) {
     // Even pass
     if (i % 2 == 0) {
       for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
         const int top = 2 * tid + 1;
         if (top < xTadLength) {
           // Get coordinates and offsets using cached shape info
           sd::LongType coordsA[SD_MAX_RANK];
           sd::LongType coordsB[SD_MAX_RANK];
           sd::LongType offsetA;
           sd::LongType offsetB;

           INDEX2COORDS(top - 1, tadRank, tadShape, coordsA);
           COORDS2INDEX(tadRank, tadStride, coordsA, offsetA);

           INDEX2COORDS(top, tadRank, tadShape, coordsB);
           COORDS2INDEX(tadRank, tadStride, coordsB, offsetB);

           // Compare & swap
           if ((!descending) == (dx[offsetA] > dx[offsetB])) {
             // Swap x
             X tempX       = dx[offsetA];
             dx[offsetA]   = dx[offsetB];
             dx[offsetB]   = tempX;

             // Swap y
             Y tempY       = dy[offsetA];
             dy[offsetA]   = dy[offsetB];
             dy[offsetB]   = tempY;
           }
         }
       }
     }
     // Odd pass
     else {
       for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
         const int top = 2 * tid + 2;
         if (top < xTadLength) {
           // Get coordinates and offsets using cached shape info
           sd::LongType coordsA[SD_MAX_RANK];
           sd::LongType coordsB[SD_MAX_RANK];
           sd::LongType offsetA;
           sd::LongType offsetB;

           INDEX2COORDS(top - 1, tadRank, tadShape, coordsA);
           COORDS2INDEX(tadRank, tadStride, coordsA, offsetA);

           INDEX2COORDS(top, tadRank, tadShape, coordsB);
           COORDS2INDEX(tadRank, tadStride, coordsB, offsetB);

           // Compare & swap
           if ((!descending) == (dx[offsetA] > dx[offsetB])) {
             // Swap x
             X tempX       = dx[offsetA];
             dx[offsetA]   = dx[offsetB];
             dx[offsetB]   = tempX;

             // Swap y
             Y tempY       = dy[offsetA];
             dy[offsetA]   = dy[offsetB];
             dy[offsetB]   = tempY;
           }
         }
       }
     }
     __syncthreads();
   }
 }
}

////////////////////////////////////////////////////////////////////////////////
// Cached kernel for execOesTadKernel that caches shape info in shared memory
template <typename T>
__global__ void execOesTadKernelCached(
   void* vx,
   sd::LongType const* xShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadShapeInfo,
   sd::LongType const* tadOffsets,
   bool descending) {

 auto x = static_cast<T*>(vx);

 // Shared memory for caching shape information
 __shared__ int xLength;
 __shared__ int xTadLength;
 __shared__ int numTads;

 __shared__ int tadRank;
 __shared__ const sd::LongType* tadShape;
 __shared__ const sd::LongType* tadStride;

 if (threadIdx.x == 0) {
   xLength     = shape::length(xShapeInfo);
   xTadLength  = shape::length(tadShapeInfo);
   numTads     = xLength / xTadLength;

   tadRank     = shape::rank(tadShapeInfo);
   tadShape    = shape::shapeOf(tadShapeInfo);
   tadStride   = shape::stride(tadShapeInfo);
 }
 __syncthreads();

 // Each block handles a subset of TADs
 for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
   auto dx = x + tadOffsets[r];

   // Naive odd-even sort for the TAD
   const int iterations = xTadLength;

   for (int i = 0; i < iterations; i++) {
     // Even pass
     if (i % 2 == 0) {
       for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
         const int top = 2 * tid + 1;
         if (top < xTadLength) {
           // Get coordinates and offsets using cached shape info
           sd::LongType coordsA[SD_MAX_RANK];
           sd::LongType coordsB[SD_MAX_RANK];
           sd::LongType offsetA;
           sd::LongType offsetB;

           INDEX2COORDS(top - 1, tadRank, tadShape, coordsA);
           COORDS2INDEX(tadRank, tadStride, coordsA, offsetA);

           INDEX2COORDS(top, tadRank, tadShape, coordsB);
           COORDS2INDEX(tadRank, tadStride, coordsB, offsetB);

           // Compare & swap
           if ((!descending) == (dx[offsetA] > dx[offsetB])) {
             // Swap x
             T tempX       = dx[offsetA];
             dx[offsetA]   = dx[offsetB];
             dx[offsetB]   = tempX;
           }
         }
       }
     }
     // Odd pass
     else {
       for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x) {
         const int top = 2 * tid + 2;
         if (top < xTadLength) {
           // Get coordinates and offsets using cached shape info
           sd::LongType coordsA[SD_MAX_RANK];
           sd::LongType coordsB[SD_MAX_RANK];
           sd::LongType offsetA;
           sd::LongType offsetB;

           INDEX2COORDS(top - 1, tadRank, tadShape, coordsA);
           COORDS2INDEX(tadRank, tadStride, coordsA, offsetA);

           INDEX2COORDS(top, tadRank, tadShape, coordsB);
           COORDS2INDEX(tadRank, tadStride, coordsB, offsetB);

           // Compare & swap
           if ((!descending) == (dx[offsetA] > dx[offsetB])) {
             // Swap x
             T tempX       = dx[offsetA];
             dx[offsetA]   = dx[offsetB];
             dx[offsetB]   = tempX;
           }
         }
       }
     }
     __syncthreads();
   }
 }
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
   const sd::LongType* tadOffsets) {

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
   const sd::LongType* tadOffsets) {

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
// Host function that launches the cached kernel for TransformSame
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
   const sd::LongType* tadOffsets) {

 // Launch the cached kernel with appropriate grid and block dimensions
 execOesTadKernelCached<X>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x, xShape, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

 // Check for any errors during kernel execution
 sd::DebugHelper::checkErrorCode(stream, "execOesTadKernelCached failed");
}

////////////////////////////////////////////////////////////////////////////////
// Host function that launches the cached execOesTadKernelKeyCached
template <typename X, typename Y>
template <typename OpType>
SD_HOST void TransformSame<X, Y>::intermediateShaped(
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
   const sd::LongType* tadOffsets) {

 // Launch the cached kernel for TransformSame with key
 execOesTadKernelKeyCached<X, Y>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x, xShape, z, zShape, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

 // Check for any errors during kernel execution
 sd::DebugHelper::checkErrorCode(stream, "execOesTadKernelKeyCached failed");
}

}  // namespace transform
}  // namespace functions

////////////////////////////////////////////////////////////////////////////////
// Host function that launches the cached execOesTadKernel
template <typename T>
SD_HOST void oesTadGeneric(
   dim3 &launchDims,
   cudaStream_t *stream,
   void* vx,
   sd::LongType const* xShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadShapeInfo,
   sd::LongType const* tadOffsets,
   bool descending) {

 execOesTadKernelCached<T>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         vx, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);

 sd::DebugHelper::checkErrorCode(stream, "execOesTadKernelCached failed");
}

////////////////////////////////////////////////////////////////////////////////
// Host function that launches the cached execOesTadKernelKeyCached
template <typename X, typename Y>
SD_HOST void oesTadGenericKey(
   dim3 &launchDims,
   cudaStream_t *stream,
   void* vx,
   sd::LongType const* xShapeInfo,
   void* vy,
   sd::LongType const* yShapeInfo,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   sd::LongType const* tadShapeInfo,
   sd::LongType const* tadOffsets,
   bool descending) {

 execOesTadKernelKeyCached<X, Y>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         vx, xShapeInfo, vy, yShapeInfo,
         dimension, dimensionLength,
         tadShapeInfo, tadOffsets,
         descending);

 sd::DebugHelper::checkErrorCode(stream, "execOesTadKernelKeyCached failed");
}

////////////////////////////////////////////////////////////////////////////////
// Instantiate templates for common types
BUILD_SINGLE_TEMPLATE(
   template void oesTadGeneric,
   (dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    sd::LongType const* xShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    bool descending),
   SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(
   template void oesTadGenericKey,
   (dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    sd::LongType const* xShapeInfo,
    void* vy,
    sd::LongType const* yShapeInfo,
    sd::LongType* dimension,
    sd::LongType dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    bool descending),
   SD_COMMON_TYPES, SD_COMMON_TYPES);
