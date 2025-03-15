/******************************************************************************
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* See the NOTICE file distributed with this work for additional
* information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
// @author raver119@gmail.com

#ifndef SCALAR_BOOL_CU
#define SCALAR_BOOL_CU

#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"
#include "../scalar_bool.h"
#include <helpers/DebugHelper.h>
#include <system/Environment.h>

using namespace simdOps;

////////////////////////////////////////////////////////////////////////
// A kernel that applies a scalar bool transform along a specific dimension (TAD).
// It uses shared memory caching for relevant shape information to reduce overhead.
template <typename X, typename Z, typename OpType>
__global__ void scalarAlongDimensionCachedKernel(
   void const* x,
   const sd::LongType* xShapeInfo,
   void* extraParams,
   void* z,
   const sd::LongType* zShapeInfo,
   void const* scalars,                     // per-TAD scalars
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets,
   const sd::LongType* tadShapeInfoZ,       // if null, same as x TAD
   const sd::LongType* tadOffsetsZ) {

 auto xTyped         = reinterpret_cast<const X*>(x);
 auto zTyped         = reinterpret_cast<Z*>(z);
 auto extra          = reinterpret_cast<X*>(extraParams);
 auto scalarsTyped   = reinterpret_cast<const X*>(scalars);

 // If not provided, fallback
 const auto* actualTadShapeInfoZ = (tadShapeInfoZ == nullptr ? tadShapeInfo : tadShapeInfoZ);
 const auto* actualTadOffsetsZ   = (tadShapeInfoZ == nullptr ? tadOffsets   : tadOffsetsZ);

 // Cache shape info in shared memory
 __shared__ sd::LongType tadLen;
 __shared__ sd::LongType numTads;

 __shared__ int   tadRank;
 __shared__ const sd::LongType* tadShapePtr;
 __shared__ const sd::LongType* tadStridePtr;

 __shared__ int   tadRankZ;
 __shared__ const sd::LongType* tadShapePtrZ;
 __shared__ const sd::LongType* tadStridePtrZ;

 if (threadIdx.x == 0) {
   tadLen      = shape::length(tadShapeInfo);
   numTads     = shape::length(xShapeInfo) / tadLen;

   tadRank     = shape::rank(tadShapeInfo);
   tadShapePtr = shape::shapeOf(tadShapeInfo);
   tadStridePtr= shape::stride(tadShapeInfo);

   tadRankZ      = shape::rank(actualTadShapeInfoZ);
   tadShapePtrZ  = shape::shapeOf(actualTadShapeInfoZ);
   tadStridePtrZ = shape::stride(actualTadShapeInfoZ);
 }
 __syncthreads();

 // Each block handles multiple TADs
 for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
   Z* zTad    = zTyped + actualTadOffsetsZ[r];
   const X* xTad = xTyped + tadOffsets[r];
   X  scalar   = scalarsTyped[r];

   // Each thread processes part of a single TAD
   for (sd::LongType f = threadIdx.x; f < tadLen; f += blockDim.x) {
     sd::LongType coordsX[SD_MAX_RANK];
     sd::LongType coordsZ[SD_MAX_RANK];

     sd::LongType offsetX;
     sd::LongType offsetZ;

     // Compute offset for X TAD
     INDEX2COORDS(f, tadRank, tadShapePtr, coordsX);
     COORDS2INDEX(tadRank, tadStridePtr, coordsX, offsetX);

     // Compute offset for Z TAD
     INDEX2COORDS(f, tadRankZ, tadShapePtrZ, coordsZ);
     COORDS2INDEX(tadRankZ, tadStridePtrZ, coordsZ, offsetZ);

     zTad[offsetZ] = OpType::op(xTad[offsetX], scalar, extra);
   }
 }
}

////////////////////////////////////////////////////////////////////////
// A kernel to apply a scalar transform to a shaped buffer, with caching logic
// for shape info in shared memory to reduce overhead.
template <typename X, typename Z, typename OpType>
__global__ void scalarSimpleShapedCachedKernel(
   void const* x,                 // the "scalar" input
   void const* y,                 // the "array" input
   const sd::LongType* xShapeInfo,// we just read rank from here if needed
   void* params,
   void* z,
   const sd::LongType* zShapeInfo,
   sd::LongType* allocationBuffer) {

 auto scalar = reinterpret_cast<const X*>(x)[0];
 auto yTyped = reinterpret_cast<const X*>(y);
 auto zTyped = reinterpret_cast<Z*>(z);
 auto extra  = reinterpret_cast<X*>(params);

 __shared__ sd::LongType length;
 __shared__ int yRank;
 __shared__ const sd::LongType* yShapePtr;
 __shared__ const sd::LongType* yStridePtr;

 __shared__ int zRank;
 __shared__ const sd::LongType* zShapePtr;
 __shared__ const sd::LongType* zStridePtr;

 if (threadIdx.x == 0) {
   length      = shape::length(xShapeInfo);  // or maybe shape::length(zShapeInfo)
   // Actually we only need length from either input array's shape
   // but let's assume x is scalar, so let's do it from z shape if that's a shaped array
   // For now we keep as is.

   yRank       = shape::rank(xShapeInfo);    // or we do: shape::rank(some-other-shape)
   yShapePtr   = shape::shapeOf(xShapeInfo);
   yStridePtr  = shape::stride(xShapeInfo);

   zRank       = shape::rank(zShapeInfo);
   zShapePtr   = shape::shapeOf(zShapeInfo);
   zStridePtr  = shape::stride(zShapeInfo);
 }
 __syncthreads();

 const auto tid          = blockDim.x * blockIdx.x + threadIdx.x;
 const auto totalThreads = gridDim.x * blockDim.x;

 for (sd::LongType i = tid; i < length; i += totalThreads) {
   sd::LongType coordsY[SD_MAX_RANK];
   sd::LongType coordsZ[SD_MAX_RANK];
   sd::LongType offsetY;
   sd::LongType offsetZ;

   // get offset for Y
   INDEX2COORDS(i, yRank, yShapePtr, coordsY);
   COORDS2INDEX(yRank, yStridePtr, coordsY, offsetY);

   // get offset for Z
   INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
   COORDS2INDEX(zRank, zStridePtr, coordsZ, offsetZ);

   zTyped[offsetZ] = OpType::op(yTyped[offsetY], scalar, extra);
 }
}

// *********************************************************************//
// *********************************************************************//
namespace functions {
namespace scalar {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__device__ void ScalarBoolTransform<X,Z>::transformCuda(
   void const* vscalar,
   void const* vy,
   const sd::LongType* yShapeInfo,
   void* vparams,
   void* vz,
   const sd::LongType* zShapeInfo,
   sd::LongType* allocationBuffer) {

 auto scalar = reinterpret_cast<const X*>(vscalar)[0];
 auto yTyped = reinterpret_cast<const X*>(vy);
 auto zTyped = reinterpret_cast<Z*>(vz);
 auto extra  = reinterpret_cast<const X*>(vparams);

 // store shape info in shared memory
 __shared__ sd::LongType length;
 __shared__ int yRank;
 __shared__ const sd::LongType* yShapePtr;
 __shared__ const sd::LongType* yStridePtr;

 __shared__ int zRank;
 __shared__ const sd::LongType* zShapePtr;
 __shared__ const sd::LongType* zStridePtr;

 if (threadIdx.x == 0) {
   length      = shape::length(yShapeInfo);

   yRank       = shape::rank(yShapeInfo);
   yShapePtr   = shape::shapeOf(yShapeInfo);
   yStridePtr  = shape::stride(yShapeInfo);

   zRank       = shape::rank(zShapeInfo);
   zShapePtr   = shape::shapeOf(zShapeInfo);
   zStridePtr  = shape::stride(zShapeInfo);
 }
 __syncthreads();

 const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
 const auto totalThreads = blockDim.x * gridDim.x;

 for (sd::LongType i = tid; i < length; i += totalThreads) {
   sd::LongType coordsY[SD_MAX_RANK];
   sd::LongType coordsZ[SD_MAX_RANK];
   sd::LongType offsetY;
   sd::LongType offsetZ;

   INDEX2COORDS(i, yRank, yShapePtr, coordsY);
   COORDS2INDEX(yRank, yStridePtr, coordsY, offsetY);

   INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
   COORDS2INDEX(zRank, zStridePtr, coordsZ, offsetZ);

   zTyped[offsetZ] = OpType::op(yTyped[offsetY], scalar, const_cast<X*>(extra));
 }
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__host__ void ScalarBoolTransform<X,Z>::intermediateAlongDimension(
   dim3& launchDims,
   cudaStream_t* stream,
   void const* x,
   const sd::LongType* xShapeInfo,
   void* z,
   const sd::LongType* zShapeInfo,
   void const* scalars,
   void* extraParams,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets,
   const sd::LongType* tadShapeInfoZ,
   const sd::LongType* tadOffsetsZ)
{
 scalarAlongDimensionCachedKernel<X,Z,OpType>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         x,
         xShapeInfo,
         extraParams,
         z,
         zShapeInfo,
         scalars,
         dimension,
         dimensionLength,
         tadShapeInfo,
         tadOffsets,
         tadShapeInfoZ,
         tadOffsetsZ);

 sd::DebugHelper::checkErrorCode(stream, "scalarAlongDimensionCachedKernel(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
__host__ void ScalarBoolTransform<X,Z>::intermediateShaped(
   dim3& launchDims,
   cudaStream_t* stream,
   void const* vx,
   const sd::LongType* xShapeInfo,
   void* vz,
   const sd::LongType* zShapeInfo,
   void const* vscalar,
   void* vextraParams,
   sd::LongType* allocPointer)
{
 scalarSimpleShapedCachedKernel<X,Z,OpType>
     <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
         vx,
         vscalar,
         xShapeInfo,
         vextraParams,
         vz,
         zShapeInfo,
         allocPointer);

 sd::DebugHelper::checkErrorCode(stream, "scalarSimpleShapedCachedKernel(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void ScalarBoolTransform<X, Y>::executeCudaShaped(
   dim3& launchDims,
   cudaStream_t* stream,
   int opNum,
   void const* vx,
   const sd::LongType* xShapeInfo,
   void* vz,
   const sd::LongType* zShapeInfo,
   void const* vscalar,
   void const* vextraParams)
{
 if (sd::Environment::getInstance().isDebugAndVerbose()) {
   printf("H14 opNum:[%i]\n", opNum);
 }

 DISPATCH_BY_OPNUM_TT(
     intermediateShaped,
     PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo,
            vscalar, const_cast<void*>(vextraParams), nullptr),
     SCALAR_BOOL_OPS);
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
__host__ void ScalarBoolTransform<X, Y>::executeCudaAlongDimension(
   dim3& launchDims,
   cudaStream_t* stream,
   int opNum,
   void const* vx,
   const sd::LongType* xShapeInfo,
   void* vz,
   const sd::LongType* zShapeInfo,
   void const* vscalars,
   void* vextraParams,
   sd::LongType* dimension,
   sd::LongType dimensionLength,
   const sd::LongType* tadShapeInfo,
   const sd::LongType* tadOffsets,
   const sd::LongType* tadShapeInfoZ,
   const sd::LongType* tadOffsetsZ)
{
 DISPATCH_BY_OPNUM_TT(
     intermediateAlongDimension,
     PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo,
            vscalars, vextraParams, dimension, dimensionLength,
            tadShapeInfo, tadOffsets,
            tadShapeInfoZ, tadOffsetsZ),
     SCALAR_BOOL_OPS);
}

BUILD_DOUBLE_TEMPLATE(template class ScalarBoolTransform, , SD_COMMON_TYPES, SD_BOOL_TYPES);

}  // namespace scalar
}  // namespace functions

#endif  // SCALAR_BOOL_CU
