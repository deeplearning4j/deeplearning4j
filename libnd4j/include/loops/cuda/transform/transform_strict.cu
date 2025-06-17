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
// @author raver119@gmail.com
//
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <loops/transform_strict.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>


using namespace simdOps;

template <typename X, typename OpType>
SD_KERNEL void transformStrictSimple(const void *x, const sd::LongType *xShapeInfo, long long int xRank, void *params, void *z,
                                    const sd::LongType *zShapeInfo, long long int zRank,
                                    sd::LongType *allocationPointer,
                                    void *reductionPointer, const sd::LongType *tadShapeInfo,
                                    const sd::LongType *tadOffsets) {
 functions::transform::TransformStrict<X>::template transformCuda<OpType>(
     x, xShapeInfo, params, z, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
}

namespace functions {
namespace transform {

template <typename X>
SD_HOST void TransformStrict<X>::executeTransformShaped(dim3 launchDims, cudaStream_t *stream, const int opNum,
                                                       const void *x, const sd::LongType *xShape, sd::LongType xRank,
                                                       void *extraParams, void *z, const sd::LongType *zShape,
                                                       sd::LongType zRank, sd::LongType *allocationPointer, void *reductionPointer,
                                                       const sd::LongType *tadShapeInfo,
                                                       const sd::LongType *tadOffsets) {
 DISPATCH_BY_OPNUM_T(intermediateShaped,
                     PARAMS(launchDims, stream, x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer,
                            reductionPointer, tadShapeInfo, tadOffsets),
                     TRANSFORM_STRICT_OPS);

 sd::DebugHelper::checkErrorCode(stream, "transformStrict(...) failed");
}

template <typename X>
template <typename OpType>
SD_DEVICE void TransformStrict<X>::transformCuda(const void *vx, const sd::LongType *xShapeInfo, void *vparams,
                                                 void *vz, const sd::LongType *zShapeInfo,
                                                 sd::LongType *allocationPointer,
                                                 void *vreductionPointer, const sd::LongType *tadShapeInfo,
                                                 const sd::LongType *tadOffsets) {
 auto x = static_cast<const X *>(vx);
 auto z = static_cast<X *>(vz);
 auto params = static_cast<X *>(vparams);
 auto reductionPointer = static_cast<X *>(vreductionPointer);

 if (OpType::requiresSpecial) {
   OpType::execSpecialCuda(x, xShapeInfo, z, zShapeInfo, params, allocationPointer, reductionPointer, tadShapeInfo,
                           tadOffsets);
   return;
 } else {
   __shared__ sd::LongType length;

   // Cache shape information for x buffer
   __shared__ sd::LongType xRank;
   __shared__ const sd::LongType* xShapePtr;
   __shared__ const sd::LongType* xStridePtr;

   // Cache shape information for z buffer
   __shared__ sd::LongType zRank;
   __shared__ const sd::LongType* zShapePtr;
   __shared__ const sd::LongType* zStridePtr;

   if (threadIdx.x == 0) {
     length = shape::length(xShapeInfo);

     // Cache x shape information
     xRank = shape::rank(xShapeInfo);
     xShapePtr = shape::shapeOf(xShapeInfo);
     xStridePtr = shape::stride(xShapeInfo);

     // Cache z shape information
     zRank = shape::rank(zShapeInfo);
     zShapePtr = shape::shapeOf(zShapeInfo);
     zStridePtr = shape::stride(zShapeInfo);
   }
   __syncthreads();

   auto tid = blockIdx.x * blockDim.x + threadIdx.x;
   int totalThreads = gridDim.x * blockDim.x;

   for (sd::LongType i = tid; i < length; i += totalThreads) {
     sd::LongType xCoords[SD_MAX_RANK];
     sd::LongType zCoords[SD_MAX_RANK];
     sd::LongType xOffset;
     sd::LongType zOffset;

     INDEX2COORDS(i, xRank, xShapePtr, xCoords);
     COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);
     INDEX2COORDS(i, zRank, zShapePtr, zCoords);
     COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);

     z[zOffset] = OpType::op(x[xOffset], params);
   }
 }
};
template <typename X>
template <typename OpType>
SD_HOST void TransformStrict<X>::intermediateShaped(dim3 launchDims, cudaStream_t *stream, const void *x,
                                                   const sd::LongType *xShape, sd::LongType xRank, void *extraParams, void *z,
                                                   const sd::LongType *zShape, sd::LongType zRank,
                                                   sd::LongType *allocationPointer,
                                                   void *reductionPointer, const sd::LongType *tadShapeInfo,
                                                   const sd::LongType *tadOffsets) {
 transformStrictSimple<X, OpType><<<launchDims.x, launchDims.x, launchDims.z, *stream>>>(
     x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
 sd::DebugHelper::checkErrorCode(stream, "transformStrict(...) failed");
}

BUILD_SINGLE_TEMPLATE(template class TransformStrict, , SD_FLOAT_TYPES);
}  // namespace transform
}  // namespace functions
