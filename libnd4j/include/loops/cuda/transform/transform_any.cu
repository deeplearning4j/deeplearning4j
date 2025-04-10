/******************************************************************************
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// @author raver119@gmail.com
//
#include <helpers/DebugHelper.h>
#include <loops/legacy_ops.h>
#include <loops/transform_any.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <execution/cuda/DeviceValidator.h>

using namespace simdOps;

////////////////////////////////////////////////////////////////////////////////
// The kernel that calls the transform CUDA method,
// caching shape info in shared memory for offset computations.
template <typename X, typename Z, typename OpType>
__global__ void transformAnySimpleCached(
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
  // Just delegate to transformCuda,
  // which will do the shape caching logic for coords->offset conversions.
  functions::transform::TransformAny<X, Z>::template transformCuda<OpType>(
      x, xShapeInfo, params, z, zShapeInfo, allocationPointer, reductionPointer, tadShapeInfo, tadOffsets);
}

namespace functions {
namespace transform {

////////////////////////////////////////////////////////////////////////////////
// Implementation of the "executeTransformShaped" that calls the new cached kernel
template <typename X, typename Y>
SD_HOST void TransformAny<X, Y>::executeTransformShaped(
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
  DISPATCH_BY_OPNUM_TT(
      intermediateShaped,
      PARAMS(launchDims, stream, x, xShape, xRank, extraParams, z, zShape, zRank, allocationPointer,
             reductionPointer, tadShapeInfo, tadOffsets),
      TRANSFORM_ANY_OPS);

  sd::DebugHelper::checkErrorCode(stream, "transformAny executeTransformShaped(...) failed");
}

////////////////////////////////////////////////////////////////////////////////
// The transformCuda method that uses shared memory for shape/stride caching,
// then does coords->offset conversions.
template <typename X, typename Z>
template <typename OpType>
SD_DEVICE void TransformAny<X, Z>::transformCuda(
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
  // cast pointers
  auto x      = reinterpret_cast<const X*>(vx);
  auto z      = reinterpret_cast<Z*>(vz);
  auto params = reinterpret_cast<X*>(vparams);

  if (x == nullptr || z == nullptr) return;

  // cache shape info in shared memory
  __shared__ sd::LongType length;
  __shared__ int xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ int zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    length      = shape::length(xShapeInfo);

    xRank       = shape::rank(xShapeInfo);
    xShapePtr   = shape::shapeOf(xShapeInfo);
    xStridePtr  = shape::stride(xShapeInfo);

    zRank       = shape::rank(zShapeInfo);
    zShapePtr   = shape::shapeOf(zShapeInfo);
    zStridePtr  = shape::stride(zShapeInfo);
  }
  __syncthreads();

  // do the transform
  const auto tid          = blockIdx.x * blockDim.x + threadIdx.x;
  const auto totalThreads = gridDim.x * blockDim.x;

  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType coordsX[SD_MAX_RANK];
    sd::LongType coordsZ[SD_MAX_RANK];
    sd::LongType offsetX;
    sd::LongType offsetZ;

    // convert i -> coords -> offset for x
    INDEX2COORDS(i, xRank, xShapePtr, coordsX);
    COORDS2INDEX(xRank, xStridePtr, coordsX, offsetX);

    // convert i -> coords -> offset for z
    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, offsetZ);

    z[offsetZ] = OpType::op(x[offsetX], params);
  }
}

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template <typename OpType>
SD_HOST void TransformAny<X, Z>::intermediateShaped(
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
  // We call the new transformAnySimpleCached kernel
  transformAnySimpleCached<X, Z, OpType>
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
      tadOffsets);

  sd::DebugHelper::checkErrorCode(stream, "transformAny(...) cached kernel failed");
}

BUILD_DOUBLE_TEMPLATE(template class TransformAny, , SD_COMMON_TYPES, SD_COMMON_TYPES);

}  // namespace transform
}  // namespace functions
