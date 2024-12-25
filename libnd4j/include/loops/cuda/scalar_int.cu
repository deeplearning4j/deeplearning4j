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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
// @author raver119@gmail.com
//
#ifndef SCALAR_INT_CU
#define SCALAR_INT_CU

#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"
#include "../scalar_int.h"
#include <helpers/DebugHelper.h>
#include <system/Environment.h>

    using namespace simdOps;

////////////////////////////////////////////////////////////////////////
// A kernel that applies an integer-based scalar transform along specified dimension (via TAD).
// We'll cache shape info in shared memory to reduce repeated calls to shapeOf, strideOf, etc.
template <typename X, typename OpType>
__global__ void scalarAlongDimensionCachedKernel(
    void const* x,
    sd::LongType const* xShapeInfo,
    void* extraParams,
    void* z,
    sd::LongType const* zShapeInfo,
    void const* scalars,
    sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadShapeInfoZ,
    sd::LongType const* tadOffsetsZ)
{
  // delegate the actual transform to the transformCuda method,
  // caching shape info inside that method.
  functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(
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
      tadOffsetsZ
  );
}

////////////////////////////////////////////////////////////////////////
// A kernel to handle shaped transforms: x is "scalar," y is "array," but
// we store shape data in shared memory.
template <typename X, typename OpType>
__global__ void scalarSimpleShapedCachedKernel(
    void const* x,
    void const* y,
    sd::LongType const* xShapeInfo,
    void* params,
    void* z,
    sd::LongType const* zShapeInfo,
    sd::LongType* allocationBuffer)
{
  // We'll call the transformCuda method (which will do caching).
  functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(
      y,
      x,
      xShapeInfo,
      params,
      z,
      zShapeInfo,
      allocationBuffer
  );
}

namespace functions {
namespace scalar {

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ScalarIntTransform<X>::transformCuda(
    void const* vscalar,
    void const* vy,
    sd::LongType const* yShapeInfo,
    void* vparams,
    void* vz,
    sd::LongType const* zShapeInfo,
    sd::LongType* allocationBuffer)
{
  auto scalar = reinterpret_cast<const X*>(vscalar)[0];
  auto yTyped = reinterpret_cast<const X*>(vy);
  auto zTyped = reinterpret_cast<X*>(vz);
  auto extra  = reinterpret_cast<X*>(vparams);

  // cache shape info in shared memory
  __shared__ sd::LongType length;

  __shared__ int yRank;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  __shared__ int zRank;
  __shared__ const sd::LongType* zShapePtr;
  __shared__ const sd::LongType* zStridePtr;

  if (threadIdx.x == 0) {
    length    = shape::length(yShapeInfo);

    yRank     = shape::rank(yShapeInfo);
    yShapePtr = shape::shapeOf(yShapeInfo);
    yStridePtr= shape::stride(yShapeInfo);

    zRank     = shape::rank(zShapeInfo);
    zShapePtr = shape::shapeOf(zShapeInfo);
    zStridePtr= shape::stride(zShapeInfo);
  }
  __syncthreads();

  const auto tid          = blockDim.x * blockIdx.x + threadIdx.x;
  const auto totalThreads = gridDim.x * blockDim.x;

  // now we do the transform
  for (sd::LongType i = tid; i < length; i += totalThreads) {
    sd::LongType coordsY[SD_MAX_RANK];
    sd::LongType coordsZ[SD_MAX_RANK];

    sd::LongType offsetY;
    sd::LongType offsetZ;

    INDEX2COORDS(i, yRank, yShapePtr, coordsY);
    COORDS2INDEX(yRank, yStridePtr, coordsY, offsetY);

    INDEX2COORDS(i, zRank, zShapePtr, coordsZ);
    COORDS2INDEX(zRank, zStridePtr, coordsZ, offsetZ);

    zTyped[offsetZ] = OpType::op(yTyped[offsetY], scalar, extra);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ScalarIntTransform<X>::transformCuda(
    sd::LongType len,
    void const* vx,
    void const* vy,
    sd::LongType yEWS,
    void* vparams,
    void* vz,
    sd::LongType zEWS,
    sd::LongType* allocationBuffer)
{
  auto x     = reinterpret_cast<const X*>(vx)[0];  // scalar
  auto yTyped= reinterpret_cast<const X*>(vy);
  auto zTyped= reinterpret_cast<X*>(vz);
  auto extra = reinterpret_cast<X*>(vparams);

  const int tid          = blockDim.x * blockIdx.x + threadIdx.x;
  const int totalThreads = blockDim.x * gridDim.x;

  for (sd::LongType i = tid; i < len; i += totalThreads) {
    zTyped[i * zEWS] = OpType::op(yTyped[i * yEWS], x, extra);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__device__ void ScalarIntTransform<X>::transformCuda(
    void const* vx,
    sd::LongType const* xShapeInfo,
    void* vextraParams,
    void* vz,
    sd::LongType const* zShapeInfo,
    void const* vscalars,
    sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadShapeInfoZ,
    sd::LongType const* tadOffsetsZ)
{
  auto x        = reinterpret_cast<const X*>(vx);
  auto scalars  = reinterpret_cast<const X*>(vscalars);
  auto zTyped   = reinterpret_cast<X*>(vz);
  auto extra    = reinterpret_cast<X*>(vextraParams);

  // if z TAD not provided, fallback
  const auto* actualTadShapeInfoZ = (tadShapeInfoZ == nullptr ? tadShapeInfo : tadShapeInfoZ);
  const auto* actualTadOffsetsZ   = (tadShapeInfoZ == nullptr ? tadOffsets : tadOffsetsZ);

  // cache shape info in shared memory
  __shared__ sd::LongType tadLen;
  __shared__ sd::LongType numTads;

  __shared__ int tadRank;
  __shared__ const sd::LongType* tadShapePtr;
  __shared__ const sd::LongType* tadStridePtr;

  __shared__ int tadRankZ;
  __shared__ const sd::LongType* tadShapePtrZ;
  __shared__ const sd::LongType* tadStridePtrZ;

  if (threadIdx.x == 0) {
    tadLen       = shape::length(tadShapeInfo);
    numTads      = shape::length(xShapeInfo) / tadLen;

    tadRank      = shape::rank(tadShapeInfo);
    tadShapePtr  = shape::shapeOf(tadShapeInfo);
    tadStridePtr = shape::stride(tadShapeInfo);

    tadRankZ      = shape::rank(actualTadShapeInfoZ);
    tadShapePtrZ  = shape::shapeOf(actualTadShapeInfoZ);
    tadStridePtrZ = shape::stride(actualTadShapeInfoZ);
  }
  __syncthreads();

  for (sd::LongType r = blockIdx.x; r < numTads; r += gridDim.x) {
    X* zTad       = zTyped + actualTadOffsetsZ[r];
    const X* xTad = x       + tadOffsets[r];
    X    scalar   = scalars[r];

    // each thread processes part of TAD
    for (sd::LongType i = threadIdx.x; i < tadLen; i += blockDim.x) {
      sd::LongType coordsX[SD_MAX_RANK];
      sd::LongType coordsZ[SD_MAX_RANK];

      sd::LongType offsetX;
      sd::LongType offsetZ;

      INDEX2COORDS(i, tadRank, tadShapePtr, coordsX);
      COORDS2INDEX(tadRank, tadStridePtr, coordsX, offsetX);

      INDEX2COORDS(i, tadRankZ, tadShapePtrZ, coordsZ);
      COORDS2INDEX(tadRankZ, tadStridePtrZ, coordsZ, offsetZ);

      zTad[offsetZ] = OpType::op(xTad[offsetX], scalar, extra);
    }
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__host__ void ScalarIntTransform<X>::intermediateAlongDimension(
    dim3& launchDims,
    cudaStream_t* stream,
    void const* x,
    sd::LongType const* xShapeInfo,
    void* z,
    sd::LongType const* zShapeInfo,
    void const* scalars,
    void* extraParams,
    sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadShapeInfoZ,
    sd::LongType const* tadOffsetsZ)
{
  // we use the new, cached version
  scalarAlongDimensionCachedKernel<X,OpType>
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

  sd::DebugHelper::checkErrorCode(stream, "ScalarIntTransform intermediateAlongDimension(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X>
template <typename OpType>
__host__ void ScalarIntTransform<X>::intermediateShaped(
    dim3& launchDims,
    cudaStream_t* stream,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void const* vscalar,
    void* vextraParams,
    sd::LongType* allocPointer)
{
  // call the new cached kernel
  scalarSimpleShapedCachedKernel<X,OpType>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          vscalar,
          xShapeInfo,
          vextraParams,
          vz,
          zShapeInfo,
          allocPointer);

  sd::DebugHelper::checkGlobalErrorCode("scalarSimpleShapedCachedKernel(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X>
__host__ void ScalarIntTransform<X>::executeCudaShaped(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void const* vscalar,
    void* vextraParams)
{
  if (sd::Environment::getInstance().isDebugAndVerbose()) {
    printf("H14 scalar int transform opNum:[%i]\n", opNum);
  }

  DISPATCH_BY_OPNUM_T(
      intermediateShaped,
      PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalar, vextraParams, nullptr),
      SCALAR_INT_OPS);

  sd::DebugHelper::checkErrorCode(stream, "ScalarIntTransform executeCudaShaped(...) failed");
}

////////////////////////////////////////////////////////////////////////
template <typename X>
__host__ void ScalarIntTransform<X>::executeCudaAlongDimension(
    dim3& launchDims,
    cudaStream_t* stream,
    int opNum,
    void const* vx,
    sd::LongType const* xShapeInfo,
    void* vz,
    sd::LongType const* zShapeInfo,
    void const* vscalars,
    void* vextraParams,
    sd::LongType* dimension,
    long long int dimensionLength,
    sd::LongType const* tadShapeInfo,
    sd::LongType const* tadOffsets,
    sd::LongType const* tadShapeInfoZ,
    sd::LongType const* tadOffsetsZ)
{
  DISPATCH_BY_OPNUM_T(
      intermediateAlongDimension,
      PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo,
             vscalars, vextraParams, dimension, dimensionLength,
             tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ),
      SCALAR_INT_OPS);

  sd::DebugHelper::checkErrorCode(stream, "ScalarIntTransform executeCudaAlongDimension(...) failed");
}

BUILD_SINGLE_TEMPLATE(template class ScalarIntTransform, , SD_INTEGER_TYPES);

}  // namespace scalar
}  // namespace functions

#endif  // SCALAR_INT_CU
