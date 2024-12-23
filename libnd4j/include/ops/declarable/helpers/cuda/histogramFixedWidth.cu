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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/histogramFixedWidth.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void histogramFixedWidthCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                              const LongType* zShapeInfo, const X leftEdge, const X rightEdge) {
  const auto x = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);

  // Shared memory caching
  __shared__ LongType xLen, zLen, totalThreads, nbins;
  __shared__ X binWidth, secondEdge, lastButOneEdge;
  __shared__ LongType xRank, zRank;
  __shared__ const LongType* xShapePtr;
  __shared__ const LongType* xStridePtr;
  __shared__ const LongType* zShapePtr;
  __shared__ const LongType* zStridePtr;

  if (threadIdx.x == 0) {
    xLen = shape::length(xShapeInfo);
    nbins = shape::length(zShapeInfo);  // nbins = zLen
    totalThreads = gridDim.x * blockDim.x;

    binWidth = (rightEdge - leftEdge) / nbins;
    secondEdge = leftEdge + binWidth;
    lastButOneEdge = rightEdge - binWidth;

    xRank = shape::rank(xShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xShapePtr = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    zShapePtr = shape::shapeOf(zShapeInfo);
    zStridePtr = shape::stride(zShapeInfo);
  }

  __syncthreads();

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < xLen; i += totalThreads) {
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;

    // Use cached rank and shape to compute coordinates and offset for x
    INDEX2COORDS(i, xRank, xShapePtr, xCoords);
    COORDS2INDEX(xRank, xStridePtr, xCoords, xOffset);

    const X value = x[xOffset];

    // Determine the histogram bin index
    LongType zIndex;
    if (value < secondEdge)
      zIndex = 0;
    else if (value >= lastButOneEdge)
      zIndex = nbins - 1;
    else
      zIndex = static_cast<LongType>((value - leftEdge) / binWidth);

    LongType zCoords[SD_MAX_RANK];
    LongType zOffset;

    // Use cached rank and shape to compute coordinates and offset for z
    INDEX2COORDS(zIndex, zRank, zShapePtr, zCoords);
    COORDS2INDEX(zRank, zStridePtr, zCoords, zOffset);

    // Atomic addition to the histogram bin
    math::atomics::sd_atomicAdd<Z>(&z[zOffset], 1);
  }
}


///////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_HOST static void histogramFixedWidthCudaLauncher(const cudaStream_t* stream, NDArray& input,
                                                    NDArray& range, NDArray& output) {
  const X leftEdge = range.e<X>(0);
  const X rightEdge = range.e<X>(1);

  dim3 launchDims = getLaunchDims("histogram_fixed_width");
  histogramFixedWidthCuda<X, Z><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(input.specialBuffer(), input.specialShapeInfo(),
                                                             output.specialBuffer(), output.specialShapeInfo(),
                                                             leftEdge, rightEdge);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream),"histogramKernel failed");

}

////////////////////////////////////////////////////////////////////////
void histogramFixedWidth(LaunchContext* context, NDArray& input, NDArray& range, NDArray& output) {
  // firstly initialize output with zeros
  output.nullify();

  PointersManager manager(context, "histogramFixedWidth");

  NDArray::prepareSpecialUse({&output}, {&input});
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), histogramFixedWidthCudaLauncher,
                        (context->getCudaStream(), input, range, output), SD_COMMON_TYPES, SD_INDEXING_TYPES);
  NDArray::registerSpecialUse({&output}, {&input});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
