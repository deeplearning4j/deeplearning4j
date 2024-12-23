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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <array/DataTypeUtils.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/dilation2d.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
SD_KERNEL static void dilation2dCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                     const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                     const int sH, const int sW, const int pH, const int pW, const int dH,
                                     const int dW) {
  // x [bS, iH, iW, iC]
  // y [kH, kW, iC]
  // z [bS, oH, oW, iC]

  const X* x = reinterpret_cast<const X*>(vx);
  const X* y = reinterpret_cast<const X*>(vy);
  Z* z = reinterpret_cast<Z*>(vz);

  __shared__ LongType xRank, yRank, zRank;
  __shared__ const LongType *xShape, *xStride, *yShape, *yStride, *zShape, *zStride;
  __shared__ LongType iH, iW, kH, kW, zLen;

  if (threadIdx.x == 0) {
    xRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);
    zRank = shape::rank(zShapeInfo);

    xShape = shape::shapeOf(xShapeInfo);
    xStride = shape::stride(xShapeInfo);
    yShape = shape::shapeOf(yShapeInfo);
    yStride = shape::stride(yShapeInfo);
    zShape = shape::shapeOf(zShapeInfo);
    zStride = shape::stride(zShapeInfo);

    zLen = shape::length(zShapeInfo);

    iH = xShape[1];
    iW = xShape[2];

    kH = yShape[0];
    kW = yShape[1];
  }
  __syncthreads();

  const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (zInd >= zLen) return;

  LongType zCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType xCoords[SD_MAX_RANK];
  LongType zOffset;

  INDEX2COORDS(zInd, zRank, zShape, zCoords);
  COORDS2INDEX(zRank, zStride, zCoords, zOffset);

  yCoords[2] = zCoords[3]; // iC coordinate is the same for x, y, and z
  const auto oh = zCoords[1];
  const auto ow = zCoords[2];

  X max = -DataTypeUtils::max<X>();

  for (yCoords[0] = 0; yCoords[0] < kH; ++yCoords[0]) {
    xCoords[1] = oh * sH - pH + yCoords[0] * dH;
    if (xCoords[1] < 0 || xCoords[1] >= iH) continue;

    for (yCoords[1] = 0; yCoords[1] < kW; ++yCoords[1]) {
      xCoords[2] = ow * sW - pW + yCoords[1] * dW;
      if (xCoords[2] < 0 || xCoords[2] >= iW) continue;

      LongType xOffset, yOffset;
      COORDS2INDEX(xRank, xStride, xCoords, xOffset);
      COORDS2INDEX(yRank, yStride, yCoords, yOffset);

      const X val = x[xOffset] + y[yOffset];
      if (val > max) max = val;
    }
  }

  z[zOffset] = static_cast<Z>(max);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void dilation2dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                   const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                   const void* vy, const LongType* yShapeInfo, void* vz,
                                   const LongType* zShapeInfo, const LongType sH, const LongType sW, const LongType pH,
                                   const LongType pW, const LongType dH, const LongType dW) {
  dilation2dCuda<X, Z><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                               zShapeInfo, sH, sW, pH, pW, dH, dW);
  DebugHelper::checkGlobalErrorCode( "dilation2d(...) failed");

}

void dilation2d(LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, const LongType sH,
                const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW) {
  PointersManager manager(context, "dilation2d");
  dim3 dilation = getDilation(output->lengthOf(),weights->rankOf(),output->rankOf());

  NDArray::prepareSpecialUse({output}, {input, weights});
  BUILD_SINGLE_SELECTOR_TWICE(
      input->dataType(), dilation2dCudaLauncher,
      (dilation.y, dilation.x, dilation.z, context->getCudaStream(), input->specialBuffer(),
       input->specialShapeInfo(), weights->specialBuffer(), weights->specialShapeInfo(), output->specialBuffer(),
       output->specialShapeInfo(), sH, sW, pH, pW, dH, dW),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {input, weights});

  manager.synchronize();
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
