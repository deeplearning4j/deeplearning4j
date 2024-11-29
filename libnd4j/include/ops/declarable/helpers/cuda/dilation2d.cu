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

  __shared__ LongType xzRank, yRank, *sharedMem;
  __shared__ LongType iH, iW, kH, kW;
  __shared__ LongType zLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    zLen = shape::length(zShapeInfo);

    xzRank = shape::rank(xShapeInfo);
    yRank = shape::rank(yShapeInfo);

    iH = xShapeInfo[2];
    iW = xShapeInfo[3];

    kH = yShapeInfo[1];
    kW = yShapeInfo[2];
  }
  __syncthreads();

  const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

  if (zInd >= zLen) return;

  auto xzCoords = sharedMem + threadIdx.x * (xzRank + yRank);
  auto yCoords = xzCoords + xzRank;

  INDEX2COORDS(zInd, xzRank, zShapeInfo, xzCoords);

  LongType zOffset;
  COORDS2INDEX(xzRank, shape::shapeOf(zShapeInfo), xzCoords, zOffset);

  yCoords[2] = xzCoords[3];  // iC coordinate is same for x, y and z

  const auto oh = xzCoords[1];
  const auto ow = xzCoords[2];

  X max = -DataTypeUtils::max<X>();

  for (yCoords[0] = 0; yCoords[0] < kH; ++yCoords[0]) {
    xzCoords[1] = oh * sH - pH + yCoords[0] * dH;
    if (xzCoords[1] < 0 || xzCoords[1] >= iH) continue;

    for (yCoords[1] = 0; yCoords[1] < kW; ++yCoords[1]) {
      xzCoords[2] = ow * sW - pW + yCoords[1] * dW;
      if (xzCoords[2] < 0 || xzCoords[2] >= iW) continue;

      LongType xOffset, yOffset;
      COORDS2INDEX(xzRank, shape::shapeOf(xShapeInfo), xzCoords, xOffset);
      COORDS2INDEX(yRank, shape::shapeOf(yShapeInfo), yCoords, yOffset);

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
