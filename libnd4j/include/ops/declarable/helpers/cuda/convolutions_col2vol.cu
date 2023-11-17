/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/PointersManager.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/convolutions.h>

#include "execution/cuda/LaunchDims.h"
#include <helpers/DebugHelper.h>
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// columns [bS, iC, kD, kH, kW, oD, oH, oW] to be de-convoluted to volume [bS, iC, iD, iH, iW]
template <typename T>
static SD_KERNEL void col2volCuda(const void* columns, const LongType* colShapeInfo, void* volume,
                                  const LongType* volShapeInfo, const int sD, const int sH, const int sW,
                                  const int pD, const int pH, const int pW, const int dD, const int dH, const int dW) {
  const T* col = reinterpret_cast<const T*>(columns);
  T* vol = reinterpret_cast<T*>(volume);

  __shared__ LongType kD, kH, kW, oD, oH, oW, *sharedMem;
  __shared__ LongType volLen;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    oD = colShapeInfo[6];
    oH = colShapeInfo[7];
    oW = colShapeInfo[8];

    kD = dD * (colShapeInfo[3] - 1) + 1;
    kH = dH * (colShapeInfo[4] - 1) + 1;
    kW = dW * (colShapeInfo[5] - 1) + 1;

    volLen = shape::length(volShapeInfo);
  }
  __syncthreads();

  auto coords = sharedMem + threadIdx.x * 8;

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (LongType i = tid; i < volLen; i += gridDim.x * blockDim.x) {
    shape::index2coords(i, volShapeInfo, coords);

    const auto volOffset = shape::getOffset(volShapeInfo, coords);

    const auto bSiCoffset = coords[0] * colShapeInfo[9] + coords[1] * colShapeInfo[10];

    const LongType imD = coords[2] + pD;
    const LongType imH = coords[3] + pH;
    const LongType imW = coords[4] + pW;

    const LongType colDstart = (imD < kD) ? 0 : (imD - kD) / sD + 1;
    const LongType colHstart = (imH < kH) ? 0 : (imH - kH) / sH + 1;
    const LongType colWstart = (imW < kW) ? 0 : (imW - kW) / sW + 1;

    const LongType colDend = sd::math::sd_min<LongType>(imD / sD + 1, oD);
    const LongType colHend = sd::math::sd_min<LongType>(imH / sH + 1, oH);
    const LongType colWend = sd::math::sd_min<LongType>(imW / sW + 1, oW);

    T val = 0;

    for (LongType colD = colDstart; colD < colDend; ++colD) {
      coords[2] = imD - colD * sD;
      if (coords[2] % dD != 0) continue;

      for (LongType colH = colHstart; colH < colHend; ++colH) {
        coords[3] = imH - colH * sH;
        if (coords[3] % dH != 0) continue;

        for (LongType colW = colWstart; colW < colWend; ++colW) {
          coords[4] = imW - colW * sW;
          if (coords[4] % dW != 0) continue;

          val += col[bSiCoffset + (coords[2] / dD) * colShapeInfo[11] + (coords[3] / dH) * colShapeInfo[12] +
                     (coords[4] / dW) * colShapeInfo[13] + colD * colShapeInfo[14] + colH * colShapeInfo[15] +
                     colW * colShapeInfo[16]];
        }
      }
    }

    vol[volOffset] = val;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void col2volCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                const cudaStream_t* stream, const void* columns, const LongType* colShapeInfo,
                                void* volume, const LongType* volShapeInfo, const LongType sD, const LongType sH,
                                const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH,
                                const LongType dW) {
  col2volCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(columns, colShapeInfo, volume, volShapeInfo,
                                                                         sD, sH, sW, pD, pH, pW, dD, dH, dW);
  DebugHelper::checkGlobalErrorCode( "col2vol(...) failed");

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::col2vol(graph::Context& block, const NDArray& col, NDArray& vol, const LongType sD, const LongType sH,
                               const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH,
                               const LongType dW) {
  PointersManager manager(block.launchContext(), "col2vol");


  dim3 col2VolDims = getCol2VolDims(vol.lengthOf(), col.rankOf());

  NDArray::prepareSpecialUse({&vol}, {&col});
  BUILD_SINGLE_SELECTOR(
      vol.dataType(), col2volCudaLauncher,
      (col2VolDims.x, col2VolDims.y, col2VolDims.z, block.launchContext()->getCudaStream(), col.specialBuffer(),
       col.specialShapeInfo(), vol.specialBuffer(), vol.specialShapeInfo(), sD, sH, sW, pD, pH, pW, dD, dH, dW),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({&vol}, {&col});

  manager.synchronize();
  DebugHelper::checkGlobalErrorCode( "col2vol(...) failed");

}

}  // namespace ops
}  // namespace sd
