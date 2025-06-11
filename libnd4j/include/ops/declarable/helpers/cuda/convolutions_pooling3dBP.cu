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
#include <execution/cuda/LaunchDims.h>
#include <helpers/PointersManager.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/convolutions.h>

#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void pooling3dBPCuda(const void* vx, const LongType* xShapeInfo, const void* vy,
                                     const LongType* yShapeInfo, void* vz, const LongType* zShapeInfo,
                                     const int kD, const int kH, const int kW, const int sD, const int sH,
                                     const int sW, const int pD, const int pH, const int pW, const int dD,
                                     const int dH, const int dW, const int poolingMode, const int extraParam0) {
 const T* x = reinterpret_cast<const T*>(vx);
 const T* y = reinterpret_cast<const T*>(vy);
 T* z = reinterpret_cast<T*>(vz);

 LongType coord2, coord3, coord4;
 __shared__ int rank, kDeff, kHeff, kWeff, iD, iH, iW, kProd;
 __shared__ LongType yLen, *sharedMem;
 __shared__ LongType* xShape;
 __shared__ LongType* yShape;
 __shared__ LongType* zShape;
 __shared__ LongType* xStride;
 __shared__ LongType* yStride;
 __shared__ LongType* zStride;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   sharedMem = reinterpret_cast<LongType*>(shmem);

   yLen = shape::length(yShapeInfo);
   rank = 5;

   kDeff = kD + (kD - 1) * (dD - 1);
   kHeff = kH + (kH - 1) * (dH - 1);
   kWeff = kW + (kW - 1) * (dW - 1);

   iD = xShapeInfo[3];
   iH = xShapeInfo[4];
   iW = xShapeInfo[5];

   kProd = kD * kH * kW;

   // Cache shape information
   xShape = shape::shapeOf(xShapeInfo);
   yShape = shape::shapeOf(yShapeInfo);
   zShape = shape::shapeOf(zShapeInfo);
   xStride = shape::stride(xShapeInfo);
   yStride = shape::stride(yShapeInfo);
   zStride = shape::stride(zShapeInfo);
 }
 __syncthreads();

 const auto yInd = threadIdx.x + blockIdx.x * blockDim.x;

 if (yInd >= yLen) return;

 auto coords = sharedMem + threadIdx.x * rank;

 INDEX2COORDS(yInd, rank, yShape, coords);

 LongType yOffset;
 COORDS2INDEX(rank, yStride, coords, yOffset);

 int dstart = coords[2] * sD - pD;
 int hstart = coords[3] * sH - pH;
 int wstart = coords[4] * sW - pW;
 int dend = dstart + kDeff;
 int hend = hstart + kHeff;
 int wend = wstart + kWeff;

 if (dstart < 0) dstart += dD * ((-dstart + dD - 1) / dD);
 if (hstart < 0) hstart += dH * ((-hstart + dH - 1) / dH);
 if (wstart < 0) wstart += dW * ((-wstart + dW - 1) / dW);
 if (dend > iD) dend -= dD * ((dend - iD + dD - 1) / dD);
 if (hend > iH) hend -= dH * ((hend - iH + dH - 1) / dH);
 if (wend > iW) wend -= dW * ((wend - iW + dW - 1) / dW);

 switch (poolingMode) {
   /*** max ***/
   case 0: {
     T max = -DataTypeUtils::max<T>();
     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD) {
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH) {
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType xOffset;
           COORDS2INDEX(rank, xStride, coords, xOffset);
           T val = x[xOffset];
           if (val > max) {
             max = val;
             coord2 = coords[2];
             coord3 = coords[3];
             coord4 = coords[4];
           }
         }
       }
     }
     coords[2] = coord2;
     coords[3] = coord3;
     coords[4] = coord4;
     LongType zOffset;
     COORDS2INDEX(rank, zStride, coords, zOffset);
     math::atomics::sd_atomicAdd<T>(&z[zOffset], y[yOffset]);
   } break;

   /*** avg ***/
   case 1: {
     T val = y[yOffset];

     if (extraParam0 == 0)  // Exclude padding
       val /= static_cast<T>(math::sd_ceil<double, int>(static_cast<double>(dend - dstart) / static_cast<double>(dD)) *
                             math::sd_ceil<double, int>(static_cast<double>(hend - hstart) / static_cast<double>(dH)) *
                             math::sd_ceil<double, int>(static_cast<double>(wend - wstart) / static_cast<double>(dW)));
     else if (extraParam0 == 1)  // Include padding
       val /= static_cast<T>(kProd);

     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType zOffset;
           COORDS2INDEX(rank, zStride, coords, zOffset);
           math::atomics::sd_atomicAdd<T>(&z[zOffset], val);
         }
   } break;

   /*** pnorm ***/
   case 2: {
     T sum = static_cast<T>(0.);
     T val = y[yOffset];

     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType xOffset;
           COORDS2INDEX(rank, xStride, coords, xOffset);
           sum += math::sd_pow<T, T, T>(math::sd_abs<T,T>(x[xOffset]), static_cast<T>(extraParam0));
         }

     val *= math::sd_pow<T, T, T>(sum, (static_cast<T>(1.0f) - static_cast<T>(extraParam0)) / static_cast<T>(extraParam0));

     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType xOffset, zOffset;
           COORDS2INDEX(rank, xStride, coords, xOffset);
           COORDS2INDEX(rank, zStride, coords, zOffset);
           math::atomics::sd_atomicAdd<T>(
               &z[zOffset], val * math::sd_pow<T, T, T>(math::sd_abs<T,T>(x[xOffset]), static_cast<T>(extraParam0) - static_cast<T>(1.0f)) *
                                math::sd_sgn<T, T>(x[xOffset]));
         }
   } break;
 }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3dBPCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                   const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                                   const void* vy, const LongType* yShapeInfo, void* vz,
                                   const LongType* zShapeInfo, const int kD, const int kH, const int kW,
                                   const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                                   const int dD, const int dH, const int dW, const int poolingMode,
                                   const int extraParam0) {
 pooling3dBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, vz,
                                                                            zShapeInfo, kD, kH, kW, sD, sH, sW, pD, pH,
                                                                            pW, dD, dH, dW, poolingMode, extraParam0);
 DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream),"pooling3dBPCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling3dBP(graph::Context& block, NDArray& input, NDArray& gradO,
                                  NDArray& gradI, const LongType kD, const LongType kH, const LongType kW, const LongType sD, const LongType sH,
                                  const LongType sW, const LongType pD, const LongType pH, const LongType pW, const LongType dD, const LongType dH,
                                  const LongType dW, const int poolingMode, const int extraParam0) {
 // initial zeroing of gradI
 gradI.nullify();

 PointersManager manager(block.launchContext(), "pooling3dBP");
 dim3 poolingDims = getPoolingDims(gradO.lengthOf(),gradO.rankOf());
 NDArray::prepareSpecialUse({&gradI}, {&input, &gradO});
 BUILD_SINGLE_SELECTOR(
     input.dataType(), pooling3dBPCudaLauncher,
     (poolingDims.y, poolingDims.x, poolingDims.z, block.launchContext()->getCudaStream(), input.specialBuffer(),
      input.specialShapeInfo(), gradO.specialBuffer(), gradO.specialShapeInfo(), gradI.specialBuffer(),
      gradI.specialShapeInfo(), kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0),
     SD_FLOAT_TYPES);
 NDArray::registerSpecialUse({&gradI}, {&input, &gradO});

 manager.synchronize();
}

}  // namespace ops
}  // namespace s