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
SD_KERNEL static void pooling3dCuda(const void* vx, const LongType* xShapeInfo, void* vz,
                                   const LongType* zShapeInfo, const int kD, const int kH, const int kW,
                                   const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                                   const int dD, const int dH, const int dW, const int poolingMode,
                                   const int extraParam0) {
 const T* x = reinterpret_cast<const T*>(vx);
 T* z = reinterpret_cast<T*>(vz);

 __shared__ int rank, kDeff, kHeff, kWeff, iD, iH, iW, kProd;
 __shared__ LongType zLen, *sharedMem;
 __shared__ LongType* xShape;
 __shared__ LongType* zShape;
 __shared__ LongType* xStride;
 __shared__ LongType* zStride;

 if (threadIdx.x == 0) {
   extern __shared__ unsigned char shmem[];
   sharedMem = reinterpret_cast<LongType*>(shmem);

   zLen = shape::length(zShapeInfo);
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
   zShape = shape::shapeOf(zShapeInfo);
   xStride = shape::stride(xShapeInfo);
   zStride = shape::stride(zShapeInfo);
 }
 __syncthreads();

 const auto zInd = threadIdx.x + blockIdx.x * blockDim.x;

 if (zInd >= zLen) return;

 auto coords = sharedMem + threadIdx.x * rank;

 INDEX2COORDS(zInd, rank, zShape, coords);

 LongType zOffset;
 COORDS2INDEX(rank, zStride, coords, zOffset);

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
           if (val > max) max = val;
         }
       }
     }
     z[zOffset] = max;
   } break;

   /*** avg ***/
   case 1: {
     T sum = static_cast<T>(0.);
     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType xOffset;
           COORDS2INDEX(rank, xStride, coords, xOffset);
           sum += x[xOffset];
         }

     if (extraParam0 == 0) {  // Exclude padding
       LongType a = (dend - dstart) / dD + ((dend - dstart) % dD == 0 ? 0 : 1);
       LongType b = (hend - hstart) / dH + ((hend - hstart) % dH == 0 ? 0 : 1);
       LongType c = (wend - wstart) / dW + ((wend - wstart) % dW == 0 ? 0 : 1);
       sum /= static_cast<T>(a * b * c);  // Accounts for dilation
     } else if (extraParam0 == 1)  // Include padding
       sum /= kProd;

     z[zOffset] = sum;
   } break;

   /*** pnorm ***/
   case 2: {
     T sum = static_cast<T>(0.);
     for (coords[2] = dstart; coords[2] < dend; coords[2] += dD)
       for (coords[3] = hstart; coords[3] < hend; coords[3] += dH)
         for (coords[4] = wstart; coords[4] < wend; coords[4] += dW) {
           LongType xOffset;
           COORDS2INDEX(rank, xStride, coords, xOffset);
           sum += math::sd_pow<T, T, T>(math::sd_abs<T, T>(x[xOffset]), static_cast<T>(extraParam0));
         }

     sum = math::sd_pow<T, T, T>(sum, static_cast<T>(1.0f) / static_cast<T>(extraParam0));

     z[zOffset] = sum;
   } break;
 }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void pooling3dCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                 const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, void* vz,
                                 const LongType* zShapeInfo, const int kD, const int kH, const int kW,
                                 const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                                 const int dD, const int dH, const int dW, const int poolingMode,
                                 const int extraParam0) {
 pooling3dCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
     vx, xShapeInfo, vz, zShapeInfo, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode, extraParam0);
 DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream),"pooling3dBPCudaLauncher failed");

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling3d(graph::Context& block, NDArray& input, NDArray& output, const LongType kD,
                                const LongType kH, const LongType kW, const LongType sD, const LongType sH, const LongType sW, const LongType pD,
                                const LongType pH, const LongType pW, const LongType dD, const LongType dH, const LongType dW,
                                const int poolingMode, const int extraParam0) {
 PointersManager manager(block.launchContext(), "pooling3d");

 dim3 poolingDims = getPoolingDims(output.lengthOf(),output.rankOf());

 NDArray::prepareSpecialUse({&output}, {&input});
 BUILD_SINGLE_SELECTOR(
     input.dataType(), pooling3dCudaLauncher,
     (poolingDims.x, poolingDims.y, poolingDims.z, block.launchContext()->getCudaStream(), input.specialBuffer(),
      input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), kD, kH, kW, sD, sH, sW, pD, pH, pW,
      dD, dH, dW, poolingMode, extraParam0),
     SD_FLOAT_TYPES);
 NDArray::registerSpecialUse({&output}, {&input});

 manager.synchronize();
}

}  // namespace ops
}  // namespace sd