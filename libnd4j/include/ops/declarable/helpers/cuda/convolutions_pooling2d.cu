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
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/convolutions.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static SD_KERNEL void avgPooling2dCuda(const void *vx, const LongType *xShapeInfo, void *vz, const LongType *zShapeInfo,
                                       const LongType kH, const LongType kW, const LongType sH, const LongType sW,
                                       const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                                       const int extraParam0) {
  // input is  [bS, iC, iH, iW]
  // output is [bS, iC, oH, oW]

  const auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);

  __shared__ LongType bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY,
      strideOX, length, kHEff, kWEff;

  if (threadIdx.x == 0) {
    bS = shape::sizeAt(xShapeInfo, 0);
    iC = shape::sizeAt(xShapeInfo, 1);
    oH = shape::sizeAt(zShapeInfo, 2);
    oW = shape::sizeAt(zShapeInfo, 3);
    iH = shape::sizeAt(xShapeInfo, 2);
    iW = shape::sizeAt(xShapeInfo, 3);

    strideB = shape::stride(xShapeInfo)[0];
    strideC = shape::stride(xShapeInfo)[1];
    strideY = shape::stride(xShapeInfo)[2];
    strideX = shape::stride(xShapeInfo)[3];

    strideOB = shape::stride(zShapeInfo)[0];
    strideOC = shape::stride(zShapeInfo)[1];
    strideOY = shape::stride(zShapeInfo)[2];
    strideOX = shape::stride(zShapeInfo)[3];

    length = shape::length(zShapeInfo);

    // Replace kernel H/W with *effective* kernel H/W accounting for dilation
    kHEff = kH + (kH - 1) * (dH - 1);
    kWEff = kW + (kW - 1) * (dW - 1);
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
    const LongType pw = index % oW;
    const LongType ph = (index / oW) % oH;
    const LongType c = (index / oW / oH) % iC;
    const LongType n = index / oW / oH / iC;

    LongType hstart = sH * ph - pH;
    LongType wstart = sW * pw - pW;
    LongType hend = hstart + kHEff;
    LongType wend = wstart + kWEff;

    if (hstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-hstart / (Z)dH);
      hstart += f * dH;
    }
    if (wstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-wstart / (Z)dW);
      wstart += f * dW;
    }
    if (hend > iH) {
      int f = math::sd_ceil<Z, LongType>((Z)(hend - iH) / (Z)dH);
      hend -= f * dH;
    }
    if (wend > iW) {
      int f = math::sd_ceil<Z, LongType>((Z)(wend - iW) / (Z)dW);
      wend -= f * dW;
    }

    // Accounts for dilation
    int pool_size = sd::math::sd_ceil<double, LongType>((double)(hend - hstart) / (double)dH) *
                    sd::math::sd_ceil<double, LongType>((double)(wend - wstart) / (double)dW);

    Z sum = 0.0f;

    const X *inSlice = x + (n * strideB + c * strideC);

    for (int h = hstart; h < hend; h += dH)
      for (int w = wstart; w < wend; w += dW) sum += static_cast<Z>(inSlice[h * strideY + w * strideX]);

    int divide_factor = pool_size;  // Case 0: exclude padding
    if (extraParam0 == 1)           // Case 1: include padding
      divide_factor = kH * kW;

    z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = sum / static_cast<Z>(divide_factor);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void avgPooling2dCudaLauncher(LaunchContext &block, const void *vx, const LongType *vxShapeInfo, void *vz,
                                     const LongType *vzShapeInfo, const LongType kH, const LongType kW,
                                     const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                     const LongType dH, const LongType dW, const int extraParam0) {
  dim3 launchDims = getLaunchDims("avg_pooling");
  avgPooling2dCuda<X, Z><<<launchDims.y, launchDims.x, launchDims.z, *block.getCudaStream()>>>(
      vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
  DebugHelper::checkErrorCode(block.getCudaStream(), "avgb pooling 2d failed");

}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static SD_KERNEL void pnormPooling2dCuda(const void *vx, const LongType *xShapeInfo, void *vz,
                                         const LongType *zShapeInfo, const LongType kH, const LongType kW,
                                         const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                         const LongType dH, const LongType dW, const int extraParam0) {
  // input is  [bS, iC, iH, iW]
  // output is [bS, iC, oH, oW]

  const auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);

  __shared__ LongType bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY,
      strideOX, length, kHEff, kWEff;
  __shared__ bool fOrder;

  if (threadIdx.x == 0) {
    bS = shape::sizeAt(xShapeInfo, 0);
    iC = shape::sizeAt(xShapeInfo, 1);
    oH = shape::sizeAt(zShapeInfo, 2);
    oW = shape::sizeAt(zShapeInfo, 3);
    iH = shape::sizeAt(xShapeInfo, 2);
    iW = shape::sizeAt(xShapeInfo, 3);

    strideB = shape::stride(xShapeInfo)[0];
    strideC = shape::stride(xShapeInfo)[1];
    strideY = shape::stride(xShapeInfo)[2];
    strideX = shape::stride(xShapeInfo)[3];

    strideOB = shape::stride(zShapeInfo)[0];
    strideOC = shape::stride(zShapeInfo)[1];
    strideOY = shape::stride(zShapeInfo)[2];
    strideOX = shape::stride(zShapeInfo)[3];

    length = shape::length(zShapeInfo);

    // Replace kernel H/W with *effective* kernel H/W accounting for dilation
    kHEff = kH + (kH - 1) * (dH - 1);
    kWEff = kW + (kW - 1) * (dW - 1);
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
    const LongType pw = index % oW;
    const LongType ph = (index / oW) % oH;
    const LongType c = (index / oW / oH) % iC;
    const LongType n = index / oW / oH / iC;

    LongType hstart = sH * ph - pH;
    LongType wstart = sW * pw - pW;
    LongType hend = hstart + kHEff;
    LongType wend = wstart + kWEff;

    if (hstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-hstart / (Z)dH);
      hstart += f * dH;
    }
    if (wstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-wstart / (Z)dW);
      wstart += f * dW;
    }
    if (hend > iH) {
      int f = math::sd_ceil<Z, LongType>((Z)(hend - iH) / (Z)dH);
      hend -= f * dH;
    }
    if (wend > iW) {
      int f = math::sd_ceil<Z, LongType>((Z)(wend - iW) / (Z)dW);
      wend -= f * dW;
    }

    Z sum = 0.f;

    const X *inSlice = x + (n * strideB + c * strideC);

    for (int h = hstart; h < hend; h += dH)
      for (int w = wstart; w < wend; w += dW)
        sum += math::sd_pow<Z, Z, Z>(static_cast<Z>(math::sd_abs<X>(inSlice[h * strideY + w * strideX])), extraParam0);

    z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = math::sd_pow<Z, Z, Z>(sum, (Z)1.0f / extraParam0);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void pnormPooling2dCudaLauncher(LaunchContext &block, const void *vx, const LongType *vxShapeInfo, void *vz,
                                       const LongType *vzShapeInfo, const LongType kH, const LongType kW,
                                       const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                       const LongType dH, const LongType dW, const int extraParam0) {
  dim3 launchDims = getLaunchDims("avg_pooling");
  pnormPooling2dCuda<X, Z><<<launchDims.y, launchDims.x, launchDims.z, *block.getCudaStream()>>>(
      vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
  DebugHelper::checkErrorCode(block.getCudaStream(), "pnorm pooling 2d failed");

}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static SD_KERNEL void maxPooling2dCuda(const void *vx, const LongType *xShapeInfo, void *vz, const LongType *zShapeInfo,
                                       const int kH, const LongType kW, const LongType sH, const LongType sW,
                                       const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                                       const int extraParam0) {
  // input is  [bS, iC, iH, iW]
  // output is [bS, iC, oH, oW]

  const auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);

  __shared__ LongType bS, iC, oH, oW, iH, iW, strideB, strideC, strideY, strideX, strideOB, strideOC, strideOY,
      strideOX, length, kHEff, kWEff;
  __shared__ bool fOrder;

  if (threadIdx.x == 0) {
    bS = shape::sizeAt(xShapeInfo, 0);
    iC = shape::sizeAt(xShapeInfo, 1);
    oH = shape::sizeAt(zShapeInfo, 2);
    oW = shape::sizeAt(zShapeInfo, 3);
    iH = shape::sizeAt(xShapeInfo, 2);
    iW = shape::sizeAt(xShapeInfo, 3);

    strideB = shape::stride(xShapeInfo)[0];
    strideC = shape::stride(xShapeInfo)[1];
    strideY = shape::stride(xShapeInfo)[2];
    strideX = shape::stride(xShapeInfo)[3];

    strideOB = shape::stride(zShapeInfo)[0];
    strideOC = shape::stride(zShapeInfo)[1];
    strideOY = shape::stride(zShapeInfo)[2];
    strideOX = shape::stride(zShapeInfo)[3];

    length = shape::length(zShapeInfo);

    // Replace kernel H/W with *effective* kernel H/W accounting for dilation
    kHEff = kH + (kH - 1) * (dH - 1);
    kWEff = kW + (kW - 1) * (dW - 1);
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int index = tid; index < length; index += blockDim.x * gridDim.x) {
    const LongType pw = index % oW;
    const LongType ph = (index / oW) % oH;
    const LongType c = (index / oW / oH) % iC;
    const LongType n = index / oW / oH / iC;

    LongType hstart = sH * ph - pH;
    LongType wstart = sW * pw - pW;
    LongType hend = hstart + kHEff;
    LongType wend = wstart + kWEff;

    if (hstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-hstart / (Z)dH);
      hstart += f * dH;
    }
    if (wstart < 0) {
      int f = math::sd_ceil<Z, LongType>((Z)-wstart / (Z)dW);
      wstart += f * dW;
    }
    if (hend > iH) {
      int f = math::sd_ceil<Z, LongType>((Z)(hend - iH) / (Z)dH);
      hend -= f * dH;
    }
    if (wend > iW) {
      int f = math::sd_ceil<Z, LongType>((Z)(wend - iW) / (Z)dW);
      wend -= f * dW;
    }
    // Accounts for dilation
    int pool_size = sd::math::sd_ceil<double, LongType>((double)(hend - hstart) / (double)dH) *
                    sd::math::sd_ceil<double, LongType>((double)(wend - wstart) / (double)dW);

    Z max = -DataTypeUtils::max<Z>();

    const X *inSlice = x + (n * strideB + c * strideC);

    for (int h = hstart; h < hend; h += dH) {
      for (int w = wstart; w < wend; w += dW) {
        Z v = static_cast<Z>(inSlice[h * strideY + w * strideX]);
        if (v > max) max = v;
      }
    }

    z[n * strideOB + c * strideOC + pw * strideOX + ph * strideOY] = max;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
static void maxPooling2dCudaLauncher(LaunchContext &block, const void *vx, const LongType *vxShapeInfo, void *vz,
                                     const LongType *vzShapeInfo, const LongType kH, const LongType kW,
                                     const LongType sH, const LongType sW, const LongType pH, const LongType pW,
                                     const LongType dH, const LongType dW, const int extraParam0, const int rank,
                                     const int len) {
  dim3 poolingDims = getPoolingDims(len, rank);
  maxPooling2dCuda<X, Z><<<poolingDims.y, poolingDims.x, poolingDims.z, *block.getCudaStream()>>>(
      vx, vxShapeInfo, vz, vzShapeInfo, kH, kW, sH, sW, pH, pW, dH, dW, extraParam0);
  DebugHelper::checkErrorCode(block.getCudaStream(), "max pooling 2d failed");
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::pooling2d(graph::Context &block, const NDArray &input, NDArray &output, const LongType kH,
                                 const LongType kW, const LongType sH, const LongType sW, const LongType pH,
                                 const LongType pW, const LongType dH, const LongType dW, const PoolingType poolingMode,
                                 const int extraParam0) {
  if (!input.isActualOnDeviceSide()) input.syncToDevice();

  switch (poolingMode) {
    case MAX_POOL: {
      BUILD_SINGLE_SELECTOR_TWICE(
          input.dataType(), maxPooling2dCudaLauncher,
          (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(),
           output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0, output.rankOf(), output.lengthOf()),
          SD_NUMERIC_TYPES);

    } break;
    case AVG_POOL: {
      BUILD_SINGLE_SELECTOR_TWICE(
          input.dataType(), avgPooling2dCudaLauncher,
          (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(),
           output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0),
          SD_NUMERIC_TYPES);
    } break;
    case PNORM_POOL: {
      BUILD_SINGLE_SELECTOR_TWICE(
          input.dataType(), pnormPooling2dCudaLauncher,
          (*block.launchContext(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(),
           output.specialShapeInfo(), kH, kW, sH, sW, pH, pW, dH, dW, extraParam0),
          SD_FLOAT_TYPES);
    } break;
    default:
      THROW_EXCEPTION("Pooling2D: Unknown PoolingType used");
  }

  output.tickWriteDevice();
  input.tickReadDevice();

  auto result = cudaStreamSynchronize(*block.launchContext()->getCudaStream());
  if (result != 0) throw cuda_exception::build("Pooling2D failed", result);
}

}  // namespace ops
}  // namespace sd
