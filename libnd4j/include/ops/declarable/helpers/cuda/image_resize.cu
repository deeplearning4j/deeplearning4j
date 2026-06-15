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
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <ops/declarable/helpers/image_resize.h>

#include "execution/cuda/LaunchDims.h"
#include <system/selective_rendering.h>

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// computeInterpolationWeights kernel
//      outSize - output length
//      inSize - input size
//      scale - input scale
//      interporationData - result
//
template <class Scaler>
static SD_KERNEL void computeInterpolationWeights(LongType outSize, LongType inSize, double scale, LongType channels, BilinearInterpolationData* interpolationData) {
  interpolationData[outSize].bottomIndex = 0;
  interpolationData[outSize].topIndex = 0;
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  Scaler scaler;
  for (LongType i = outSize - tid; i >= 0; i -= step) {
    double in = scaler(i, scale);
    double const in_f = sd::math::p_floor<double>(in);
    double const in_c = sd::math::p_ceil<double>(in);
    interpolationData[i].bottomIndex =
        math::sd_max(static_cast<LongType>(in_f), (LongType)0LL);  // static_cast<sd::LongType>(in);
    interpolationData[i].topIndex = math::sd_min(static_cast<LongType>(in_c), inSize - 1);
    interpolationData[i].interpolarValue = in - in_f;

    if (channels) {
      math::atomics::sd_atomicMul(&interpolationData[i].bottomIndex, channels);
      math::atomics::sd_atomicMul(&interpolationData[i].topIndex, channels);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm
//
static void resizeImage(LaunchContext* context, NDArray * images, LongType batchSize, LongType inHeight,
                        LongType inWidth, LongType outHeight, LongType outWidth, LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                        NDArray* output);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm kernel
//
template <typename T, typename Z>
static SD_KERNEL void resizeImageKernel(T const* input, LongType const* inputShape, Z* outputYptr,
                                        LongType const* outputShape, LongType batchSize, LongType outWidth,
                                        LongType outHeight, LongType channels, LongType inRowSize, LongType outRowSize,
                                        LongType inBatchNumValues,
                                        BilinearInterpolationData* xs_, BilinearInterpolationData* ys_) {
  for (auto batch = blockIdx.x; batch < batchSize; batch += gridDim.x) {  // blockIdx.x as batch index
    auto pX = input + batch * inBatchNumValues;
    for (LongType y = threadIdx.x; y < outHeight; y += blockDim.x) {
      const T* ys_input_lower_ptr = pX + ys_[y].bottomIndex * inRowSize;
      const T* ys_input_upper_ptr = pX + ys_[y].topIndex * inRowSize;
      double yVal = ys_[y].interpolarValue;
      auto pZ = outputYptr + (batch * outHeight + y) * outRowSize;
      for (LongType x = 0; x < outWidth; x++) {
        auto xsBottom = xs_[x].bottomIndex;
        auto xsTop = xs_[x].topIndex;
        auto xVal = xs_[x].interpolarValue;
        // process interpolation for all channels
        for (int c = 0; c < channels; c++) {
          Z topLeft(ys_input_lower_ptr[xsBottom + c]);
          Z topRight(ys_input_lower_ptr[xsTop + c]);
          Z bottomLeft(ys_input_upper_ptr[xsBottom + c]);
          Z bottomRight(ys_input_upper_ptr[xsTop + c]);
          Z top = topLeft + (topRight - topLeft) * xVal;
          Z bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
          Z resVal = Z(top + (bottom - top) * yVal);
          pZ[x * channels + c] = resVal;
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with
template <typename T, typename F>
static void resizeImage_(LaunchContext* context, NDArray * images, LongType batchSize, LongType inHeight,
                         LongType inWidth, LongType outHeight, LongType outWidth, LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                         NDArray* output) {
  LongType inRowSize = inWidth * channels;
  LongType inBatchNumValues = inHeight * inRowSize;
  LongType outRowSize = outWidth * channels;
  auto stream = context->getCudaStream();
  T const* pInput = images->getDataBuffer()->template specialAsT<T>();
  dim3 launchDims = getLaunchDims("image_resize");

                                                               // // this works only with 'c' direction
  F* pOutput = output->dataBuffer()->template specialAsT<F>();
  resizeImageKernel<T, F><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(pInput, images->specialShapeInfo(), pOutput,
                                                      output->specialShapeInfo(), batchSize, outWidth, outHeight,
                                                      channels, inRowSize, outRowSize, inBatchNumValues, xs_, ys_);

  // During CUDA graph capture, stream sync is illegal. Stream ordering guarantees correctness.
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    auto err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      { std::string msg = "helpers::resizeImage_: Cannot synchronize kernel execution; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename F>
static Status resizeBilinearFunctor_(LaunchContext* context, NDArray * images, int const width,
                                         int const height, bool const alignCorners, bool const halfPixelCenter,
                                         NDArray* output) {
  const LongType batchSize = images->sizeAt(0);
  const LongType inHeight = images->sizeAt(1);
  const LongType inWidth = images->sizeAt(2);
  const LongType channels = images->sizeAt(3);

  const LongType outHeight = output->sizeAt(1);
  const LongType outWidth = output->sizeAt(2);

  // Handle no-op resizes efficiently.
  if (outHeight == inHeight && outWidth == inWidth) {
    output->assign(images);
    return Status::OK;
  }

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorners);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorners);

  BilinearInterpolationData* xs_;  // = xs.data();
  BilinearInterpolationData* ys_;  // = xs.data();

  int irDevId = 0; cudaGetDevice(&irDevId);
  xs_ = reinterpret_cast<BilinearInterpolationData*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(BilinearInterpolationData) * (outWidth + 1), irDevId, nullptr));
  if (xs_ == nullptr) THROW_EXCEPTION("helpers::resize_image: Cannot allocate memory for vertical parts rectangulars");

  ys_ = reinterpret_cast<BilinearInterpolationData*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(BilinearInterpolationData) * (outHeight + 1), irDevId, nullptr));
  if (ys_ == nullptr) THROW_EXCEPTION("helpers::resize_image: Cannot allocate memory for horizontal parts rectangulars");
  cudaError_t err;
  dim3 launchDims = getLaunchDims("image_resize_interp_weights");

  auto stream = context->getCudaStream();
  // Compute the cached interpolation weights on the x and y dimensions.
  if (halfPixelCenter) {
    computeInterpolationWeights<HalfPixelScaler><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(outHeight, inHeight, heightScale, 0, ys_);
    computeInterpolationWeights<HalfPixelScaler>
        <<<launchDims.x,launchDims.y, launchDims.z, *stream>>>(outWidth, inWidth, widthScale, channels, xs_);
  } else {
    computeInterpolationWeights<LegacyScaler><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(outHeight, inHeight, heightScale, 0, ys_);
    computeInterpolationWeights<LegacyScaler><<<launchDims.x, launchDims.y,launchDims.z, *stream>>>(outWidth, inWidth, widthScale, channels, xs_);
  }

  NDArray::prepareSpecialUse({output}, {images});
  resizeImage_<T, F>(context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    err = cudaStreamSynchronize(*stream);
  }
  NDArray::registerSpecialUse({output}, {images});

  sd::memory::CudaMemoryPool::getInstance().free(xs_, irDevId, nullptr);
  sd::memory::CudaMemoryPool::getInstance().free(ys_, irDevId, nullptr);

  return Status::OK;
}

typedef float (*MODE_FUNC)(float);

SD_DEVICE MODE_FUNC mode_functions[4] = {sd::math::p_floor<float>, sd::math::p_round_prefer_floor<float>,
                                         sd::math::p_round_prefer_ceil<float>, sd::math::p_ceil<float>};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by interpolation nearest neighbor algorithm kernel
//
template <typename T, typename Scaler>
static SD_KERNEL void resizeNeighborKernel(T const* input, LongType const* inputShape, T* output,
                                           LongType const* outputShape, LongType batchSize, LongType inWidth,
                                           LongType inHeight, LongType outWidth, LongType outHeight, LongType channels, double widthScale,
                                           double heightScale, NearestMode nearestMode) {
  constexpr bool halfPixelCenter =
      std::is_same<Scaler, HalfPixelScaler>::value || std::is_same<Scaler, HalfPixelScalerNN>::value;
  MODE_FUNC modeFunc;
  switch (nearestMode) {
    case FLOOR:
      modeFunc = mode_functions[0];
      break;
    case ROUND_PREFER_FLOOR:
      modeFunc = mode_functions[1];
      break;
    case ROUND_PREFER_CEIL:
      modeFunc = mode_functions[2];
      break;
    case CEIL:
      modeFunc = mode_functions[3];
      break;
    default:
      modeFunc = mode_functions[0];
  }
  Scaler scaler;

  if (blockIdx.x < batchSize) {
    auto b = blockIdx.x;
    for (int y = threadIdx.x; y < outHeight; y += blockDim.x) {
      auto posY = static_cast<LongType>(modeFunc(scaler(y, heightScale)));
      LongType inY = math::sd_min(posY, inHeight - 1);
      if (halfPixelCenter) {
        inY = math::sd_max(0LL, inY);
      }

      for (int x = threadIdx.y; x < outWidth; x += blockDim.y) {
        auto posX = static_cast<LongType>(modeFunc(scaler(x, widthScale)));
        LongType inX = math::sd_min(posX, inWidth - 1);
        if (halfPixelCenter) {
          inX = math::sd_max(0LL, inX);
        }

        auto start = blockIdx.z * blockDim.z + threadIdx.z;
        auto step = blockDim.z * gridDim.z;

        for (LongType e = start; e < channels; e += step) {
          LongType posX[] = {b, inY, inX, e};
          LongType posZ[] = {b, y, x, e};
          LongType xIndex, zIndex;
          COORDS2INDEX(shape::rank(inputShape), shape::stride(inputShape), posX, xIndex);
          COORDS2INDEX(shape::rank(outputShape), shape::stride(outputShape), posZ, zIndex);
          output[zIndex] = input[xIndex];
        }
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resizeNeighborFunctor - main algorithm by nearest neighbor
//
template <typename T>
Status resizeNeighborFunctor_(LaunchContext* context, NDArray * images, int const width, int const height,
                                  CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                  NDArray* output) {
  const LongType batchSize = images->sizeAt(0);
  const LongType inHeight = images->sizeAt(1);
  const LongType inWidth = images->sizeAt(2);
  const LongType channels = images->sizeAt(3);

  const LongType outHeight = output->sizeAt(1);
  const LongType outWidth = output->sizeAt(2);

  // Handle no-op resizes efficiently.
  if (outHeight == inHeight && outWidth == inWidth) {
    output->assign(images);
    return Status::OK;
  }

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorner);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorner);

  auto imagesBuffer = images->getDataBuffer()->template specialAsT<T>();
  auto outputBuffer = output->dataBuffer()->template specialAsT<T>();
  auto stream = context->getCudaStream();

  dim3 neightborDims = resizeNeighborDims(batchSize, outHeight, outWidth);
  NDArray::prepareSpecialUse({output}, {images});
  switch (coorMode) {
    case ASYMMETRIC:
      resizeNeighborKernel<T, LegacyScaler><<<neightborDims.x, neightborDims.y,neightborDims.z, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL:
      resizeNeighborKernel<T, HalfPixelScaler><<<neightborDims.x, neightborDims.y,neightborDims.z, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL_NN:
      resizeNeighborKernel<T, HalfPixelScalerNN><<<neightborDims.x, neightborDims.y,neightborDims.z, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    default:
      resizeNeighborKernel<T, HalfPixelScaler><<<neightborDims.x, neightborDims.y,neightborDims.z, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
  };

  NDArray::registerSpecialUse({output}, {images});

  return Status::OK;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resizeImage - resize bilinear algorithm caller
//
void resizeImage(LaunchContext* context, NDArray * images, LongType batchSize, LongType inHeight, LongType inWidth,
                 LongType outHeight, LongType outWidth, LongType channels,
                 BilinearInterpolationData* xs_, BilinearInterpolationData* ys_, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(
      images->dataType(), output->dataType(), resizeImage_,
      (context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output),
      SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
}

BUILD_DOUBLE_TEMPLATE( void resizeImage_,
                      (sd::LaunchContext * context, NDArray * images, sd::LongType batchSize,
                       sd::LongType inHeight, sd::LongType inWidth, sd::LongType outHeight, sd::LongType outWidth,
                       sd::LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                       NDArray* output),
                      SD_NUMERIC_TYPES, SD_FLOAT_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status resizeBilinearFunctor(LaunchContext* context, NDArray * images, int width, int height,
                                 bool const alignCorners, bool const halfPixelCenter, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_,
                        (context, images, width, height, alignCorners, halfPixelCenter, output), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status resizeNeighborFunctor(LaunchContext* context, NDArray * images, int const width, int const height,
                                 CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                 NDArray* output) {
  BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_,
                        (context, images, width, height, coorMode, nearestMode, alignCorner, output), SD_COMMON_TYPES);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bicubic interpolation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static SD_KERNEL void initCoefTableKernel(const float a, float* table, LongType tableSize) {
  KeysCubicKernelFunc<float> kernel(a);
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  for (int i = start; i <= tableSize; i += step) {
    float x = i * 1.0 / tableSize;
    table[i * 2] = kernel.calc_less1pt0(x);
    x += 1.0;
    table[i * 2 + 1] = kernel.calc_less2pt0(x);
  }
}

float* initCoeffsTable(const double a, cudaStream_t* stream) {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  float* coeffs_table;  // = new float[(kTableSize + 1) * 2];
  int irDevId2 = 0; cudaGetDevice(&irDevId2);
  coeffs_table = reinterpret_cast<float*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(float) * ((kTableSize + 1) * 2), irDevId2, nullptr));
  if (coeffs_table == nullptr) THROW_EXCEPTION("helpers::initCoeffsTable: Cannot allocate memory for coeffs table");

  dim3 launchDims = getLaunchDims("image_resize_init_coeffs");
  initCoefTableKernel<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(static_cast<float>(a), coeffs_table, kTableSize);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    cudaError_t err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      { std::string msg = "helpers::initCoeffsTable: Cannot synchronize kernel; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }
  }

  return coeffs_table;
}

static SD_KERNEL void accumulateChannelsKernel(WeightsAndIndices* pXWais, LongType outWidth, LongType channels) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto x = start; x < outWidth; x += step) {
    pXWais[x]._index0 *= channels;
    pXWais[x]._index1 *= channels;
    pXWais[x]._index2 *= channels;
    pXWais[x]._index3 *= channels;
  }
}

template <typename Scaler>
static SD_KERNEL void advanceWeightsAndIndicesKernel(float const* cacheTable, CachedInterpolationCalculator* calc,
                                                     WeightsAndIndices* pXWais, LongType inWidth, float widthScale,
                                                     LongType outWidth, LongType channels,
                                                     bool exclude_outside) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (auto x = start; x < outWidth; x += step) {
    getWeightsAndIndices<Scaler>(cacheTable, widthScale, x, inWidth, pXWais + x, exclude_outside);
  }
  __syncthreads();
  if (start == 0) {
    // update only in one thread
    for (auto i = 0; i < outWidth; i++) {
      pXWais[i]._advance = calc->Advance(pXWais[i]._index0, pXWais[i]._index1, pXWais[i]._index2, pXWais[i]._index3);
    }
  }
}
// resizerState and xWais are device allocated
template <typename Scaler>
static void computeXWeightsAndIndices(float const* coeffsTable, const ImageResizerState& resizerState,
                                      WeightsAndIndices* pXWais, bool exclude_outside) {
  auto stream = resizerState.stream;
  auto outWidth = resizerState.outWidth;
  CachedInterpolationCalculator calc;  // = new CachedInterpolationCalculator;
  CachedInterpolationCalculator* pCalcD;
  int irDevId3 = 0; cudaGetDevice(&irDevId3);
  pCalcD = reinterpret_cast<CachedInterpolationCalculator*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(CachedInterpolationCalculator), irDevId3, nullptr));
  if (pCalcD == nullptr) THROW_EXCEPTION("helpers::computeXWeightsAndIndices: Cannot allocate device memory for interpolate calculator");
  cudaError_t err;
  err = cudaMemcpyAsync(pCalcD, &calc, sizeof(CachedInterpolationCalculator), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    std::string msg = "helpers::computeXWeightsAndIndices: Cannot set up device memory for interpolate calculator; Error code: [" + std::to_string(err) + "]";
    THROW_EXCEPTION(msg.c_str());
  }
  dim3 launchDims = getLaunchDims("image_resize_init_coeffs");

  advanceWeightsAndIndicesKernel<Scaler><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(coeffsTable, pCalcD, pXWais, resizerState.inWidth,
                                                                     resizerState.widthScale, outWidth,
                                                                     resizerState.channels, exclude_outside);
  sd::memory::CudaMemoryPool::getInstance().free(pCalcD, irDevId3, nullptr);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      std::string msg = "helpers::computeXWeightsAndIndices: Cannot synchronize stream after advance weights and indicers; Error code: [" + std::to_string(err) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
  }
  dim3 launchDims2 = getLaunchDims("image_resize_coeffs_accum");
  // Scale the values so they can be used as offsets into buffers.
  accumulateChannelsKernel<<<launchDims2.x,launchDims.y,launchDims.z, *stream>>>(pXWais, outWidth, resizerState.wStride);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      std::string msg = "helpers::computeXWeightsAndIndices: Cannot synchronize stream after accumulate channels; Error code: [" + std::to_string(err) + "]";
      THROW_EXCEPTION(msg.c_str());
    }
  }
}

template <typename T, typename Scaler>
static SD_KERNEL void bicubicInterpolateWithCachingKernel(float const* cachedTable, T const* inputPtr,
                                                          ImageResizerState* pResizerState, WeightsAndIndices* xWais,
                                                          bool exclude_outside, float* outputPtr) {
  const auto batchStride = pResizerState->bStride;
  const auto hStride = pResizerState->hStride;
  const auto cStride = pResizerState->cStride;
  for (LongType b = blockIdx.x; b < pResizerState->batchSize; b += gridDim.x) {
    auto pInput = inputPtr + b * batchStride;

    float* cachedValue;
    for (LongType y = threadIdx.x; y < pResizerState->outHeight; y += blockDim.x) {
      if (threadIdx.x == 0) {
        extern __shared__ char sharedChar[];
        cachedValue = reinterpret_cast<float*>(sharedChar);
      }
      auto pos = (b * pResizerState->outHeight + y) * pResizerState->outWidth * pResizerState->channels;
      auto pOutput = &outputPtr[pos];
      struct WeightsAndIndices yWai;

      getWeightsAndIndices<Scaler>(cachedTable, pResizerState->heightScale, y, pResizerState->inHeight, &yWai,
                                   exclude_outside);

      // Make pointers represent offsets of data in inputBPtr.
      const T* y_ptr_0 = pInput + yWai._index0 * hStride;
      const T* y_ptr_1 = pInput + yWai._index1 * hStride;
      const T* y_ptr_2 = pInput + yWai._index2 * hStride;
      const T* y_ptr_3 = pInput + yWai._index3 * hStride;

      if (pResizerState->channels == 100) {
        // Manually unroll case of 3 channels.
        float cached_value_0[4] = {0};
        float cached_value_1[4] = {0};
        float cached_value_2[4] = {0};
        for (LongType x = 0; x < pResizerState->outWidth; ++x) {
          const WeightsAndIndices& xWai = xWais[x];
          // Shift values in cached_value_* to fill first '_advance' values.
          switch (xWai._advance) {
            case 3:
              cached_value_0[0] = cached_value_0[1];
              cached_value_0[1] = cached_value_0[2];
              cached_value_0[2] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[1];
              cached_value_1[1] = cached_value_1[2];
              cached_value_1[2] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[1];
              cached_value_2[1] = cached_value_2[2];
              cached_value_2[2] = cached_value_2[3];
              break;
            case 2:
              cached_value_0[0] = cached_value_0[2];
              cached_value_0[1] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[2];
              cached_value_1[1] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[2];
              cached_value_2[1] = cached_value_2[3];
              break;
            case 1: {
              cached_value_0[0] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[3];
              break;
            }
          }

          // Set the remaining '4-_advance' values by computing.
          switch (xWai._advance) {
            case 0:
              cached_value_0[0] = computeYInterpolation(0, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_1[0] = computeYInterpolation(0, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_2[0] = computeYInterpolation(0, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
            case 1:
              cached_value_0[1] = computeYInterpolation(1, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_1[1] = computeYInterpolation(1, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_2[1] = computeYInterpolation(1, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
            case 2:
              cached_value_0[2] = computeYInterpolation(2, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_1[2] = computeYInterpolation(2, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_2[2] = computeYInterpolation(2, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
            case 3:
              cached_value_0[3] = computeYInterpolation(3, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_1[3] = computeYInterpolation(3, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              cached_value_2[3] = computeYInterpolation(3, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              //        break;
          }
          pOutput[x * pResizerState->channels + 0] =
              compute(cached_value_0, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
          pOutput[x * pResizerState->channels + 1] =
              compute(cached_value_1, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
          pOutput[x * pResizerState->channels + 2] =
              compute(cached_value_2, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
        }
      } else {
        for (LongType x = 0; x < pResizerState->outWidth; ++x) {
          const WeightsAndIndices& xWai = xWais[x];
          // Shift values in cachedValue to fill first '_advance' values.
          switch (xWai._advance) {
            case 3:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 1];
                cachedValue[4 * c + 1] = cachedValue[4 * c + 2];
                cachedValue[4 * c + 2] = cachedValue[4 * c + 3];
              }
              break;
            case 2:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 2];
                cachedValue[4 * c + 1] = cachedValue[4 * c + 3];
              }
              break;
            case 1: {
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 3];
              }
              break;
            }
          }

          // Set the remaining '4-_advance' values by computing.
          switch (xWai._advance) {
            case 0:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] =
                    computeYInterpolation(0, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 1:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 1] =
                    computeYInterpolation(1, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 2:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 2] =
                    computeYInterpolation(2, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 3:
              for (LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 3] =
                    computeYInterpolation(3, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
              // break;
          }
          for (LongType c = 0; c < pResizerState->channels; ++c) {
            auto res = compute(&cachedValue[4 * c], xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
            pOutput[x * pResizerState->channels + c] = res;
          }
        }
      }
    }
  }
}

template <typename T, typename Scaler>
static void bicubicInterpolateWithCaching(NDArray * image, const ImageResizerState& resizerState,
                                          const double coefficient, bool exclude_outside, NDArray* output) {
  const auto numChannels = resizerState.channels;
  auto stream = resizerState.stream;  // output->getContext()->getCudaStream();
  ImageResizerState* resizerStateD;
  int irDevId4 = 0; cudaGetDevice(&irDevId4);
  resizerStateD = reinterpret_cast<ImageResizerState*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(ImageResizerState), irDevId4, nullptr));
  if (resizerStateD == nullptr) THROW_EXCEPTION("helpers::bicubicInterpolateWithCaching: Cannot allocate memory for resizerState");
  cudaError_t err;
  err = cudaMemcpyAsync(resizerStateD, &resizerState, sizeof(ImageResizerState), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    { std::string msg = "helpers::bicubicInterpolateWithCaching: Cannot set up memory for resizerState; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }


  WeightsAndIndices* xWais;
  xWais = reinterpret_cast<WeightsAndIndices*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(WeightsAndIndices) * resizerState.outWidth, irDevId4, nullptr));
  if (xWais == nullptr) THROW_EXCEPTION("helpers::bicubicInterpolateWithCaching: Cannot allocate memory for weights and indices");

  auto coeffsTable = initCoeffsTable(
      coefficient, stream);
  if (err != 0) {
    { std::string msg = "helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
  computeXWeightsAndIndices<Scaler>(coeffsTable, resizerState, xWais, exclude_outside);
  err = cudaStreamQuery(*stream);
  if (err != 0) {
    { std::string msg = "helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }

  const T* pInput = image->getDataBuffer()->template specialAsT<T>();
  float* pOutput = output->dataBuffer()->specialAsT<float>();
  dim3 bicubDims = getLaunchDims("image_resize_bicubic");
  //128,1,512
  bicubicInterpolateWithCachingKernel<T, Scaler>
      <<<bicubDims.x, bicubDims.y, bicubDims.z, *stream>>>(coeffsTable, pInput, resizerStateD, xWais, exclude_outside, pOutput);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      { std::string msg = "helpers::bicubicInterpolateWithCaching: Kernels finished with error; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }
  }

  sd::memory::CudaMemoryPool::getInstance().free(resizerStateD, irDevId4, nullptr);
  sd::memory::CudaMemoryPool::getInstance().free(xWais, irDevId4, nullptr);
  sd::memory::CudaMemoryPool::getInstance().free(coeffsTable, irDevId4, nullptr);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
Status resizeBicubicFunctor_(LaunchContext* context, NDArray * image, int width, int height,
                             bool preserveAspectRatio, bool antialias, NDArray* output) {
  return Status::OK;
}

Status resizeBicubicFunctor(LaunchContext* context, NDArray * image, int width, int height,
                                bool preserveAspectRatio, bool antialias, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctor_,
                        (context, image, width, height, preserveAspectRatio, antialias, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( sd::Status resizeBicubicFunctor_,
                      (sd::LaunchContext * context, NDArray * image, int width, int height,
                       bool preserveAspectRatio, bool antialias, NDArray* output),
                      SD_NUMERIC_TYPES);
// ------------------------------------------------------------------------------------------------------------------ //

static SD_KERNEL void fillInterpolationCache(CachedInterpolation* xCached, LongType cacheLen, LongType inWidth,
                                             float widthScale) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto increment = blockDim.x * gridDim.x;

  for (auto x = start; x < cacheLen; x += increment) {
    auto& xCache = xCached[x];
    const float inX = x * widthScale;
    const float inX1 = (x + 1) * widthScale;

    LongType v = math::sd_floor<float, LongType>(inX);
    xCache.start = v;
    xCache.startScale = v < inX ? (v + 1 > inX1 ? widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
    v = math::sd_ceil<float, LongType>(inX1);
    xCache.end = v--;
    xCache.endMinusOneScale = v < inX ? (v + 1 > inX1 ? widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
    xCache.needsBounding =
        bound(xCache.start, inWidth) != xCache.start || bound(xCache.end - 1, inWidth) != (xCache.end - 1);
  }
}

// ------------------------------------------------------------------------------------------------------------------ //

template <typename T>
static SD_KERNEL void resizeAreaKernel(ImageResizerState const* pSt, CachedInterpolation const* caches, float scale,
                                       T const* inputPtr, LongType const* inputShape, float* outputPtr,
                                       LongType const* outputShape,
                                       ScaleCache<T>* cachePool) {  // batch * outWidth * outHeight

  for (auto batch = blockIdx.x; batch < pSt->batchSize; batch += gridDim.x) {
    for (auto y = threadIdx.x; y < pSt->outHeight; y += blockDim.x) {
      const float inY = y * pSt->heightScale;
      const float inY1 = (y + 1) * pSt->heightScale;
      // The start and end height indices of all the cells that could
      // contribute to the target cell.
      const LongType yStart = math::sd_floor<float, LongType>(inY);
      const LongType yEnd = math::sd_ceil<float, LongType>(inY1);
      auto scalesDim = yEnd - yStart;
      auto yScaleCache = cachePool + (batch * pSt->outHeight + y) * pSt->outWidth;

      float* output = outputPtr + (batch * pSt->outHeight + y) * pSt->channels * pSt->outWidth;
      for (LongType i = yStart, k = 0; i < yEnd; ++i, ++k) {
        float scaleY;
        if (i < inY) {
          scaleY = (i + 1 > inY1 ? pSt->heightScale : i + 1 - inY);
        } else {
          scaleY = (i + 1 > inY1 ? inY1 - i : 1.0);
        }
        yScaleCache[k].yScale = scaleY;
        yScaleCache[k].yPtr = inputPtr + (batch * pSt->bStride + bound(i, pSt->inHeight) * pSt->hStride);
      }

      if (pSt->channels == 3) {
        for (LongType x = 0; x < pSt->outWidth; ++x) {
          const CachedInterpolation& xCache = caches[x];
          computePatchSumOf3Channels<T>(scale, *pSt, yScaleCache, scalesDim, xCache, output);
          output += pSt->channels;
        }
      } else {
        for (LongType x = 0; x < pSt->outWidth; ++x) {
          const CachedInterpolation& xCache = caches[x];
          computePatchSum<T>(scale, *pSt, yScaleCache, scalesDim, xCache, output);
          output += pSt->channels;
        }
      }
    }
  }
}

template <typename T>
static void resizeArea(cudaStream_t* stream, ImageResizerState const& st, CachedInterpolation* cache,
                       NDArray * input, NDArray* output) {
  T const* inputPtr = reinterpret_cast<T const*>(input->specialBuffer());
  float scale = 1.f / (st.heightScale * st.widthScale);
  auto outputPtr =
      reinterpret_cast<float*>(output->specialBuffer());  // output is always float. TO DO: provide another float types
                                                          // also with  template <typename X, typename Z> declaration
  ImageResizerState* pSt;
  int irDevId5 = 0; cudaGetDevice(&irDevId5);
  pSt = reinterpret_cast<ImageResizerState*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(ImageResizerState), irDevId5, nullptr));
  if (pSt == nullptr) THROW_EXCEPTION("helpers::resizeArea: Cannot allocate memory for ImageResizerState");
  cudaError_t err;

  err = cudaMemcpyAsync(pSt, &st, sizeof(ImageResizerState), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    { std::string msg = "helpers::resizeArea: Cannot copy to device memory; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
  }
  ScaleCache<T>* cachePool;
  auto cachePoolSize = sizeof(ScaleCache<T>) * st.batchSize * st.outWidth * st.outHeight;
  cachePool = reinterpret_cast<ScaleCache<T>*>(sd::memory::CudaMemoryPool::getInstance().allocate(cachePoolSize, irDevId5, nullptr));
  if (cachePool == nullptr) THROW_EXCEPTION("helpers::resizeArea: Cannot allocate memory for cache");
  resizeAreaKernel<T><<<128, 128, 2048, *stream>>>(pSt, cache, scale, inputPtr, input->specialShapeInfo(), outputPtr,
                                                   output->specialShapeInfo(), cachePool);
  if (!tl_graphExecutionActive && !tl_dspReplayActive) {
    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      { std::string msg = "helpers::resizeArea: An error occured with kernel running; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
    }
  }
  sd::memory::CudaMemoryPool::getInstance().free(cachePool, irDevId5, nullptr);
  sd::memory::CudaMemoryPool::getInstance().free(pSt, irDevId5, nullptr);
}
// ------------------------------------------------------------------------------------------------------------------ //
template <typename T>
Status resizeAreaFunctor_(LaunchContext* context, NDArray * image, int const width, int const height,
                          bool const alignCorners, NDArray* output) {
  ImageResizerState st(alignCorners, false);  // Create resize info
  auto res = st.validateAndCalculateOutputSize(image, width, height);
  auto stream = context->getCudaStream();
  if (Status::OK == res) {
    CachedInterpolation* xCached;
    //(st.outWidth);
    int irDevId6 = 0; cudaGetDevice(&irDevId6);
    xCached = reinterpret_cast<CachedInterpolation*>(sd::memory::CudaMemoryPool::getInstance().allocate(sizeof(CachedInterpolation) * st.outWidth, irDevId6, nullptr));
    if (xCached == nullptr) THROW_EXCEPTION("helpers::resizeAreaFunctor_: Cannot allocate memory for cached interpolations");
    NDArray::prepareSpecialUse({output}, {image});
    dim3 launchDims = getLaunchDims("image_resize_fill_interp");
    fillInterpolationCache<<<128, 128, 256, *stream>>>(xCached, st.outWidth, st.inWidth, st.widthScale);
    resizeArea<T>(stream, st, xCached, image, output);
    if (!tl_graphExecutionActive && !tl_dspReplayActive) {
      cudaError_t err = cudaStreamSynchronize(*stream);
      if (err != 0) {
        { std::string msg = "helpers::resizeAreaFunctor_: Error occured when kernel was running; Error code: [" + std::to_string(err) + "]"; THROW_EXCEPTION(msg.c_str()); }
      }
    }
    sd::memory::CudaMemoryPool::getInstance().free(xCached, irDevId6, nullptr);
    NDArray::registerSpecialUse({output}, {image});
  }

  return res;
}
Status resizeAreaFunctor(LaunchContext* context, NDArray * image, int const width, int const height,
                             bool const alignCorners, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_,
                        (context, image, width, height, alignCorners, output), SD_NUMERIC_TYPES);
}

// ------------------------------------------------------------------------------------------------------------------ //
// simplified bicubic resize without antialiasing
//
template <typename T>
Status resizeBicubicFunctorA_(LaunchContext* context, NDArray * image, int const width, int const height,
                              bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                              double coefficient, NDArray* output) {
  ImageResizerState st(alignCorners, coorMode == HALF_PIXEL,
                       context->getCudaStream());  // align_corners, half_pixel_align
  NDArray::prepareSpecialUse({output}, {image});
  Status res = st.validateAndCreateOutput(image, width, height);
  if (res == Status::OK) {
    switch (coorMode) {
      case ASYMMETRIC:
        bicubicInterpolateWithCaching<T, LegacyScaler>(image, st, coefficient, exclude_outside, output);
        break;
      case HALF_PIXEL:
        bicubicInterpolateWithCaching<T, HalfPixelScaler>(image, st, coefficient, exclude_outside, output);
        break;
      case HALF_PIXEL_NN:
        bicubicInterpolateWithCaching<T, HalfPixelScalerNN>(image, st, coefficient, exclude_outside, output);
        break;
      default:
        break;
    }
  }
  NDArray::registerSpecialUse({output}, {image});
  return res;
}
Status resizeBicubicFunctorA(LaunchContext* context, NDArray * image, int const width, int const height,
                                 bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                                 double coefficient, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctorA_,
                        (context, image, width, height, alignCorners, coorMode, exclude_outside, coefficient, output),
                        SD_NUMERIC_TYPES);
}
// ------------------------------------------------------------------------------------------------------------------ //
Status resizeImagesFunctor(LaunchContext* context, NDArray * image, int const width, int const height,
                               ImageResizeMethods method, bool alignCorners, NDArray* output) {
  switch (method) {
    case kResizeBilinear:
      return resizeBilinearFunctor(context, image, width, height, alignCorners, false, output);
    case kResizeNearest:
      return resizeNeighborFunctor(context, image, width, height, ASYMMETRIC,
                                   alignCorners ? ROUND_PREFER_CEIL : FLOOR, alignCorners,
                                   output);
    case kResizeBicubic:
      return resizeBicubicFunctor(context, image, width, height, alignCorners, false, output);
    case kResizeArea:
      return resizeAreaFunctor(context, image, width, height, alignCorners, output);
    default:
      THROW_EXCEPTION("helper::resizeImagesFunctor: Wrong resize method.");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// --------------------------------------------------------------------------------------------------------------- //
// Crop and Resize helper implementation
// -------------------------------------------------------------------------------------------------------------- //
// cropAndResize kernel   type of input(images) and output should be the same
//
template <typename T, typename Z, typename I>
static SD_KERNEL void cropAndResizeKernel(T const* images, LongType const* imagesShape, Z const* boxes,
                                          LongType const* boxesShape, I const* indices, LongType const* indexShape, I const* cropSize, LongType const* cropShape, int method, double extrapolationVal, T* output, LongType const* outputShape, int numBoxes, int cropHeight, int cropWidth,
                                          int batchSize, int imageHeight, int imageWidth, int depth) {
  for (int b = blockIdx.x; b < numBoxes; b += gridDim.x) {
    LongType x1Pos[] = {b, 1};
    LongType y1Pos[] = {b, 0};
    LongType y2Pos[] = {b, 2};
    LongType x2Pos[] = {b, 3};
    LongType y1Offset, x1Offset, y2Offset, x2Offset;
    COORDS2INDEX(2, shape::stride(boxesShape), y1Pos, y1Offset);
    COORDS2INDEX(2, shape::stride(boxesShape), x1Pos, x1Offset);
    COORDS2INDEX(2, shape::stride(boxesShape), y2Pos, y2Offset);
    COORDS2INDEX(2, shape::stride(boxesShape), x2Pos, x2Offset);
    Z y1 = boxes[y1Offset];
    Z x1 = boxes[x1Offset];
    Z y2 = boxes[y2Offset];
    Z x2 = boxes[x2Offset];

    int bIn = indices[b];
    if (bIn >= batchSize) {
      continue;
    }

    Z heightScale = (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / Z(cropHeight - 1) : Z(0);
    Z widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / Z(cropWidth - 1) : Z(0);

    for (int y = threadIdx.x; y < cropHeight; y += blockDim.x) {
      const float inY =
          (cropHeight > 1) ? y1 * (imageHeight - 1) + y * heightScale : 0.5 * (y1 + y2) * (imageHeight - 1);
      if (inY < 0 || inY > imageHeight - 1) {
        for (int x = threadIdx.y; x < cropWidth; x += blockDim.y) {
          auto start = blockIdx.z * blockDim.x + threadIdx.z;
          auto step = blockDim.z * gridDim.z;
          for (int d = start; d < depth; d += step) {
            LongType zPos[] = {b, y, x, d};
            LongType zOffset;
            COORDS2INDEX(4, shape::stride(outputShape), zPos, zOffset);
            output[zOffset] = (Z)extrapolationVal;
          }
        }
        continue;
      }

      if (method == 0 /* bilinear */) {
        const int topYIndex = math::p_floor(inY);
        const int bottomYIndex = math::p_ceil(inY);
        const float y_lerp = inY - topYIndex;

        for (int x = 0; x < cropWidth; ++x) {
          const float in_x =
              (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageWidth - 1);
          if (in_x < 0 || in_x > imageWidth - 1) {
            auto start = blockIdx.z * blockDim.x + threadIdx.z;
            auto step = blockDim.z * gridDim.z;
            for (int d = start; d < depth; d += step) {
              LongType zPos[] = {b, y, x, d};
              LongType zOffset;
              COORDS2INDEX(4, shape::stride(outputShape), zPos, zOffset);
              output[zOffset] = (Z)extrapolationVal;
            }
            continue;
          }
          int left_x_index = math::p_floor(in_x);
          int right_x_index = math::p_ceil(in_x);
          T x_lerp = static_cast<T>(in_x) - static_cast<T>(left_x_index);

          auto start = blockIdx.z * blockDim.x + threadIdx.z;
          auto step = blockDim.z * gridDim.z;
          for (int d = start; d < depth; d += step) {
            LongType topLeftPos[] = {bIn, topYIndex, left_x_index, d};
            LongType topRightPos[] = {bIn, topYIndex, right_x_index, d};
            LongType bottomLeftPos[] = {bIn, bottomYIndex, left_x_index, d};
            LongType bottomRightPos[] = {bIn, bottomYIndex, right_x_index, d};
            LongType topLeftOffset, topRightOffset, bottomLeftOffset, bottomRightOffset;
            COORDS2INDEX(4, shape::stride(imagesShape), topLeftPos, topLeftOffset);
            COORDS2INDEX(4, shape::stride(imagesShape), topRightPos, topRightOffset);
            COORDS2INDEX(4, shape::stride(imagesShape), bottomLeftPos, bottomLeftOffset);
            COORDS2INDEX(4, shape::stride(imagesShape), bottomRightPos, bottomRightOffset);
            const T topLeft = images[topLeftOffset];
            const T topRight = images[topRightOffset];
            const T bottomLeft = images[bottomLeftOffset];
            const T bottomRight = images[bottomRightOffset];
            const T top = topLeft + (topRight - topLeft) * x_lerp;
            const T bottom = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
            LongType zPos[] = {b, y, x, d};
            LongType zOffset;
            COORDS2INDEX(4, shape::stride(outputShape), zPos, zOffset);
            output[zOffset] = Z(top + (bottom - top) * y_lerp);
          }
        }
      } else {  // method is "nearest neighbor"
        for (int x = 0; x < cropWidth; ++x) {
          const float inX =
              (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageWidth - 1);
          if (inX < 0 || inX > imageWidth - 1) {
            auto start = blockIdx.z * blockDim.x + threadIdx.z;
            auto step = blockDim.z * gridDim.z;
            for (int d = start; d < depth; d += step) {
              LongType zPos[] = {b, y, x, d};
              LongType zOffset;
              COORDS2INDEX(4, shape::stride(outputShape), zPos, zOffset);
              output[zOffset] = (Z)extrapolationVal;
            }
            continue;
          }
          const int closestXIndex = roundf(inX);
          const int closestYIndex = roundf(inY);
          auto start = blockIdx.z * blockDim.x + threadIdx.z;
          auto step = blockDim.z * gridDim.z;
          for (int d = start; d < depth; d += step) {
            LongType zPos[] = {b, y, x, d};
            LongType xPos[] = {bIn, closestYIndex, closestXIndex, d};
            LongType zOffset, xOffset;
            COORDS2INDEX(4, shape::stride(outputShape), zPos, zOffset);
            COORDS2INDEX(4, shape::stride(imagesShape), xPos, xOffset);
            output[zOffset] = images[xOffset];
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cropAndResizeFunctor main algorithm
//      context - launch context
//      images - batch of images (4D tensor - [batch, width, height, pixels])
//      boxes - 2D tensor with boxes for crop
//      indices - 2D int tensor with indices of boxes to crop
//      cropSize - 2D int tensor with crop box sizes
//      method - (one of 0 - bilinear, 1 - nearest)
//      extrapolationVal - double value of extrapolation
//      crops - output (4D tensor - [batch, outWidth, outHeight, pixels])
//
template <typename T, typename Z, typename I>
void cropAndResizeFunctor_(LaunchContext* context, NDArray * images, NDArray * boxes,
                           NDArray * indices, NDArray * cropSize, int method, double extrapolationVal,
                           NDArray* crops) {
  const int batchSize = images->sizeAt(0);
  const int imageHeight = images->sizeAt(1);
  const int imageWidth = images->sizeAt(2);

  const int numBoxes = crops->sizeAt(0);
  const int cropHeight = crops->sizeAt(1);
  const int cropWidth = crops->sizeAt(2);
  const int depth = crops->sizeAt(3);
  auto stream = context->getCudaStream();
  T const* imagesBuf = reinterpret_cast<T const*>(images->specialBuffer());
  Z const* boxesBuf = reinterpret_cast<Z const*>(boxes->specialBuffer());
  I const* indexBuf = reinterpret_cast<I const*>(indices->specialBuffer());
  I const* cropSizes = reinterpret_cast<I const*>(cropSize->specialBuffer());
  T* outBuf = reinterpret_cast<T*>(crops->specialBuffer());

  int threadsPerBlock = math::sd_max(imageHeight * imageWidth, cropHeight * cropWidth);
  if (threadsPerBlock > SD_MAX_NUM_THREADS / 4) threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  dim3 cropAndResizeDims = cropAndResize(batchSize,imageHeight,imageWidth,cropHeight,cropWidth);
  NDArray::prepareSpecialUse({crops}, {images, boxes, indices, cropSize});
  cropAndResizeKernel<T, Z, I><<<cropAndResizeDims.y, cropAndResizeDims.x, cropAndResizeDims.z, *stream>>>(
      imagesBuf, images->specialShapeInfo(), boxesBuf, boxes->specialShapeInfo(), indexBuf, indices->specialShapeInfo(),
      cropSizes, cropSize->specialShapeInfo(), method, extrapolationVal, outBuf, crops->specialShapeInfo(), numBoxes,
      cropHeight, cropWidth, batchSize, imageHeight, imageWidth, depth);
  NDArray::registerSpecialUse({crops}, {images, boxes, indices, cropSize});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cropAndResizeFunctor(LaunchContext* context, NDArray * images, NDArray * boxes,
                          NDArray * indices, NDArray * cropSize, int method, double extrapolationVal,
                          NDArray* crops) {

auto imagesDType = images->dataType();
auto boxesDType = boxes->dataType();
auto indicesDType = indices->dataType();
  BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                        (context, images, boxes, indices, cropSize, method, extrapolationVal, crops), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES, SD_INTEGER_TYPES);

}
BUILD_TRIPLE_TEMPLATE( void cropAndResizeFunctor_,
                      (sd::LaunchContext * context, NDArray * images, NDArray * boxes, NDArray * indices,
                       NDArray * cropSize, int method, double extrapolationVal, NDArray* crops),
                      SD_NUMERIC_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
