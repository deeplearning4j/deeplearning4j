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
#include <exceptions/cuda_exception.h>
#include <ops/declarable/helpers/image_resize.h>

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
static SD_KERNEL void computeInterpolationWeights(sd::LongType outSize, sd::LongType inSize, double scale,
                                                  sd::LongType channels, BilinearInterpolationData* interpolationData) {
  interpolationData[outSize].bottomIndex = 0;
  interpolationData[outSize].topIndex = 0;
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;
  Scaler scaler;
  for (sd::LongType i = outSize - tid; i >= 0; i -= step) {
    double in = scaler(i, scale);
    //            interpolationData[i].bottomIndex = static_cast<sd::LongType>(in);
    //            interpolationData[i].topIndex = sd::math::sd_min(interpolationData[i].bottomIndex + 1, inSize - 1);
    //            interpolationData[i].interpolarValue = in - interpolationData[i].bottomIndex;
    double const in_f = sd::math::p_floor<double>(in);
    double const in_c = sd::math::p_ceil<double>(in);
    interpolationData[i].bottomIndex =
        sd::math::sd_max(static_cast<sd::LongType>(in_f), (sd::LongType)0LL);  // static_cast<sd::LongType>(in);
    interpolationData[i].topIndex = sd::math::sd_min(static_cast<sd::LongType>(in_c), inSize - 1);
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
static void resizeImage(sd::LaunchContext* context, NDArray const* images, sd::LongType batchSize,
                        sd::LongType inHeight, sd::LongType inWidth, sd::LongType outHeight, sd::LongType outWidth,
                        sd::LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                        NDArray* output);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm kernel
//
template <typename T, typename Z>
static SD_KERNEL void resizeImageKernel(T const* input, sd::LongType const* inputShape, Z* outputYptr,
                                        sd::LongType const* outputShape, sd::LongType batchSize, sd::LongType outWidth,
                                        sd::LongType outHeight, sd::LongType channels, sd::LongType inRowSize,
                                        sd::LongType outRowSize, sd::LongType inBatchNumValues,
                                        BilinearInterpolationData* xs_, BilinearInterpolationData* ys_) {
  for (auto batch = blockIdx.x; batch < batchSize; batch += gridDim.x) {  // blockIdx.x as batch index
    auto pX = input + batch * inBatchNumValues;
    for (sd::LongType y = threadIdx.x; y < outHeight; y += blockDim.x) {
      const T* ys_input_lower_ptr = pX + ys_[y].bottomIndex * inRowSize;
      const T* ys_input_upper_ptr = pX + ys_[y].topIndex * inRowSize;
      double yVal = ys_[y].interpolarValue;
      auto pZ = outputYptr + (batch * outHeight + y) * outRowSize;
      for (sd::LongType x = 0; x < outWidth; x++) {
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
static void resizeImage_(sd::LaunchContext* context, NDArray const* images, sd::LongType batchSize,
                         sd::LongType inHeight, sd::LongType inWidth, sd::LongType outHeight, sd::LongType outWidth,
                         sd::LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                         NDArray* output) {
  sd::LongType inRowSize = inWidth * channels;
  sd::LongType inBatchNumValues = inHeight * inRowSize;
  sd::LongType outRowSize = outWidth * channels;
  auto stream = context->getCudaStream();
  T const* pInput = images->getDataBuffer()->specialAsT<T>();
                                                               // // this works only with 'c' direction
  F* pOutput = output->dataBuffer()->specialAsT<F>();
  resizeImageKernel<T, F><<<256, 256, 256, *stream>>>(pInput, images->specialShapeInfo(), pOutput,
                                                      output->specialShapeInfo(), batchSize, outWidth, outHeight,
                                                      channels, inRowSize, outRowSize, inBatchNumValues, xs_, ys_);

  auto err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeImage_: Cannot synchronize kernel execution", err);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename F>
static sd::Status resizeBilinearFunctor_(sd::LaunchContext* context, NDArray const* images, int const width,
                                         int const height, bool const alignCorners, bool const halfPixelCenter,
                                         NDArray* output) {
  const sd::LongType batchSize = images->sizeAt(0);
  const sd::LongType inHeight = images->sizeAt(1);
  const sd::LongType inWidth = images->sizeAt(2);
  const sd::LongType channels = images->sizeAt(3);

  const sd::LongType outHeight = output->sizeAt(1);
  const sd::LongType outWidth = output->sizeAt(2);

  // Handle no-op resizes efficiently.
  if (outHeight == inHeight && outWidth == inWidth) {
    output->assign(images);
    return sd::Status::OK;
  }

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorners);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorners);

  BilinearInterpolationData* xs_;  // = xs.data();
  BilinearInterpolationData* ys_;  // = xs.data();

  cudaError_t err = cudaMalloc(&xs_, sizeof(BilinearInterpolationData) * (outWidth + 1));
  if (err != 0) {
    throw cuda_exception::build("helpers::resize_image: Cannot allocate memory for vertical parts rectangulars", err);
  }

  err = cudaMalloc(&ys_, sizeof(BilinearInterpolationData) * (outHeight + 1));
  if (err != 0) {
    throw cuda_exception::build("helpers::resize_image: Cannot allocate memory for horizontal parts rectangulars", err);
  }
  auto stream = context->getCudaStream();
  // Compute the cached interpolation weights on the x and y dimensions.
  if (halfPixelCenter) {
    computeInterpolationWeights<HalfPixelScaler><<<256, 512, 512, *stream>>>(outHeight, inHeight, heightScale, 0, ys_);
    computeInterpolationWeights<HalfPixelScaler>
        <<<256, 512, 512, *stream>>>(outWidth, inWidth, widthScale, channels, xs_);
  } else {
    computeInterpolationWeights<LegacyScaler><<<256, 512, 512, *stream>>>(outHeight, inHeight, heightScale, 0, ys_);
    computeInterpolationWeights<LegacyScaler><<<256, 512, 512, *stream>>>(outWidth, inWidth, widthScale, channels, xs_);
  }

  NDArray::prepareSpecialUse({output}, {images});
  resizeImage_<T, F>(context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output);
  err = cudaStreamSynchronize(*stream);
  NDArray::registerSpecialUse({output}, {images});

  err = cudaFree(xs_);
  if (err != 0) {
    throw cuda_exception::build("helpers::resize_image: Cannot deallocate memory for vertical parts rectangulars", err);
  }

  err = cudaFree(ys_);
  if (err != 0) {
    throw cuda_exception::build("helpers::resize_image: Cannot deallocate memory for horizontical parts rectangulars",
                                err);
  }

  return sd::Status::OK;
}

typedef float (*MODE_FUNC)(float);

SD_DEVICE MODE_FUNC mode_functions[4] = {sd::math::p_floor<float>, sd::math::p_round_prefer_floor<float>,
                                         sd::math::p_round_prefer_ceil<float>, sd::math::p_ceil<float>};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by interpolation nearest neighbor algorithm kernel
//
template <typename T, typename Scaler>
static SD_KERNEL void resizeNeighborKernel(T const* input, sd::LongType const* inputShape, T* output,
                                           sd::LongType const* outputShape, sd::LongType batchSize,
                                           sd::LongType inWidth, sd::LongType inHeight, sd::LongType outWidth,
                                           sd::LongType outHeight, sd::LongType channels, double widthScale,
                                           double heightScale, NearestMode nearestMode) {
  constexpr bool halfPixelCenter =
      std::is_same<Scaler, HalfPixelScaler>::value || std::is_same<Scaler, HalfPixelScalerNN>::value;
  MODE_FUNC modeFunc;
  switch (nearestMode) {
    case NearestMode::FLOOR:
      modeFunc = mode_functions[0];
      break;
    case NearestMode::ROUND_PREFER_FLOOR:
      modeFunc = mode_functions[1];
      break;
    case NearestMode::ROUND_PREFER_CEIL:
      modeFunc = mode_functions[2];
      break;
    case NearestMode::CEIL:
      modeFunc = mode_functions[3];
      break;
    default:
      modeFunc = mode_functions[0];
  }
  Scaler scaler;

  if (blockIdx.x < batchSize) {
    auto b = blockIdx.x;
    for (int y = threadIdx.x; y < outHeight; y += blockDim.x) {
      auto posY = static_cast<sd::LongType>(modeFunc(scaler(y, heightScale)));
      sd::LongType inY = sd::math::sd_min(posY, inHeight - 1);
      if (halfPixelCenter) {
        inY = sd::math::sd_max(0LL, inY);
      }

      for (int x = threadIdx.y; x < outWidth; x += blockDim.y) {
        auto posX = static_cast<sd::LongType>(modeFunc(scaler(x, widthScale)));
        sd::LongType inX = sd::math::sd_min(posX, inWidth - 1);
        if (halfPixelCenter) {
          inX = sd::math::sd_max(0LL, inX);
        }

        auto start = blockIdx.z * blockDim.z + threadIdx.z;
        auto step = blockDim.z * gridDim.z;

        for (sd::LongType e = start; e < channels; e += step) {
          sd::LongType posX[] = {b, inY, inX, e};
          sd::LongType posZ[] = {b, y, x, e};
          auto xIndex = shape::getOffset(inputShape, posX);
          auto zIndex = shape::getOffset(outputShape, posZ);
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
sd::Status resizeNeighborFunctor_(sd::LaunchContext* context, NDArray const* images, int const width, int const height,
                                  CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                  NDArray* output) {
  const sd::LongType batchSize = images->sizeAt(0);
  const sd::LongType inHeight = images->sizeAt(1);
  const sd::LongType inWidth = images->sizeAt(2);
  const sd::LongType channels = images->sizeAt(3);

  const sd::LongType outHeight = output->sizeAt(1);
  const sd::LongType outWidth = output->sizeAt(2);

  // Handle no-op resizes efficiently.
  if (outHeight == inHeight && outWidth == inWidth) {
    output->assign(images);
    return sd::Status::OK;
  }

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorner);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorner);

  auto imagesBuffer = images->getDataBuffer()->specialAsT<T>();  // reinterpret_cast<T const*>(images->specialBuffer());
  auto outputBuffer = output->dataBuffer()->specialAsT<T>();     // reinterpret_cast<T*>(output->specialBuffer());
  auto stream = context->getCudaStream();

  NDArray::prepareSpecialUse({output}, {images});
  switch (coorMode) {
    case ASYMMETRIC:
      resizeNeighborKernel<T, LegacyScaler><<<batchSize, outHeight * outWidth, 512, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL:
      resizeNeighborKernel<T, HalfPixelScaler><<<batchSize, outHeight * outWidth, 512, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL_NN:
      resizeNeighborKernel<T, HalfPixelScalerNN><<<batchSize, outHeight * outWidth, 512, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    default:
      resizeNeighborKernel<T, HalfPixelScaler><<<batchSize, outHeight * outWidth, 512, *stream>>>(
          imagesBuffer, images->specialShapeInfo(), outputBuffer, output->specialShapeInfo(), batchSize, inWidth,
          inHeight, outWidth, outHeight, channels, widthScale, heightScale, nearestMode);
      break;
  };

  NDArray::registerSpecialUse({output}, {images});

  return sd::Status::OK;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resizeImage - resize bilinear algorithm caller
//
void resizeImage(sd::LaunchContext* context, NDArray const* images, sd::LongType batchSize, sd::LongType inHeight,
                 sd::LongType inWidth, sd::LongType outHeight, sd::LongType outWidth, sd::LongType channels,
                 BilinearInterpolationData* xs_, BilinearInterpolationData* ys_, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(
      images->dataType(), output->dataType(), resizeImage_,
      (context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output),
      SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
}

BUILD_DOUBLE_TEMPLATE(template void resizeImage_,
                      (sd::LaunchContext * context, NDArray const* images, sd::LongType batchSize,
                       sd::LongType inHeight, sd::LongType inWidth, sd::LongType outHeight, sd::LongType outWidth,
                       sd::LongType channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_,
                       NDArray* output),
                      SD_NUMERIC_TYPES, SD_FLOAT_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
sd::Status resizeBilinearFunctor(sd::LaunchContext* context, NDArray const* images, int width, int height,
                                 bool const alignCorners, bool const halfPixelCenter, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_,
                        (context, images, width, height, alignCorners, halfPixelCenter, output), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
sd::Status resizeNeighborFunctor(sd::LaunchContext* context, NDArray const* images, int const width, int const height,
                                 CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                 NDArray* output) {
  BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_,
                        (context, images, width, height, coorMode, nearestMode, alignCorner, output), SD_COMMON_TYPES);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bicubic interpolation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static SD_KERNEL void initCoefTableKernel(const float a, float* table, sd::LongType tableSize) {
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
  auto err = cudaMalloc(&coeffs_table, sizeof(float) * ((kTableSize + 1) * 2));
  if (err != 0) {
    throw cuda_exception::build("helpers::initCoeffsTable: Cannot allocate memory for vertical parts rectangulars",
                                err);
  }

  initCoefTableKernel<<<128, 128, 128, *stream>>>(static_cast<float>(a), coeffs_table, kTableSize);
  err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::initCoeffsTable: Cannot syncronize kernel", err);
  }

  return coeffs_table;
}

static SD_KERNEL void accumulateChannelsKernel(WeightsAndIndices* pXWais, sd::LongType outWidth,
                                               sd::LongType channels) {
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
                                                     WeightsAndIndices* pXWais, sd::LongType inWidth, float widthScale,
                                                     sd::LongType outWidth, sd::LongType channels,
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
  auto err = cudaMalloc(&pCalcD, sizeof(CachedInterpolationCalculator));
  if (err != 0) {
    cuda_exception::build(
        "helpers::computeXWeightsAndIndices: Cannot allocated device memory for interpolate calculator", err);
  }
  err = cudaMemcpyAsync(pCalcD, &calc, sizeof(CachedInterpolationCalculator), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot set up device memory for interpolate calculator",
                          err);
  }

  advanceWeightsAndIndicesKernel<Scaler><<<128, 128, 128, *stream>>>(coeffsTable, pCalcD, pXWais, resizerState.inWidth,
                                                                     resizerState.widthScale, outWidth,
                                                                     resizerState.channels, exclude_outside);
  err = cudaFree(pCalcD);
  if (err != 0) {
    cuda_exception::build(
        "helpers::computeXWeightsAndIndices: Cannot deallocated device memory for interpolate calculator", err);
  }
  err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    cuda_exception::build(
        "helpers::computeXWeightsAndIndices: Cannot synchronize stream after advance weights and indicers", err);
  }
  // Scale the values so they can be used as offsets into buffers.
  accumulateChannelsKernel<<<128, 128, 512, *stream>>>(pXWais, outWidth, resizerState.wStride);
  err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot synchronize stream after accumulate channels",
                          err);
  }
}

template <typename T, typename Scaler>
static SD_KERNEL void bicubicInterpolateWithCachingKernel(float const* cachedTable, T const* inputPtr,
                                                          ImageResizerState* pResizerState, WeightsAndIndices* xWais,
                                                          bool exclude_outside, float* outputPtr) {
  //        auto numChannels = pResizerState->channels;
  const auto batchStride = pResizerState->bStride;
  const auto hStride = pResizerState->hStride;
  const auto cStride = pResizerState->cStride;
  for (sd::LongType b = blockIdx.x; b < pResizerState->batchSize; b += gridDim.x) {
    auto pInput = inputPtr + b * batchStride;

    float* cachedValue;
    for (sd::LongType y = threadIdx.x; y < pResizerState->outHeight; y += blockDim.x) {
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
        for (sd::LongType x = 0; x < pResizerState->outWidth; ++x) {
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
        for (sd::LongType x = 0; x < pResizerState->outWidth; ++x) {
          const WeightsAndIndices& xWai = xWais[x];
          // Shift values in cachedValue to fill first '_advance' values.
          switch (xWai._advance) {
            case 3:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 1];
                cachedValue[4 * c + 1] = cachedValue[4 * c + 2];
                cachedValue[4 * c + 2] = cachedValue[4 * c + 3];
              }
              break;
            case 2:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 2];
                cachedValue[4 * c + 1] = cachedValue[4 * c + 3];
              }
              break;
            case 1: {
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] = cachedValue[4 * c + 3];
              }
              break;
            }
          }

          // Set the remaining '4-_advance' values by computing.
          switch (xWai._advance) {
            case 0:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 0] =
                    computeYInterpolation(0, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 1:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 1] =
                    computeYInterpolation(1, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 2:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 2] =
                    computeYInterpolation(2, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
            case 3:
              for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
                cachedValue[4 * c + 3] =
                    computeYInterpolation(3, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
              }
              // break;
          }
          for (sd::LongType c = 0; c < pResizerState->channels; ++c) {
            auto res = compute(&cachedValue[4 * c], xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
            pOutput[x * pResizerState->channels + c] = res;
          }
        }
      }
    }
  }
}

template <typename T, typename Scaler>
static void bicubicInterpolateWithCaching(NDArray const* image, const ImageResizerState& resizerState,
                                          const double coefficient, bool exclude_outside, NDArray* output) {
  const auto numChannels = resizerState.channels;
  auto stream = resizerState.stream;  // output->getContext()->getCudaStream();
  ImageResizerState* resizerStateD;
  auto err = cudaMalloc(&resizerStateD, sizeof(ImageResizerState));
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot allocate memory for resizerState", err);
  }
  err = cudaMemcpyAsync(resizerStateD, &resizerState, sizeof(ImageResizerState), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot set up memory for resizerState", err);
  }


  WeightsAndIndices* xWais;
  err = cudaMalloc(&xWais, sizeof(WeightsAndIndices) * resizerState.outWidth);
  if (err != 0) {
    throw cuda_exception::build(
        "helpers::bicubicInterpolateWithCaching: Cannot allocate memory for weights and indices", err);
  }

  auto coeffsTable = initCoeffsTable(
      coefficient, stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error",
                                err);
  }
  computeXWeightsAndIndices<Scaler>(coeffsTable, resizerState, xWais, exclude_outside);
  err = cudaStreamQuery(*stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error",
                                err);
  }

  const T* pInput = image->getDataBuffer()->specialAsT<T>();
  float* pOutput = output->dataBuffer()->specialAsT<float>();  //_data.data();
  bicubicInterpolateWithCachingKernel<T, Scaler>
      <<<128, 1, 512, *stream>>>(coeffsTable, pInput, resizerStateD, xWais, exclude_outside, pOutput);
  err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Kernels finished with error", err);
  }

  err = cudaFree(resizerStateD);
  if (err != 0) {
    throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for resizerState",
                                err);
  }


  err = cudaFree(xWais);
  if (err != 0) {
    throw cuda_exception::build(
        "helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for weights and indices", err);
  }

  err = cudaFree(coeffsTable);
  if (err != 0) {
    throw cuda_exception::build(
        "helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for coefficients table", err);
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
sd::Status resizeBicubicFunctor_(sd::LaunchContext* context, NDArray const* image, int width, int height,
                                 bool preserveAspectRatio, bool antialias, NDArray* output) {
  return sd::Status::OK;
}

sd::Status resizeBicubicFunctor(sd::LaunchContext* context, NDArray const* image, int width, int height,
                                bool preserveAspectRatio, bool antialias, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctor_,
                        (context, image, width, height, preserveAspectRatio, antialias, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE(template sd::Status resizeBicubicFunctor_,
                      (sd::LaunchContext * context, NDArray const* image, int width, int height,
                       bool preserveAspectRatio, bool antialias, NDArray* output),
                      SD_NUMERIC_TYPES);
// ------------------------------------------------------------------------------------------------------------------ //

static SD_KERNEL void fillInterpolationCache(CachedInterpolation* xCached, sd::LongType cacheLen, sd::LongType inWidth,
                                             float widthScale) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto increment = blockDim.x * gridDim.x;

  for (auto x = start; x < cacheLen; x += increment) {
    auto& xCache = xCached[x];
    const float inX = x * widthScale;
    const float inX1 = (x + 1) * widthScale;

    sd::LongType v = math::sd_floor<float, sd::LongType>(inX);
    xCache.start = v;
    xCache.startScale = v < inX ? (v + 1 > inX1 ? widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
    v = math::sd_ceil<float, sd::LongType>(inX1);
    xCache.end = v--;
    xCache.endMinusOneScale = v < inX ? (v + 1 > inX1 ? widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
    xCache.needsBounding =
        bound(xCache.start, inWidth) != xCache.start || bound(xCache.end - 1, inWidth) != (xCache.end - 1);
  }
}

// ------------------------------------------------------------------------------------------------------------------ //

template <typename T>
static SD_KERNEL void resizeAreaKernel(ImageResizerState const* pSt, CachedInterpolation const* caches, float scale,
                                       T const* inputPtr, sd::LongType const* inputShape, float* outputPtr,
                                       sd::LongType const* outputShape,
                                       ScaleCache<T>* cachePool) {  // batch * outWidth * outHeight

  for (auto batch = blockIdx.x; batch < pSt->batchSize; batch += gridDim.x) {
    for (auto y = threadIdx.x; y < pSt->outHeight; y += blockDim.x) {
      const float inY = y * pSt->heightScale;
      const float inY1 = (y + 1) * pSt->heightScale;
      // The start and end height indices of all the cells that could
      // contribute to the target cell.
      const sd::LongType yStart = math::sd_floor<float, sd::LongType>(inY);
      const sd::LongType yEnd = math::sd_ceil<float, sd::LongType>(inY1);
      auto scalesDim = yEnd - yStart;
      auto yScaleCache = cachePool + (batch * pSt->outHeight + y) * pSt->outWidth;

      float* output = outputPtr + (batch * pSt->outHeight + y) * pSt->channels * pSt->outWidth;
      // int k = 0;
      for (sd::LongType i = yStart, k = 0; i < yEnd; ++i, ++k) {
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
        for (sd::LongType x = 0; x < pSt->outWidth; ++x) {
          const CachedInterpolation& xCache = caches[x];
          computePatchSumOf3Channels<T>(scale, *pSt, yScaleCache, scalesDim, xCache, output);
          output += pSt->channels;
        }
      } else {
        for (sd::LongType x = 0; x < pSt->outWidth; ++x) {
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
                       NDArray const* input, NDArray* output) {
  T const* inputPtr = reinterpret_cast<T const*>(input->specialBuffer());
  float scale = 1.f / (st.heightScale * st.widthScale);
  auto outputPtr =
      reinterpret_cast<float*>(output->specialBuffer());  // output is always float. TO DO: provide another float types
                                                          // also with  template <typename X, typename Z> declaration
  ImageResizerState* pSt;
  auto err = cudaMalloc(&pSt, sizeof(ImageResizerState));
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: Cannot allocate memory for ImageResizerState", err);
  }

  err = cudaMemcpyAsync(pSt, &st, sizeof(ImageResizerState), cudaMemcpyHostToDevice, *stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: Cannot copy to device memory", err);
  }
  ScaleCache<T>* cachePool;
  auto cachePoolSize = sizeof(ScaleCache<T>) * st.batchSize * st.outWidth * st.outHeight;
  err = cudaMalloc(&cachePool, cachePoolSize);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: Cannot allocate memory for cache", err);
  }
  resizeAreaKernel<T><<<128, 128, 2048, *stream>>>(pSt, cache, scale, inputPtr, input->specialShapeInfo(), outputPtr,
                                                   output->specialShapeInfo(), cachePool);
  err = cudaStreamSynchronize(*stream);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: An error occured with kernel running", err);
  }
  err = cudaFree(cachePool);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: Cannot deallocate memory for cache", err);
  }
  err = cudaFree(pSt);
  if (err != 0) {
    throw cuda_exception::build("helpers::resizeArea: Cannot deallocate memory for ImageResizeState", err);
  }
}
// ------------------------------------------------------------------------------------------------------------------ //
template <typename T>
sd::Status resizeAreaFunctor_(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, NDArray* output) {
  ImageResizerState st(alignCorners, false);  // Create resize info
  auto res = st.validateAndCalculateOutputSize(image, width, height);
  auto stream = context->getCudaStream();
  if (sd::Status::OK == res) {
    CachedInterpolation* xCached;
    //(st.outWidth);
    auto err = cudaMalloc(&xCached, sizeof(CachedInterpolation) * st.outWidth);
    if (err != 0) {
      throw cuda_exception::build("helpers::resizeAreaFunctor_: Cannot allocate memory for cached interpolations", err);
    }
    NDArray::prepareSpecialUse({output}, {image});
    fillInterpolationCache<<<128, 128, 256, *stream>>>(xCached, st.outWidth, st.inWidth, st.widthScale);
    resizeArea<T>(stream, st, xCached, image, output);
    err = cudaStreamSynchronize(*stream);
    if (err != 0) {
      throw cuda_exception::build("helpers::resizeAreaFunctor_: Error occured when kernel was running", err);
    }
    err = cudaFree(xCached);
    if (err != 0) {
      throw cuda_exception::build("helpers::resizeAreaFunctor_: Cannot deallocate memory for cached interpolations",
                                  err);
    }
    NDArray::registerSpecialUse({output}, {image});
  }

  return res;
}
sd::Status resizeAreaFunctor(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                             bool const alignCorners, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_,
                        (context, image, width, height, alignCorners, output), SD_NUMERIC_TYPES);
}

// ------------------------------------------------------------------------------------------------------------------ //
// simplified bicubic resize without antialiasing
//
template <typename T>
sd::Status resizeBicubicFunctorA_(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                  bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                                  double coefficient, NDArray* output) {
  ImageResizerState st(alignCorners, coorMode == HALF_PIXEL,
                       context->getCudaStream());  // align_corners, half_pixel_align
  NDArray::prepareSpecialUse({output}, {image});
  sd::Status res = st.validateAndCreateOutput(image, width, height);
  if (res == sd::Status::OK) {
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
sd::Status resizeBicubicFunctorA(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                                 double coefficient, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctorA_,
                        (context, image, width, height, alignCorners, coorMode, exclude_outside, coefficient, output),
                        SD_NUMERIC_TYPES);
}
// ------------------------------------------------------------------------------------------------------------------ //
sd::Status resizeImagesFunctor(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                               ImageResizeMethods method, bool alignCorners, NDArray* output) {
  switch (method) {
    case kResizeBilinear:
      return resizeBilinearFunctor(context, image, width, height, alignCorners, false, output);
    case kResizeNearest:
      return resizeNeighborFunctor(context, image, width, height, CoordinateTransformationMode::ASYMMETRIC,
                                   alignCorners ? NearestMode::ROUND_PREFER_CEIL : NearestMode::FLOOR, alignCorners,
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
static SD_KERNEL void cropAndResizeKernel(T const* images, sd::LongType const* imagesShape, Z const* boxes,
                                          sd::LongType const* boxesShape, I const* indices,
                                          sd::LongType const* indexShape, I const* cropSize,
                                          sd::LongType const* cropShape, int method, double extrapolationVal, T* output,
                                          sd::LongType const* outputShape, int numBoxes, int cropHeight, int cropWidth,
                                          int batchSize, int imageHeight, int imageWidth, int depth) {
  for (int b = blockIdx.x; b < numBoxes; b += gridDim.x) {
    sd::LongType x1Pos[] = {b, 1};
    sd::LongType y1Pos[] = {b, 0};
    sd::LongType y2Pos[] = {b, 2};
    sd::LongType x2Pos[] = {b, 3};
    Z y1 = boxes[shape::getOffset(boxesShape, y1Pos)];  //->t<T>(b, 0)];
    Z x1 = boxes[shape::getOffset(boxesShape, x1Pos)];
    Z y2 = boxes[shape::getOffset(boxesShape, y2Pos)];
    Z x2 = boxes[shape::getOffset(boxesShape, x2Pos)];

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
            sd::LongType zPos[] = {b, y, x, d};
            auto zIndex = shape::getOffset(outputShape, zPos);
            output[zIndex] = (Z)extrapolationVal;
          }
        }
        continue;
      }

      if (method == 0 /* bilinear */) {
        const int topYIndex = sd::math::p_floor(inY);
        const int bottomYIndex = sd::math::p_ceil(inY);
        const float y_lerp = inY - topYIndex;

        for (int x = 0; x < cropWidth; ++x) {
          const float in_x =
              (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale : 0.5 * (x1 + x2) * (imageWidth - 1);
          if (in_x < 0 || in_x > imageWidth - 1) {
            auto start = blockIdx.z * blockDim.x + threadIdx.z;
            auto step = blockDim.z * gridDim.z;
            for (int d = start; d < depth; d += step) {
              sd::LongType zPos[] = {b, y, x, d};
              auto zIndex = shape::getOffset(outputShape, zPos);
              output[zIndex] = (Z)extrapolationVal;
              //                                crops->p(b, y, x, d, extrapolationVal);
            }
            continue;
          }
          int left_x_index = math::p_floor(in_x);
          int right_x_index = math::p_ceil(in_x);
          T x_lerp = in_x - left_x_index;

          auto start = blockIdx.z * blockDim.x + threadIdx.z;
          auto step = blockDim.z * gridDim.z;
          for (int d = start; d < depth; d += step) {
            sd::LongType topLeftPos[] = {bIn, topYIndex, left_x_index, d};
            sd::LongType topRightPos[] = {bIn, topYIndex, right_x_index, d};
            sd::LongType bottomLeftPos[] = {bIn, bottomYIndex, left_x_index, d};
            sd::LongType bottomRightPos[] = {bIn, bottomYIndex, right_x_index, d};
            const T topLeft(
                images[shape::getOffset(imagesShape, topLeftPos)]);  //->e<float>(bIn, topYIndex, left_x_index, d));
            const T topRight(
                images[shape::getOffset(imagesShape, topRightPos)]);  //->e<float>(bIn, topYIndex, right_x_index, d));
            const T bottomLeft(images[shape::getOffset(
                imagesShape, bottomLeftPos)]);  //->e<float>(bIn, bottomYIndex, left_x_index, d));
            const T bottomRight(images[shape::getOffset(
                imagesShape, bottomRightPos)]);  //->e<float>(bIn, bottomYIndex, right_x_index, d));
            const T top = topLeft + (topRight - topLeft) * x_lerp;
            const T bottom = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
            sd::LongType zPos[] = {b, y, x, d};
            auto zIndex = shape::getOffset(outputShape, zPos);
            output[zIndex] = Z(top + (bottom - top) * y_lerp);
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
              sd::LongType zPos[] = {b, y, x, d};
              auto zIndex = shape::getOffset(outputShape, zPos);
              output[zIndex] = (Z)extrapolationVal;
            }
            continue;
          }
          const int closestXIndex = roundf(inX);
          const int closestYIndex = roundf(inY);
          auto start = blockIdx.z * blockDim.x + threadIdx.z;
          auto step = blockDim.z * gridDim.z;
          for (int d = start; d < depth; d += step) {
            sd::LongType zPos[] = {b, y, x, d};
            sd::LongType xPos[] = {bIn, closestYIndex, closestXIndex, d};
            auto zIndex = shape::getOffset(outputShape, zPos);
            auto xIndex = shape::getOffset(imagesShape, xPos);
            output[zIndex] = images[xIndex];
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
void cropAndResizeFunctor_(sd::LaunchContext* context, NDArray const* images, NDArray const* boxes,
                           NDArray const* indices, NDArray const* cropSize, int method, double extrapolationVal,
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

  NDArray::prepareSpecialUse({crops}, {images, boxes, indices, cropSize});
  cropAndResizeKernel<T, Z, I><<<batchSize, threadsPerBlock, 256, *stream>>>(
      imagesBuf, images->specialShapeInfo(), boxesBuf, boxes->specialShapeInfo(), indexBuf, indices->specialShapeInfo(),
      cropSizes, cropSize->specialShapeInfo(), method, extrapolationVal, outBuf, crops->specialShapeInfo(), numBoxes,
      cropHeight, cropWidth, batchSize, imageHeight, imageWidth, depth);
  NDArray::registerSpecialUse({crops}, {images, boxes, indices, cropSize});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cropAndResizeFunctor(sd::LaunchContext* context, NDArray const* images, NDArray const* boxes,
                          NDArray const* indices, NDArray const* cropSize, int method, double extrapolationVal,
                          NDArray* crops) {
  BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                        (context, images, boxes, indices, cropSize, method, extrapolationVal, crops), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES, SD_INTEGER_TYPES);
  //
}
BUILD_TRIPLE_TEMPLATE(template void cropAndResizeFunctor_,
                      (sd::LaunchContext * context, NDArray const* images, NDArray const* boxes, NDArray const* indices,
                       NDArray const* cropSize, int method, double extrapolationVal, NDArray* crops),
                      SD_NUMERIC_TYPES, SD_FLOAT_TYPES, SD_INTEGER_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
