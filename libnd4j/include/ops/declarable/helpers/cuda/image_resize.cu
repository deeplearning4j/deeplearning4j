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
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/image_resize.h>

#include "execution/cuda/LaunchDims.h"
#include <system/selective_rendering.h>

#include <type_traits>

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The images are [batch, height, width, channels] arrays. Every kernel below reads the input and writes the output
// through the strides of its array, from the buffer pointer of the array (which already includes the offset of a
// view), so views and arrays of any order work; the kernels that need a dense output (area) get a dense copy for
// other layouts. The temporary device arrays come from the PointersManager, which is safe in a CUDA graph capture
// (and frees them stream-ordered behind the kernels), and the kernels run on the context's stream with no
// synchronization of their own (the one barrier is documented in resizeAreaFunctor_).

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// launch errors of an asynchronous launch; inside a CUDA graph capture nothing has run yet
static SD_INLINE void checkLaunch(cudaStream_t* stream, const char* message) {
  if (!DebugHelper::inGraphCapture(stream)) {
    DebugHelper::checkGlobalErrorCode(message);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// blocks and threads of a launch with one thread per output pixel: the named launch dimensions (x = blocks,
// y = threads) give the thread count and cap the grid, the kernels stride over the pixels the grid does not cover
static void pixelLaunch(const dim3& named, LongType pixels, unsigned int& blocks, unsigned int& threads) {
  threads = named.y > 0 ? named.y : 1;
  if (threads > SD_MAX_NUM_THREADS) threads = SD_MAX_NUM_THREADS;
  const LongType needed = math::sd_max<LongType>(1, (pixels + threads - 1) / threads);
  const LongType cap = math::sd_max<LongType>(1, named.x);
  blocks = static_cast<unsigned int>(math::sd_min<LongType>(needed, cap));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// true when the elements of the array lie as in a dense C-order array (axes of size one may have any stride): its
// buffer pointer plus the linear logical index addresses them
static bool isDenseCOrder(NDArray* array) {
  LongType expected = 1;
  for (int d = array->rankOf() - 1; d >= 0; d--) {
    const LongType extent = array->sizeAt(d);
    if (extent != 1 && array->strideAt(d) != expected) return false;
    expected *= extent;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// computeInterpolationWeights kernel: one thread per entry
//      outSize - output length
//      inSize - input size
//      scale - input scale
//      indexStride - the entries' source indices are multiplied by it (the input's stride along the axis) so that
//                    they are offsets into the input
//      interporationData - result (outSize + 1 entries; the last one is a sentinel that holds zeros, as on the CPU)
//
template <class Scaler>
static SD_KERNEL void computeInterpolationWeights(LongType outSize, LongType inSize, double scale,
                                                  LongType indexStride, BilinearInterpolationData* interpolationData) {
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  if (start == 0) {
    interpolationData[outSize].bottomIndex = 0;
    interpolationData[outSize].topIndex = 0;
  }
  Scaler scaler;
  for (LongType i = start; i < outSize; i += step) {
    double const in = scaler(static_cast<int>(i), static_cast<float>(scale));
    double const in_f = math::p_floor<double>(in);
    double const in_c = math::p_ceil<double>(in);
    interpolationData[i].bottomIndex = math::sd_max(static_cast<LongType>(in_f), (LongType)0LL) * indexStride;
    interpolationData[i].topIndex = math::sd_min(static_cast<LongType>(in_c), inSize - 1) * indexStride;
    interpolationData[i].interpolarValue = in - in_f;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm kernel: one thread per output pixel (grid-stride over batch *
// outHeight * outWidth), the channels of the pixel in a loop. The weights xs_ hold offsets along the input's width
// axis, the weights ys_ plain row indices. The arithmetic is done in double, as on the CPU, and rounded once into Z.
//
template <typename T, typename Z>
static SD_KERNEL void resizeImageKernel(T const* input, Z* output, LongType batchSize, LongType outHeight,
                                        LongType outWidth, LongType channels, LongType inBatchStride,
                                        LongType inRowStride, LongType inChannelStride, LongType outBatchStride,
                                        LongType outRowStride, LongType outColumnStride, LongType outChannelStride,
                                        BilinearInterpolationData const* xs_, BilinearInterpolationData const* ys_) {
  const LongType pixels = batchSize * outHeight * outWidth;
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType pixel = start; pixel < pixels; pixel += step) {
    const LongType x = pixel % outWidth;
    const LongType y = (pixel / outWidth) % outHeight;
    const LongType batch = pixel / (outWidth * outHeight);

    const T* pX = input + batch * inBatchStride;
    const T* ys_input_lower_ptr = pX + ys_[y].bottomIndex * inRowStride;
    const T* ys_input_upper_ptr = pX + ys_[y].topIndex * inRowStride;
    const double yVal = ys_[y].interpolarValue;
    const LongType xsBottom = xs_[x].bottomIndex;
    const LongType xsTop = xs_[x].topIndex;
    const double xVal = xs_[x].interpolarValue;
    Z* pZ = output + batch * outBatchStride + y * outRowStride + x * outColumnStride;
    // process interpolation for all channels
    for (LongType c = 0; c < channels; c++) {
      const LongType channelOffset = c * inChannelStride;
      const double topLeft = static_cast<double>(ys_input_lower_ptr[xsBottom + channelOffset]);
      const double topRight = static_cast<double>(ys_input_lower_ptr[xsTop + channelOffset]);
      const double bottomLeft = static_cast<double>(ys_input_upper_ptr[xsBottom + channelOffset]);
      const double bottomRight = static_cast<double>(ys_input_upper_ptr[xsTop + channelOffset]);
      const double top = imageResizeLerp<double>(topLeft, topRight, xVal);
      const double bottom = imageResizeLerp<double>(bottomLeft, bottomRight, xVal);
      pZ[c * outChannelStride] = static_cast<Z>(imageResizeLerp<double>(top, bottom, yVal));
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
  if (output->lengthOf() == 0) return Status::OK;

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorners);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorners);

  auto stream = context->getCudaStream();
  PointersManager pm(context, "resizeBilinear");
  auto xs_ = reinterpret_cast<BilinearInterpolationData*>(
      pm.allocateDevMem(sizeof(BilinearInterpolationData) * (outWidth + 1)));
  auto ys_ = reinterpret_cast<BilinearInterpolationData*>(
      pm.allocateDevMem(sizeof(BilinearInterpolationData) * (outHeight + 1)));

  NDArray::prepareSpecialUse({output}, {images});

  // Compute the cached interpolation weights on the x and y dimensions: the x weights as offsets along the input's
  // width axis, the y weights as row indices.
  const dim3 weightDims = getLaunchDims("image_resize_interp_weights");
  const LongType inColumnStride = images->strideAt(2);
  if (halfPixelCenter) {
    computeInterpolationWeights<HalfPixelScaler><<<weightDims.x, weightDims.y, 0, *stream>>>(outHeight, inHeight, heightScale, 1, ys_);
    computeInterpolationWeights<HalfPixelScaler>
        <<<weightDims.x, weightDims.y, 0, *stream>>>(outWidth, inWidth, widthScale, inColumnStride, xs_);
  } else {
    computeInterpolationWeights<LegacyScaler><<<weightDims.x, weightDims.y, 0, *stream>>>(outHeight, inHeight, heightScale, 1, ys_);
    computeInterpolationWeights<LegacyScaler><<<weightDims.x, weightDims.y, 0, *stream>>>(outWidth, inWidth, widthScale, inColumnStride, xs_);
  }
  checkLaunch(stream, "computeInterpolationWeights failed: ");

  unsigned int blocks, threads;
  pixelLaunch(getLaunchDims("image_resize"), batchSize * outHeight * outWidth, blocks, threads);
  resizeImageKernel<T, F><<<blocks, threads, 0, *stream>>>(
      reinterpret_cast<T const*>(images->specialBuffer()), reinterpret_cast<F*>(output->specialBuffer()), batchSize,
      outHeight, outWidth, channels, images->strideAt(0), images->strideAt(1), images->strideAt(3),
      output->strideAt(0), output->strideAt(1), output->strideAt(2), output->strideAt(3), xs_, ys_);
  checkLaunch(stream, "resizeImageKernel failed: ");

  NDArray::registerSpecialUse({output}, {images});

  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status resizeBilinearFunctor(LaunchContext* context, NDArray * images, int width, int height,
                                 bool const alignCorners, bool const halfPixelCenter, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_,
                        (context, images, width, height, alignCorners, halfPixelCenter, output), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the source index along one axis of an output index for the nearest neighbor resize
template <typename Scaler>
static SD_DEVICE SD_INLINE LongType nearestSourceIndex(Scaler& scaler, LongType outIndex, float scale,
                                                       NearestMode nearestMode, LongType inSize) {
  constexpr bool halfPixelCenter =
      std::is_same<Scaler, HalfPixelScaler>::value || std::is_same<Scaler, HalfPixelScalerNN>::value;
  const float source = scaler(static_cast<int>(outIndex), scale);
  float rounded;
  switch (nearestMode) {
    case ROUND_PREFER_FLOOR:
      rounded = math::p_round_prefer_floor<float>(source);
      break;
    case ROUND_PREFER_CEIL:
      rounded = math::p_round_prefer_ceil<float>(source);
      break;
    case CEIL:
      rounded = math::p_ceil<float>(source);
      break;
    case FLOOR:
    default:
      rounded = math::p_floor<float>(source);
      break;
  }
  LongType index = math::sd_min(static_cast<LongType>(rounded), inSize - 1);
  if (halfPixelCenter) {
    index = math::sd_max(0LL, index);
  }
  return index;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by interpolation nearest neighbor algorithm kernel: one thread per output pixel (grid-stride over
// batch * outHeight * outWidth), so any batch size and output size is covered by a capped grid; the channels of the
// pixel are copied in a loop. Input and output are read and written through their strides.
//
template <typename T, typename Scaler>
static SD_KERNEL void resizeNeighborKernel(T const* input, LongType const* inputShape, T* output,
                                           LongType const* outputShape, LongType batchSize, LongType inWidth,
                                           LongType inHeight, LongType outWidth, LongType outHeight, LongType channels,
                                           double widthScale, double heightScale, NearestMode nearestMode) {
  const LongType* inStride = shape::stride(inputShape);
  const LongType* outStride = shape::stride(outputShape);
  const LongType pixels = batchSize * outHeight * outWidth;
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  Scaler scaler;

  for (LongType pixel = start; pixel < pixels; pixel += step) {
    const LongType x = pixel % outWidth;
    const LongType y = (pixel / outWidth) % outHeight;
    const LongType b = pixel / (outWidth * outHeight);

    const LongType inY = nearestSourceIndex<Scaler>(scaler, y, static_cast<float>(heightScale), nearestMode, inHeight);
    const LongType inX = nearestSourceIndex<Scaler>(scaler, x, static_cast<float>(widthScale), nearestMode, inWidth);

    const T* source = input + b * inStride[0] + inY * inStride[1] + inX * inStride[2];
    T* target = output + b * outStride[0] + y * outStride[1] + x * outStride[2];
    for (LongType c = 0; c < channels; c++) {
      target[c * outStride[3]] = source[c * inStride[3]];
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
  if (output->lengthOf() == 0) return Status::OK;

  float heightScale = ImageResizerState::calculateResizeScale(inHeight, outHeight, alignCorner);
  float widthScale = ImageResizerState::calculateResizeScale(inWidth, outWidth, alignCorner);

  auto stream = context->getCudaStream();
  NDArray::prepareSpecialUse({output}, {images});

  const T* imagesBuffer = reinterpret_cast<const T*>(images->specialBuffer());
  T* outputBuffer = reinterpret_cast<T*>(output->specialBuffer());
  const LongType* imagesShapeInfo = images->specialShapeInfo();
  const LongType* outputShapeInfo = output->specialShapeInfo();

  dim3 neightborDims = resizeNeighborDims(batchSize, outHeight, outWidth);
  switch (coorMode) {
    case ASYMMETRIC:
      resizeNeighborKernel<T, LegacyScaler><<<neightborDims.x, neightborDims.y, neightborDims.z, *stream>>>(
          imagesBuffer, imagesShapeInfo, outputBuffer, outputShapeInfo, batchSize, inWidth, inHeight, outWidth,
          outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL:
      resizeNeighborKernel<T, HalfPixelScaler><<<neightborDims.x, neightborDims.y, neightborDims.z, *stream>>>(
          imagesBuffer, imagesShapeInfo, outputBuffer, outputShapeInfo, batchSize, inWidth, inHeight, outWidth,
          outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    case HALF_PIXEL_NN:
      resizeNeighborKernel<T, HalfPixelScalerNN><<<neightborDims.x, neightborDims.y, neightborDims.z, *stream>>>(
          imagesBuffer, imagesShapeInfo, outputBuffer, outputShapeInfo, batchSize, inWidth, inHeight, outWidth,
          outHeight, channels, widthScale, heightScale, nearestMode);
      break;
    default:
      resizeNeighborKernel<T, HalfPixelScaler><<<neightborDims.x, neightborDims.y, neightborDims.z, *stream>>>(
          imagesBuffer, imagesShapeInfo, outputBuffer, outputShapeInfo, batchSize, inWidth, inHeight, outWidth,
          outHeight, channels, widthScale, heightScale, nearestMode);
      break;
  };
  checkLaunch(stream, "resizeNeighborKernel failed: ");

  NDArray::registerSpecialUse({output}, {images});

  return Status::OK;
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// weights and source indices of the output columns (or rows) for the bicubic resize: one thread per entry. The
// indices are multiplied by indexStride (the input's stride along the axis), so they are offsets into the input.
template <typename Scaler>
static SD_KERNEL void computeBicubicWeightsKernel(float const* coeffsTable, float scale, LongType count,
                                                  LongType limit, bool excludeOutside, LongType indexStride,
                                                  WeightsAndIndices* weights) {
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType i = start; i < count; i += step) {
    WeightsAndIndices* wai = weights + i;
    getWeightsAndIndices<Scaler>(coeffsTable, scale, i, limit, wai, excludeOutside);
    wai->_index0 *= indexStride;
    wai->_index1 *= indexStride;
    wai->_index2 *= indexStride;
    wai->_index3 *= indexStride;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bicubic interpolation: one thread per output pixel (grid-stride over batch * outHeight * outWidth), the channels of
// the pixel in a loop. Each of the four columns the pixel reads is interpolated along y first and the four results
// along x, as the CPU does (which caches the columns it shares with the previous pixel: the values are the same). The
// interpolation is computed in float and stored as the output type Z (FLOAT32 or DOUBLE).
template <typename T, typename Z>
static SD_KERNEL void bicubicInterpolateKernel(T const* inputPtr, Z* outputPtr, LongType batchSize,
                                               LongType outHeight, LongType outWidth, LongType channels,
                                               LongType inBatchStride, LongType inChannelStride,
                                               LongType outBatchStride, LongType outRowStride,
                                               LongType outColumnStride, LongType outChannelStride,
                                               WeightsAndIndices const* yWais, WeightsAndIndices const* xWais) {
  const LongType pixels = batchSize * outHeight * outWidth;
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType pixel = start; pixel < pixels; pixel += step) {
    const LongType x = pixel % outWidth;
    const LongType y = (pixel / outWidth) % outHeight;
    const LongType batch = pixel / (outWidth * outHeight);

    const WeightsAndIndices& yWai = yWais[y];
    const WeightsAndIndices& xWai = xWais[x];
    const T* pInput = inputPtr + batch * inBatchStride;
    // the row indices of yWai are offsets into the input
    const T* y_ptr_0 = pInput + yWai._index0;
    const T* y_ptr_1 = pInput + yWai._index1;
    const T* y_ptr_2 = pInput + yWai._index2;
    const T* y_ptr_3 = pInput + yWai._index3;
    Z* pOutput = outputPtr + batch * outBatchStride + y * outRowStride + x * outColumnStride;

    for (LongType c = 0; c < channels; ++c) {
      float cachedValue[4];
      for (int i = 0; i < 4; ++i) {
        cachedValue[i] = computeYInterpolation(i, c * inChannelStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
      }
      pOutput[c * outChannelStride] =
          static_cast<Z>(compute(cachedValue, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename Z, typename Scaler>
static void bicubicInterpolateWithCaching(LaunchContext* context, NDArray * image,
                                          const ImageResizerState& resizerState, const double coefficient,
                                          bool exclude_outside, NDArray* output) {
  const LongType batchSize = resizerState.batchSize;
  const LongType outHeight = resizerState.outHeight;
  const LongType outWidth = resizerState.outWidth;
  const LongType channels = resizerState.channels;
  if (output->lengthOf() == 0) return;

  auto stream = context->getCudaStream();
  PointersManager pm(context, "resizeBicubic");

  // Coefficients table, computed with the Bicubic convolution algorithm on the host (a table of 2 * 1025 floats):
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  std::vector<float> table((kTableSize + 1) * 2);
  KeysCubicKernelFunc<float> kernel(static_cast<float>(coefficient));
  for (LongType i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    table[i * 2] = kernel.calc_less1pt0(x);
    x += 1.0;
    table[i * 2 + 1] = kernel.calc_less2pt0(x);
  }
  auto coeffsTable = reinterpret_cast<float*>(pm.replicatePointer(table.data(), table.size() * sizeof(float)));
  auto xWais = reinterpret_cast<WeightsAndIndices*>(pm.allocateDevMem(sizeof(WeightsAndIndices) * outWidth));
  auto yWais = reinterpret_cast<WeightsAndIndices*>(pm.allocateDevMem(sizeof(WeightsAndIndices) * outHeight));

  const dim3 weightDims = getLaunchDims("image_resize_interp_weights");
  computeBicubicWeightsKernel<Scaler><<<weightDims.x, weightDims.y, 0, *stream>>>(
      coeffsTable, resizerState.widthScale, outWidth, resizerState.inWidth, exclude_outside, resizerState.wStride,
      xWais);
  computeBicubicWeightsKernel<Scaler><<<weightDims.x, weightDims.y, 0, *stream>>>(
      coeffsTable, resizerState.heightScale, outHeight, resizerState.inHeight, exclude_outside,
      resizerState.hStride, yWais);
  checkLaunch(stream, "computeBicubicWeightsKernel failed: ");

  unsigned int blocks, threads;
  pixelLaunch(getLaunchDims("image_resize"), batchSize * outHeight * outWidth, blocks, threads);
  bicubicInterpolateKernel<T, Z><<<blocks, threads, 0, *stream>>>(
      reinterpret_cast<T const*>(image->specialBuffer()), reinterpret_cast<Z*>(output->specialBuffer()),
      batchSize, outHeight, outWidth, channels, resizerState.bStride, resizerState.cStride, output->strideAt(0),
      output->strideAt(1), output->strideAt(2), output->strideAt(3), yWais, xWais);
  checkLaunch(stream, "bicubicInterpolateKernel failed: ");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The legacy bicubic resize of resize_images: the coordinates of the ASYMMETRIC mode, the border pixels repeated and
// the ordinary (OpenCV) coefficient, as resize_bicubic does without half pixel centers. (The two flags are the
// resize_images arguments: align corners, and antialias, which does not apply to this resize.)
Status resizeBicubicFunctor(LaunchContext* context, NDArray * image, int width, int height,
                                bool alignCorners, bool antialias, NDArray* output) {
  return resizeBicubicFunctorA(context, image, width, height, alignCorners, ASYMMETRIC, false,
                               KeysCubicKernelFunc<double>::ORDINARY_COEF, output);
}
// ------------------------------------------------------------------------------------------------------------------ //

static SD_KERNEL void fillInterpolationCache(CachedInterpolation* xCached, LongType cacheLen, LongType inWidth,
                                             float widthScale) {
  const LongType start = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const LongType increment = static_cast<LongType>(gridDim.x) * blockDim.x;

  for (LongType x = start; x < cacheLen; x += increment) {
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

// resizeAreaKernel: one thread per output row of every image. The input rows that contribute to an output row (at
// most cacheStride of them) are cached in the thread's slice of cachePool, a [batch, outHeight, cacheStride] array.
template <typename T>
static SD_KERNEL void resizeAreaKernel(ImageResizerState const* pSt, CachedInterpolation const* caches, float scale,
                                       T const* inputPtr, float* outputPtr, ScaleCache<T>* cachePool,
                                       LongType cacheStride) {
  for (LongType batch = blockIdx.x; batch < pSt->batchSize; batch += gridDim.x) {
    for (LongType y = threadIdx.x; y < pSt->outHeight; y += blockDim.x) {
      const float inY = y * pSt->heightScale;
      const float inY1 = (y + 1) * pSt->heightScale;
      // The start and end height indices of all the cells that could
      // contribute to the target cell.
      const LongType yStart = math::sd_floor<float, LongType>(inY);
      const LongType yEnd = math::sd_ceil<float, LongType>(inY1);
      auto scalesDim = yEnd - yStart;
      auto yScaleCache = cachePool + (batch * pSt->outHeight + y) * cacheStride;

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// the area resize into a dense C-order float output (outputPtr)
template <typename T>
static void resizeArea(LaunchContext* context, PointersManager& pm, ImageResizerState const& st,
                       CachedInterpolation* cache, NDArray * input, float* outputPtr) {
  auto stream = context->getCudaStream();
  T const* inputPtr = reinterpret_cast<T const*>(input->specialBuffer());
  float scale = 1.f / (st.heightScale * st.widthScale);

  // the resize state on the device (the manager stages the host copy so that it survives a CUDA graph capture)
  auto pSt = reinterpret_cast<ImageResizerState*>(pm.replicatePointer(&st, sizeof(ImageResizerState)));

  // An output row reads at most ceil(heightScale) + 1 input rows (+ 1 for the rounding of the float coordinates)
  const LongType cacheStride = static_cast<LongType>(math::sd_ceil<float, LongType>(st.heightScale)) + 2;
  auto cachePool = reinterpret_cast<ScaleCache<T>*>(
      pm.allocateDevMem(sizeof(ScaleCache<T>) * st.batchSize * st.outHeight * cacheStride));

  const dim3 launchDims = getLaunchDims("image_resize");
  resizeAreaKernel<T><<<launchDims.x, launchDims.y, 0, *stream>>>(pSt, cache, scale, inputPtr, outputPtr, cachePool,
                                                                   cacheStride);
  checkLaunch(stream, "resizeAreaKernel failed: ");
}
// ------------------------------------------------------------------------------------------------------------------ //
template <typename T>
Status resizeAreaFunctor_(LaunchContext* context, NDArray * image, int const width, int const height,
                          bool const alignCorners, NDArray* output) {
  ImageResizerState st(alignCorners, false);  // Create resize info
  auto res = st.validateAndCalculateOutputSize(image, width, height);
  if (Status::OK != res) return res;
  if (output->lengthOf() == 0) return Status::OK;

  // the kernel writes the output as a dense C-order array: other layouts go through a dense copy
  NDArray* target = output;
  NDArray* staged = nullptr;
  if (!isDenseCOrder(output)) {
    staged = NDArrayFactory::create('c', {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)},
                                    DataType::FLOAT32, context);
    target = staged;
  }

  auto stream = context->getCudaStream();
  PointersManager pm(context, "resizeArea");
  auto xCached = reinterpret_cast<CachedInterpolation*>(pm.allocateDevMem(sizeof(CachedInterpolation) * st.outWidth));
  NDArray::prepareSpecialUse({target}, {image});

  const dim3 cacheDims = getLaunchDims("image_resize_fill_interp");
  fillInterpolationCache<<<cacheDims.x, cacheDims.y, 0, *stream>>>(xCached, st.outWidth, st.inWidth, st.widthScale);
  checkLaunch(stream, "fillInterpolationCache failed: ");
  resizeArea<T>(context, pm, st, xCached, image, reinterpret_cast<float*>(target->specialBuffer()));

  NDArray::registerSpecialUse({target}, {image});

  if (staged != nullptr) {
    // The one barrier of this file: the copy into the output is an NDArray op on the output's own context, whose
    // stream need not be this helper's, so the staged array must be complete before it starts (a capture-aware
    // barrier: skipped while a graph capture is recorded; it is only reached for an output that is not a dense array).
    pm.synchronize();
    output->assign(staged);
    delete staged;
  }
  return res;
}
Status resizeAreaFunctor(LaunchContext* context, NDArray * image, int const width, int const height,
                             bool const alignCorners, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_,
                        (context, image, width, height, alignCorners, output), SD_NUMERIC_TYPES);
}

// ------------------------------------------------------------------------------------------------------------------ //
// simplified bicubic resize without antialiasing, into an output of type Z
//
template <typename T, typename Z>
static Status resizeBicubicByMode(LaunchContext* context, NDArray * image, const ImageResizerState& st,
                                  CoordinateTransformationMode coorMode, bool exclude_outside, double coefficient,
                                  NDArray* output) {
  switch (coorMode) {
    case ASYMMETRIC:
      bicubicInterpolateWithCaching<T, Z, LegacyScaler>(context, image, st, coefficient, exclude_outside, output);
      return Status::OK;
    case HALF_PIXEL:
      bicubicInterpolateWithCaching<T, Z, HalfPixelScaler>(context, image, st, coefficient, exclude_outside, output);
      return Status::OK;
    case HALF_PIXEL_NN:
      bicubicInterpolateWithCaching<T, Z, HalfPixelScalerNN>(context, image, st, coefficient, exclude_outside, output);
      return Status::OK;
  }
  return Logger::logStatusMsg(Status::BAD_INPUT, "resize_bicubic: Wrong coordinate transformation mode");
}

template <typename T>
Status resizeBicubicFunctorA_(LaunchContext* context, NDArray * image, int const width, int const height,
                              bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                              double coefficient, NDArray* output) {
  ImageResizerState st(alignCorners, coorMode == HALF_PIXEL,
                       context->getCudaStream());  // align_corners, half_pixel_align
  NDArray::prepareSpecialUse({output}, {image});
  Status res = st.validateAndCreateOutput(image, width, height);
  if (res == Status::OK) {
    // the op's output types: FLOAT32 and DOUBLE
    if (output->dataType() == DataType::FLOAT32) {
      res = resizeBicubicByMode<T, float>(context, image, st, coorMode, exclude_outside, coefficient, output);
    } else if (output->dataType() == DataType::DOUBLE) {
      res = resizeBicubicByMode<T, double>(context, image, st, coorMode, exclude_outside, coefficient, output);
    } else {
      res = Logger::logStatusMsg(Status::BAD_INPUT, "resize_bicubic: The output should be of type FLOAT32 or DOUBLE");
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
// cropAndResize kernel: a block per box (grid-stride over the boxes), its threads stride over the pixels of the box's
// crop, and a thread computes all the channels of its pixel. The images, the boxes, the indices and the crops are read
// and written through their strides, from their buffer pointers (which include the offset of a view). Type of
// input(images) and output should be the same.
//
// The sample positions and the interpolation are computed in double when the images or the boxes are DOUBLE, in float
// otherwise, as on the CPU. The crop of a box that names no image of the batch, and every sample position outside the
// image, is the extrapolation value.
template <typename T, typename Z, typename I>
static SD_KERNEL void cropAndResizeKernel(T const* images, LongType const* imagesShape, Z const* boxes,
                                          LongType const* boxesShape, I const* indices, LongType const* indexShape,
                                          int method, double extrapolationVal, T* output, LongType const* outputShape,
                                          LongType numBoxes, LongType cropHeight, LongType cropWidth,
                                          LongType batchSize, LongType imageHeight, LongType imageWidth,
                                          LongType depth) {
  using PosT =
      typename std::conditional<std::is_same<T, double>::value || std::is_same<Z, double>::value, double, float>::type;
  const LongType* imageStride = shape::stride(imagesShape);
  const LongType* boxStride = shape::stride(boxesShape);
  const LongType* outStride = shape::stride(outputShape);
  const LongType iRank = shape::rank(indexShape);
  const LongType* iShape = shape::shapeOf(indexShape);
  const LongType* iStride = shape::stride(indexShape);
  const LongType cropPixels = cropHeight * cropWidth;

  for (LongType b = blockIdx.x; b < numBoxes; b += gridDim.x) {
    const Z* box = boxes + b * boxStride[0];
    const PosT y1 = static_cast<PosT>(box[0]);
    const PosT x1 = static_cast<PosT>(box[boxStride[1]]);
    const PosT y2 = static_cast<PosT>(box[2 * boxStride[1]]);
    const PosT x2 = static_cast<PosT>(box[3 * boxStride[1]]);

    // the box's image, read through the strides of the indices
    LongType iCoords[SD_MAX_RANK];
    LongType iOffset;
    INDEX2COORDS(b, iRank, iShape, iCoords);
    COORDS2INDEX(iRank, iStride, iCoords, iOffset);
    const LongType bIn = static_cast<LongType>(indices[iOffset]);
    const bool inBatch = bIn >= 0 && bIn < batchSize;
    const T* image = inBatch ? images + bIn * imageStride[0] : images;

    const PosT heightScale = cropResizeScale<PosT>(y1, y2, imageHeight, cropHeight);
    const PosT widthScale = cropResizeScale<PosT>(x1, x2, imageWidth, cropWidth);

    for (LongType pixel = threadIdx.x; pixel < cropPixels; pixel += blockDim.x) {
      const LongType y = pixel / cropWidth;
      const LongType x = pixel % cropWidth;
      T* crop = output + b * outStride[0] + y * outStride[1] + x * outStride[2];

      const PosT inY = cropResizeCoordinate<PosT>(y1, y2, imageHeight, cropHeight, y, heightScale);
      const PosT inX = cropResizeCoordinate<PosT>(x1, x2, imageWidth, cropWidth, x, widthScale);

      // outside the image (a position that is not a number is outside too, so it never addresses the images)
      if (!inBatch || !(inY >= 0 && inY <= imageHeight - 1 && inX >= 0 && inX <= imageWidth - 1)) {
        for (LongType d = 0; d < depth; d++) {
          crop[d * outStride[3]] = static_cast<T>(extrapolationVal);
        }
        continue;
      }

      if (method == 0 /* bilinear */) {
        const LongType topYIndex = static_cast<LongType>(math::p_floor<PosT>(inY));
        const LongType bottomYIndex = static_cast<LongType>(math::p_ceil<PosT>(inY));
        const PosT yLerp = inY - topYIndex;
        const LongType leftXIndex = static_cast<LongType>(math::p_floor<PosT>(inX));
        const LongType rightXIndex = static_cast<LongType>(math::p_ceil<PosT>(inX));
        const PosT xLerp = inX - leftXIndex;

        const T* topRow = image + topYIndex * imageStride[1];
        const T* bottomRow = image + bottomYIndex * imageStride[1];
        const LongType leftOffset = leftXIndex * imageStride[2];
        const LongType rightOffset = rightXIndex * imageStride[2];
        for (LongType d = 0; d < depth; d++) {
          const LongType channelOffset = d * imageStride[3];
          const PosT topLeft = static_cast<PosT>(topRow[leftOffset + channelOffset]);
          const PosT topRight = static_cast<PosT>(topRow[rightOffset + channelOffset]);
          const PosT bottomLeft = static_cast<PosT>(bottomRow[leftOffset + channelOffset]);
          const PosT bottomRight = static_cast<PosT>(bottomRow[rightOffset + channelOffset]);
          const PosT top = imageResizeLerp<PosT>(topLeft, topRight, xLerp);
          const PosT bottom = imageResizeLerp<PosT>(bottomLeft, bottomRight, xLerp);
          crop[d * outStride[3]] = static_cast<T>(imageResizeLerp<PosT>(top, bottom, yLerp));
        }
      } else {  // method is "nearest neighbor"
        const LongType closestXIndex = static_cast<LongType>(math::p_round<PosT>(inX));
        const LongType closestYIndex = static_cast<LongType>(math::p_round<PosT>(inY));
        const T* source = image + closestYIndex * imageStride[1] + closestXIndex * imageStride[2];
        for (LongType d = 0; d < depth; d++) {
          crop[d * outStride[3]] = source[d * imageStride[3]];
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
  const LongType batchSize = images->sizeAt(0);
  const LongType imageHeight = images->sizeAt(1);
  const LongType imageWidth = images->sizeAt(2);

  const LongType numBoxes = crops->sizeAt(0);
  const LongType cropHeight = crops->sizeAt(1);
  const LongType cropWidth = crops->sizeAt(2);
  const LongType depth = crops->sizeAt(3);
  if (crops->lengthOf() == 0) return;
  auto stream = context->getCudaStream();

  // a block per box, its threads stride over the pixels of the crop
  dim3 cropAndResizeDims = cropAndResize(static_cast<int>(numBoxes), static_cast<int>(imageHeight),
                                         static_cast<int>(imageWidth), static_cast<int>(cropHeight),
                                         static_cast<int>(cropWidth));
  NDArray::prepareSpecialUse({crops}, {images, boxes, indices});
  T const* imagesBuf = reinterpret_cast<T const*>(images->specialBuffer());
  Z const* boxesBuf = reinterpret_cast<Z const*>(boxes->specialBuffer());
  I const* indexBuf = reinterpret_cast<I const*>(indices->specialBuffer());
  T* outBuf = reinterpret_cast<T*>(crops->specialBuffer());
  cropAndResizeKernel<T, Z, I><<<cropAndResizeDims.y, cropAndResizeDims.x, 0, *stream>>>(
      imagesBuf, images->specialShapeInfo(), boxesBuf, boxes->specialShapeInfo(), indexBuf, indices->specialShapeInfo(),
      method, extrapolationVal, outBuf, crops->specialShapeInfo(), numBoxes, cropHeight, cropWidth, batchSize,
      imageHeight, imageWidth, depth);
  checkLaunch(stream, "cropAndResizeKernel failed: ");
  NDArray::registerSpecialUse({crops}, {images, boxes, indices});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cropAndResizeFunctor(LaunchContext* context, NDArray * images, NDArray * boxes,
                          NDArray * indices, NDArray * cropSize, int method, double extrapolationVal,
                          NDArray* crops) {
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
