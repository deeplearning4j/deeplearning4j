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
//  @author sgazeos@gmail.com
//
#include <execution/Threads.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/image_resize.h>

#include "../cross.h"

namespace sd {
namespace ops {
namespace helpers {

template <class Scaler>
static inline void computeInterpolationWeights(const Scaler scaler, sd::LongType outSize, sd::LongType inSize,
                                               double scale, BilinearInterpolationData* interpolationData) {
  interpolationData[outSize].bottomIndex = 0;
  interpolationData[outSize].topIndex = 0;

  auto func = PRAGMA_THREADS_FOR {
    for (auto k = start; k < stop; k++) {
      auto i = (outSize - k - 1);
      double const in = scaler(i, scale);
      double const in_f = sd::math::sd_floor<double, double>(in);
      double const in_c = sd::math::sd_ceil<double, double>(in);
      interpolationData[i].bottomIndex =
          sd::math::sd_max(static_cast<sd::LongType>(in_f), (sd::LongType)0LL);  // static_cast<sd::LongType>(in);
      interpolationData[i].topIndex = sd::math::sd_min(static_cast<sd::LongType>(in_c), inSize - 1);
      interpolationData[i].interpolarValue = in - in_f;
    }
  };
  samediff::Threads::parallel_for(func, 0, outSize);
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
//    static void
//    resizeImage(NDArray const *images, sd::LongType batchSize, sd::LongType inHeight, sd::LongType inWidth,
//    sd::LongType outHeight,
//                sd::LongType outWidth, sd::LongType channels,
//                std::vector<BilinearInterpolationData> const& xs,
//                std::vector<BilinearInterpolationData> const& ys,
//                NDArray *output);

template <typename T, typename Z>
static void resizeImage_(T const* pInputBuf, sd::LongType batchSize, sd::LongType inHeight, sd::LongType inWidth,
                         sd::LongType outHeight, sd::LongType outWidth, sd::LongType channels,
                         std::vector<BilinearInterpolationData> const& xs,
                         std::vector<BilinearInterpolationData> const& ys, Z* pOutputBuf) {
  sd::LongType inRowSize = inWidth * channels;
  sd::LongType inBatchNumValues = inHeight * inRowSize;
  sd::LongType outRowSize = outWidth * channels;

  //        T const *pInputBuf = images->getDataBuffer()->primaryAsT<T>(); // this works only with 'c' direction
  BilinearInterpolationData const* xsPtr = xs.data();

  //        T* pOutputBuf = output->dataBuffer()->primaryAsT<T>();
  auto computeBilinear = [](double topLeft, double topRight, double bottomLeft, double bottomRight, double xVal,
                            double yVal) {
    double top = topLeft + (topRight - topLeft) * xVal;
    double bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
    return top + (bottom - top) * yVal;
  };

  auto func = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; ++batch) {
      auto pInput = pInputBuf + batch * inBatchNumValues;
      for (sd::LongType y = 0; y < outHeight; ++y) {
        auto pOutput = pOutputBuf + (batch * outHeight + y) * outRowSize;
        const T* ysInputLowerPtr = pInput + ys[y].bottomIndex * inRowSize;
        const T* ysInputUpperPtr = pInput + ys[y].topIndex * inRowSize;
        double yVal = ys[y].interpolarValue;
        for (sd::LongType x = 0; x < outWidth; ++x) {
          auto xsBottom = xsPtr[x].bottomIndex;
          auto xsTop = xsPtr[x].topIndex;
          auto xVal = xsPtr[x].interpolarValue;
          for (sd::LongType c = 0; c < channels; ++c) {
            double topLeft(ysInputLowerPtr[xsBottom + c]);
            double topRight(ysInputLowerPtr[xsTop + c]);
            double bottomLeft(ysInputUpperPtr[xsBottom + c]);
            double bottomRight(ysInputUpperPtr[xsTop + c]);
            pOutput[x * channels + c] = computeBilinear(topLeft, topRight, bottomLeft, bottomRight, xVal, yVal);
          }
        }
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, batchSize);
}

template <typename X, typename Z>
static sd::Status resizeBilinearFunctor_(NDArray const* images, int const width, int const height,
                                         bool const alignCorners, bool const halfPixelCenter, NDArray* output) {
  ImageResizerState st(alignCorners, halfPixelCenter);
  st.validateAndCalculateOutputSize(images, width, height);

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

  std::vector<BilinearInterpolationData> ys(outHeight + 1);
  std::vector<BilinearInterpolationData> xs(outWidth + 1);
  if (halfPixelCenter) {
    computeInterpolationWeights(HalfPixelScaler(), outHeight, inHeight, st.heightScale, ys.data());
    computeInterpolationWeights(HalfPixelScaler(), outWidth, inWidth, st.widthScale, xs.data());

  } else {
    // Compute the cached interpolation weights on the x and y dimensions.
    computeInterpolationWeights(LegacyScaler(), outHeight, inHeight, st.heightScale, ys.data());
    computeInterpolationWeights(LegacyScaler(), outWidth, inWidth, st.widthScale, xs.data());
  }
  int xsSize = xs.size();
  // Scale x interpolation weights to avoid a multiplication during iteration.
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      xs[i].bottomIndex *= channels;
      xs[i].topIndex *= channels;
    }
  };
  samediff::Threads::parallel_for(func, 0, xsSize);

  resizeImage_<X, Z>(images->getDataBuffer()->primaryAsT<X>(), batchSize, inHeight, inWidth, outHeight, outWidth,
                     channels, xs, ys, output->dataBuffer()->primaryAsT<Z>());
  return sd::Status::OK;
}

template <class Scaler, typename T>
void resizeNeighborImpl(ImageResizerState const& st, NDArray const* images, NearestMode nearestMode, NDArray* output) {
  const sd::LongType batchSize = st.batchSize;
  const sd::LongType inHeight = st.inHeight;
  const sd::LongType inWidth = st.inWidth;
  const sd::LongType channels = st.channels;

  const sd::LongType outHeight = st.outHeight;
  const sd::LongType outWidth = st.outWidth;
  Scaler scaler;
  constexpr bool halfPixelCenter =
      std::is_same<Scaler, HalfPixelScaler>::value || std::is_same<Scaler, HalfPixelScalerNN>::value;

  float (*modeFunc)(float);
  switch (nearestMode) {
    case NearestMode::FLOOR:
      modeFunc = [](float x){return sd::math::p_floor<float>(x);};
      break;
    case NearestMode::ROUND_PREFER_FLOOR:
      modeFunc = [](float x){return sd::math::p_round_prefer_floor<float>(x);};
      break;
    case NearestMode::ROUND_PREFER_CEIL:
      modeFunc = [](float x){return sd::math::p_round_prefer_ceil<float>(x);};
      break;
    case NearestMode::CEIL:
      modeFunc = [](float x){return sd::math::p_ceil<float>(x);};
      break;
    default:
      modeFunc = [](float x){return sd::math::p_floor<float>(x);};
  }

  auto func = PRAGMA_THREADS_FOR_2D {
    for (auto b = start_x; b < stop_x; b += inc_x) {
      for (auto y = start_y; y < stop_y; y += inc_y) {
        auto posY = static_cast<sd::LongType>(modeFunc(scaler(y, st.heightScale)));
        sd::LongType inY = sd::math::sd_min(posY, inHeight - 1);
        if (halfPixelCenter) {
          inY = sd::math::sd_max(0LL, inY);
        }
        for (sd::LongType x = 0; x < outWidth; ++x) {
          auto posX = static_cast<sd::LongType>(modeFunc(scaler(x, st.widthScale)));
          sd::LongType inX = sd::math::sd_min(posX, inWidth - 1);
          if (halfPixelCenter) {
            inX = sd::math::sd_max(0LL, inX);
          }
          // copy pixel over all channels
          for (sd::LongType e = 0; e < channels; e++) output->r<T>(b, y, x, e) = images->t<T>(b, inY, inX, e);
        }
      }
    }
  };
  samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, outHeight, 1);
}

template <typename T>
sd::Status resizeNeighborFunctor_(NDArray const* images, int const width, int const height,
                                  CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                  NDArray* output) {
  ImageResizerState st(alignCorner, (coorMode == HALF_PIXEL_NN));
  st.validateAndCalculateOutputSize(images, width, height);

  // Handle no-op resizes efficiently.
  if (output->sizeAt(1) == images->sizeAt(1) && output->sizeAt(2) == images->sizeAt(2)) {
    output->assign(images);
    return sd::Status::OK;
  }

  switch (coorMode) {
    case ASYMMETRIC:
      resizeNeighborImpl<LegacyScaler, T>(st, images, nearestMode, output);
      break;
    case HALF_PIXEL:
      resizeNeighborImpl<HalfPixelScaler, T>(st, images, nearestMode, output);
      break;
    case HALF_PIXEL_NN:
      resizeNeighborImpl<HalfPixelScalerNN, T>(st, images, nearestMode, output);
      break;
    default:
      resizeNeighborImpl<HalfPixelScaler, T>(st, images, nearestMode, output);
      break;
  };
  return sd::Status::OK;
}

//    void resizeImage(NDArray const *images, sd::LongType batchSize, sd::LongType inHeight, sd::LongType inWidth,
//    sd::LongType outHeight,
//                     sd::LongType outWidth, sd::LongType channels,
//                     std::vector<BilinearInterpolationData> const &xs,
//                     std::vector<BilinearInterpolationData> const &ys,
//                     NDArray *output) {
//        BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), resizeImage_,
//                              (images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output),
//                              SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
//    }

sd::Status resizeBilinearFunctor(sd::LaunchContext* context, NDArray const* images, int const width, int const height,
                                 bool const alignCorners, bool const halfPixelCenter, NDArray* output) {
  BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_,
                        (images, width, height, alignCorners, halfPixelCenter, output), SD_NUMERIC_TYPES,
                        SD_FLOAT_TYPES);
  return sd::Status::OK;
}

sd::Status resizeNeighborFunctor(sd::LaunchContext* context, NDArray const* images, int const width, int const height,
                                 CoordinateTransformationMode coorMode, NearestMode nearestMode, bool alignCorner,
                                 NDArray* output) {
  BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_,
                        (images, width, height, coorMode, nearestMode, alignCorner, output), SD_COMMON_TYPES);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------------------------ //
// Bicubic interpolation
// ------------------------------------------------------------------------------------------------------------------ //

template <typename T>
std::unique_ptr<T[]> initCoeffsTable(const double a) {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  KeysCubicKernelFunc<T> kernel(static_cast<T>(a));
  std::unique_ptr<T[]> coeffsTableUniq(new T[(kTableSize + 1) * 2]);
  T* coeffsTable = coeffsTableUniq.get();
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i <= stop; ++i) {
      float x = i * 1.0 / kTableSize;
      coeffsTable[i * 2] = kernel.calc_less1pt0(x);
      x += 1.0;
      coeffsTable[i * 2 + 1] = kernel.calc_less2pt0(x);
    }
  };
  samediff::Threads::parallel_for(func, 0, kTableSize);
  return coeffsTableUniq;
}

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
// ------------------------------------------------------------------------------------------------------------------ //

template <typename Scaler>
static void computeXWeightsAndIndices(const ImageResizerState& resizer_state, const float* coeffs_table,
                                      std::vector<WeightsAndIndices>* x_wais, bool exclude_outside) {
  CachedInterpolationCalculator calc;
  for (auto x = 0; x < resizer_state.outWidth; ++x) {
    WeightsAndIndices& x_wai = (*x_wais)[x];
    ;
    getWeightsAndIndices<Scaler>(coeffs_table, resizer_state.widthScale, x, resizer_state.inWidth, &x_wai,
                                 exclude_outside);
    x_wai._advance = calc.Advance(x_wai._index0, x_wai._index1, x_wai._index2, x_wai._index3);
    (*x_wais)[x]._index0 *= resizer_state.wStride;
    (*x_wais)[x]._index1 *= resizer_state.wStride;
    (*x_wais)[x]._index2 *= resizer_state.wStride;
    (*x_wais)[x]._index3 *= resizer_state.wStride;
  }
}

template <typename T, typename F, typename Scaler>
static void bicubicInterpolateWithCaching(NDArray const* image, ImageResizerState const& resizerState,
                                          const double coefficient, bool exclude_outside, NDArray* output) {
  std::vector<WeightsAndIndices> xWais(resizerState.outWidth);
  auto coeffs_table_uniq = initCoeffsTable<float>(coefficient);
  float* coeffs_table = coeffs_table_uniq.get();

  computeXWeightsAndIndices<Scaler>(resizerState, coeffs_table, &xWais, exclude_outside);

  const auto numChannels = resizerState.channels;
  const auto batchNum = resizerState.batchSize;
  const auto outHeight = resizerState.outHeight;
  const auto outWidth = resizerState.outWidth;
  const auto batchStride = image->strideAt(0);
  const auto hStride = image->strideAt(1);
  const auto cStride = image->strideAt(3);

  auto func = PRAGMA_THREADS_FOR {
    const T* inputPtr = image->getDataBuffer()->primaryAsT<T>();
    F* pOutputY = output->dataBuffer()->primaryAsT<F>();  // output is float anyway
    std::vector<float> cachedValue(numChannels == 3 ? 0 : 4 * numChannels, 0);
    for (auto b = start; b < stop; ++b) {
      auto pInput = inputPtr + b * batchStride;
      for (sd::LongType y = 0; y < outHeight; ++y) {
        auto pOutput = &pOutputY[(b * outHeight + y) * outWidth * numChannels];

        WeightsAndIndices yWai;
        getWeightsAndIndices<Scaler>(coeffs_table, resizerState.heightScale, y, resizerState.inHeight, &yWai,
                                     exclude_outside);
        // Make pointers represent offsets of data in inputBPtr.
        const T* y_ptr_0 = pInput + yWai._index0 * hStride;
        const T* y_ptr_1 = pInput + yWai._index1 * hStride;
        const T* y_ptr_2 = pInput + yWai._index2 * hStride;
        const T* y_ptr_3 = pInput + yWai._index3 * hStride;

        if (numChannels == 3) {
          // Manually unroll case of 3 channels.
          F cached_value_0[4] = {0};
          F cached_value_1[4] = {0};
          F cached_value_2[4] = {0};
          for (sd::LongType x = 0; x < resizerState.outWidth; ++x) {
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
                cached_value_2[0] =
                    computeYInterpolation(0, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

              case 1:
                cached_value_0[1] = computeYInterpolation(1, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_1[1] = computeYInterpolation(1, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_2[1] =
                    computeYInterpolation(1, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

              case 2:
                cached_value_0[2] = computeYInterpolation(2, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_1[2] = computeYInterpolation(2, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_2[2] =
                    computeYInterpolation(2, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

              case 3:
                cached_value_0[3] = computeYInterpolation(3, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_1[3] = computeYInterpolation(3, cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                cached_value_2[3] =
                    computeYInterpolation(3, 2 * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                break;
            }
            pOutput[x * numChannels + 0] =
                compute(cached_value_0, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
            pOutput[x * numChannels + 1] =
                compute(cached_value_1, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
            pOutput[x * numChannels + 2] =
                compute(cached_value_2, xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
          }
        } else {
          for (sd::LongType x = 0; x < resizerState.outWidth; ++x) {
            const WeightsAndIndices& xWai = xWais[x];
            // Shift values in cachedValue to fill first '_advance' values.
            switch (xWai._advance) {
              case 3:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 0] = cachedValue[4 * c + 1];
                  cachedValue[4 * c + 1] = cachedValue[4 * c + 2];
                  cachedValue[4 * c + 2] = cachedValue[4 * c + 3];
                }
                break;
              case 2:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 0] = cachedValue[4 * c + 2];
                  cachedValue[4 * c + 1] = cachedValue[4 * c + 3];
                }
                break;
              case 1: {
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 0] = cachedValue[4 * c + 3];
                }
                break;
              }
            }

            // Set the remaining '4-_advance' values by computing.
            switch (xWai._advance) {
              case 0:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 0] =
                      computeYInterpolation(0, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                }
              case 1:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 1] =
                      computeYInterpolation(1, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                }
              case 2:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 2] =
                      computeYInterpolation(2, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                }
              case 3:
                for (auto c = 0; c < numChannels; ++c) {
                  cachedValue[4 * c + 3] =
                      computeYInterpolation(3, c * cStride, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                }
                break;
            }
            for (auto c = 0; c < numChannels; ++c) {
              pOutput[x * numChannels + c] =
                  (F)compute(&cachedValue[4 * c], xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
            }
          }
        }
      }
    }
  };
  samediff::Threads::parallel_tad(func, 0, batchNum);
}

// simplified bicubic resize without antialiasing
//
template <typename T>
sd::Status resizeBicubicFunctorA_(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                  bool const alignCorners, CoordinateTransformationMode coorMode, bool exclude_outside,
                                  double coefficient, NDArray* output) {
  ImageResizerState st(alignCorners, coorMode == HALF_PIXEL);  // align_corners, half_pixel_align
  auto res = st.validateAndCreateOutput(image, width, height);
  if (res == sd::Status::OK) {
    switch (coorMode) {
      case ASYMMETRIC:
        bicubicInterpolateWithCaching<T, float, LegacyScaler>(image, st, coefficient, exclude_outside, output);
        break;
      case HALF_PIXEL:
        bicubicInterpolateWithCaching<T, float, HalfPixelScaler>(image, st, coefficient, exclude_outside, output);
        break;
      case HALF_PIXEL_NN:
        bicubicInterpolateWithCaching<T, float, HalfPixelScalerNN>(image, st, coefficient, exclude_outside, output);
        break;
      default:
        break;
    }
  }

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

template <typename T>
static void resizeArea(ImageResizerState const& st, std::vector<CachedInterpolation> const& caches,
                       NDArray const* input, NDArray* output) {
  T const* inputPtr = input->bufferAsT<T>();
  float scale = 1.f / (st.heightScale * st.widthScale);
  auto outputPtr = output->bufferAsT<float>();  // output is always float. TO DO: provide another float types also with
                                                // template <typename X, typename Z> declaration

  auto batchProcess = PRAGMA_THREADS_FOR {
    for (auto batch = start; batch < stop; batch++) {
      for (auto y = 0; y < st.outHeight; ++y) {
        const float inY = y * st.heightScale;
        const float inY1 = (y + 1) * st.heightScale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        const sd::LongType yStart = math::sd_floor<float, sd::LongType>(inY);
        const sd::LongType yEnd = math::sd_ceil<float, sd::LongType>(inY1);

        std::vector<ScaleCache<T>> yCaches;
        auto cacheLen = yEnd - yStart;
        if (cacheLen) {
          yCaches.resize(cacheLen);
        };
        ScaleCache<T>* yCachesPtr = yCaches.data();
        sd::LongType yCachesSize = yCaches.size();
        for (auto i = yStart, k = 0LL; i < yEnd; ++i, ++k) {
          ScaleCache<T> scaleCache;
          if (i < inY) {
            scaleCache.yScale = (i + 1 > inY1 ? st.heightScale : i + 1 - inY);
          } else {
            scaleCache.yScale = (i + 1 > inY1 ? inY1 - i : 1.0);
          }
          scaleCache.yPtr = inputPtr + (batch * st.bStride + bound(i, st.inHeight) * st.hStride);
          yCaches[k] = scaleCache;
        }
        float* output = outputPtr + (batch * st.outHeight + y) * st.channels * st.outWidth;

        if (st.channels == 3) {
          for (sd::LongType x = 0; x < st.outWidth; ++x) {
            const CachedInterpolation& xCache = caches[x];
            computePatchSumOf3Channels<T>(scale, st, yCachesPtr, yCachesSize, xCache, output);
            output += st.channels;
          }
        } else {
          for (sd::LongType x = 0; x < st.outWidth; ++x) {
            const CachedInterpolation& xCache = caches[x];
            computePatchSum<T>(scale, st, yCachesPtr, yCachesSize, xCache, output);
            output += st.channels;
          }
        }
      }
    }
  };
  samediff::Threads::parallel_tad(batchProcess, 0, st.batchSize, 1);
}

template <typename X>
sd::Status resizeAreaFunctor_(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, NDArray* output) {
  ImageResizerState st(alignCorners, false);  // Create resize info
  auto res = st.validateAndCalculateOutputSize(image, width, height);
  if (Status::OK == res) {
    std::vector<CachedInterpolation> xCached(st.outWidth);
    auto cachingProcedure = PRAGMA_THREADS_FOR {
      for (auto x = start; x < stop; x++) {
        auto& xCache = xCached[x];
        const float inX = x * st.widthScale;
        const float inX1 = (x + 1) * st.widthScale;

        sd::LongType v = math::sd_floor<float, sd::LongType>(inX);
        xCache.start = v;
        xCache.startScale = v < inX ? (v + 1 > inX1 ? st.widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
        v = math::sd_ceil<float, sd::LongType>(inX1);
        xCache.end = v--;
        xCache.endMinusOneScale =
            v < inX ? (v + 1 > inX1 ? st.widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v : 1.f);
        xCache.needsBounding =
            bound(xCache.start, st.inWidth) != xCache.start || bound(xCache.end - 1, st.inWidth) != (xCache.end - 1);
      }
    };
    samediff::Threads::parallel_for(cachingProcedure, 0, xCached.size(), 1);

    resizeArea<X>(st, xCached, image, output);
  }
  return res;
}

sd::Status resizeAreaFunctor(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                             bool const alignCorners, NDArray* output) {
  BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_,
                        (context, image, width, height, alignCorners, output), SD_NUMERIC_TYPES);
}

static sd::Status computeSpans(IKernelFunc<float>* kernel, sd::LongType const outSize, sd::LongType const inSize,
                               float const scale, float const translate, bool const antialias, Spans& spans) {
  // When sampling, we need the inverse scale and translation, to map from an
  // output to an input pixel.
  float const invScale = 1.f / scale;
  float const invTranslate = -invScale * translate;
  // When downsampling the kernel should be scaled since we want to low pass
  // filter and interpolate, but when upsampling it should not be since we only
  // want to interpolate.
  float const kernelScale = antialias ? math::sd_max(invScale, 1.f) : 1.f;
  spans._spanSize =
      math::sd_min(2 * static_cast<int>(std::ceil(kernel->radius() * kernelScale)) + 1, static_cast<int>(inSize));
  spans._starts = NDArrayFactory::create<int>('c', {outSize});
  spans._weights = NDArrayFactory::create<float>('c', {outSize, spans._spanSize});

  auto startsVec = spans._starts.bufferAsT<int>();
  auto weightsVector = spans._weights.bufferAsT<float>();
  spans._weights.nullify();

  const float invKernelScale = 1.f / kernelScale;
  int maxSpanSize = 0;
  std::vector<float> tempWeights;

  // return value if within bounds or bounds otherwise
  auto boundsAmp = [](sd::LongType const low, sd::LongType const high, sd::LongType const value) {
    if (high < value) return high;
    if (value < low) return low;
    return value;
  };

  for (auto x = 0LL; x < outSize; ++x) {
    const float columnFloat = x + 0.5f;
    const float sampleFloat = columnFloat * invScale + invTranslate;

    // Don't sample when the sampling location is outside the source image.
    if (sampleFloat < 0 || sampleFloat > inSize) {
      // Add an empty span.
      startsVec[x] = 0;
      continue;
    }
    sd::LongType spanStart = math::sd_ceil<float, float>(sampleFloat - kernel->radius() * kernelScale - 0.5f);
    sd::LongType spanEnd = math::sd_floor<float, float>(sampleFloat + kernel->radius() * kernelScale - 0.5f);
    spanStart = boundsAmp(0LL, inSize - 1, spanStart);
    spanEnd = boundsAmp(0LL, inSize - 1, spanEnd) + 1;
    int const spanSize = spanEnd - spanStart;
    if (spanSize > spans._spanSize) {
      return Logger::logStatusMsg(
          Status::BAD_INPUT,
          "Span is too large: ");  // + spanSize + " vs " + spans._spanSize);//, spanSize, spans._spanSize));
    }
    float totalWeightSum = 0.f;
    tempWeights.clear();
    for (int source = spanStart; source < spanEnd; ++source) {
      float kernelPos = static_cast<float>(source) + 0.5f - sampleFloat;
      float weight = (*kernel)(kernelPos * invKernelScale);
      totalWeightSum += weight;
      tempWeights.push_back(weight);
    }
    maxSpanSize = std::max(maxSpanSize, spanSize);
    if (math::sd_abs(totalWeightSum) >= 1000.f * DataTypeUtils::min_positive<float>()) {  //
      auto totalWeightSumInverted = 1.0f / totalWeightSum;
      auto outIndex = spans._spanSize * x;
      for (auto weight : tempWeights) {
        weightsVector[outIndex] = weight * totalWeightSumInverted;
        ++outIndex;
      }
    }
    startsVec[x] = spanStart;
  }
  return sd::Status::OK;
}

template <typename X, typename Z>
static void gatherSpans(int const rowSpanSize, NDArray const& rowStarts, NDArray const& rowWeights,
                        int const colSpanSize, NDArray const& columnStarts, NDArray const& columnWeights,
                        NDArray const* images, NDArray& intermediate, NDArray* output) {
  auto batchSize = images->sizeAt(0);
  auto inputHeight = images->sizeAt(1);
  auto inputWidth = images->sizeAt(2);
  auto channels = images->sizeAt(3);

  auto outputHeight = output->sizeAt(1);
  auto outputWidth = output->sizeAt(2);

  auto inputPixPerBatch = images->strideAt(0);
  auto intermediatePixPerBatch = inputWidth * outputHeight * channels;
  auto outputPixPerBatch = outputWidth * outputHeight * channels;
  Z* intermediatePtr = intermediate.bufferAsT<Z>();
  bool inputEws1 = images->ews() == 1;
  auto inRowStride = images->strideAt(1);
  auto wStride = images->strideAt(2);
  auto cStride = images->strideAt(3);
  const X* imagePtr = images->bufferAsT<X>();
  Z* outPtr = output->bufferAsT<Z>();
  for (int b = 0; b < batchSize;
       ++b, imagePtr += inputPixPerBatch, intermediatePtr += intermediatePixPerBatch, outPtr += outputPixPerBatch) {
    gatherRows<X, Z>(rowSpanSize, rowStarts.bufferAsT<int>(), rowWeights.bufferAsT<Z>(), imagePtr, inputHeight,
                     inputWidth, outputHeight, inputWidth, channels, intermediatePtr, inputEws1, inRowStride, wStride,
                     cStride);
    gatherColumns<Z>(colSpanSize, columnStarts.bufferAsT<int>(), columnWeights.bufferAsT<Z>(), intermediatePtr,
                     outputHeight, inputWidth, outputHeight, outputWidth, channels, outPtr);
  }
}

template <typename X, typename Z>
static sd::Status resizeKernel(IKernelFunc<float>* transformationKernel, NDArray const* input, sd::LongType outWidth,
                               sd::LongType outHeight, bool antialias, NDArray* output) {
  sd::LongType const batchSize = input->sizeAt(0);
  sd::LongType const inputHeight = input->sizeAt(1);
  sd::LongType const inputWidth = input->sizeAt(2);
  sd::LongType const channels = input->sizeAt(3);

  Z rowScale = Z(outHeight) / Z(inputHeight);
  Z columnScale = Z(outWidth) / Z(inputWidth);

  // Return if the output is empty.
  if (output->lengthOf() == 0) return sd::Status::OK;

  Spans colSpans;

  auto res = computeSpans(transformationKernel, outWidth, inputWidth, columnScale, 0.f, antialias, colSpans);
  if (res != sd::Status::OK) return res;
  Spans rowSpans;
  res = computeSpans(transformationKernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

  NDArray intermediate = NDArrayFactory::create<Z>('c', {batchSize, outHeight, inputWidth, channels});

  // const functor::Spans& const_row_spans = row_spans;
  // typename TTypes<int32, 1>::ConstTensor row_starts(
  // const_row_spans.starts.tensor<int32, 1>());
  auto& rowStarts = rowSpans._starts;       // shape {outWidth}
  auto& rowWeights = rowSpans._weights;     // shape {outWidth, numSpans}
  auto& columnStarts = colSpans._starts;    // shape {outHeights}
  auto& columnWeights = colSpans._weights;  // shape {outHeights, numSpans}

  gatherSpans<X, Z>(rowSpans._spanSize, rowStarts, rowWeights, colSpans._spanSize, columnStarts, columnWeights, input,
                    intermediate, output);
  return res;
}
#if defined(HAS_FLOAT32)
static sd::Status resizeBilinear(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 bool const antialias, NDArray* output) {
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new TriangleKernelFunc());
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
  return Logger::logStatusMsg(Status::VALIDATION, "helpers::resizeBilinear: Unknown error occured.");
}

static sd::Status resizeBicubicA(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 CoordinateTransformationMode coorMode, bool exclude_outside, double coefficient,
                                 NDArray* output) {
  constexpr bool alignCorners = false;
  return resizeBicubicFunctorA(context, image, width, height, alignCorners, coorMode, exclude_outside, coefficient,
                               output);
}

static sd::Status resizeBicubicAntialias(sd::LaunchContext* context, NDArray const* image, int const width,
                                         int const height, bool const antialias, double coefficient, NDArray* output) {
  // coorMode is HALF_PIXEL exlude_outside is True
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new KeysCubicKernelFunc<float>(coefficient));
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
}
#endif

static sd::Status resizeArea(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                             bool const antialias, NDArray* output) {
  return resizeAreaFunctor(context, image, width, height, false, output);
}
#if defined(HAS_FLOAT32)
static sd::Status resizeLanczos3(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 bool const antialias, NDArray* output) {
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new LanczosKernelFunc(3.f));
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
  return Logger::logStatusMsg(Status::VALIDATION, "helpers::resizeLanczos3: Unknown error occured.");
}

static sd::Status resizeLanczos5(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 bool const antialias, NDArray* output) {
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new LanczosKernelFunc(5.f));
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
  return Logger::logStatusMsg(Status::VALIDATION, "helpers::resizeLanczos5: Unknown error occured.");
}

static sd::Status resizeGaussian(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                                 bool const antialias, NDArray* output) {
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new GaussianKernelFunc());
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
  return Logger::logStatusMsg(Status::VALIDATION, "helpers::resizeGaussian: Unknown error occured.");
}

static sd::Status resizeMitchellcubic(sd::LaunchContext* context, NDArray const* image, int const width,
                                      int const height, bool const antialias, NDArray* output) {
  auto kernel = std::unique_ptr<IKernelFunc<float>>(new MitchellCubicKernelFunc());
  BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                        (kernel.get(), image, (sd::LongType)width, (sd::LongType)height, antialias, output),
                        SD_NUMERIC_TYPES, SKIP_FIRST_COMMA(TTYPE_FLOAT32));
  return Logger::logStatusMsg(Status::VALIDATION, "helpers::ResizeMitchellcubic: Unknown error occured.");
}
#endif
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
  }
  sd_printf("helper::resizeImagesFunctor: Wrong resize method %i\n", (int)method);
  return Logger::logStatusMsg(Status::BAD_INPUT, "helper::resizeImagesFunctor: Wrong resize method");
}
// ------------------------------------------------------------------------------------------------------------------ //
sd::Status resizeFunctor(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                         ImageResizeMethods method, CoordinateTransformationMode coorMode, bool exclude_outside,
                         NearestMode nearestMode, double coefficient, bool antialias, NDArray* output) {
  switch (method) {
    case kResizeNearest:
      return resizeNeighborFunctor(context, image, width, height, coorMode, nearestMode, false, output);
    case kResizeArea:
      return resizeArea(context, image, width, height, antialias, output);
#if defined(HAS_FLOAT32)
    case kResizeBilinear:
      return resizeBilinear(context, image, width, height, antialias, output);
    case kResizeBicubic: {
      // if antialias then coorMode is HALF_PIXEL and exlude_outside is true
      if (antialias) {
        return resizeBicubicAntialias(context, image, width, height, antialias, coefficient, output);
      } else {
        // use modified v1
        return resizeBicubicA(context, image, width, height, coorMode, exclude_outside, coefficient, output);
      }
    }
    case kResizeLanczos3:
      return resizeLanczos3(context, image, width, height, antialias, output);
    case kResizeLanczos5:
      return resizeLanczos5(context, image, width, height, antialias, output);
    case kResizeGaussian:
      return resizeGaussian(context, image, width, height, antialias, output);
    case kResizeMitchellcubic:
      return resizeMitchellcubic(context, image, width, height, antialias, output);
#else
    case kResizeBilinear:
    case kResizeBicubic:
    case kResizeLanczos3:
    case kResizeLanczos5:
    case kResizeGaussian:
    case kResizeMitchellcubic: {
      sd_printf("helper::resizeFunctor: only float type is supported by this resize method %i\n", (int)method);
      return Logger::logStatusMsg(Status::BAD_INPUT, "helper::resizeFunctor: only float type supported");
    }

#endif
  }
  sd_printf("helper::resizeFunctor: Wrong resize method %i\n", (int)method);
  return Logger::logStatusMsg(Status::BAD_INPUT, "helper::resizeFunctor: Wrong resize method");
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
