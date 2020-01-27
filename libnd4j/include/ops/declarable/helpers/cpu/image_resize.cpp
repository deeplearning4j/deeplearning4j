/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
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

#include <ops/declarable/helpers/image_resize.h>
#include <execution/Threads.h>
#include <ops/declarable/headers/parity_ops.h>
#include "../cross.h"

namespace nd4j {
namespace ops {
namespace helpers {

    struct BilinearInterpolationData {
        Nd4jLong _bottomIndex;  // Lower source index used in the interpolation
        Nd4jLong _topIndex;  // Upper source index used in the interpolation
        // 1D linear iterpolation scale (see:
        // https://en.wikipedia.org/wiki/Bilinear_interpolation)
        double _interpolarValue;
    };
    // calculateResizeScale determines the float scaling factor.
    inline float calculateResizeScale(Nd4jLong inSize, Nd4jLong outSize,
                                      bool alignCorners) {
        return (alignCorners && outSize > 1)
               ? (inSize - 1) / static_cast<float>(outSize - 1)
               : inSize / static_cast<float>(outSize);
    }

    template <typename I, typename F>
    struct ImageResizerStateCommon {
        explicit ImageResizerStateCommon(bool alignCorners, bool halfPixelCenters)
                : _alignCorners(alignCorners),
                  _halfPixelCenters(halfPixelCenters) {}

        // ValidateAndCalculateOutputSize checks the bounds on the input tensors
        // and requested size, sets up some of the resizing state such as the
        // heightScale and widthScale, and calculates the output size.
        // If any of these operations fails, it sets an error status in
        // the context, which the caller must check.
        int validateAndCalculateOutputSize(NDArray const* input, int const width, int const height) {
            //
            batchSize = input->sizeAt(0);//.dim_size(0);
            outHeight = height;
            outWidth = width; //internal::SubtleMustCopy(Svec(1));
            inHeight = static_cast<int32_t>(input->sizeAt(1));
            inWidth = static_cast<int32_t>(input->sizeAt(2));
            channels = input->sizeAt(3); //.dim_size(3);
            heightScale = calculateResizeScale(inHeight, outHeight, _alignCorners);
            widthScale = calculateResizeScale(inWidth, outWidth, _alignCorners);

            // Guard against overflows
            if (ceilf((outHeight - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
                nd4j_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize height");
            }
            if (ceilf((outWidth - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
                nd4j_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize width");
            }

            return Status::OK();
        }

        // Calculates all the required variables, and allocates the output.
        int validateAndCreateOutput(NDArray const* input, int const width, int const height) {
            return validateAndCalculateOutputSize(input, width, height);
        }

        I batchSize;
        I outHeight;
        I outWidth;
        I inHeight;
        I inWidth;
        I channels;
        F heightScale;
        F widthScale;
        NDArray* output = nullptr;

    private:
        bool _alignCorners;
        bool _halfPixelCenters;
    };

    typedef ImageResizerStateCommon<Nd4jLong, float> ImageResizerState;

    // Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
    struct HalfPixelScaler {
        HalfPixelScaler(){};
        inline float operator()(const int x, const float scale) const {
            // Note that we subtract 0.5 from the return value, as the existing bilinear
            // sampling code etc assumes pixels are in the old coordinate system.
            return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
        }
    };

    // Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
    struct HalfPixelScalerNN {
        HalfPixelScalerNN(){};
        inline float operator()(const int x, const float scale) const {
            // Note that we subtract 0.5 from the return value, as the existing bilinear
            // sampling code etc assumes pixels are in the old coordinate system.
            return (static_cast<float>(x) + 0.5f) * scale;
        }
    };

// Older incorrect scaling method that causes all resizes to have a slight
// translation leading to inconsistent results. For example, a flip then a
// resize gives different results then a resize then a flip.
    struct LegacyScaler {
        LegacyScaler(){};
        inline float operator()(const int x, const float scale) const {
            return static_cast<float>(x) * scale;
        }
    };

    struct WeightsAndIndices {
        float _weight0;
        float _weight1;
        float _weight2;
        float _weight3;
        Nd4jLong _index0;
        Nd4jLong _index1;
        Nd4jLong _index2;
        Nd4jLong _index3;

        int _advance;  // advance value.
    };

    template <class Scaler>
    inline void computeInterpolationWeights(const Scaler scaler, Nd4jLong outSize,
                                            Nd4jLong inSize,
                                            double scale,
                                            BilinearInterpolationData *interpolationData) {
        interpolationData[outSize]._bottomIndex = 0;
        interpolationData[outSize]._topIndex = 0;

        auto func = PRAGMA_THREADS_FOR {
       	    for (auto k = start; k < stop; k++) {
                auto i = (outSize - k - 1);
                double  const in =  scaler(i, scale);
                double const in_f = nd4j::math::nd4j_floor<double, double>(in);
                double const in_c = nd4j::math::nd4j_ceil<double, double>(in);
                interpolationData[i]._bottomIndex = nd4j::math::nd4j_max(static_cast<Nd4jLong>(in_f), (Nd4jLong)0LL);//static_cast<Nd4jLong>(in);
                interpolationData[i]._topIndex = nd4j::math::nd4j_min(static_cast<Nd4jLong>(in_c), inSize - 1);
                interpolationData[i]._interpolarValue = in - in_f;
     	    }
	    };
	    samediff::Threads::parallel_for(func, 0, outSize);
    }

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
//    static void
//    resizeImage(NDArray const *images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
//                Nd4jLong outWidth, Nd4jLong channels,
//                std::vector<BilinearInterpolationData> const& xs,
//                std::vector<BilinearInterpolationData> const& ys,
//                NDArray *output);

    template<typename T, typename Z>
    static void
    resizeImage_(T const* pInputBuf, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                 Nd4jLong outWidth, Nd4jLong channels,
                 std::vector<BilinearInterpolationData> const &xs,
                 std::vector<BilinearInterpolationData> const &ys,
                 Z* pOutputBuf) {

        Nd4jLong inRowSize = inWidth * channels;
        Nd4jLong inBatchNumValues = inHeight * inRowSize;
        Nd4jLong outRowSize = outWidth * channels;

//        T const *pInputBuf = images->getDataBuffer()->primaryAsT<T>(); // this works only with 'c' direction
        BilinearInterpolationData const* xsPtr = xs.data();

//        T* pOutputBuf = output->dataBuffer()->primaryAsT<T>();
        auto computeBilinear = [](double topLeft, double topRight,
                                      double bottomLeft, double bottomRight,
                                      double xVal, double yVal) {
            double top = topLeft + (topRight - topLeft) * xVal;
            double bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
            return top + (bottom - top) * yVal;
        };

        auto func = PRAGMA_THREADS_FOR {
            for (auto batch = start; batch < stop; ++batch) {
                auto pInput = pInputBuf + batch * inBatchNumValues;
                for (auto y = 0; y < outHeight; ++y) {
                    auto pOutput = pOutputBuf + (batch * outHeight + y) * outRowSize;
                    const T* ysInputLowerPtr = pInput + ys[y]._bottomIndex * inRowSize;
                    const T* ysInputUpperPtr = pInput + ys[y]._topIndex * inRowSize;
                    double yVal = ys[y]._interpolarValue;
                    for (auto x = 0; x < outWidth; ++x) {
                        auto xsBottom = xsPtr[x]._bottomIndex;
                        auto xsTop = xsPtr[x]._topIndex;
                        auto xVal = xsPtr[x]._interpolarValue;
                        for (auto c = 0; c < channels; ++c) {
                            double topLeft(ysInputLowerPtr[xsBottom + c]);
                            double topRight(ysInputLowerPtr[xsTop + c]);
                            double bottomLeft(ysInputUpperPtr[xsBottom + c]);
                            double bottomRight(ysInputUpperPtr[xsTop + c]);
                            pOutput[x * channels + c] = computeBilinear(topLeft, topRight, bottomLeft, bottomRight,
                                    xVal, yVal);
                        }
                    }
                }
            }
        };
        samediff::Threads::parallel_tad(func, 0, batchSize);
    }

    template<typename X, typename Z>
    static int resizeBilinearFunctor_(NDArray const *images, int const width, int const height, bool const alignCorners,
            bool const halfPixelCenter, NDArray *output) {
        ImageResizerState st(alignCorners, halfPixelCenter);
        st.validateAndCalculateOutputSize(images, width, height);

        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight = images->sizeAt(1);
        const Nd4jLong inWidth = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong outHeight = output->sizeAt(1);
        const Nd4jLong outWidth = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (outHeight == inHeight && outWidth == inWidth) {
            output->assign(images);
            return Status::OK();
        }

        std::vector<BilinearInterpolationData> ys(outHeight + 1);
        std::vector<BilinearInterpolationData> xs(outWidth + 1);
        if (halfPixelCenter) {
            computeInterpolationWeights(HalfPixelScaler(), outHeight, inHeight, st.heightScale,
                                        ys.data());
            computeInterpolationWeights(HalfPixelScaler(), outWidth, inWidth, st.widthScale, xs.data());

        }
        else {
            // Compute the cached interpolation weights on the x and y dimensions.
            computeInterpolationWeights(LegacyScaler(), outHeight, inHeight, st.heightScale,
                                        ys.data());
            computeInterpolationWeights(LegacyScaler(), outWidth, inWidth, st.widthScale, xs.data());
        }
        int xsSize = xs.size();
        // Scale x interpolation weights to avoid a multiplication during iteration.
        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                xs[i]._bottomIndex *= channels;
                xs[i]._topIndex *= channels;
            }
        };
        samediff::Threads::parallel_for(func, 0, xsSize);

        resizeImage_<X,Z>(images->getDataBuffer()->primaryAsT<X>(), batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output->dataBuffer()->primaryAsT<Z>());
        return Status::OK();
    }

    template <class Scaler, typename T>
    void resizeNeighbor(ImageResizerState const& st, NDArray const *images, bool const alignCorners, bool const halfPixelCenter, NDArray *output) {
        const Nd4jLong batchSize = st.batchSize;
        const Nd4jLong inHeight = st.inHeight;
        const Nd4jLong inWidth = st.inWidth;
        const Nd4jLong channels = st.channels;

        const Nd4jLong outHeight = st.outHeight;
        const Nd4jLong outWidth = st.outWidth;
        Scaler scaler;

        auto func = PRAGMA_THREADS_FOR_2D {
            for (auto b = start_x; b < stop_x; b += inc_x) {
                for (auto y = start_y; y < stop_y; y += inc_y) {
                    auto posY = alignCorners ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(scaler(y, st.heightScale))) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(scaler(y, st.heightScale)));
                    Nd4jLong inY = nd4j::math::nd4j_min(posY, inHeight - 1);
                    if (halfPixelCenter) {
                        inY = nd4j::math::nd4j_max(0LL, inY);
                    }
                    for (auto x = 0; x < outWidth; ++x) {
                        auto posX = alignCorners ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(scaler(x, st.widthScale))) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(scaler(x, st.widthScale)));
                        Nd4jLong inX = nd4j::math::nd4j_min(posX,inWidth - 1);
                        if (halfPixelCenter) {
                            inX = nd4j::math::nd4j_max(0LL, inX);
                        }
                        // copy pixel over all channels
                        for (auto e = 0; e < channels; e++)
                            output->t<T>(b, y, x, e) = images->t<T>(b, inY, inX, e);
                    }
                }
            }
        };
        samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, outHeight, 1);
    }

    template<typename T>
    int resizeNeighborFunctor_(NDArray const *images, int const width, int const height, bool const alignCorners, bool const halfPixelCenter, NDArray *output) {
        ImageResizerState st(alignCorners, halfPixelCenter);
        st.validateAndCalculateOutputSize(images, width, height);

        // Handle no-op resizes efficiently.
        if (output->sizeAt(1) == images->sizeAt(1) && output->sizeAt(2) == images->sizeAt(2)) {
            output->assign(images);
            return Status::OK();
        }

        if (halfPixelCenter)
            resizeNeighbor<HalfPixelScalerNN, T>(st, images, alignCorners, true, output);
        else
            resizeNeighbor<LegacyScaler, T>(st, images, alignCorners, false, output);

        return Status::OK();
    }

//    void resizeImage(NDArray const *images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
//                     Nd4jLong outWidth, Nd4jLong channels,
//                     std::vector<BilinearInterpolationData> const &xs,
//                     std::vector<BilinearInterpolationData> const &ys,
//                     NDArray *output) {
//        BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), resizeImage_,
//                              (images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output),
//                              NUMERIC_TYPES, FLOAT_TYPES);
//    }

    int resizeBilinearFunctor(nd4j::LaunchContext * context, NDArray const *images, int const width, int const height,
            bool const alignCorners, bool const halfPixelCenter, NDArray *output) {
        BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_, (images, width, height, alignCorners, halfPixelCenter, output), NUMERIC_TYPES, FLOAT_TYPES);
        return Status::OK();
    }

    int resizeNeighborFunctor(nd4j::LaunchContext * context, NDArray const *images, int const width, int const height,
            bool const alignCorners,  bool const halfPixelCenter, NDArray *output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_, (images, width, height, alignCorners, halfPixelCenter, output), LIBND4J_TYPES);
    }


    template<typename T, typename F, typename I>
    static void cropAndResizeFunctor_(NDArray const *images, NDArray const *boxes, NDArray const *indices,
                                      NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops) {
        const int batchSize = images->sizeAt(0);
        const int imageHeight = images->sizeAt(1);
        const int imageWidth = images->sizeAt(2);

        const int numBoxes = crops->sizeAt(0);
        const int cropHeight = crops->sizeAt(1);
        const int cropWidth = crops->sizeAt(2);
        const int depth = crops->sizeAt(3);

        for (auto b = 0; b < numBoxes; ++b) {
            T y1 = boxes->t<F>(b, 0);
            T x1 = boxes->t<F>(b, 1);
            T y2 = boxes->t<F>(b, 2);
            T x2 = boxes->t<F>(b, 3);

            int bIn = indices->e<int>(b);
            if (bIn >= batchSize) {
                continue;
            }

            T heightScale = (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : T(0);
            T widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : T(0);

            auto func = PRAGMA_THREADS_FOR {
                for (auto y = start; y < stop; y += increment) {
                    const float inY = (cropHeight > 1)
                                      ? y1 * (imageHeight - 1) + y * heightScale
                                      : 0.5 * (y1 + y2) * (imageHeight - 1);

                    if (inY < 0 || inY > imageHeight - 1) {
                        for (auto x = 0; x < cropWidth; ++x) {
                            for (auto d = 0; d < depth; ++d) {
                                crops->p(b, y, x, d, extrapolationVal);
                            }
                        }
                        continue;
                    }
                    if (method == 0 /* bilinear */) {
                        const int topYIndex = nd4j::math::p_floor(inY);
                        const int bottomYIndex = nd4j::math::p_ceil(inY);
                        const float y_lerp = inY - topYIndex;

                        for (auto x = 0; x < cropWidth; ++x) {
                            const float in_x = (cropWidth > 1)
                                               ? x1 * (imageWidth - 1) + x * widthScale
                                               : 0.5 * (x1 + x2) * (imageWidth - 1);

                            if (in_x < 0 || in_x > imageWidth - 1) {
                                for (auto d = 0; d < depth; ++d) {
                                    crops->p(b, y, x, d, extrapolationVal);
                                }
                                continue;
                            }
                            int left_x_index = math::p_floor(in_x);
                            int right_x_index = math::p_ceil(in_x);
                            T x_lerp = in_x - left_x_index;

                            for (auto d = 0; d < depth; ++d) {
                                const float topLeft(images->e<float>(bIn, topYIndex, left_x_index, d));
                                const float topRight(images->e<float>(bIn, topYIndex, right_x_index, d));
                                const float bottomLeft(images->e<float>(bIn, bottomYIndex, left_x_index, d));
                                const float bottomRight(images->e<float>(bIn, bottomYIndex, right_x_index, d));
                                const float top = topLeft + (topRight - topLeft) * x_lerp;
                                const float bottom = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
                                crops->p(b, y, x, d, top + (bottom - top) * y_lerp);
                            }
                        }
                    } else {  // method is "nearest neighbor"
                        for (auto x = 0; x < cropWidth; ++x) {
                            const float inX = (cropWidth > 1)
                                              ? x1 * (imageWidth - 1) + x * widthScale
                                              : 0.5 * (x1 + x2) * (imageWidth - 1);

                            if (inX < 0 || inX > imageWidth - 1) {
                                for (auto d = 0; d < depth; ++d) {
                                    crops->p(b, y, x, d, extrapolationVal);
                                }
                                continue;
                            }
                            const int closestXIndex = roundf(inX);
                            const int closestYIndex = roundf(inY);
                            for (auto d = 0; d < depth; ++d) {
                                crops->p(b, y, x, d, images->e<T>(bIn, closestYIndex, closestXIndex, d));
                            }
                        }
                    }
                }
            };

            samediff::Threads::parallel_for(func, 0, cropHeight);
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------------------------------------------------------------------------------------------------------------ //
// Bicubic interpolation
// ------------------------------------------------------------------------------------------------------------------ //
    class CachedInterpolationCalculator {
    public:
        CachedInterpolationCalculator() : _indexes{-1, -1, -1, -1} {}

        // Advances iteration. Returns the number of values that should be copied from
        // the current point to the next point. The copying should always be done by
        // copying the last <retval> values from the old point to the first <retval>
        // values of the new point.
        inline int Advance(const Nd4jLong x0, const Nd4jLong x1, const Nd4jLong x2,
                           const Nd4jLong x3) {
            // We use 2 hands and walk through, copying from one to another where
            // we already have values.
            // Invariant, new_indicies_hand <= cached_values_hand
            const Nd4jLong new_x_indices[] = {x0, x1, x2, x3};
            int cachedValuesHand = 0;
            int newIndiciesHand = 0;
            while (cachedValuesHand < 4) {
                if (_indexes[cachedValuesHand] == new_x_indices[newIndiciesHand]) {
                    if (newIndiciesHand < cachedValuesHand) {
                        _indexes[newIndiciesHand] = _indexes[cachedValuesHand];
                    }
                    newIndiciesHand++;
                }
                cachedValuesHand++;
            }
            switch (newIndiciesHand) {
                case 0:
                    _indexes[0] = x0;
                case 1:
                    _indexes[1] = x1;
                case 2:
                    _indexes[2] = x2;
                case 3:
                    _indexes[3] = x3;
                    break;
            }
            return newIndiciesHand;
        }

    private:
        Nd4jLong _indexes[4];
    };
    static const Nd4jLong kTableSize = 1024LL; //(1 << 10);

    const float* initCoeffsTable(const double a) {
        // Allocate and initialize coefficients table using Bicubic
        // convolution algorithm.
        // https://en.wikipedia.org/wiki/Bicubic_interpolation
        float* coeffs_table = new float[(kTableSize + 1) * 2];
        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i <= stop; ++i) {
                float x = i * 1.0 / kTableSize;
                coeffs_table[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
                x += 1.0;
                coeffs_table[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
            }
        };
        samediff::Threads::parallel_for(func, 0, kTableSize);
        return coeffs_table;
    }

    const float* getCoeffsTable(const bool use_keys_cubic) {
        // Static so that we initialize it on first use
        if (use_keys_cubic) {
            // http://ieeexplore.ieee.org/document/1163711/
            // R. G. Keys. Cubic convolution interpolation for digital image
            // processing. IEEE Transactions on Acoustics, Speech, and Signal
            // Processing, 29(6):1153–1160, 1981.
            static const float* coeffs_table = initCoeffsTable(-0.5f);
            return coeffs_table;
        } else {
            static const float* coeffs_table = initCoeffsTable(-0.75f);
            return coeffs_table;
        }
    }

    inline Nd4jLong bound(Nd4jLong val, Nd4jLong limit) {
        return math::nd4j_min(limit - 1ll, math::nd4j_max(Nd4jLong{0}, val));
    }

    template <typename T>
    int resizeBicubicFunctor_(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                             bool preserveAspectRatio, bool antialias, NDArray* output) {
        return ND4J_STATUS_OK;
    }

    int resizeBicubicFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                             bool preserveAspectRatio, bool antialias, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctor_, (context, image,
                width, height, preserveAspectRatio, antialias, output), NUMERIC_TYPES);
    }
// ------------------------------------------------------------------------------------------------------------------ //

        template <typename T>
        inline float interpolate1D(const float weight0, const float weight1, const float weight2, const float weight3,
                                   const T value0, const T value1, const T value2, const T value3) {
            return static_cast<float>(value0) * weight0 +
                   static_cast<float>(value1) * weight1 +
                   static_cast<float>(value2) * weight2 +
                   static_cast<float>(value3) * weight3;
        }

// Compute the 1D interpolation for a given X index using the y_weights
        static float compute(float values[4], const float xW0, const float xW1, const float xW2, const float xW3) {
            return interpolate1D(xW0, xW1, xW2, xW3, values[0], values[1],values[2], values[3]);
        }

        template <typename Scaler, bool use_keys_cubic>
        inline void getWeightsAndIndices(const float scale, const Nd4jLong out_loc, const Nd4jLong limit, WeightsAndIndices* out) {
            const Scaler scaler;
            const float in_loc_f = scaler(out_loc, scale);
            const Nd4jLong in_loc = std::floor(in_loc_f);
            const float delta = in_loc_f - in_loc;
            const Nd4jLong offset = lrintf(delta * kTableSize);
            const float* coeffs_table = getCoeffsTable(use_keys_cubic);
            if (use_keys_cubic) {
                // The legacy code placed more weight on the edge pixels, since bounding
                // the set of inputs to sample could cause an edge pixel to be repeated.
                // Here we change the behavior at borders to match that used by the
                // scale_and_translate_op, where sampling locations outside the image have
                // their weight set to 0, and the weights are renormalized so that their sum
                // is 1.0.
                out->_index0 = bound(in_loc - 1, limit);
                out->_weight0 =
                        (out->_index0 == in_loc - 1 ? coeffs_table[offset * 2 + 1] : 0.0f);
                out->_index1 = bound(in_loc, limit);
                out->_weight1 = (out->_index1 == in_loc ? coeffs_table[offset * 2] : 0.0f);
                out->_index2 = bound(in_loc + 1, limit);
                out->_weight2 =
                        (out->_index2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * 2]
                                                    : 0.0f);
                out->_index3 = bound(in_loc + 2, limit);
                out->_weight3 = (out->_index3 == in_loc + 2
                                 ? coeffs_table[(kTableSize - offset) * 2 + 1]
                                 : 0.0f);

                const float weight_sum =
                        out->_weight0 + out->_weight1 + out->_weight2 + out->_weight3;
                if (std::abs(weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
                    const float one_over_weight_sum = 1.0f / weight_sum;
                    out->_weight0 *= one_over_weight_sum;
                    out->_weight1 *= one_over_weight_sum;
                    out->_weight2 *= one_over_weight_sum;
                    out->_weight3 *= one_over_weight_sum;
                }
            } else {
                out->_weight0 = coeffs_table[offset * 2 + 1];
                out->_weight1 = coeffs_table[offset * 2];
                out->_weight2 = coeffs_table[(kTableSize - offset) * 2];
                out->_weight3 = coeffs_table[(kTableSize - offset) * 2 + 1];
                out->_index0 = bound(in_loc - 1, limit);
                out->_index1 = bound(in_loc, limit);
                out->_index2 = bound(in_loc + 1, limit);
                out->_index3 = bound(in_loc + 2, limit);
            }
        }

        static void computeXWeightsAndIndices(const ImageResizerState& resizer_state,
                                              const bool half_pixel_centers,
                                              std::vector<WeightsAndIndices>* x_wais) {
            CachedInterpolationCalculator calc;
            if (half_pixel_centers) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto x = start; x < stop; ++x) {
                        getWeightsAndIndices<HalfPixelScaler, true>(
                                resizer_state.widthScale, x, resizer_state.inWidth, &(*x_wais)[x]);
                        auto &x_wai = (*x_wais)[x];
                        x_wai._advance = calc.Advance(x_wai._index0, x_wai._index1, x_wai._index2,
                                                      x_wai._index3);
                    }
                };
                samediff::Threads::parallel_for(func, 0, resizer_state.outWidth);
            } else {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto x = start; x < stop; ++x) {
                        getWeightsAndIndices<LegacyScaler, false>(
                                resizer_state.widthScale, x, resizer_state.inWidth, &(*x_wais)[x]);
                        auto& x_wai = (*x_wais)[x];
                        x_wai._advance = calc.Advance(x_wai._index0, x_wai._index1, x_wai._index2,
                                                      x_wai._index3);
                    }
                };
                samediff::Threads::parallel_for(func, 0, resizer_state.outWidth);
            }
            // Scale the values so they can be used as offsets into buffers.
            auto func = PRAGMA_THREADS_FOR {
                for (auto x = start; x < stop; ++x) {
                    (*x_wais)[x]._index0 *= resizer_state.channels;
                    (*x_wais)[x]._index1 *= resizer_state.channels;
                    (*x_wais)[x]._index2 *= resizer_state.channels;
                    (*x_wais)[x]._index3 *= resizer_state.channels;
                }
            };
            samediff::Threads::parallel_for(func, 0, resizer_state.outWidth);
        }

        template <typename T>
        static FORCEINLINE float computeYInterpolation(
                int which, int channelNum, const WeightsAndIndices& yWai,
                const T* pY0, const T* pY1, const T* pY2, const T* pY3,
                const WeightsAndIndices& xWai) {
            int xIndex;
            switch (which) {
                case 0:
                    xIndex = xWai._index0;
                    break;
                case 1:
                    xIndex = xWai._index1;
                    break;
                case 2:
                    xIndex = xWai._index2;
                    break;
                default:
                    xIndex = xWai._index3;
                    break;
            }
            const Nd4jLong pt_index = xIndex + channelNum;
            return interpolate1D<T>(yWai._weight0, yWai._weight1, yWai._weight2,
                                    yWai._weight3, pY0[pt_index], pY1[pt_index],
                                    pY2[pt_index], pY3[pt_index]);
        }

        template <typename T, typename F>
        static void
    bicubicInterpolateWithCaching(NDArray const* image, ImageResizerState const& resizerState, bool const halfPixelCenters, NDArray* output) {
        std::vector<WeightsAndIndices> xWais(resizerState.outWidth);
        computeXWeightsAndIndices(resizerState, halfPixelCenters, &xWais);

        const auto numChannels = resizerState.channels;
        const Nd4jLong inRowWidth = resizerState.inWidth * numChannels;
        const Nd4jLong inBatchWidth = resizerState.inHeight * inRowWidth;
        const auto batchNum = resizerState.batchSize;
        const auto outHeight = resizerState.outHeight;
        const auto outWidth = resizerState.outWidth;

       auto func = PRAGMA_THREADS_FOR {
            const T* inputPtr = image->getDataBuffer()->primaryAsT<T>();
            F* pOutputY = output->dataBuffer()->primaryAsT<F>(); // output is float anyway
            std::vector<float> cachedValue(numChannels == 3 ? 0 : 4 * numChannels, 0);

            for (auto b = start; b < stop; ++b) {
                auto pInput = inputPtr + b * inBatchWidth;

                for (auto y = 0; y < outHeight; ++y) {
                    auto pOutput = &pOutputY[(b * outHeight + y) * outWidth * numChannels];

                    WeightsAndIndices yWai;
                    if (halfPixelCenters) {
                        getWeightsAndIndices<HalfPixelScaler, true>(
                                resizerState.heightScale, y, resizerState.inHeight, &yWai);
                    } else {
                        getWeightsAndIndices<LegacyScaler, false>(
                                resizerState.heightScale, y, resizerState.inHeight, &yWai);
                    }
                    // Make pointers represent offsets of data in inputBPtr.
                    const T* y_ptr_0 = pInput + yWai._index0 * inRowWidth;
                    const T* y_ptr_1 = pInput + yWai._index1 * inRowWidth;
                    const T* y_ptr_2 = pInput + yWai._index2 * inRowWidth;
                    const T* y_ptr_3 = pInput + yWai._index3 * inRowWidth;

                    if (numChannels == 3) {
                        // Manually unroll case of 3 channels.
                        F cached_value_0[4] = {0};
                        F cached_value_1[4] = {0};
                        F cached_value_2[4] = {0};
                        for (auto x = 0; x < resizerState.outWidth; ++x) {
                            const WeightsAndIndices &xWai = xWais[x];
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
                                    cached_value_0[0] = computeYInterpolation(
                                            0, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_1[0] = computeYInterpolation(
                                            0, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_2[0] = computeYInterpolation(
                                            0, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

                                case 1:
                                    cached_value_0[1] = computeYInterpolation(
                                            1, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_1[1] = computeYInterpolation(
                                            1, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_2[1] = computeYInterpolation(
                                            1, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

                                case 2:
                                    cached_value_0[2] = computeYInterpolation(
                                            2, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_1[2] = computeYInterpolation(
                                            2, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_2[2] = computeYInterpolation(
                                            2, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);

                                case 3:
                                    cached_value_0[3] = computeYInterpolation(
                                            3, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_1[3] = computeYInterpolation(
                                            3, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    cached_value_2[3] = computeYInterpolation(
                                            3, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    break;
                            }
                            pOutput[x * numChannels + 0] =
                                    compute(cached_value_0, xWai._weight0, xWai._weight1,
                                            xWai._weight2, xWai._weight3);
                            pOutput[x * numChannels + 1] =
                                    compute(cached_value_1, xWai._weight0, xWai._weight1,
                                            xWai._weight2, xWai._weight3);
                            pOutput[x * numChannels + 2] =
                                    compute(cached_value_2, xWai._weight0, xWai._weight1,
                                            xWai._weight2, xWai._weight3);
                        }
                    } else {
                        for (auto x = 0; x < resizerState.outWidth; ++x) {
                            const WeightsAndIndices &xWai = xWais[x];
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
                                        cachedValue[4 * c + 0] = computeYInterpolation(
                                                0, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    }

                                case 1:
                                    for (auto c = 0; c < numChannels; ++c) {
                                        cachedValue[4 * c + 1] = computeYInterpolation(
                                                1, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    }

                                case 2:
                                    for (auto c = 0; c < numChannels; ++c) {
                                        cachedValue[4 * c + 2] = computeYInterpolation(
                                                2, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    }

                                case 3:
                                    for (auto c = 0; c < numChannels; ++c) {
                                        cachedValue[4 * c + 3] = computeYInterpolation(
                                                3, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                    }
                                    break;
                            }
                            for (auto c = 0; c < numChannels; ++c) {
                                pOutput[x * numChannels + c] =
                                        (F)compute(&cachedValue[4 * c], xWai._weight0, xWai._weight1,
                                                xWai._weight2, xWai._weight3);
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
    int resizeBicubicFunctorA_(nd4j::LaunchContext * context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, bool const halfPixelAlign, NDArray* output) {
        ImageResizerState st(alignCorners, halfPixelAlign); // align_corners, half_pixel_align
        int res = st.validateAndCreateOutput(image, width, height);
        if (res == Status::OK())
            bicubicInterpolateWithCaching<T, float>(image, st, halfPixelAlign, output);

        return res;
    }
    int resizeBicubicFunctorA(nd4j::LaunchContext * context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, bool const halfPixelAlign, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctorA_, (context, image, width, height, alignCorners, halfPixelAlign, output), NUMERIC_TYPES);
    }
// ------------------------------------------------------------------------------------------------------------------ //
    struct CachedInterpolation {
        Nd4jLong start;
        Nd4jLong end;
        float startScale;
        float endMinusOneScale;
        bool needsBounding;
    };

    template <typename T>
    struct ScaleCache {
        float yScale;
        T const* yPtr;
    };
    // Computes the sum of all x values defined by <x_interp> taken across
    // the y offsets and scales defined by y_ptrs and y_scales, for channel c.
    //
    // Note that <NeedsXBounding> is a template parameter to avoid a performance
    // penalty from dynamically checking it.
    template <typename T>
    static void computePatchSumOf3Channels(float scale,
                                           ImageResizerState const& st,
                                           std::vector<ScaleCache<T>> const& yPtrs,
                                           CachedInterpolation const& xCache,
                                           float* outputPtr) {

        bool const needsXBounding = xCache.needsBounding;

        auto boundIfNeeded = [needsXBounding](Nd4jLong x, Nd4jLong y) -> Nd4jLong {
            return (needsXBounding ? bound(x, y) : (x));
        };

        float sum_0 = 0;
        float sum_1 = 0;
        float sum_2 = 0;
        for (int i = 0; i < yPtrs.size(); ++i) {
            const T* ptr = yPtrs[i].yPtr;
            float scaleX = xCache.startScale;
            Nd4jLong offset = 3 * boundIfNeeded(xCache.start, st.inWidth);
            float sum_y_0 = static_cast<float>(ptr[offset + 0]) * scaleX;
            float sum_y_1 = static_cast<float>(ptr[offset + 1]) * scaleX;
            float sum_y_2 = static_cast<float>(ptr[offset + 2]) * scaleX;

            if (xCache.start + 1 != xCache.end) {
                for (Nd4jLong x = xCache.start + 1; x < xCache.end - 1; ++x) {
                    Nd4jLong offset = 3 * boundIfNeeded(x, st.inWidth);
                    sum_y_0 += static_cast<float>(ptr[offset + 0]);
                    sum_y_1 += static_cast<float>(ptr[offset + 1]);
                    sum_y_2 += static_cast<float>(ptr[offset + 2]);
                }
                scaleX = xCache.endMinusOneScale;
                offset = st.channels * boundIfNeeded(xCache.end - 1, st.inWidth);
                sum_y_0 += static_cast<float>(ptr[offset + 0]) * scaleX;
                sum_y_1 += static_cast<float>(ptr[offset + 1]) * scaleX;
                sum_y_2 += static_cast<float>(ptr[offset + 2]) * scaleX;
            }
            sum_0 += sum_y_0 * yPtrs[i].yScale;
            sum_1 += sum_y_1 * yPtrs[i].yScale;
            sum_2 += sum_y_2 * yPtrs[i].yScale;
        }

        outputPtr[0] = sum_0 * scale;
        outputPtr[1] = sum_1 * scale;
        outputPtr[2] = sum_2 * scale;
    }

        // Computes the sum of all x values defined by <x_interp> taken across
        // the y offsets and scales defined by y_ptrs and y_scales, for channel c.
        //
        // Note that <NeedsXBounding> is a template parameter to avoid a performance
        // penalty from dynamically checking it.
        template <typename T>
        static void computePatchSum(float scale, const ImageResizerState& st,
                                    const std::vector<ScaleCache<T>>& yPtrs,
                                    const CachedInterpolation& xCache,
                                    float* outputPtr) {

            bool const needsXBounding = xCache.needsBounding;

            auto boundIfNeeded = [needsXBounding](Nd4jLong x, Nd4jLong y) -> Nd4jLong {
                return (needsXBounding ? bound(x, y) : (x));
            };

            const auto numChannels = st.channels;
            for (Nd4jLong c = 0; c < numChannels; ++c) {
                float sum = 0;
                for (int i = 0; i < yPtrs.size(); ++i) {
                    T const* ptr = yPtrs[i].yPtr;
                    float scaleX = xCache.startScale;
                    float sumY = static_cast<float>(ptr[numChannels * boundIfNeeded(xCache.start, st.inWidth) + c]) * scaleX;
                    if (xCache.start + 1 != xCache.end) {
                        for (Nd4jLong x = xCache.start + 1; x < xCache.end - 1; ++x) {
                            sumY += static_cast<float>(
                                    ptr[numChannels * boundIfNeeded(x, st.inWidth) + c]);
                        }
                        scaleX = xCache.endMinusOneScale;
                        sumY += static_cast<float>(ptr[numChannels * boundIfNeeded(xCache.end - 1, st.inWidth) + c]) * scaleX;
                    }
                    sum += sumY * yPtrs[i].yScale;
                }
                outputPtr[c] = sum * scale;
            }
        }



    template <typename T>
    static void resizeArea(ImageResizerState const& st, std::vector<CachedInterpolation> const& caches, NDArray const* input, NDArray* output) {
        T const* inputPtr = input->bufferAsT<T>();
        float scale = 1.f / (st.heightScale * st.widthScale);
        auto outputPtr = output->bufferAsT<float>(); // output is always float. TO DO: provide another float types also with  template <typename X, typename Z> declaration

        auto batchProcess = PRAGMA_THREADS_FOR {
            for (auto batch = start; batch < stop; batch += increment) {
                for (auto y = 0; y < st.outHeight; ++y) {
                    const float inY = y * st.heightScale;
                    const float inY1 = (y + 1) * st.heightScale;
                    // The start and end height indices of all the cells that could
                    // contribute to the target cell.
                    const Nd4jLong yStart = math::nd4j_floor<float, Nd4jLong>(inY);
                    const Nd4jLong yEnd = math::nd4j_ceil<float, Nd4jLong>(inY1);

                    std::vector<ScaleCache<T>> yCaches;
                    auto cacheLen = yEnd - yStart;
                    if (cacheLen) {
                        yCaches.resize(cacheLen);
                    };

                    for (auto i = yStart, k = 0LL; i < yEnd; ++i, ++k) {
                        ScaleCache<T> scaleCache;
                        if (i < inY) {
                            scaleCache.yScale = (i + 1 > inY1 ? st.heightScale : i + 1 - inY);
                        } else {
                            scaleCache.yScale = (i + 1 > inY1 ? inY1 - i : 1.0);
                        }
                        scaleCache.yPtr = inputPtr + (batch * st.inHeight * st.inWidth * st.channels +
                                bound(i, st.inHeight) * st.inWidth * st.channels);
                        yCaches[k] = scaleCache;
                    }
                    float* output = outputPtr + (batch * st.outHeight  +  y)  * st.channels * st.outWidth;

                    if (st.channels == 3) {
                        for (Nd4jLong x = 0; x < st.outWidth; ++x) {
                            const CachedInterpolation &xCache = caches[x];
                            computePatchSumOf3Channels<T>(scale, st, yCaches, xCache, output);
                            output += st.channels;
                        }
                    } else {
                        for (Nd4jLong x = 0; x < st.outWidth; ++x) {
                            const CachedInterpolation &xCache = caches[x];
                            computePatchSum<T>(scale, st, yCaches, xCache, output);
                            output += st.channels;
                        }
                    }
                }
            }
        };
        samediff::Threads::parallel_tad(batchProcess, 0, st.batchSize, 1);
    }

    template <typename X>
    int resizeAreaFunctor_(nd4j::LaunchContext* context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, NDArray* output) {
            ImageResizerState st(alignCorners, false); // Create resize info
            auto res = st.validateAndCalculateOutputSize(image, width, height);
            if (Status::OK() == res) {
                std::vector<CachedInterpolation> xCached(st.outWidth);
                auto cachingProcedure = PRAGMA_THREADS_FOR {
                    for (auto x = start; x < stop; x += increment) {
                        auto &xCache = xCached[x];
                        const float inX = x * st.widthScale;
                        const float inX1 = (x + 1) * st.widthScale;

                        Nd4jLong v = math::nd4j_floor<float, Nd4jLong>(inX);
                        xCache.start = v;
                        xCache.startScale =
                                v < inX ? (v + 1 > inX1 ? st.widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v
                                                                                                       : 1.f);
                        v = math::nd4j_ceil<float, Nd4jLong>(inX1);
                        xCache.end = v--;
                        xCache.endMinusOneScale =
                                v < inX ? (v + 1 > inX1 ? st.widthScale : v + 1 - inX) : (v + 1 > inX1 ? inX1 - v
                                                                                                       : 1.f);
                        xCache.needsBounding = bound(xCache.start, st.inWidth) != xCache.start ||
                                               bound(xCache.end - 1, st.inWidth) != (xCache.end - 1);

                    }
                };
                samediff::Threads::parallel_for(cachingProcedure, 0, xCached.size(), 1);

                resizeArea<X>(st, xCached, image, output);
            }
            return res;
    }

    int resizeAreaFunctor(nd4j::LaunchContext * context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_, (context, image, width, height, alignCorners, output), NUMERIC_TYPES);
    }

// ------------------------------------------------------------------------------------------------------------------ //
    int resizeFunctor(nd4j::LaunchContext * context, NDArray const* image, int const width, int const height,
                      ImageResizeMethods method, bool preserveAspectRatio, bool antialias, NDArray* output) {
        switch (method) {
            case kResizeBilinear: return resizeBilinearFunctor(context, image, width, height, false, false, output); break;
            case kResizeNearest: return resizeNeighborFunctor(context, image, width, height, false, false, output); break;
            case kResizeBicubic: return resizeBicubicFunctor(context, image, width, height, preserveAspectRatio, antialias, output); break;
            case kResizeArea: return resizeAreaFunctor(context, image, width, height, preserveAspectRatio, output);
            case kResizeLanczos5:
            case kResizeGaussian:
            case kResizeMitchelcubic:
                throw std::runtime_error("helper::resizeFunctor: Non implemented yet.");
        }
        return ND4J_STATUS_OK;
    }

// ------------------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------------------ //
// crop and resize helper functor:
// \@param context - launch context for operation
// \@param images - batch of images (4D tensor) with shape {batch, width, height, channels} with given type
// \@param boxes - float boxes for crop
// \@param indices - integer boxes indices for crop
// \@param cropSize - integer size (newWidth, newHeight)
// \@param method - one of bilinear (0) or nearest neighbour (1) interpolation algorithm
// \@param extrapolationVal - radix to increase/decrease image
// \@param crops - output image batch (4D with given type)
//
    void
    cropAndResizeFunctor(nd4j::LaunchContext * context, NDArray const *images, NDArray const *boxes,
            NDArray const *indices, NDArray const *cropSize,
                         int method, double extrapolationVal, NDArray *crops) {
        BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                              (images, boxes, indices, cropSize, method, extrapolationVal, crops), NUMERIC_TYPES, FLOAT_TYPES, INTEGER_TYPES);
    }
}
}
}