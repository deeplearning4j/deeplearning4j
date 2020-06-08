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

namespace sd {
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
                double const in_f = sd::math::nd4j_floor<double, double>(in);
                double const in_c = sd::math::nd4j_ceil<double, double>(in);
                interpolationData[i]._bottomIndex = sd::math::nd4j_max(static_cast<Nd4jLong>(in_f), (Nd4jLong)0LL);//static_cast<Nd4jLong>(in);
                interpolationData[i]._topIndex = sd::math::nd4j_min(static_cast<Nd4jLong>(in_c), inSize - 1);
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
                for (Nd4jLong y = 0; y < outHeight; ++y) {
                    auto pOutput = pOutputBuf + (batch * outHeight + y) * outRowSize;
                    const T* ysInputLowerPtr = pInput + ys[y]._bottomIndex * inRowSize;
                    const T* ysInputUpperPtr = pInput + ys[y]._topIndex * inRowSize;
                    double yVal = ys[y]._interpolarValue;
                    for (Nd4jLong x = 0; x < outWidth; ++x) {
                        auto xsBottom = xsPtr[x]._bottomIndex;
                        auto xsTop = xsPtr[x]._topIndex;
                        auto xVal = xsPtr[x]._interpolarValue;
                        for (Nd4jLong c = 0; c < channels; ++c) {
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
            for (auto i = start; i < stop; i++) {
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
                    auto posY = alignCorners ? static_cast<Nd4jLong>(sd::math::p_round<float>(scaler(y, st.heightScale))) : static_cast<Nd4jLong>(sd::math::p_floor<float>(scaler(y, st.heightScale)));
                    Nd4jLong inY = sd::math::nd4j_min(posY, inHeight - 1);
                    if (halfPixelCenter) {
                        inY = sd::math::nd4j_max(0LL, inY);
                    }
                    for (Nd4jLong x = 0; x < outWidth; ++x) {
                        auto posX = alignCorners ? static_cast<Nd4jLong>(sd::math::p_round<float>(scaler(x, st.widthScale))) : static_cast<Nd4jLong>(sd::math::p_floor<float>(scaler(x, st.widthScale)));
                        Nd4jLong inX = sd::math::nd4j_min(posX,inWidth - 1);
                        if (halfPixelCenter) {
                            inX = sd::math::nd4j_max(0LL, inX);
                        }
                        // copy pixel over all channels
                        for (Nd4jLong e = 0; e < channels; e++)
                            output->r<T>(b, y, x, e) = images->t<T>(b, inY, inX, e);
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

    int resizeBilinearFunctor(sd::LaunchContext * context, NDArray const *images, int const width, int const height,
            bool const alignCorners, bool const halfPixelCenter, NDArray *output) {
        BUILD_DOUBLE_SELECTOR(images->dataType(), output->dataType(), return resizeBilinearFunctor_, (images, width, height, alignCorners, halfPixelCenter, output), NUMERIC_TYPES, FLOAT_TYPES);
        return Status::OK();
    }

    int resizeNeighborFunctor(sd::LaunchContext * context, NDArray const *images, int const width, int const height,
            bool const alignCorners,  bool const halfPixelCenter, NDArray *output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_, (images, width, height, alignCorners, halfPixelCenter, output), LIBND4J_TYPES);
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
        float* coeffsTable = new float[(kTableSize + 1) * 2];
        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i <= stop; ++i) {
                float x = i * 1.0 / kTableSize;
                coeffsTable[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
                x += 1.0;
                coeffsTable[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
            }
        };
        samediff::Threads::parallel_for(func, 0, kTableSize);
        return coeffsTable;
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
    int resizeBicubicFunctor_(sd::LaunchContext * context, NDArray const* image, int width, int height,
                             bool preserveAspectRatio, bool antialias, NDArray* output) {
        return ND4J_STATUS_OK;
    }

    int resizeBicubicFunctor(sd::LaunchContext * context, NDArray const* image, int width, int height,
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

                for (Nd4jLong y = 0; y < outHeight; ++y) {
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
                        for (Nd4jLong x = 0; x < resizerState.outWidth; ++x) {
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
                        for (Nd4jLong x = 0; x < resizerState.outWidth; ++x) {
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
    int resizeBicubicFunctorA_(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, bool const halfPixelAlign, NDArray* output) {
        ImageResizerState st(alignCorners, halfPixelAlign); // align_corners, half_pixel_align
        int res = st.validateAndCreateOutput(image, width, height);
        if (res == Status::OK())
            bicubicInterpolateWithCaching<T, float>(image, st, halfPixelAlign, output);

        return res;
    }
    int resizeBicubicFunctorA(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
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
        for (size_t i = 0; i < yPtrs.size(); ++i) {
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
                for (size_t i = 0; i < yPtrs.size(); ++i) {
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
            for (auto batch = start; batch < stop; batch++) {
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
    int resizeAreaFunctor_(sd::LaunchContext* context, NDArray const* image, int const width, int const height,
                              bool const alignCorners, NDArray* output) {
            ImageResizerState st(alignCorners, false); // Create resize info
            auto res = st.validateAndCalculateOutputSize(image, width, height);
            if (Status::OK() == res) {
                std::vector<CachedInterpolation> xCached(st.outWidth);
                auto cachingProcedure = PRAGMA_THREADS_FOR {
                    for (auto x = start; x < stop; x++) {
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

    int resizeAreaFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const alignCorners, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeAreaFunctor_, (context, image, width, height, alignCorners, output), NUMERIC_TYPES);
    }

    /**
     * resize as TF v.2.x implemented (with preserve aspect ratio and antialias flags routines
     * */
    // An interface for integrated scale functors.
    struct IKernelFunc {
        virtual float operator()(float x) const = 0;
        virtual float radius() const = 0;
    };

    struct LanczosKernelFunc : public IKernelFunc {
        // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
        explicit LanczosKernelFunc(float const radius) : _radius(radius) {}
        float operator()(float x) const {
            float const kPI = 3.141592653589793f;
            x = math::nd4j_abs(x);
            if (x > _radius) return 0.f;
            // Need to special case the limit case of sin(x) / x when x is zero.
            if (x <= 1.e-3f) {
                return 1.f;
            }
            return _radius * std::sin(kPI * x) * std::sin(kPI * x / _radius) / (kPI * kPI * x * x);
        }
        float radius() const { return _radius; }
        const float _radius;
    };

    struct GaussianKernelFunc : public IKernelFunc {
        static constexpr float kRadiusMultiplier = 3.0f;
        // https://en.wikipedia.org/wiki/Gaussian_function
        // We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
        // for Common Resampling Tasks" for kernels with a support of 3 pixels:
        // www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
        // This implies a radius of 1.5,
        explicit GaussianKernelFunc(float radius = 1.5f)
                : _radius(radius), _sigma(radius / kRadiusMultiplier) {}
        float operator()(float x) const {
            x = math::nd4j_abs(x);
            if (x >= _radius) return 0.0f;
            return std::exp(-x * x / (2.0 * _sigma * _sigma));
        }
        float radius() const { return _radius; }
        const float _radius;
        const float _sigma;  // Gaussian standard deviation
    };

    struct BoxKernelFunc : public IKernelFunc {
        float operator()(float x) const {
            x = math::nd4j_abs(x);
            return x < 0.5f ? 1.f : x == 0.5f ? 0.5f : 0.f;
        }
        float radius() const { return 1.f; }
    };

    struct TriangleKernelFunc : public IKernelFunc {
        // https://en.wikipedia.org/wiki/Triangle_function
        float operator()(float x) const {
            x = math::nd4j_abs(x);
            return x < 1.f ? 1.f - x : 0.f;
        }
        float radius() const { return 1.f; }
    };

    struct KeysCubicKernelFunc : public IKernelFunc {
        // http://ieeexplore.ieee.org/document/1163711/
        // R. G. Keys. Cubic convolution interpolation for digital image
        // processing. IEEE Transactions on Acoustics, Speech, and Signal
        // Processing, 29(6):1153–1160, 1981.
        float operator()(float x) const {
            x = math::nd4j_abs(x);
            if (x >= 2.0f) {
                return 0.0f;
            } else if (x >= 1.0f) {
                return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
            } else {
                return ((1.5f * x - 2.5f) * x) * x + 1.0f;
            }
        }
        float radius() const { return 2.f; }
    };

    struct MitchellCubicKernelFunc : public IKernelFunc {
        // https://doi.org/10.1145/378456.378514
        // D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
        // graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
        // 22(4):221–228, 1988.
        float operator()(float x) const {
            x = math::nd4j_abs(x);
            if (x >= 2.f) {
                return 0.f;
            } else if (x >= 1.f) {
                return (((-7.f / 18.f) * x + 2.f) * x - 10.f / 3.f) * x + 16.f / 9.f;
            } else {
                return (((7.f / 6.f) * x - 2.f) * x) * x + 8.f / 9.f;
            }
        }
        float radius() const { return 2.f; }
    };

    // A pre-computed span of pixels along a single dimension.
    // The output pixel will be the weighted sum of pixels starting from start.
    struct Spans {
        // The maximum span size of any output pixel.
        int _spanSize;
        // int32 tensor with shape {outputSize}.
        NDArray _starts;

        // float32 tensor of size {outputSize, spanSize}.
        // The output pixel at x is computed as:
        //   dot_product(input[starts[x]:starts[x]+span_size], weights[x]).
        NDArray _weights;
    };

    static int
    computeSpans(IKernelFunc* kernel, Nd4jLong const outSize, Nd4jLong const inSize, float const scale, float const translate, bool const antialias, Spans& spans) {
        // When sampling, we need the inverse scale and translation, to map from an
        // output to an input pixel.
        float const invScale = 1.f / scale;
        float const invTranslate = -invScale * translate;
        // When downsampling the kernel should be scaled since we want to low pass
        // filter and interpolate, but when upsampling it should not be since we only
        // want to interpolate.
        float  const kernelScale = antialias ? math::nd4j_max(invScale, 1.f) : 1.f;
        spans._spanSize = math::nd4j_min(2 * static_cast<int>(std::ceil(kernel->radius() * kernelScale)) + 1, static_cast<int>(inSize));
        spans._starts = NDArrayFactory::create<int>('c', {outSize});
        spans._weights = NDArrayFactory::create<float>('c', {outSize, spans._spanSize});

        auto startsVec = spans._starts.bufferAsT<int>();
        auto weightsVector = spans._weights.bufferAsT<float>();
        spans._weights.nullify();

        const float invKernelScale = 1.f / kernelScale;
        int maxSpanSize = 0;
        std::vector<float> tempWeights;

        // return value if within bounds or bounds otherwise
        auto boundsAmp = [](Nd4jLong  const low, Nd4jLong const high, Nd4jLong const value) {
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
            Nd4jLong spanStart = math::nd4j_ceil<float,float>(sampleFloat - kernel->radius() * kernelScale - 0.5f);
            Nd4jLong spanEnd = math::nd4j_floor<float, float>(sampleFloat + kernel->radius() * kernelScale - 0.5f);
            spanStart = boundsAmp(0LL, inSize - 1, spanStart);
            spanEnd = boundsAmp(0LL, inSize - 1, spanEnd) + 1;
            int const spanSize = spanEnd - spanStart;
            if (spanSize > spans._spanSize) {
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "Span is too large: "); // + spanSize + " vs " + spans._spanSize);//, spanSize, spans._spanSize));
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
            if (math::nd4j_abs(totalWeightSum) >= 1000.f * DataTypeUtils::min<float>()) { //
                auto totalWeightSumInverted = 1.0f / totalWeightSum;
                auto outIndex = spans._spanSize * x;
                for (auto weight : tempWeights) {
                    weightsVector[outIndex] = weight * totalWeightSumInverted;
                    ++outIndex;
                }
            }
            startsVec[x] = spanStart;
        }
        return Status::OK();
    }

    template <typename X, typename Z>
    static void gatherRows(int const spanSize, int const* starts, Z const* weights, X const* imagePtr, Nd4jLong const inputHeight, Nd4jLong const inputWidth, Nd4jLong const outputHeight,
                           Nd4jLong const outputWidth, Nd4jLong const channels, Z* outputPtr) {
        auto inRowSize = inputWidth * channels;
        auto outRowSize = outputWidth * channels;

        auto addScaledVector = [](const X* inVector, int vectorLen, Z weight, Z* outVector) {
            Z* outVecEnd = outVector + vectorLen;
            for (; outVector != outVecEnd; ++outVector, ++inVector) {
                *outVector += weight * static_cast<Z>(*inVector);
            }
        };

        for (int y = 0; y < outputHeight; ++y) {
            Z* outRowData = outputPtr + outRowSize * y;
            memset(outRowData, '\0', outRowSize * sizeof(Z));//            std::fill(outRowData, outRowData + outRowSize, 0.f);
            int inRow = starts[y];
            auto inRowData = imagePtr + inRowSize * inRow;
            auto weightsStart = weights + y * spanSize;
            auto realSpanSize = math::nd4j_min(starts[y] + spanSize, static_cast<int>(inputHeight)) - starts[y];
            auto weightsEnd = weightsStart + realSpanSize;
            for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
                addScaledVector(inRowData, inRowSize, *weightPtr, outRowData);
                inRowData += inRowSize;
            }
        }
    }

    template <typename Z>
    static void gatherColumns(int const spanSize, int const* starts, Z const* weights, Z const* imagesPtr, Nd4jLong const inputHeight, Nd4jLong const inputWidth, Nd4jLong const outputHeight, Nd4jLong const outputWidth, Nd4jLong channels, Z* outputPtr) {
        auto inRowSize = inputWidth * channels;
        auto outRowSize = outputWidth * channels;

        for (auto y = 0LL; y < outputHeight; ++y) {
            auto inputRowStart = imagesPtr + inRowSize * y;
            auto outPixels = outputPtr + outRowSize * y;
            for (auto x = 0LL; x < outputWidth; ++x, outPixels += channels) {
                auto inPixels = inputRowStart + starts[x] * channels;
                auto weightsStart = weights + x * spanSize;
                auto realSpanSize = math::nd4j_min(starts[x] + spanSize, static_cast<int>(inputWidth)) - starts[x];
                auto weightsEnd = weightsStart + realSpanSize;
                for (int c = 0; c < channels; ++c) {
                    outPixels[c] = 0.0f;
                }
                for (auto weightPtr = weightsStart; weightPtr != weightsEnd; ++weightPtr) {
                    Z w = *weightPtr;
                    for (int c = 0; c < channels; ++c) {
                        outPixels[c] += w * static_cast<Z>(inPixels[c]);
                    }
                    inPixels += channels;
                }
            }
        }
    }

    template <typename X, typename Z>
    static void gatherSpans(int const rowSpanSize, NDArray const& rowStarts, NDArray const& rowWeights, int const colSpanSize, NDArray const& columnStarts, NDArray const& columnWeights, NDArray const* images, NDArray& intermediate, NDArray* output) {
        auto batchSize = images->sizeAt(0);
        auto inputHeight = images->sizeAt(1);
        auto inputWidth = images->sizeAt(2);
        auto channels = images->sizeAt(3);

        auto outputHeight = output->sizeAt(1);
        auto outputWidth = output->sizeAt(2);

        auto inputPixPerBatch = inputWidth * inputHeight * channels;
        auto intermediatePixPerBatch = inputWidth * outputHeight * channels;
        auto outputPixPerBatch = outputWidth * outputHeight * channels;
        Z* intermediatePtr = intermediate.bufferAsT<Z>();

        const X* imagePtr = images->bufferAsT<X>();
        Z* outPtr = output->bufferAsT<Z>();
        for (int b = 0; b < batchSize; ++b, imagePtr += inputPixPerBatch,
                                            intermediatePtr += intermediatePixPerBatch,
                                            outPtr += outputPixPerBatch) {
            gatherRows<X,Z>(rowSpanSize, rowStarts.bufferAsT<int>(), rowWeights.bufferAsT<Z>(),
                            imagePtr, inputHeight, inputWidth, outputHeight,
                            inputWidth, channels, intermediatePtr);
            gatherColumns<Z>(colSpanSize, columnStarts.bufferAsT<int>(), columnWeights.bufferAsT<Z>(),
                               intermediatePtr, outputHeight, inputWidth, outputHeight, outputWidth, channels, outPtr);
        }
    }

    template <typename X, typename Z>
    static int resizeKernel(IKernelFunc* transformationKernel, NDArray const* input, Nd4jLong outWidth, Nd4jLong outHeight, bool antialias, NDArray* output) {
        Nd4jLong const batchSize = input->sizeAt(0);
        Nd4jLong const inputHeight = input->sizeAt(1);
        Nd4jLong const inputWidth = input->sizeAt(2);
        Nd4jLong const channels = input->sizeAt(3);

        Z rowScale = Z(outHeight) / Z(inputHeight);
        Z columnScale = Z(outWidth) / Z(inputWidth);

        // Return if the output is empty.
        if (output->lengthOf() == 0) return Status::OK();

        Spans colSpans;

        auto res = computeSpans(transformationKernel, outWidth, inputWidth, columnScale, 0.f, antialias, colSpans);
        if (res != Status::OK()) return res;
        Spans rowSpans;
        res = computeSpans(transformationKernel, outHeight, inputHeight, rowScale, 0.f, antialias, rowSpans);

        NDArray intermediate = NDArrayFactory::create<Z>('c', {batchSize, outHeight, inputWidth, channels});

        //const functor::Spans& const_row_spans = row_spans;
        //typename TTypes<int32, 1>::ConstTensor row_starts(
        //const_row_spans.starts.tensor<int32, 1>());
        auto& rowStarts = rowSpans._starts; // shape {outWidth}
        auto& rowWeights = rowSpans._weights; // shape {outWidth, numSpans}
        auto& columnStarts = colSpans._starts; // shape {outHeights}
        auto& columnWeights = colSpans._weights; // shape {outHeights, numSpans}

        gatherSpans<X, Z>(rowSpans._spanSize, rowStarts, rowWeights, colSpans._spanSize, columnStarts, columnWeights, input, intermediate, output);
        return res;
    }

    static int resizeBilinear(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        auto kernel = std::unique_ptr<IKernelFunc>(new TriangleKernelFunc());
        BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                              (kernel.get(), image, (Nd4jLong) width, (Nd4jLong) height, antialias, output),
                              NUMERIC_TYPES, FLOAT_TYPES_1);
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeBilinear: Unknown error occured.");
    }

    static int resizeBicubic(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        if (antialias) {
            auto kernel = std::unique_ptr<IKernelFunc>(new KeysCubicKernelFunc());
            BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel,
                                  (kernel.get(), image, (Nd4jLong) width, (Nd4jLong) height, antialias, output),
                                  NUMERIC_TYPES, FLOAT_TYPES_1);
        }
        else {
            return resizeBicubicFunctorA(context, image, width, height, false, true, output);
        }
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeBicubic: Unknown error occured.");
    }

    static int resizeNeighbor(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        return resizeNeighborFunctor(context, image, width, height, false, true, output);
    }

    static int resizeArea(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        return resizeAreaFunctor(context, image, width, height, false, output);
    }

    static int resizeLanczos3(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        auto kernel = std::unique_ptr<IKernelFunc>(new LanczosKernelFunc(3.f));
        BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel, (kernel.get(), image, (Nd4jLong)width, (Nd4jLong)height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos3: Unknown error occured.");
    }

    static int resizeLanczos5(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        auto kernel = std::unique_ptr<IKernelFunc>(new LanczosKernelFunc(5.f));
        BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel, (kernel.get(), image, (Nd4jLong)width, (Nd4jLong)height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeLanczos5: Unknown error occured.");
    }

    static int resizeGaussian(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        auto kernel = std::unique_ptr<IKernelFunc>(new GaussianKernelFunc());
        BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel, (kernel.get(), image, (Nd4jLong)width, (Nd4jLong)height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeGaussian: Unknown error occured.");
    }

    static int resizeMitchellcubic(sd::LaunchContext * context, NDArray const* image, int const width, int const height, bool const antialias, NDArray* output) {
        auto kernel = std::unique_ptr<IKernelFunc>(new MitchellCubicKernelFunc());
        BUILD_DOUBLE_SELECTOR(image->dataType(), output->dataType(), return resizeKernel, (kernel.get(), image, (Nd4jLong)width, (Nd4jLong)height, antialias, output), NUMERIC_TYPES, FLOAT_TYPES_1);
        return Status::CODE(ND4J_STATUS_VALIDATION, "helpers::resizeMitchelcubic: Unknown error occured.");
    }

// ------------------------------------------------------------------------------------------------------------------ //
    int resizeImagesFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                      ImageResizeMethods method, bool alignCorners, NDArray* output) {
        switch (method) {
            case kResizeBilinear:
                return resizeBilinearFunctor(context, image, width, height, alignCorners, false, output);
            case kResizeNearest:
                return resizeNeighborFunctor(context, image, width, height, alignCorners, false, output);
            case kResizeBicubic:
                return resizeBicubicFunctor(context, image, width, height, alignCorners, false, output);
            case kResizeArea:
                return resizeAreaFunctor(context, image, width, height, alignCorners, output);
        }
        nd4j_printf("helper::resizeImagesFunctor: Wrong resize method %i\n", (int)method);
        return Status::CODE(ND4J_STATUS_BAD_INPUT, "helper::resizeImagesFunctor: Wrong resize method");
    }
// ------------------------------------------------------------------------------------------------------------------ //
    int resizeFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                      ImageResizeMethods method, bool antialias, NDArray* output) {
        switch (method) {
            case kResizeBilinear:     return resizeBilinear(context, image, width, height, antialias, output);
            case kResizeNearest:      return resizeNeighbor(context, image, width, height,  antialias, output);
            case kResizeBicubic:      return resizeBicubic(context, image, width, height,  antialias, output);
            case kResizeArea:         return resizeArea(context, image, width, height, antialias, output);
            case kResizeLanczos3:     return resizeLanczos3(context, image, width, height, antialias, output);
            case kResizeLanczos5:     return resizeLanczos5(context, image, width, height, antialias, output);
            case kResizeGaussian:     return resizeGaussian(context, image, width, height, antialias, output);
            case kResizeMitchellcubic: return resizeMitchellcubic(context, image, width, height, antialias, output);
        }
        nd4j_printf("helper::resizeFunctor: Wrong resize method %i\n", (int)method);
        return Status::CODE(ND4J_STATUS_BAD_INPUT, "helper::resizeFunctor: Wrong resize method");
    }


}
}
}