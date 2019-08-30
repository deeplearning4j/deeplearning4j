/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
namespace ops {
namespace helpers {

    struct BilinearInterpolationData {
        Nd4jLong bottomIndex;  // Lower source index used in the interpolation
        Nd4jLong topIndex;  // Upper source index used in the interpolation
        // 1-D linear iterpolation scale (see:
        // https://en.wikipedia.org/wiki/Bilinear_interpolation)
        double interpolarValue;
    };

    inline void computeInterpolationWeights(Nd4jLong outSize,
                                            Nd4jLong inSize,
                                            double scale,
                                            BilinearInterpolationData *interpolationData) {
        interpolationData[outSize].bottomIndex = 0;
        interpolationData[outSize].topIndex = 0;

        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong i = outSize - 1; i >= 0; --i) {
            double in = i * scale;
            interpolationData[i].bottomIndex = static_cast<Nd4jLong>(in);
            interpolationData[i].topIndex = nd4j::math::nd4j_min(interpolationData[i].bottomIndex + 1, inSize - 1);
            interpolationData[i].interpolarValue = in - interpolationData[i].bottomIndex;
        }
    }

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
    static void
    resizeImage(NDArray const *images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                Nd4jLong outWidth, Nd4jLong channels,
                std::vector<BilinearInterpolationData> const &xs,
                std::vector<BilinearInterpolationData> const &ys,
                NDArray *output);

    template<typename T>
    static void
    resizeImage_(NDArray const *images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                 Nd4jLong outWidth, Nd4jLong channels,
                 std::vector<BilinearInterpolationData> const &xs,
                 std::vector<BilinearInterpolationData> const &ys,
                 NDArray *output) {

        Nd4jLong inRowSize = inWidth * channels;
        Nd4jLong inBatchNumValues = inHeight * inRowSize;
        Nd4jLong outRowSize = outWidth * channels;

        T const *input_b_ptr = reinterpret_cast<T const *>(images->getBuffer()); // this works only with 'c' direction
        BilinearInterpolationData const *xs_ = xs.data();

        T *output_y_ptr = reinterpret_cast<T *>(output->buffer());
        auto computeBilinear = [](double topLeft, double topRight,
                                      double bottomLeft, double bottomRight,
                                      double xVal, double yVal) {
            double top = topLeft + (topRight - topLeft) * xVal;
            double bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
            return top + (bottom - top) * yVal;
        };

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong b = 0; b < batchSize; ++b) {
            for (Nd4jLong y = 0; y < outHeight; ++y) {
                const T *ys_input_lower_ptr = input_b_ptr + ys[y].bottomIndex * inRowSize;
                const T *ys_input_upper_ptr = input_b_ptr + ys[y].topIndex * inRowSize;
                double yVal = ys[y].interpolarValue;
                for (Nd4jLong x = 0; x < outWidth; ++x) {
                    auto xsBottom = xs_[x].bottomIndex;
                    auto xsTop = xs_[x].topIndex;
                    auto xVal = xs_[x].interpolarValue;
                    for (int c = 0; c < channels; ++c) {
                        double topLeft(ys_input_lower_ptr[xsBottom + c]);
                        double topRight(ys_input_lower_ptr[xsTop + c]);
                        double bottomLeft(ys_input_upper_ptr[xsBottom + c]);
                        double bottomRight(ys_input_upper_ptr[xsTop + c]);
                        output_y_ptr[x * channels + c] =
                                computeBilinear(topLeft, topRight, bottomLeft, bottomRight,
                                                xVal, yVal);
                    }
                }
                output_y_ptr += outRowSize;
            }
            input_b_ptr += inBatchNumValues;
        }
    }

    template<typename T>
    static int resizeBilinearFunctor_(NDArray const *images, int width, int height, bool center, NDArray *output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight = images->sizeAt(1);
        const Nd4jLong inWidth = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong outHeight = output->sizeAt(1);
        const Nd4jLong outWidth = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (outHeight == inHeight && outWidth == inWidth) {
            output->assign(images);
            return ND4J_STATUS_OK;
        }

        // Special case for TF compatibility
        if((center && inHeight < 2) || (center && inWidth < 2)){
            center = false;
        }

        if ((center && inHeight < 2) || (inHeight < 1) || (outHeight < 1) || (center && outHeight < 2) ||
            (center && inWidth < 2) || (inWidth < 1) || (outWidth < 1) || (center && outWidth < 2)) {
            // wrong input data
            nd4j_printf("image.resize_bilinear: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        float heightScale = center ? (inHeight - 1.f) / double(outHeight - 1.f) : (inHeight / float(outHeight));
        float widthScale = center ? (inWidth - 1.f) / double(outWidth - 1.f) : (inWidth / float(outWidth));

        std::vector<BilinearInterpolationData> ys(outHeight + 1);
        std::vector<BilinearInterpolationData> xs(outWidth + 1);

        // Compute the cached interpolation weights on the x and y dimensions.
        computeInterpolationWeights(outHeight, inHeight, heightScale,
                                    ys.data());
        computeInterpolationWeights(outWidth, inWidth, widthScale, xs.data());

        int xsSize = xs.size();
        // Scale x interpolation weights to avoid a multiplication during iteration.
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i = 0; i < xsSize; ++i) {
            xs[i].bottomIndex *= channels;
            xs[i].topIndex *= channels;
        }

        resizeImage(images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output);
        return ND4J_STATUS_OK;
    }

    template<typename T>
    int resizeNeighborFunctor_(NDArray const *images, int width, int height, bool center, NDArray *output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight = images->sizeAt(1);
        const Nd4jLong inWidth = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong outHeight = output->sizeAt(1);
        const Nd4jLong outWidth = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (outHeight == inHeight && outWidth == inWidth) {
            output->assign(images);
            return ND4J_STATUS_OK;
        }

        if ((center && inHeight < 2) || (inHeight < 1) || (outHeight < 1) || (center && outHeight < 2) ||
            (center && inWidth < 2) || (inWidth < 1) || (outWidth < 1) || (center && outWidth < 2)) {
            // wrong input data
            nd4j_printf("image.resize_nearest_neighbor: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        double heightScale = center ? (inHeight - 1.) / double(outHeight - 1.0) : (inHeight / double(outHeight));
        double widthScale = center ? (inWidth - 1.) / double(outWidth - 1.0) : (inWidth / double(outWidth));

        PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int y = 0; y < outHeight; ++y) {
                Nd4jLong inY = nd4j::math::nd4j_min(
                        (center) ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(y * heightScale)) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(
                                y * heightScale)), inHeight - 1);
                for (int x = 0; x < outWidth; ++x) {
                    Nd4jLong inX = nd4j::math::nd4j_min(
                            (center) ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(x * widthScale)) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(
                                    x * widthScale)), inWidth - 1);
                    for (Nd4jLong e = 0; e < channels; e++)
                        output->p(b, y, x, e, images->e<T>(b, inY, inX, e));
//              std::copy_n(&input(b, in_y, in_x, 0), channels, &output(b, y, x, 0));
                }
            }
        }

        return ND4J_STATUS_OK;
    }

    void resizeImage(NDArray const *images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const &xs,
                     std::vector<BilinearInterpolationData> const &ys,
                     NDArray *output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), resizeImage_,
                              (images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output),
                              LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void resizeImage_,
                          (NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                                  Nd4jLong outWidth, Nd4jLong channels,
                                  std::vector<BilinearInterpolationData> const& xs,
                                  std::vector<BilinearInterpolationData> const& ys,
                                  NDArray* output), LIBND4J_TYPES);

    int resizeBilinearFunctor(nd4j::LaunchContext * context, NDArray const *images, int width, int height, bool center, NDArray *output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeBilinearFunctor_,
                              (images, width, height, center, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int resizeBilinearFunctor_,
                          (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

    int resizeNeighborFunctor(nd4j::LaunchContext * context, NDArray const *images, int width, int height, bool center, NDArray *output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_,
                              (images, width, height, center, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int resizeNeighborFunctor_,
                          (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

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

        for (int b = 0; b < numBoxes; ++b) {
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

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int y = 0; y < cropHeight; ++y) {
                const float inY = (cropHeight > 1)
                                  ? y1 * (imageHeight - 1) + y * heightScale
                                  : 0.5 * (y1 + y2) * (imageHeight - 1);
                if (inY < 0 || inY > imageHeight - 1) {
                    for (int x = 0; x < cropWidth; ++x) {
                        for (int d = 0; d < depth; ++d) {
                            crops->p(b, y, x, d, extrapolationVal);
                        }
                    }
                    continue;
                }
                if (method == 0 /* bilinear */) {
                    const int topYIndex = nd4j::math::p_floor(inY);
                    const int bottomYIndex = nd4j::math::p_ceil(inY);
                    const float y_lerp = inY - topYIndex;

                    for (int x = 0; x < cropWidth; ++x) {
                        const float in_x = (cropWidth > 1)
                                           ? x1 * (imageWidth - 1) + x * widthScale
                                           : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (in_x < 0 || in_x > imageWidth - 1) {
                            for (int d = 0; d < depth; ++d) {
                                crops->p(b, y, x, d, extrapolationVal);
                            }
                            continue;
                        }
                        int left_x_index = math::p_floor(in_x);
                        int right_x_index = math::p_ceil(in_x);
                        T x_lerp = in_x - left_x_index;

                        for (int d = 0; d < depth; ++d) {
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
                    for (int x = 0; x < cropWidth; ++x) {
                        const float inX = (cropWidth > 1)
                                          ? x1 * (imageWidth - 1) + x * widthScale
                                          : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (inX < 0 || inX > imageWidth - 1) {
                            for (int d = 0; d < depth; ++d) {
                                crops->p(b, y, x, d, extrapolationVal);
                            }
                            continue;
                        }
                        const int closestXIndex = roundf(inX);
                        const int closestYIndex = roundf(inY);
                        for (int d = 0; d < depth; ++d) {
                            crops->p(b, y, x, d, (F)images->e<T>(bIn, closestYIndex, closestXIndex, d));
                        }
                    }
                }
            }
        }
    }


    void
    cropAndResizeFunctor(nd4j::LaunchContext * context, NDArray const *images, NDArray const *boxes, NDArray const *indices, NDArray const *cropSize,
                         int method, double extrapolationVal, NDArray *crops) {
        BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                              (images, boxes, indices, cropSize, method, extrapolationVal, crops), NUMERIC_TYPES, FLOAT_TYPES, INTEGER_TYPES);
    }
}
}
}
