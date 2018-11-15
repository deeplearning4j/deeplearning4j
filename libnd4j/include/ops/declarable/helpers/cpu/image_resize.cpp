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

    static int gcd(int one, int two) {
        // modified Euclidian algorithm
        if (one == two) return one;
        if (one > two) {
            if (one % two == 0) return two;
            return gcd(one - two, two);
        }
        if (two % one == 0) return one;
        return gcd(one, two - one);
    }

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
                                              BilinearInterpolationData* interpolationData) {
        interpolationData[outSize].bottomIndex = 0;
        interpolationData[outSize].topIndex = 0;
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
    inline double computeBilinear(double topLeft, double topRight,
                              double bottomLeft, double bottomRight,
                              double xVal, double yVal) {
        double top = topLeft + (topRight - topLeft) * xVal;
        double bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
        return top + (bottom - top) * yVal;
    }

    static void resizeImage(NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray* output);

    template <typename T>
    static void resizeImage_(NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray* output) {

        Nd4jLong inRowSize = inWidth * channels;
        Nd4jLong inBatchNumValues = inHeight * inRowSize;
        Nd4jLong outRowSize = outWidth * channels;

        T const* input_b_ptr = reinterpret_cast<T const*>(images->getBuffer()); // this works only with 'c' direction
        BilinearInterpolationData const* xs_ = xs.data();

        T* output_y_ptr = reinterpret_cast<T*>(output->buffer());
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

    template void resizeImage(NDArray<float> const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                            Nd4jLong outWidth, Nd4jLong channels,
                            std::vector<BilinearInterpolationData> const& xs,
                            std::vector<BilinearInterpolationData> const& ys,
                            NDArray<float>* output);

    template void resizeImage(NDArray<float16> const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray<float16>* output);

    template void resizeImage(NDArray<double> const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray<double>* output);

    template <typename T>
    static int resizeBilinearFunctor_(NDArray const* images, int width, int height, bool center, NDArray* output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight  = images->sizeAt(1);
        const Nd4jLong inWidth   = images->sizeAt(2);
        const Nd4jLong channels  = images->sizeAt(3);

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
            nd4j_printf("image.resize_bilinear: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        double heightScale = center? (inHeight - 1.) / double(outHeight - 1.0): (inHeight / double(outHeight));
        double widthScale = center? (inWidth - 1.) / double(outWidth - 1.0): (inWidth / double(outWidth));

        std::vector<BilinearInterpolationData> ys(outHeight + 1);
        std::vector<BilinearInterpolationData> xs(outWidth + 1);

        // Compute the cached interpolation weights on the x and y dimensions.
        computeInterpolationWeights(outHeight, inHeight, heightScale,
                                      ys.data());
        computeInterpolationWeights(outWidth, inWidth, widthScale, xs.data());

        // Scale x interpolation weights to avoid a multiplication during iteration.
        for (int i = 0; i < xs.size(); ++i) {
            xs[i].bottomIndex *= channels;
            xs[i].topIndex    *= channels;
        }

        resizeImage(images, batchSize, inHeight, inWidth, outHeight,  outWidth, channels, xs, ys, output);
        return ND4J_STATUS_OK;
    }
    template int resizeBilinearFunctor(NDArray<float> const* image, int width, int height, bool center, NDArray<float>* output);
    template int resizeBilinearFunctor(NDArray<float16> const* image, int width, int height, bool center, NDArray<float16>* output);
    template int resizeBilinearFunctor(NDArray<double> const* image, int width, int height, bool center, NDArray<double>* output);

    template <typename T>
    int resizeNeighborFunctor(NDArray<T> const* images, int width, int height, bool center, NDArray<T>* output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight  = images->sizeAt(1);
        const Nd4jLong inWidth   = images->sizeAt(2);
        const Nd4jLong channels  = images->sizeAt(3);

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
        double heightScale = center? (inHeight - 1.) / double(outHeight - 1.0): (inHeight / double(outHeight));
        double widthScale = center? (inWidth - 1.) / double(outWidth - 1.0): (inWidth / double(outWidth));

        for (int b = 0; b < batchSize; ++b) {
          for (int y = 0; y < outHeight; ++y) {
            Nd4jLong inY = std::min((center) ? static_cast<Nd4jLong>(roundf(y * heightScale)): static_cast<Nd4jLong>(floorf(y * heightScale)), inHeight - 1);
            for (int x = 0; x < outWidth; ++x) {
              Nd4jLong inX = std::min((center) ? static_cast<Nd4jLong>(roundf(x * widthScale)) : static_cast<Nd4jLong>(floorf(x * widthScale)), inWidth - 1);
              for (Nd4jLong e = 0; e < channels; e++)
                  output->p(b, y, x, e, images->e<T>(b, inY, inX, e));
//              std::copy_n(&input(b, in_y, in_x, 0), channels, &output(b, y, x, 0));
            }
          }
        }

        return ND4J_STATUS_OK;
    }
    void resizeImage(NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), resizeImage_, (images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs, ys, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void resizeImage_,(NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray* output), LIBND4J_TYPES);

    int resizeBilinearFunctor(NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeBilinearFunctor_, (images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeBilinearFunctor_, (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

    int resizeNeighborFunctor(NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_, (images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeNeighborFunctor_, (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

}
}
}
