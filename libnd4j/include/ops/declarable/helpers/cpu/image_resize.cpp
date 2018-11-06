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

    template <typename T>
    static void resizeImage(NDArray<T> const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     std::vector<BilinearInterpolationData> const& xs,
                     std::vector<BilinearInterpolationData> const& ys,
                     NDArray<T>* output) {

        Nd4jLong inRowSize = inWidth * channels;
        Nd4jLong inBatchNumValues = inHeight * inRowSize;
        Nd4jLong outRowSize = outWidth * channels;

        T const* input_b_ptr = images->getBuffer(); // this works only with 'c' direction
        BilinearInterpolationData const* xs_ = xs.data();

        T* output_y_ptr = output->buffer();
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
    int resizeBilinearFunctor(NDArray<T> const* images, int width, int height, bool center, NDArray<T>* output) {
        const Nd4jLong batch_size = images->sizeAt(0);
        const Nd4jLong in_height = images->sizeAt(1);
        const Nd4jLong in_width = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong out_height = output->sizeAt(1);
        const Nd4jLong out_width = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (out_height == in_height && out_width == in_width) {
            output->assign(images);
            return ND4J_STATUS_OK;
        }

        if ((center && in_height < 2) || (in_height < 1) || (out_height < 1) || (center && out_height < 2) ||
            (center && in_width < 2) || (in_width < 1) || (out_width < 1) || (center && out_width < 2)) {
            // wrong input data
            nd4j_printf("image.resize_bilinear: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        double height_scale = center? (in_height - 1.) / double(out_height - 1.0): (in_height / double(out_height));
        double width_scale = center? (in_width - 1.) / double(out_width - 1.0): (in_width / double(out_width));

        std::vector<BilinearInterpolationData> ys(out_height + 1);
        std::vector<BilinearInterpolationData> xs(out_width + 1);

        // Compute the cached interpolation weights on the x and y dimensions.
        computeInterpolationWeights(out_height, in_height, height_scale,
                                      ys.data());
        computeInterpolationWeights(out_width, in_width, width_scale, xs.data());

        // Scale x interpolation weights to avoid a multiplication during iteration.
        for (int i = 0; i < xs.size(); ++i) {
            xs[i].bottomIndex *= channels;
            xs[i].topIndex *= channels;
        }

        resizeImage<T>(images, batch_size, in_height, in_width, out_height,
                        out_width, channels, xs, ys, output);
        return ND4J_STATUS_OK;
    }
    template int resizeBilinearFunctor(NDArray<float> const* image, int width, int height, bool center, NDArray<float>* output);
    template int resizeBilinearFunctor(NDArray<float16> const* image, int width, int height, bool center, NDArray<float16>* output);
    template int resizeBilinearFunctor(NDArray<double> const* image, int width, int height, bool center, NDArray<double>* output);

}
}
}