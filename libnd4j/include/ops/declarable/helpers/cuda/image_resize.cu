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

    static _CUDA_HD int gcd(int one, int two) {
        // modified Euclidian algorithm
        if (one == two) return one;
        if (one > two) {
            if (one % two == 0) return two;
            return gcd(one - two, two);
        }
        if (two % one == 0) return one;
        return gcd(one, two - one);
    }

    struct _CUDA_HD BilinearInterpolationData {
        Nd4jLong bottomIndex;  // Lower source index used in the interpolation
        Nd4jLong topIndex;  // Upper source index used in the interpolation
        // 1-D linear iterpolation scale (see:
        // https://en.wikipedia.org/wiki/Bilinear_interpolation)
        double interpolarValue;
    };

    inline _CUDA_HD void computeInterpolationWeights(Nd4jLong outSize,
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
    inline _CUDA_HD double computeBilinear(double topLeft, double topRight,
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
        //
    }

    template <typename T>
    static int resizeBilinearFunctor_(NDArray const* images, int width, int height, bool center, NDArray* output) {
        return Status::OK();
    }

    template <typename T>
    int resizeNeighborFunctor_(NDArray const* images, int width, int height, bool center, NDArray* output) {
        return Status::OK();
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

    int resizeBilinearFunctor(nd4j::LaunchContext * context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeBilinearFunctor_, (images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeBilinearFunctor_, (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

    int resizeNeighborFunctor(nd4j::LaunchContext * context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_, (images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeNeighborFunctor_, (NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

    ///////
    void cropAndResizeFunctor(nd4j::LaunchContext * context, NDArray const *images, NDArray const *boxes, NDArray const *indices, NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops) {
        //
    }
}
}
}