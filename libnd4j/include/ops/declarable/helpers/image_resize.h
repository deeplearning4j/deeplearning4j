/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
#ifndef __IMAGE_RESIZE_HELPERS__
#define __IMAGE_RESIZE_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    enum ImageResizeMethods {
        kResizeBilinear = 1,
        kResizeBicubic,
        kResizeNearest,
        kResizeGaussian,
        kResizeLanczos5,
        kResizeMitchelcubic,
        kResizeArea
    };

    int resizeBilinearFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height, bool center,
            NDArray* output);
    int resizeNeighborFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height, bool center,
            NDArray* output);
    int resizeBicubicFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                      bool preserveAspectRatio, bool antialias, NDArray* output);
    int resizeBicubicFunctorA(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                             bool const alignCorners, bool const halfPixelAlign, NDArray* output);
    int resizeFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
            ImageResizeMethods method, bool preserveAspectRatio, bool antialias, NDArray* output);

    void cropAndResizeFunctor(nd4j::LaunchContext * context, NDArray const* images, NDArray const* boxes,
            NDArray const* indices, NDArray const* cropSize, int method, double extrapolationVal, NDArray* crops);
}
}
}
#endif
