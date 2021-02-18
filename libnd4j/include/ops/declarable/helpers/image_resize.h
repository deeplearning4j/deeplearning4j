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

//
//  @author sgazeos@gmail.com
//
#ifndef __IMAGE_RESIZE_HELPERS__
#define __IMAGE_RESIZE_HELPERS__
#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

    enum ImageResizeMethods {
        kResizeBilinear = 0, // as java require
        kResizeNearest,
        kResizeBicubic,
        kResizeArea,
        kResizeGaussian,
        kResizeLanczos3,
        kResizeLanczos5,
        kResizeMitchellcubic,
        kResizeFirst = kResizeBilinear,
        kResizeLast = kResizeMitchellcubic,
        kResizeOldLast = kResizeArea
    };

    int resizeBilinearFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
            bool const alignCorners, bool const halfPixelCenter, NDArray* output);
    int resizeNeighborFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
            bool const alignCorners, bool const halfPixelCenter, NDArray* output);
    int resizeBicubicFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                      bool preserveAspectRatio, bool antialias, NDArray* output);
    int resizeBicubicFunctorA(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                             bool const alignCorners, bool const halfPixelAlign, NDArray* output);
    int resizeAreaFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                             bool const alignCorners, NDArray* output);

    int resizeFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
            ImageResizeMethods method, bool antialias, NDArray* output);

    int resizeImagesFunctor(sd::LaunchContext * context, NDArray const* image, int const width, int const height,
                      ImageResizeMethods method, bool alignCorners, NDArray* output);
}
}
}
#endif
