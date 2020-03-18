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

#include <ops/declarable/helpers/crop_and_resize.h>
#include <execution/Threads.h>

namespace sd {
    namespace ops {
        namespace helpers {
            template<typename T, typename F, typename I>
            void cropAndResizeFunctor_(NDArray const *images, NDArray const *boxes, NDArray const *indices, NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops) {
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
                        for (auto y = start; y < stop; y++) {
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
                                const int topYIndex = sd::math::p_floor(inY);
                                const int bottomYIndex = sd::math::p_ceil(inY);
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
        }
    }
}