/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//

#include <execution/Threads.h>
#include <helpers/LoopKind.h>
#include <ops/declarable/helpers/affine_grid.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void affineGrid2D_(LaunchContext* context, NDArray* theta, NDArray* size,
                           bool alignCorners, NDArray* output) {
    const auto batchSize = theta->sizeAt(0);
    const auto H = size->e<sd::LongType>(2);
    const auto W = size->e<sd::LongType>(3);

    auto thetaPtr = theta->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto h = start_y; h < stop_y; h++) {
                for (sd::LongType w = 0; w < W; w++) {
                    // Compute normalized coordinates
                    T y, x;
                    if (alignCorners) {
                        y = H > 1 ? static_cast<T>(2.0 * h / (H - 1) - 1.0) : static_cast<T>(0);
                        x = W > 1 ? static_cast<T>(2.0 * w / (W - 1) - 1.0) : static_cast<T>(0);
                    } else {
                        y = static_cast<T>((2.0 * h + 1.0) / H - 1.0);
                        x = static_cast<T>((2.0 * w + 1.0) / W - 1.0);
                    }

                    // Apply affine transformation: [x', y'] = theta @ [x, y, 1]^T
                    // theta is [N, 2, 3], we access theta[b, :, :]
                    auto thetaOffset = b * 6;  // 2 * 3 = 6 elements per batch
                    T theta00 = thetaPtr[thetaOffset + 0];
                    T theta01 = thetaPtr[thetaOffset + 1];
                    T theta02 = thetaPtr[thetaOffset + 2];
                    T theta10 = thetaPtr[thetaOffset + 3];
                    T theta11 = thetaPtr[thetaOffset + 4];
                    T theta12 = thetaPtr[thetaOffset + 5];

                    T outX = theta00 * x + theta01 * y + theta02;
                    T outY = theta10 * x + theta11 * y + theta12;

                    // Output: [N, H, W, 2]
                    auto outOffset = b * H * W * 2 + h * W * 2 + w * 2;
                    outputPtr[outOffset + 0] = outX;
                    outputPtr[outOffset + 1] = outY;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, H, 1);
}

template <typename T>
static void affineGrid3D_(LaunchContext* context, NDArray* theta, NDArray* size,
                           bool alignCorners, NDArray* output) {
    const auto batchSize = theta->sizeAt(0);
    const auto D = size->e<sd::LongType>(2);
    const auto H = size->e<sd::LongType>(3);
    const auto W = size->e<sd::LongType>(4);

    auto thetaPtr = theta->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_3D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto d = start_y; d < stop_y; d++) {
                for (auto h = start_z; h < stop_z; h++) {
                    for (sd::LongType w = 0; w < W; w++) {
                        T z, y, x;
                        if (alignCorners) {
                            z = D > 1 ? static_cast<T>(2.0 * d / (D - 1) - 1.0) : static_cast<T>(0);
                            y = H > 1 ? static_cast<T>(2.0 * h / (H - 1) - 1.0) : static_cast<T>(0);
                            x = W > 1 ? static_cast<T>(2.0 * w / (W - 1) - 1.0) : static_cast<T>(0);
                        } else {
                            z = static_cast<T>((2.0 * d + 1.0) / D - 1.0);
                            y = static_cast<T>((2.0 * h + 1.0) / H - 1.0);
                            x = static_cast<T>((2.0 * w + 1.0) / W - 1.0);
                        }

                        // theta is [N, 3, 4]
                        auto thetaOffset = b * 12;
                        T outX = thetaPtr[thetaOffset + 0] * x + thetaPtr[thetaOffset + 1] * y +
                                 thetaPtr[thetaOffset + 2] * z + thetaPtr[thetaOffset + 3];
                        T outY = thetaPtr[thetaOffset + 4] * x + thetaPtr[thetaOffset + 5] * y +
                                 thetaPtr[thetaOffset + 6] * z + thetaPtr[thetaOffset + 7];
                        T outZ = thetaPtr[thetaOffset + 8] * x + thetaPtr[thetaOffset + 9] * y +
                                 thetaPtr[thetaOffset + 10] * z + thetaPtr[thetaOffset + 11];

                        // Output: [N, D, H, W, 3]
                        auto outOffset = b * D * H * W * 3 + d * H * W * 3 + h * W * 3 + w * 3;
                        outputPtr[outOffset + 0] = outX;
                        outputPtr[outOffset + 1] = outY;
                        outputPtr[outOffset + 2] = outZ;
                    }
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, D, 1, 0, H, 1);
}

void affineGrid(LaunchContext* context, NDArray* theta, NDArray* size,
                bool alignCorners, NDArray* output) {
    bool is2D = theta->sizeAt(1) == 2;

    if (is2D) {
        BUILD_SINGLE_SELECTOR(theta->dataType(), affineGrid2D_, (context, theta, size, alignCorners, output),
                              SD_FLOAT_TYPES);
    } else {
        BUILD_SINGLE_SELECTOR(theta->dataType(), affineGrid3D_, (context, theta, size, alignCorners, output),
                              SD_FLOAT_TYPES);
    }
}

template <typename T>
static void gridSample2D_(LaunchContext* context, NDArray* input, NDArray* grid,
                           int mode, int paddingMode, bool alignCorners, NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto channels = input->sizeAt(1);
    const auto inH = input->sizeAt(2);
    const auto inW = input->sizeAt(3);
    const auto outH = grid->sizeAt(1);
    const auto outW = grid->sizeAt(2);

    auto inputPtr = input->bufferAsT<T>();
    auto gridPtr = grid->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto h = start_y; h < stop_y; h++) {
                for (sd::LongType w = 0; w < outW; w++) {
                    // Get grid coordinates
                    auto gridOffset = b * outH * outW * 2 + h * outW * 2 + w * 2;
                    T gx = gridPtr[gridOffset + 0];
                    T gy = gridPtr[gridOffset + 1];

                    // Denormalize coordinates
                    T ix, iy;
                    if (alignCorners) {
                        ix = (gx + 1) * (inW - 1) / 2;
                        iy = (gy + 1) * (inH - 1) / 2;
                    } else {
                        ix = ((gx + 1) * inW - 1) / 2;
                        iy = ((gy + 1) * inH - 1) / 2;
                    }

                    for (sd::LongType c = 0; c < channels; c++) {
                        T value = static_cast<T>(0);

                        if (mode == 0) {  // Bilinear
                            sd::LongType ix0 = static_cast<sd::LongType>(std::floor(ix));
                            sd::LongType iy0 = static_cast<sd::LongType>(std::floor(iy));
                            sd::LongType ix1 = ix0 + 1;
                            sd::LongType iy1 = iy0 + 1;

                            T wx1 = ix - ix0;
                            T wy1 = iy - iy0;
                            T wx0 = 1 - wx1;
                            T wy0 = 1 - wy1;

                            auto getVal = [&](sd::LongType py, sd::LongType px) -> T {
                                if (paddingMode == 0) {  // zeros
                                    if (px < 0 || px >= inW || py < 0 || py >= inH) return static_cast<T>(0);
                                } else if (paddingMode == 1) {  // border
                                    px = std::max(static_cast<sd::LongType>(0), std::min(px, inW - 1));
                                    py = std::max(static_cast<sd::LongType>(0), std::min(py, inH - 1));
                                } else {  // reflection
                                    if (px < 0) px = -px;
                                    if (py < 0) py = -py;
                                    px = px % (2 * inW);
                                    py = py % (2 * inH);
                                    if (px >= inW) px = 2 * inW - px - 1;
                                    if (py >= inH) py = 2 * inH - py - 1;
                                }
                                return inputPtr[b * channels * inH * inW + c * inH * inW + py * inW + px];
                            };

                            value = wx0 * wy0 * getVal(iy0, ix0) +
                                    wx1 * wy0 * getVal(iy0, ix1) +
                                    wx0 * wy1 * getVal(iy1, ix0) +
                                    wx1 * wy1 * getVal(iy1, ix1);
                        } else if (mode == 1) {  // Nearest
                            sd::LongType nearestX = static_cast<sd::LongType>(std::round(ix));
                            sd::LongType nearestY = static_cast<sd::LongType>(std::round(iy));

                            if (paddingMode == 0) {
                                if (nearestX >= 0 && nearestX < inW && nearestY >= 0 && nearestY < inH) {
                                    value = inputPtr[b * channels * inH * inW + c * inH * inW +
                                                     nearestY * inW + nearestX];
                                }
                            } else {
                                nearestX = std::max(static_cast<sd::LongType>(0), std::min(nearestX, inW - 1));
                                nearestY = std::max(static_cast<sd::LongType>(0), std::min(nearestY, inH - 1));
                                value = inputPtr[b * channels * inH * inW + c * inH * inW +
                                                 nearestY * inW + nearestX];
                            }
                        }

                        outputPtr[b * channels * outH * outW + c * outH * outW + h * outW + w] = value;
                    }
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, outH, 1);
}

void gridSample(LaunchContext* context, NDArray* input, NDArray* grid,
                int mode, int paddingMode, bool alignCorners, NDArray* output) {
    bool is2D = input->rankOf() == 4;

    if (is2D) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gridSample2D_,
                              (context, input, grid, mode, paddingMode, alignCorners, output),
                              SD_FLOAT_TYPES);
    }
    // 3D case would be similar but with additional depth dimension
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
