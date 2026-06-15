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

#include <ops/declarable/helpers/affine_grid.h>
#include <helpers/PointersManager.h>
#include <cuda_runtime.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL void affineGrid2DKernel(const T* theta, const sd::LongType* size,
                                    bool alignCorners, T* output,
                                    sd::LongType batchSize, sd::LongType H, sd::LongType W) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalElements = batchSize * H * W;

    if (tid >= totalElements) return;

    const auto b = tid / (H * W);
    const auto hw = tid % (H * W);
    const auto h = hw / W;
    const auto w = hw % W;

    // Compute normalized coordinates
    T y, x;
    if (alignCorners) {
        y = H > 1 ? static_cast<T>(2.0 * h / (H - 1) - 1.0) : static_cast<T>(0);
        x = W > 1 ? static_cast<T>(2.0 * w / (W - 1) - 1.0) : static_cast<T>(0);
    } else {
        y = static_cast<T>((2.0 * h + 1.0) / H - 1.0);
        x = static_cast<T>((2.0 * w + 1.0) / W - 1.0);
    }

    // Apply affine transformation
    auto thetaOffset = b * 6;
    T outX = theta[thetaOffset + 0] * x + theta[thetaOffset + 1] * y + theta[thetaOffset + 2];
    T outY = theta[thetaOffset + 3] * x + theta[thetaOffset + 4] * y + theta[thetaOffset + 5];

    // Output: [N, H, W, 2]
    auto outOffset = b * H * W * 2 + h * W * 2 + w * 2;
    output[outOffset + 0] = outX;
    output[outOffset + 1] = outY;
}

template <typename T>
SD_KERNEL void affineGrid3DKernel(const T* theta, const sd::LongType* size,
                                    bool alignCorners, T* output,
                                    sd::LongType batchSize, sd::LongType D, sd::LongType H, sd::LongType W) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalElements = batchSize * D * H * W;

    if (tid >= totalElements) return;

    const auto b = tid / (D * H * W);
    const auto dhw = tid % (D * H * W);
    const auto d = dhw / (H * W);
    const auto hw = dhw % (H * W);
    const auto h = hw / W;
    const auto w = hw % W;

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

    auto thetaOffset = b * 12;
    T outX = theta[thetaOffset + 0] * x + theta[thetaOffset + 1] * y +
             theta[thetaOffset + 2] * z + theta[thetaOffset + 3];
    T outY = theta[thetaOffset + 4] * x + theta[thetaOffset + 5] * y +
             theta[thetaOffset + 6] * z + theta[thetaOffset + 7];
    T outZ = theta[thetaOffset + 8] * x + theta[thetaOffset + 9] * y +
             theta[thetaOffset + 10] * z + theta[thetaOffset + 11];

    auto outOffset = b * D * H * W * 3 + d * H * W * 3 + h * W * 3 + w * 3;
    output[outOffset + 0] = outX;
    output[outOffset + 1] = outY;
    output[outOffset + 2] = outZ;
}

template <typename T>
static void affineGrid2D_(LaunchContext* context, NDArray* theta, NDArray* size,
                           bool alignCorners, NDArray* output) {
    const auto batchSize = theta->sizeAt(0);
    const auto H = size->e<sd::LongType>(2);
    const auto W = size->e<sd::LongType>(3);

    const auto totalElements = batchSize * H * W;
    const int blockSize = 256;
    const int numBlocks = (totalElements + blockSize - 1) / blockSize;

    PointersManager manager(context, "affineGrid2D");

    affineGrid2DKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(theta->specialBuffer()),
        reinterpret_cast<const sd::LongType*>(size->specialBuffer()),
        alignCorners,
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, H, W);

    manager.synchronize();
}

template <typename T>
static void affineGrid3D_(LaunchContext* context, NDArray* theta, NDArray* size,
                           bool alignCorners, NDArray* output) {
    const auto batchSize = theta->sizeAt(0);
    const auto D = size->e<sd::LongType>(2);
    const auto H = size->e<sd::LongType>(3);
    const auto W = size->e<sd::LongType>(4);

    const auto totalElements = batchSize * D * H * W;
    const int blockSize = 256;
    const int numBlocks = (totalElements + blockSize - 1) / blockSize;

    PointersManager manager(context, "affineGrid3D");

    affineGrid3DKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(theta->specialBuffer()),
        reinterpret_cast<const sd::LongType*>(size->specialBuffer()),
        alignCorners,
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, D, H, W);

    manager.synchronize();
}

void affineGrid(LaunchContext* context, NDArray* theta, NDArray* size,
                bool alignCorners, NDArray* output) {
    bool is2D = theta->sizeAt(1) == 2;

    NDArray::prepareSpecialUse({output}, {theta, size});

    if (is2D) {
        BUILD_SINGLE_SELECTOR(theta->dataType(), affineGrid2D_, (context, theta, size, alignCorners, output),
                              SD_FLOAT_TYPES);
    } else {
        BUILD_SINGLE_SELECTOR(theta->dataType(), affineGrid3D_, (context, theta, size, alignCorners, output),
                              SD_FLOAT_TYPES);
    }

    NDArray::registerSpecialUse({output}, {theta, size});
}

template <typename T>
SD_KERNEL void gridSample2DKernel(const T* input, const T* grid, T* output,
                                    int mode, int paddingMode, bool alignCorners,
                                    sd::LongType batchSize, sd::LongType channels,
                                    sd::LongType inH, sd::LongType inW,
                                    sd::LongType outH, sd::LongType outW) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalElements = batchSize * channels * outH * outW;

    if (tid >= totalElements) return;

    const auto b = tid / (channels * outH * outW);
    const auto chw = tid % (channels * outH * outW);
    const auto c = chw / (outH * outW);
    const auto hw = chw % (outH * outW);
    const auto h = hw / outW;
    const auto w = hw % outW;

    // Get grid coordinates
    auto gridOffset = b * outH * outW * 2 + h * outW * 2 + w * 2;
    T gx = grid[gridOffset + 0];
    T gy = grid[gridOffset + 1];

    // Denormalize
    T ix, iy;
    if (alignCorners) {
        ix = (gx + 1) * (inW - 1) / 2;
        iy = (gy + 1) * (inH - 1) / 2;
    } else {
        ix = ((gx + 1) * inW - 1) / 2;
        iy = ((gy + 1) * inH - 1) / 2;
    }

    T value = static_cast<T>(0);

    if (mode == 0) {  // Bilinear
        sd::LongType ix0 = static_cast<sd::LongType>(floorf(ix));
        sd::LongType iy0 = static_cast<sd::LongType>(floorf(iy));
        sd::LongType ix1 = ix0 + 1;
        sd::LongType iy1 = iy0 + 1;

        T wx1 = ix - ix0;
        T wy1 = iy - iy0;
        T wx0 = 1 - wx1;
        T wy0 = 1 - wy1;

        auto getVal = [&](sd::LongType py, sd::LongType px) -> T {
            if (paddingMode == 0) {
                if (px < 0 || px >= inW || py < 0 || py >= inH) return static_cast<T>(0);
            } else if (paddingMode == 1) {
                px = max(static_cast<sd::LongType>(0), min(px, inW - 1));
                py = max(static_cast<sd::LongType>(0), min(py, inH - 1));
            }
            return input[b * channels * inH * inW + c * inH * inW + py * inW + px];
        };

        value = wx0 * wy0 * getVal(iy0, ix0) +
                wx1 * wy0 * getVal(iy0, ix1) +
                wx0 * wy1 * getVal(iy1, ix0) +
                wx1 * wy1 * getVal(iy1, ix1);
    } else if (mode == 1) {  // Nearest
        sd::LongType nearestX = static_cast<sd::LongType>(roundf(ix));
        sd::LongType nearestY = static_cast<sd::LongType>(roundf(iy));

        if (paddingMode == 0) {
            if (nearestX >= 0 && nearestX < inW && nearestY >= 0 && nearestY < inH) {
                value = input[b * channels * inH * inW + c * inH * inW + nearestY * inW + nearestX];
            }
        } else {
            nearestX = max(static_cast<sd::LongType>(0), min(nearestX, inW - 1));
            nearestY = max(static_cast<sd::LongType>(0), min(nearestY, inH - 1));
            value = input[b * channels * inH * inW + c * inH * inW + nearestY * inW + nearestX];
        }
    }

    output[b * channels * outH * outW + c * outH * outW + h * outW + w] = value;
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

    const auto totalElements = batchSize * channels * outH * outW;
    const int blockSize = 256;
    const int numBlocks = (totalElements + blockSize - 1) / blockSize;

    PointersManager manager(context, "gridSample2D");

    gridSample2DKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<const T*>(grid->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        mode, paddingMode, alignCorners,
        batchSize, channels, inH, inW, outH, outW);

    manager.synchronize();
}

void gridSample(LaunchContext* context, NDArray* input, NDArray* grid,
                int mode, int paddingMode, bool alignCorners, NDArray* output) {
    bool is2D = input->rankOf() == 4;

    NDArray::prepareSpecialUse({output}, {input, grid});

    if (is2D) {
        BUILD_SINGLE_SELECTOR(input->dataType(), gridSample2D_,
                              (context, input, grid, mode, paddingMode, alignCorners, output),
                              SD_FLOAT_TYPES);
    }

    NDArray::registerSpecialUse({output}, {input, grid});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
