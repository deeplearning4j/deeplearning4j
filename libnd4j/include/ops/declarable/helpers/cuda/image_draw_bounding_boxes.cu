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
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static __global__ void drawBoundingBoxesKernel(T const* images, Nd4jLong* imagesShape, T const* boxes,
            Nd4jLong* boxesShape, T const* colors, Nd4jLong* colorsShape, T* output, Nd4jLong* outputShape,
            Nd4jLong batchSize, Nd4jLong width, Nd4jLong height, Nd4jLong channels, Nd4jLong colorSetSize) {

        for (auto b = blockIdx.x; b < (int)batchSize; b += gridDim.x) { // loop by batch
            for (auto c = 0; c < colorSetSize; c++) {
                // box with shape
                auto internalBox = &boxes[b * colorSetSize * 4 + c * 4];//(*boxes)(b, {0})(c, {0});//internalBoxes->at(c);
                auto color = &colors[channels * c];//colorSet->at(c);
                auto rowStart = nd4j::math::nd4j_max(Nd4jLong (0), Nd4jLong ((height - 1) * internalBox[0]));
                auto rowEnd = nd4j::math::nd4j_min(Nd4jLong (height - 1), Nd4jLong ((height - 1) * internalBox[2]));
                auto colStart = nd4j::math::nd4j_max(Nd4jLong (0), Nd4jLong ((width - 1) * internalBox[1]));
                auto colEnd = nd4j::math::nd4j_min(Nd4jLong(width - 1), Nd4jLong ((width - 1) * internalBox[3]));
                for (auto y = rowStart + threadIdx.x; y <= rowEnd; y += blockDim.x) {
                    for (auto e = 0; e < channels; ++e) {
                        Nd4jLong yMinPos[] = {b, y, colStart, e};
                        Nd4jLong yMaxPos[] = {b, y, colEnd, e};
                        auto zIndexYmin = shape::getOffset(outputShape, yMinPos);
                        auto zIndexYmax = shape::getOffset(outputShape, yMaxPos);
                        output[zIndexYmin] = color[e];
                        output[zIndexYmax] = color[e];
                    }
                }
                for (auto x = colStart + 1 + threadIdx.x; x < colEnd; x += blockDim.x) {
                    for (auto e = 0; e < channels; ++e) {
                        Nd4jLong xMinPos[] = {b, rowStart, x, e};
                        Nd4jLong xMaxPos[] = {b, rowEnd, x, e};
                        auto zIndexXmin = shape::getOffset(outputShape, xMinPos);
                        auto zIndexXmax = shape::getOffset(outputShape, xMaxPos);
                        output[zIndexXmin] = color[e];
                        output[zIndexXmax] = color[e];
                    }
                }
            }
        }

    }

    template <typename T>
    void drawBoundingBoxesH(nd4j::LaunchContext* context, NDArray const* images, NDArray const* boxes, NDArray const* colors, NDArray* output) {
        auto batchSize = images->sizeAt(0);
        auto height = images->sizeAt(1);
        auto width = images->sizeAt(2);
        auto channels = images->sizeAt(3);
        auto stream = context->getCudaStream();
        auto colorSetSize = colors->sizeAt(0);

        auto imagesBuf = images->getDataBuffer()->specialAsT<T>();
        auto boxesBuf = boxes->getDataBuffer()->specialAsT<T>();
        auto colorsBuf = colors->getDataBuffer()->specialAsT<T>();
        auto outputBuf = output->dataBuffer()->specialAsT<T>();
        drawBoundingBoxesKernel<<<batchSize > 128? 128: batchSize, 256, 1024, *stream>>>(imagesBuf, images->getSpecialShapeInfo(),
                boxesBuf, boxes->getSpecialShapeInfo(), colorsBuf, colors->getSpecialShapeInfo(),
                outputBuf, output->specialShapeInfo(), batchSize, width, height, channels, colorSetSize);
    }

    void drawBoundingBoxesFunctor(nd4j::LaunchContext * context, NDArray* images, NDArray* boxes, NDArray* colors, NDArray* output) {
        // images - batch of 3D images with BW (last dim = 1), RGB (last dim = 3) or RGBA (last dim = 4) channel set
        // boxes - batch of 2D bounds with last dim (y_start, x_start, y_end, x_end) to compute i and j as
        // floor((height - 1 ) * y_start) => rowStart, floor((height - 1) * y_end) => rowEnd
        // floor((width - 1 ) * x_start) => colStart, floor((width - 1) * x_end) => colEnd
        // height = images->sizeAt(1), width = images->sizeAt(2)
        // colors - colors for each box given
        // set up color for each box as frame
        NDArray::prepareSpecialUse({output}, {images, boxes, colors});
        output->assign(images);
        BUILD_SINGLE_SELECTOR(output->dataType(), drawBoundingBoxesH, (context, images, boxes, colors, output), FLOAT_TYPES);
        NDArray::registerSpecialUse({output}, {images, boxes, colors});
    }
    BUILD_SINGLE_TEMPLATE(template void drawBoundingBoxesH, (nd4j::LaunchContext* context, NDArray const* images, NDArray const* boxes, NDArray const* colors, NDArray* output), FLOAT_TYPES);
}
}
}
