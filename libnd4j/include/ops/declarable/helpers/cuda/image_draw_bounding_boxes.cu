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
//  @author sgazeos@gmail.com
//
#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

    typedef NDArray ColorTable_t;
    static NDArray DefaultColorTable(int depth, sd::LaunchContext* context) {
        //std::vector<std::vector<float>> colorTable;
        const Nd4jLong kDefaultTableLength = 10;
        const Nd4jLong kDefaultChannelLength = 4;
        NDArray colorTable('c', {kDefaultTableLength, kDefaultChannelLength}, {
                1,1,0,1,         // yellow
                0, 0, 1, 1,      // 1: blue
                1, 0, 0, 1,      // 2: red
                0, 1, 0, 1,      // 3: lime
                0.5, 0, 0.5, 1,  // 4: purple
                0.5, 0.5, 0, 1,  // 5: olive
                0.5, 0, 0, 1,    // 6: maroon
                0, 0, 0.5, 1,    // 7: navy blue
                0, 1, 1, 1,      // 8: aqua
                1, 0, 1, 1       // 9: fuchsia
        }, DataType::FLOAT32, context);

        if (depth == 1) {
            colorTable.assign(1.f); // all to white when black and white colors
        }
        return colorTable;
    }

    template <typename T>
    static __global__ void drawBoundingBoxesKernel(T const* images, const Nd4jLong* imagesShape,
                                                   float const* boxes, const Nd4jLong* boxesShape,
                                                   float const* colorTable, const Nd4jLong* colorTableShape,
                                                   T* output, const Nd4jLong* outputShape,
                                                   Nd4jLong batchSize, Nd4jLong width, Nd4jLong height,
                                                   Nd4jLong channels, Nd4jLong boxSize, Nd4jLong colorTableLen) {

        for (auto batch = blockIdx.x; batch < (int)batchSize; batch += gridDim.x) { // loop by batch
            for (auto boxIndex = 0; boxIndex < boxSize; ++boxIndex) {
                // box with shape
                //auto internalBox = &boxes[b * colorSetSize * 4 + c * 4];//(*boxes)(b, {0})(c, {0});//internalBoxes->at(c);
                auto colorIndex = boxIndex % colorTableLen;//colorSet->at(c);
//                auto rowStart = sd::math::nd4j_max(Nd4jLong (0), Nd4jLong ((height - 1) * internalBox[0]));
//                auto rowEnd = sd::math::nd4j_min(Nd4jLong (height - 1), Nd4jLong ((height - 1) * internalBox[2]));
//                auto colStart = sd::math::nd4j_max(Nd4jLong (0), Nd4jLong ((width - 1) * internalBox[1]));
//                auto colEnd = sd::math::nd4j_min(Nd4jLong(width - 1), Nd4jLong ((width - 1) * internalBox[3]));
                Nd4jLong indices0[] = {batch, boxIndex, 0};
                Nd4jLong indices1[] = {batch, boxIndex, 1};
                Nd4jLong indices2[] = {batch, boxIndex, 2};
                Nd4jLong indices3[] = {batch, boxIndex, 3};
                auto rowStart = Nd4jLong ((height - 1) * boxes[shape::getOffset(boxesShape, indices0, 0)]);
                auto rowStartBound = sd::math::nd4j_max(Nd4jLong (0), rowStart);
                auto rowEnd = Nd4jLong ((height - 1) * boxes[shape::getOffset(boxesShape, indices2, 0)]);
                auto rowEndBound = sd::math::nd4j_min(Nd4jLong (height - 1), rowEnd);
                auto colStart = Nd4jLong ((width - 1) * boxes[shape::getOffset(boxesShape, indices1, 0)]);
                auto colStartBound = sd::math::nd4j_max(Nd4jLong (0), colStart);
                auto colEnd = Nd4jLong ((width - 1) * boxes[shape::getOffset(boxesShape, indices3, 0)]);
                auto colEndBound = sd::math::nd4j_min(Nd4jLong(width - 1), colEnd);
                if (rowStart > rowEnd || colStart > colEnd) {
//                    printf("helpers::drawBoundingBoxesFunctor: Bounding box (%lld, %lld, %lld, %lld) is inverted "
//                                "and will not be drawn\n", rowStart, colStart, rowEnd, colEnd);
                    continue;
                }
                if (rowStart >= height || rowEnd < 0 || colStart >= width ||
                    colEnd < 0) {
//                    printf("helpers::drawBoundingBoxesFunctor: Bounding box (%lld, %lld, %lld, %lld) is completely "
//                                "outside the image and not be drawn\n", rowStart, colStart, rowEnd, colEnd);
                    continue;
                }

                // Draw upper line
                if (rowStart >= 0) {
                    for (auto j = colStartBound + threadIdx.x; j <= colEndBound; j += blockDim.x)
                        for (auto c = 0; c < channels; c++) {
                            Nd4jLong zPos[] = {batch, rowStart, j, c};
                            Nd4jLong cPos[] = {colorIndex, c};
                            auto cIndex = shape::getOffset(colorTableShape, cPos, 0);
                            auto zIndex = shape::getOffset(outputShape, zPos, 0);
                            output[zIndex] = (T)colorTable[cIndex];
                        }
                }
                // Draw bottom line.
                if (rowEnd < height) {
                    for (auto j = colStartBound + threadIdx.x; j <= colEndBound; j += blockDim.x)
                        for (auto c = 0; c < channels; c++) {
                            Nd4jLong zPos[] = {batch, rowEnd, j, c};
                            Nd4jLong cPos[] = {colorIndex, c};
                            auto cIndex = shape::getOffset(colorTableShape, cPos, 0);
                            auto zIndex = shape::getOffset(outputShape, zPos, 0);
                            output[zIndex] = (T)colorTable[cIndex];
                        }
                }

                // Draw left line.
                if (colStart >= 0) {
                    for (auto i = rowStartBound + threadIdx.x; i <= rowEndBound; i += blockDim.x)
                        for (auto c = 0; c < channels; c++) {
                            Nd4jLong zPos[] = {batch, i, colStart, c};
                            Nd4jLong cPos[] = {colorIndex, c};
                            auto cIndex = shape::getOffset(colorTableShape, cPos, 0);
                            auto zIndex = shape::getOffset(outputShape, zPos, 0);
                            output[zIndex] = (T)colorTable[cIndex];
                        }
                }
                // Draw right line.
                if (colEnd < width) {
                    for (auto i = rowStartBound + threadIdx.x; i <= rowEndBound; i += blockDim.x)
                        for (auto c = 0; c < channels; c++) {
                            Nd4jLong zPos[] = {batch, i, colEnd, c};
                            Nd4jLong cPos[] = {colorIndex, c};
                            auto cIndex = shape::getOffset(colorTableShape, cPos, 0);
                            auto zIndex = shape::getOffset(outputShape, zPos, 0);
                            output[zIndex] = (T)colorTable[cIndex];
                        }
                }
            }
        }

    }

    template <typename T>
    void drawBoundingBoxesH(sd::LaunchContext* context, NDArray const* images, NDArray const* boxes, NDArray const* colors, NDArray* output) {
        auto batchSize = images->sizeAt(0);
        auto height = images->sizeAt(1);
        auto width = images->sizeAt(2);
        auto channels = images->sizeAt(3);
        auto stream = context->getCudaStream();
        auto boxSize = boxes->sizeAt(1);
        NDArray colorsTable = DefaultColorTable(channels, context);
        if ((colors != nullptr && colors->lengthOf() > 0)) {
            colorsTable = *colors;
        }

        auto imagesBuf = images->getDataBuffer()->specialAsT<T>();
        auto boxesBuf = boxes->getDataBuffer()->specialAsT<float>(); // boxes should be float32
        auto colorsTableBuf = colorsTable.getDataBuffer()->specialAsT<float>(); // color table is float32
        auto outputBuf = output->dataBuffer()->specialAsT<T>();
        drawBoundingBoxesKernel<<<128, 128, 1024, *stream>>>(imagesBuf, images->specialShapeInfo(),
                boxesBuf, boxes->specialShapeInfo(), colorsTableBuf, colorsTable.specialShapeInfo(),
                outputBuf, output->specialShapeInfo(), batchSize, width, height, channels, boxSize, colorsTable.lengthOf());
    }

    void drawBoundingBoxesFunctor(sd::LaunchContext * context, NDArray* images, NDArray* boxes, NDArray* colors, NDArray* output) {
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

}
}
}
