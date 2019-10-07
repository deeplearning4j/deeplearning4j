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

    void drawBoundingBoxesFunctor(nd4j::LaunchContext * context, NDArray* images, NDArray* boxes, NDArray* colors, NDArray* output) {
        // images - batch of 3D images with BW (last dim = 1), RGB (last dim = 3) or RGBA (last dim = 4) channel set
        // boxes - batch of 2D bounds with last dim (y_start, x_start, y_end, x_end) to compute i and j as
        // floor((height - 1 ) * y_start) => rowStart, floor((height - 1) * y_end) => rowEnd
        // floor((width - 1 ) * x_start) => colStart, floor((width - 1) * x_end) => colEnd
        // height = images->sizeAt(1), width = images->sizeAt(2)
        // colors - colors for each box given
        // set up color for each box as frame
        auto batchSize = images->sizeAt(0);
        auto height = images->sizeAt(1);
        auto width = images->sizeAt(2);
        auto channels = images->sizeAt(3);
        //auto imageList = images->allTensorsAlongDimension({1, 2, 3}); // split images by batch
//        auto boxList = boxes->allTensorsAlongDimension({1, 2}); // split boxes by batch
        auto colorSet = colors->allTensorsAlongDimension({1});
        output->assign(images); // fill up all output with input images, then fill up boxes
        for (auto b = 0; b < batchSize; ++b) { // loop by batch
//            auto image = imageList->at(b);

            for (auto c = 0; c < colorSet->size(); ++c) {
                // box with shape
                auto internalBox = (*boxes)(b, {0})(c, {0});//internalBoxes->at(c);
                auto color = colorSet->at(c);
                auto rowStart = nd4j::math::nd4j_max(Nd4jLong (0), Nd4jLong ((height - 1) * internalBox.e<float>(0)));
                auto rowEnd = nd4j::math::nd4j_min(Nd4jLong (height - 1), Nd4jLong ((height - 1) * internalBox.e<float>(2)));
                auto colStart = nd4j::math::nd4j_max(Nd4jLong (0), Nd4jLong ((width - 1) * internalBox.e<float>(1)));
                auto colEnd = nd4j::math::nd4j_min(Nd4jLong(width - 1), Nd4jLong ((width - 1) * internalBox.e<float>(3)));
                for (auto y = rowStart; y <= rowEnd; y++) {
                    for (auto e = 0; e < color->lengthOf(); ++e) {
                        output->p(b, y, colStart, e, color->e(e));
                        output->p(b, y, colEnd, e, color->e(e));
                    }
                }
                for (auto x = colStart + 1; x < colEnd; x++) {
                    for (auto e = 0; e < color->lengthOf(); ++e) {
                        output->p(b, rowStart, x, e, color->e(e));
                        output->p(b, rowEnd, x, e, color->e(e));
                    }
                }
            }
//            delete internalBoxes;
        }
        delete colorSet;
//        delete imageList;
//        delete boxList;
    }

}
}
}
