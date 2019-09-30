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

#include <ops/declarable/helpers/image_suppression.h>
//#include <blas/NDArray.h>
#include <algorithm>
#include <numeric>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void nonMaxSuppressionV2_(NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        std::vector<Nd4jLong> indices(scales->lengthOf());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [scales](int i, int j) {return scales->e<T>(i) > scales->e<T>(j);});

//        std::vector<int> selected(output->lengthOf());
        std::vector<int> selectedIndices(output->lengthOf(), 0);
        auto needToSuppressWithThreshold = [] (NDArray& boxes, int previousIndex, int nextIndex, T threshold) -> bool {
            T minYPrev = nd4j::math::nd4j_min(boxes.e<T>(previousIndex, 0), boxes.e<T>(previousIndex, 2));
            T minXPrev = nd4j::math::nd4j_min(boxes.e<T>(previousIndex, 1), boxes.e<T>(previousIndex, 3));
            T maxYPrev = nd4j::math::nd4j_max(boxes.e<T>(previousIndex, 0), boxes.e<T>(previousIndex, 2));
            T maxXPrev = nd4j::math::nd4j_max(boxes.e<T>(previousIndex, 1), boxes.e<T>(previousIndex, 3));
            T minYNext = nd4j::math::nd4j_min(boxes.e<T>(nextIndex, 0), boxes.e<T>(nextIndex, 2));
            T minXNext = nd4j::math::nd4j_min(boxes.e<T>(nextIndex, 1), boxes.e<T>(nextIndex, 3));
            T maxYNext = nd4j::math::nd4j_max(boxes.e<T>(nextIndex, 0), boxes.e<T>(nextIndex, 2));
            T maxXNext = nd4j::math::nd4j_max(boxes.e<T>(nextIndex, 1), boxes.e<T>(nextIndex, 3));
            T areaPrev = (maxYPrev - minYPrev) * (maxXPrev - minXPrev);
            T areaNext = (maxYNext - minYNext) * (maxXNext - minXNext);

            if (areaNext <= T(0.f) || areaPrev <= T(0.f)) return false;

            T minIntersectionY = nd4j::math::nd4j_max(minYPrev, minYNext);
            T minIntersectionX = nd4j::math::nd4j_max(minXPrev, minXNext);
            T maxIntersectionY = nd4j::math::nd4j_min(maxYPrev, maxYNext);
            T maxIntersectionX = nd4j::math::nd4j_min(maxXPrev, maxXNext);
            T intersectionArea =
                    nd4j::math::nd4j_max(T(maxIntersectionY - minIntersectionY), T(0.0f)) *
                            nd4j::math::nd4j_max(T(maxIntersectionX - minIntersectionX), T(0.0f));
            T intersectionValue = intersectionArea / (areaPrev + areaNext - intersectionArea);
            return intersectionValue > threshold;

        };
//        int numSelected = 0;
        int numBoxes = boxes->sizeAt(0);
        int numSelected = 0;

        for (int i = 0; i < numBoxes; ++i) {
            bool shouldSelect = numSelected < output->lengthOf();

            // FIXME: add parallelism here
            for (int j = numSelected - 1; j >= 0; --j) {
                if (shouldSelect)
                if (needToSuppressWithThreshold(*boxes, indices[i], indices[selectedIndices[j]], T(threshold))) {
                    shouldSelect = false;
                }
            }
            if (shouldSelect) {
                output->p(numSelected, indices[i]);
                selectedIndices[numSelected++] = i;
            }
        }
    }

    void nonMaxSuppressionV2(nd4j::LaunchContext * context, NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output) {
        BUILD_SINGLE_SELECTOR(boxes->dataType(), nonMaxSuppressionV2_, (boxes, scales, maxSize, threshold, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void nonMaxSuppressionV2_, (NDArray* boxes, NDArray* scales, int maxSize, double threshold, NDArray* output), NUMERIC_TYPES);

}
}
}