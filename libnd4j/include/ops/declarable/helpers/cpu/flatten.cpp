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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/flatten.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            static void flatten_(std::vector<NDArray*> &inputs, NDArray *output, const char order) {

                int numArrays = inputs.size();
                std::vector<Nd4jLong> offsets(numArrays);
                Nd4jLong cOffset = 0;

                // calculating offsets in output
                for (int e = 0; e < numArrays; e++) {
                    offsets[e] = cOffset;
                    cOffset += inputs[e]->lengthOf();
                }

                Nd4jLong xCoord[MAX_RANK];

                // actually transferring data
                for (int e = 0; e < numArrays; e++) {
                    auto z = reinterpret_cast<T *>(output->bufferWithOffset(offsets[e]));

                    auto xBuffer = inputs[e]->bufferAsT<T>();
                    auto xShapeInfo = inputs[e]->shapeInfo();
                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    auto xRank = shape::rank(xShapeInfo);
                    auto xLength = inputs[e]->lengthOf();
                    
                    for (uint i = 0; i < xLength; i++) {
                        shape::index2coords(xRank, xShape, i, xLength, xCoord, order);
                        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        z[i] = xBuffer[xOffset];
                    }                    
                }
            }

            void flatten(std::vector<NDArray*> &inputs, NDArray *output, char order) {
                BUILD_SINGLE_SELECTOR(output->dataType(), flatten_, (inputs, output, order), LIBND4J_TYPES);
            }
        }
    }
}