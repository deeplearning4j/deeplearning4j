/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com
//


#include <ops/declarable/helpers/sparse_to_dense.h>
#include <helpers/StringUtils.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename X, typename I>
            static void fill_(const void *vvalues, const void *vindices, void *voutput, const Nd4jLong *zShapeInfo, uint8_t rank, uint64_t length) {
                auto values = reinterpret_cast<const X*>(vvalues);
                auto indices = reinterpret_cast<const I*>(vindices);
                auto output = reinterpret_cast<X*>(voutput);

                Nd4jLong coords[MAX_RANK];
                uint64_t pos = 0;
                for (uint64_t e = 0L; e < length; e++) {
                    // indices come in blocks
                    for (uint8_t p = 0; p < rank; p++) {
                        coords[p] = indices[pos++];
                    }

                    // fill output at given coords with sparse value
                    output[shape::getOffset(zShapeInfo, coords)] = values[e];
                }

            }

            void compat_sparse_to_dense(const NDArray &values, const NDArray &indices, NDArray *def, NDArray &output) {
                // make sure host buffer is updated
                values.syncToHost();
                indices.syncToHost();

                auto rank = output.rankOf();

                if (output.isS()) {
                    // string case is not so trivial, since elements might, and probably will, have different sizes
                    auto numValues = values.lengthOf();
                    auto numElements = output.lengthOf();

                    // first of all we calculate final buffer sizes and offsets
                    auto defaultLength = def == nullptr ? 0 : StringUtils::byteLength(*def);
                    auto valuesLength = StringUtils::byteLength(values);
                    auto bufferLength = defaultLength * (output.lengthOf() - numValues) + valuesLength;
                    auto headerLength = ShapeUtils::stringBufferHeaderRequirements(numElements);

                    // now we make sure our output buffer can hold results
                    output.dataBuffer()->expand( bufferLength + headerLength);

                    std::vector<Nd4jLong> outputCoords(rank);
                    std::vector<Nd4jLong> valueCoords(rank);

                    auto offsetsBuffer = output.bufferAsT<Nd4jLong>();
                    auto dataBuffer = reinterpret_cast<uint8_t*>(offsetsBuffer + output.lengthOf());

                    offsetsBuffer[0] = 0;

                    // getting initial value coords
                    for (int e = 0; e < rank; e++)
                        valueCoords[e] = indices.e<Nd4jLong>(e);

                    // write results individually
                    for (uint64_t e = 0; e < numElements; e++) {
                        auto vIndex = shape::coords2index(output.shapeInfo(), valueCoords.data());
                        auto cLength = 0L;
                        std::string str;
                        if (vIndex == e) {
                            // we're writing down sparse value here
                             str = values.e<std::string>(e);
                        } else {
                            // we're writing down default value if it exists
                            if (def != nullptr)
                                str = def->e<std::string>(0);
                            else
                                str = "";
                        }

                        // TODO: make it unicode compliant
                        memcpy(&dataBuffer[offsetsBuffer[e]], str.c_str(), str.length());

                        // writing down offset
                        offsetsBuffer[e+1] = cLength;
                    }
                } else {
                    // numeric case is trivial, since all elements have equal sizes

                    // write out default values, if they are present
                    if (def != nullptr) {
                        output.assign(def);

                        // make sure output is synced back
                        output.syncToHost();
                    }

                    // write out values
                    BUILD_DOUBLE_SELECTOR(values.dataType(), indices.dataType(), fill_, (values.getBuffer(), indices.getBuffer(), output.buffer(), output.getShapeInfo(), rank, values.lengthOf()), LIBND4J_TYPES, INDEXING_TYPES);
                }
                // copy back to device, if there's any
                output.syncToDevice();
            }
        }
    }
}