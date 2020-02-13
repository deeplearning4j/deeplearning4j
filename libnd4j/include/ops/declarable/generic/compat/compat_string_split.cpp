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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_split_string)

#include <ops/declarable/CustomOperations.h>
#include <helpers/StringUtils.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(compat_string_split, 2, 2, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            auto indices = OUTPUT_VARIABLE(0);
            auto values = OUTPUT_VARIABLE(1);

            auto d = delim->e<std::string>(0);

            input->syncToHost();
            delim->syncToHost();

            // output rank N+1 wrt input rank
            std::vector<Nd4jLong> ocoords(input->rankOf() + 1);
            std::vector<Nd4jLong> icoords(input->rankOf());

            // getting buffer lengths
            // FIXME: it'll be bigger, since it'll include delimiters,
            auto outputLength = StringUtils::byteLength(*input);

            uint64_t ss = 0L;
            Nd4jLong ic = 0L;
            // loop through each string within tensor
            for (auto e = 0L; e < input->lengthOf(); e++) {
                // now we should map substring to indices
                auto s = input->e<std::string>(e);

                // getting base index
                shape::index2coords(e, input->shapeInfo(), icoords.data());

                // getting number of substrings
                auto cnt = StringUtils::countSubarrays(s.c_str(), s.length(), d.c_str(), d.length()) + 1;

                // filling output indices
                for (uint64_t f = 0; f < cnt; f++) {
                    for (auto v: icoords)
                        indices->p(ic++, v);

                    // last index
                    indices->p(ic++, f);
                }

                ss += cnt;
            }

            // process strings now
            std::vector<std::string> strings;
            for (auto e = 0L; e < input->lengthOf(); e++) {
                auto split = StringUtils::split(input->e<std::string>(e), d);

                for (const auto &s:split)
                    strings.emplace_back(s);
            }

            // now once we have all strings in single vector time to fill
            auto tmp = NDArrayFactory::string({(Nd4jLong) strings.size()}, strings);
            auto blen = StringUtils::byteLength(tmp) + ShapeUtils::stringBufferHeaderRequirements(strings.size());

            // for CUDA mostly
            values->dataBuffer()->allocatePrimary();
            values->dataBuffer()->expand(blen);
            memcpy(values->buffer(), tmp.buffer(), blen);
            values->tickWriteHost();

            // special case, for future use
            indices->syncToDevice();
            values->syncToDevice();

            // we have to tick buffers
            values->dataBuffer()->writePrimary();
            values->dataBuffer()->readSpecial();

            return Status::OK();
        };

        DECLARE_SHAPE_FN(compat_string_split) {
            auto input = INPUT_VARIABLE(0);
            auto delim = INPUT_VARIABLE(1);

            auto d = delim->e<std::string>(0);

            // count number of delimiter substrings in all strings within input tensor
            uint64_t cnt = 0;
            for (auto e = 0L; e < input->lengthOf(); e++) {
                // FIXME: bad, not UTF-compatible
                auto s = input->e<std::string>(e);

                // each substring we see in haystack, splits string in two parts. so we should add 1 to the number of subarrays
                cnt += StringUtils::countSubarrays(s.c_str(), s.length(), d.c_str(), d.length()) + 1;
            }

            // shape calculations
            // virtual tensor rank will be N+1, for N rank input array, where data will be located at the biggest dimension
            // values tensor is going to be vector always
            // indices tensor is going to be vector with length equal to values.length * output rank

            auto valuesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(cnt, nd4j::DataType::UTF8);
            auto indicesShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(cnt * (input->rankOf() + 1), nd4j::DataType::INT64);

            return SHAPELIST(indicesShape, valuesShape);
        }

        DECLARE_TYPES(compat_string_split) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_STRINGS})
                    ->setAllowedOutputTypes(0, {ALL_INDICES})
                    ->setAllowedOutputTypes(1, {ALL_STRINGS});
        }
    }
}

#endif