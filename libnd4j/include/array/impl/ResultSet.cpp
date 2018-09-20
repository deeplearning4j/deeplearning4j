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
// @author raver119@gmail.com
//

#include <array/ResultSet.h>
#include <graph/FlatUtils.h>

namespace nd4j {
    ResultSet::ResultSet(const nd4j::graph::FlatResult* result) {
        if (result != nullptr) {
            for (int e = 0; e < result->variables()->size(); e++) {
                auto var = result->variables()->Get(e);

                NDArray* array;

                if (var->ndarray() != nullptr) {
                    array = nd4j::graph::FlatUtils::fromFlatArray(var->ndarray());
                } else if (var->shape() != nullptr) {
                    std::vector<Nd4jLong> shapeInfo;
                    for (int i = 0; i < var->shape()->size(); i++) {
                        shapeInfo.emplace_back(var->shape()->Get(i));
                    }

                    // we just create empty array here
                    std::vector<Nd4jLong> shape;
                    for (int i = 0; i < shapeInfo.at(0); i++) {
                        shape.emplace_back(shapeInfo.at(i + 1));
                    }

                    array = new NDArray((char) shapeInfo.at(shapeInfo.size() - 1), shape, );
                } else {
                    nd4j_printf("Either shape or NDArray should be defined in FlatResult variable\n","");
                    throw std::runtime_error("Empty variable");
                }


                _content.push_back(array);
            }
        }
    }

    ResultSet::~ResultSet() {
        if (_removable)
            for (auto v: _content)
                delete v;
    }

    void ResultSet::setNonRemovable() {
        _removable = false;
    }

    int ResultSet::size() {
        return (int) _content.size();
    }

    nd4j::NDArray* ResultSet::at(unsigned long idx) {
        return _content.at(idx);
    }

    void ResultSet::push_back(nd4j::NDArray *array) {
        _content.emplace_back(array);
    }

    Nd4jStatus ResultSet::status() {
        return _status;
    }

    void ResultSet::setStatus(Nd4jStatus status) {
        _status = status;
    }

    void ResultSet::purge() {
        _content.clear();
    }
}

