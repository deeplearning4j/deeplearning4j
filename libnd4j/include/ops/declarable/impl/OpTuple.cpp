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
// Created by raver119 on 11.10.2017.
//

#include "ops/declarable/OpTuple.h"

nd4j::ops::OpTuple::OpTuple(const char *opName) {
    _opName = opName;
}

nd4j::ops::OpTuple::OpTuple(const char *opName, std::initializer_list<nd4j::NDArray<float> *> &&inputs, std::initializer_list<double> &&tArgs, std::initializer_list<Nd4jLong> &&iArgs) {
    _opName = opName;
    _inputs = inputs;
    _iArgs = iArgs;
    _tArgs = tArgs;
}

nd4j::ops::OpTuple::~OpTuple() {
    for (auto v: _inputs)
        delete v;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::addInput(nd4j::NDArray *array) {
    _inputs.emplace_back(array);
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::addOutput(nd4j::NDArray *array) {
    _outputs.emplace_back(array);
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::setTArgs(std::initializer_list<double> tArgs) {
    _tArgs = tArgs;
    return this;
}

nd4j::ops::OpTuple *nd4j::ops::OpTuple::setIArgs(std::initializer_list<Nd4jLong> iArgs) {
    _iArgs = iArgs;
    return this;
}
