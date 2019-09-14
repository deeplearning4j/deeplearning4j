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

#include "../SameDiff.h"
#include <graph/Variable.h>
#include <GraphExecutioner.h>

namespace samediff {
    SameDiff::SameDiff() {
        _graph = new Graph();
    }

    SameDiff::~SameDiff() {
        if (_graph != nullptr)
            delete _graph;
    }

    SDVariable SameDiff::variable(const nd4j::NDArray &array, bool trainable, const std::string &name) {
        auto variable = new Variable(array.dup(), name.c_str());
        return SDVariable(*this, _graph->addVariableNode(variable));
    }

    SDVariable SameDiff::placeholder(const std::string &name, const nd4j::DataType dataType, const std::vector<Nd4jLong> shape) {
        auto variable = new Variable(name, dataType, shape);
        return SDVariable(*this, _graph->addPlaceholderNode(variable));
    }

    nd4j::graph::Graph* SameDiff::graph() {
        return _graph;
    }

    void SameDiff::execute() {
        nd4j::graph::GraphExecutioner::execute(graph(), graph()->getVariableSpace());
    }

    void SameDiff::train() {
        //
    }

    void SameDiff::executeWithDictionary(const std::unordered_map<const char*, nd4j::NDArray> &args) {
        //
    }

    void SameDiff::save(const char *filename) {
        //
    }

    SameDiff SameDiff::load(const char *filename) {
        return {};
    }
}