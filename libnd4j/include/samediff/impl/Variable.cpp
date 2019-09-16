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

#include "samediff/Variable.h"
#include "../SameDiff.h"
#include <samediff/samediff_cpp.h>
#include <NDArrayFactory.h>
#include <exceptions/precondition_exception.h>

namespace samediff {
    Variable::Variable(SameDiff &sd, nd4j::graph::Node *node, int index) {
        precondition_exception::check(node != nullptr, "Variable: Node passed in is null");
        precondition_exception::check(node->parentGraph() != nullptr, "Variable: Node passed in has no Graph defined");
        _node = node;
        _sd = &sd;
        _index = index;
    }

    nd4j::NDArray Variable::array() {
        auto var = _node->parentGraph()->getVariableSpace()->getVariable(_node->id(), _index);
        return *var->getNDArray();
    }

    int Variable::nodeId() const{
        return _node->id();
    }

    SameDiff* Variable::sd() const {
        return _sd;
    }

    Variable Variable::operator+(const Variable& other) const {
        return samediff::arithmetic::Add(*this, other);
    }

    std::pair<int, int> Variable::index() const {
        return {nodeId(), _index};
    }
}

samediff::Variable operator+(const float &scalar, const samediff::Variable &var) {
    auto sd = var.sd();
    auto x = sd->variable(nd4j::NDArrayFactory::create<float>(scalar));
    return samediff::arithmetic::Add(x, var);
}

samediff::Variable operator+(const samediff::Variable &var, const float &scalar) {
    auto sd = var.sd();
    auto y = sd->variable(nd4j::NDArrayFactory::create<float>(scalar));
    return samediff::arithmetic::Add(var, y);
}