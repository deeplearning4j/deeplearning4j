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

#include <exceptions/precondition_exception.h>
#include "../Tuple.h"

namespace samediff {
    Tuple::Tuple(SameDiff &sd, nd4j::graph::Node *node) {
        precondition_exception::check(node != nullptr, "Tuple: Node passed in is null");
        precondition_exception::check(node->parentGraph() != nullptr, "Tuple: Node passed in has no Graph defined");
        _node = node;
        _sd = &sd;
    }

    uint32_t Tuple::size() const {
        return 0;
    }

    Variable Tuple::at(uint32_t index) const {
        return Variable(*_sd, _node, index);
    }

    Variable Tuple::operator[](const uint32_t index) const {
        return this->at(index);
    }
}