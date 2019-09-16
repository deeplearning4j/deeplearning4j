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

#ifndef SAMEDIFF_TUPLE_H
#define SAMEDIFF_TUPLE_H

#include <graph/Node.h>
#include <samediff/Variable.h>
#include <vector>

namespace samediff {
    class SameDiff;

    class Tuple {
    private:
        // TODO: use shared_ptr here
        nd4j::graph::Node *_node = nullptr;
        SameDiff *_sd = nullptr;

        // only used for Tuple-as-input scenario
        std::vector<Variable> _variables;
        std::vector<std::pair<int, int>> _indices;
    public:
        Tuple(std::initializer_list<Variable> variables);
        Tuple(const std::vector<Variable> &variables = {});
        Tuple(SameDiff &sd, nd4j::graph::Node *node);
        ~Tuple() = default;

        SameDiff* sd() const;
        uint32_t size() const;

        int nodeId() const;

        std::vector<std::pair<int, int>> indices() const;

        Variable at(uint32_t index) const;

        Variable operator[](const uint32_t index) const;
    };
}


#endif //SAMEDIFF_TUPLE_H
