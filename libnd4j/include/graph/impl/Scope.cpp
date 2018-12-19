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
// Created by raver119 on 14.10.2017.
//

#include "Scope.h"

namespace nd4j {
    namespace graph {
        Scope::Scope(int id, const char *name) {
            _id = id;

            if (name != nullptr)
                _name = name;
            else
                name = "";
        }

        Scope::~Scope() {
            for (auto v: _nodes)
                delete v;
        }

        void Scope::push_back(Node *node) {
            _nodes.emplace_back(node);
        }

        std::vector<Node *>* Scope::nodes() {
            return &_nodes;
        }

        int Scope::size() {
            return (int) _nodes.size();
        }

        int Scope::id() {
            return _id;
        }

        std::string* Scope::name() {
            return &_name;
        }

        void Scope::forgetNodes() {
            _nodes.clear();
        }

        Scope* Scope::clone() {
            auto clone = new Scope(_id, _name.c_str());

            for (auto v: _nodes)
                clone->_nodes.emplace_back(v->clone());

            return clone;
        }
    }
}

