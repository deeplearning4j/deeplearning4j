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

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>
#include <graph/ExecutionResult.h>

namespace nd4j {
    namespace graph {
        ExecutionResult::ExecutionResult(const FlatResult* flatResult) {
            if (flatResult->variables() != nullptr) {
                for (int e = 0; e < flatResult->variables()->size(); e++) {
                    auto fv = flatResult->variables()->Get(e);
                    auto v = new Variable(fv);
                    this->emplace_back(v);
                }

                _releasable = true;
            }
        }

        ExecutionResult::~ExecutionResult(){
            if (_releasable)
                for (auto v : _variables)
                    delete v;
        }

        Nd4jLong ExecutionResult::size() {
            return _variables.size();
        }

        ExecutionResult::ExecutionResult(std::initializer_list<Variable *> variables) {
            for (auto v: variables)
                this->emplace_back(v);
        }

        void ExecutionResult::emplace_back(Variable *variable) {
            _variables.emplace_back(variable);

            if (variable->getName() != nullptr)
                _stringIdMap[*variable->getName()] = variable;

            std::pair<int,int> p(variable->id(), variable->index());
            _pairIdMap[p] = variable;
        }

        Variable* ExecutionResult::at(int position) {
            if (position >= _variables.size())
                throw std::runtime_error("Position index is higher then number of variables stored");

            return _variables.at(position);
        }

        Variable* ExecutionResult::byId(std::string &id) {
            if (_stringIdMap.count(id) == 0)
                throw std::runtime_error("Can't find specified ID");

            return _stringIdMap.at(id);
        }
        
        Variable* ExecutionResult::byId(std::pair<int, int> &id) {
            if (_pairIdMap.count(id) == 0)
                throw std::runtime_error("Can't find specified ID");

            return _pairIdMap.at(id);
        }

        Variable* ExecutionResult::byId(int id) {
            std::pair<int,int> p(id, 0);
            return byId(p);
        }

        Variable* ExecutionResult::byId(const char *str) {
            std::string p(str);
            return byId(p);
        }

        flatbuffers::Offset<FlatResult> ExecutionResult::asFlatResult(flatbuffers::FlatBufferBuilder &builder) {

            std::vector<flatbuffers::Offset<FlatVariable>> vec;
            for (Variable* v : _variables) {
                vec.emplace_back(v->asFlatVariable(builder));
            }

            auto vecOffset = builder.CreateVector(vec);

            return CreateFlatResult(builder, 0, vecOffset);
        }
    }
}