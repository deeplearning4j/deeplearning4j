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
        template <typename T>
        ExecutionResult<T>::ExecutionResult(const FlatResult* flatResult) {
            if (flatResult->variables() != nullptr) {
                for (int e = 0; e < flatResult->variables()->size(); e++) {
                    auto fv = flatResult->variables()->Get(e);
                    auto v = new Variable<T>(fv);
                    this->emplace_back(v);
                }

                _releasable = true;
            }
        }

        template <typename T>
        ExecutionResult<T>::~ExecutionResult(){
            if (_releasable)
                for (auto v : _variables)
                    delete v;
        }

        template <typename T>
        Nd4jLong ExecutionResult<T>::size() {
            return _variables.size();
        }

        template <typename T>
        ExecutionResult<T>::ExecutionResult(std::initializer_list<Variable<T> *> variables) {
            for (auto v: variables)
                this->emplace_back(v);
        }

        template <typename T>
        void ExecutionResult<T>::emplace_back(Variable<T> *variable) {
            _variables.emplace_back(variable);

            if (variable->getName() != nullptr)
                _stringIdMap[*variable->getName()] = variable;

            std::pair<int,int> p(variable->id(), variable->index());
            _pairIdMap[p] = variable;
        }

        template <typename T>
        Variable<T>* ExecutionResult<T>::at(int position) {
            if (position >= _variables.size())
                throw std::runtime_error("Position index is higher then number of variables stored");

            return _variables.at(position);
        }

        template <typename T>
        Variable<T>* ExecutionResult<T>::byId(std::string &id) {
            if (_stringIdMap.count(id) == 0)
                throw std::runtime_error("Can't find specified ID");

            return _stringIdMap.at(id);
        }
        
        template <typename T>
        Variable<T>* ExecutionResult<T>::byId(std::pair<int, int> &id) {
            if (_pairIdMap.count(id) == 0)
                throw std::runtime_error("Can't find specified ID");

            return _pairIdMap.at(id);
        }

        template <typename T>
        Variable<T>* ExecutionResult<T>::byId(int id) {
            std::pair<int,int> p(id, 0);
            return byId(p);
        }

        template <typename T>
        Variable<T>* ExecutionResult<T>::byId(const char *str) {
            std::string p(str);
            return byId(p);
        }

        template <typename T>
        flatbuffers::Offset<FlatResult> ExecutionResult<T>::asFlatResult(flatbuffers::FlatBufferBuilder &builder) {

            std::vector<flatbuffers::Offset<FlatVariable>> vec;
            for (Variable<T>* v : _variables) {
                vec.emplace_back(v->asFlatVariable(builder));
            }

            auto vecOffset = builder.CreateVector(vec);

            return CreateFlatResult(builder, 0, vecOffset);
        }


        template class ND4J_EXPORT ExecutionResult<float>;
        template class ND4J_EXPORT ExecutionResult<float16>;
        template class ND4J_EXPORT ExecutionResult<double>;
    }
}