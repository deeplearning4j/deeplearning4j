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

#include <graph/VariableSpace.h>
#include <NativeOps.h>

namespace nd4j {
    namespace graph {
        std::vector<nd4j::graph::Variable *> * nd4j::graph::VariableSpace::getExternalVariables() {
            return &_external;
        }

        nd4j::graph::Stash* nd4j::graph::VariableSpace::getStash() {
            return &_stash;
        }

        nd4j::graph::VariableSpace* nd4j::graph::VariableSpace::clone() {
            auto result = new VariableSpace();

            for (auto const& x : _paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                Variable* clonedVar = x.second->clone();

                result->injectVariable(pair, clonedVar);
            }

            return result;
        }

        void VariableSpace::setWorkspace(nd4j::memory::Workspace *workspace) {
            //_workspace = *workspace;
        }

        
        nd4j::graph::VariableSpace* nd4j::graph::VariableSpace::asT() {
            auto result = new VariableSpace();

            for (auto const& x : _paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                //Variable* clonedVar = x.second->template asT<N>();

                //result->injectVariable(pair, clonedVar);
            }

            return result;
        }

        
        void nd4j::graph::VariableSpace::injectVariable(std::pair<int, int> &pair, Variable* variable) {
            if (pair.second == 0) {
                if (pair.first < 0)
                    this->_variables[pair.first] = variable;
                else
                    this->_temporary[pair.first] = variable;
            }

            if (variable->getName() != nullptr && variable->getName()->length() > 0)
                this->_symbolic[*(variable->getName())] = variable;

            this->_paired[pair] = variable;

            this->_handles->push_back(variable);
        }

        std::vector<nd4j::graph::Variable*> * nd4j::graph::VariableSpace::getPlaceholders() {
            return &_placeholders;
        }

        int nd4j::graph::VariableSpace ::numberOfPlaceholders() {
            return _placeholders.size();
        }

        bool nd4j::graph::VariableSpace::hasVariable(std::string *symbol) {
            return _symbolic.count(*symbol) == 1;
        }

        nd4j::graph::Variable * nd4j::graph::VariableSpace::getVariable(std::string *symbol) {
            return _symbolic.at(*symbol);
        }

        bool nd4j::graph::VariableSpace::hasVariable(int id, int index) {
            std::pair<int, int> pair(id, index);
            return hasVariable(pair);
        }

        bool VariableSpace::hasExternalVariable(int id) {
            if (!hasVariable(id))
                return false;

            auto var = getVariable(id);
            return var->isExternal();
        }

        bool VariableSpace::hasExternalVariable(std::pair<int,int>& pair) {
            if (!hasVariable(pair))
                return false;

            auto var = getVariable(pair);
            return var->isExternal();
        }

        bool VariableSpace::hasExternalVariable(std::string *symbol) {
            if (!hasVariable(symbol))
                return false;

            auto var = getVariable(symbol);
            return var->isExternal();
        }

        nd4j::graph::Variable * nd4j::graph::VariableSpace::getVariable(int id, int index) {
            std::pair<int, int> pair(id, index);
            return getVariable(pair);
        }

        nd4j::graph::Variable * nd4j::graph::VariableSpace::getVariable(std::pair<int, int>& pair) {
//            if (pair.first == 0)
//                throw "0 requested";

            //nd4j_debug("Requested variable: [%i:%i]\n", pair.first, pair.second);

            if (pair.first < 0)
                return getVariable(pair.first);
            else if (_paired.count(pair) > 0)
                return _paired.at(pair);
            else {
                if (hasVariable(pair.first) && pair.second == 0)
                    return getVariable(pair.first);
            }

            nd4j_printf("Unknown variable requested: [%i,%i]\n", pair.first, pair.second);

            return nullptr;
        }

        bool nd4j::graph::VariableSpace::hasVariable(int id) {
            return _variables.count(id) == 1 || _temporary.count(id) == 1;
        }

        bool nd4j::graph::VariableSpace::hasVariable(std::pair<int,int>& id) {
            return _paired.count(id) > 0;
        }

        void nd4j::graph::VariableSpace::putOutputVariable(Variable *variable) {
            //putVariable(_auto_counter--, variable);
            putVariable(variable->id(), variable);
        }

        int nd4j::graph::VariableSpace::externalEntries() {
            return _external.size();
        }

        int nd4j::graph::VariableSpace::internalEntries() {
            return _internal.size();
        }

        int nd4j::graph::VariableSpace::totalEntries() {
            return externalEntries() + internalEntries();
        }

        Nd4jLong nd4j::graph::VariableSpace::externalMemory() {
            Nd4jLong size = 0;
            for (auto n: _external) {
                size += n->getNDArray()->memoryFootprint();
            }

            return size;
        }

        std::vector<Variable*> VariableSpace::getVariables() {
            std::vector<Variable*> result;

            for (auto v: _internal)
                result.emplace_back(v);

            for (auto v: _external)
                result.emplace_back(v);

            return result;
        }

        Nd4jLong nd4j::graph::VariableSpace::internalMemory() {
            Nd4jLong size = 0;
            for (auto n: _internal) {
                size += n->getNDArray()->memoryFootprint();
            }

            return size;
        }

        Nd4jLong nd4j::graph::VariableSpace::totalMemory() {
            return externalMemory() + internalMemory();
        }

        void nd4j::graph::VariableSpace::putVariable(std::pair<int,int>& pair, NDArray *array) {
            auto variable = new Variable(array, nullptr, pair.first, pair.second);
            this->putVariable(pair, variable);
        }

        void nd4j::graph::VariableSpace::putVariable(int node, int idx, NDArray *array) {
            std::pair<int, int> pair(node, idx);
            this->putVariable(pair, array);
        }

        void nd4j::graph::VariableSpace::putVariable(int node, int idx, Variable *variable) {
            std::pair<int, int> pair(node, idx);
            this->putVariable(pair, variable);
        }

        void nd4j::graph::VariableSpace::silentPutVariable(std::pair<int,int>& pair, Variable *variable) {
            _varmap.lock();

            //std::pair<std::pair<int, int>, nd4j::graph::Variable *> p(pair, variable);
            _paired[pair] = variable;

            _varmap.unlock();
        }

        void nd4j::graph::VariableSpace::putVariable(std::pair<int,int>& pair, Variable *variable) {
            silentPutVariable(pair, variable);

            if (variable->isPlaceholder())
                _placeholders.push_back(variable);

            // copying duplicate for compatibility
            if (pair.second == 0 && !this->hasVariable(pair.first)) {
                this->putVariable(pair.first, variable);
            } else {
                if (variable->getName() != nullptr && variable->getName()->length() != 0) {
                    _symbolic[*(variable->getName())] = variable;
                }

                _varmap.lock();

                _handles->push_back(variable);

                _varmap.unlock();
            }
        }

        void VariableSpace::trackList(nd4j::NDArrayList* list) {
            _lists.emplace_back(list);
        }

        void nd4j::graph::VariableSpace::putVariable(int id, Variable *variable) {
            // we don't want to add variables more then once
            if (_variables.count(id) > 0 || _temporary.count(id) > 0) {
                // nd4j_verbose("Trying to update variable for node_%i\n", id);

                auto local = id < 0 ? _variables.at(id) : _temporary.at(id);

                if (local->getNDArray() == nullptr && variable->getNDArray() != nullptr) {
                    // nd4j_verbose("Saving variable for node_%i\n", id);
                    local->setNDArray(variable->getNDArray());
                }
                return;
            }

            //nd4j_debug("Adding Variable to Space: id: %i; Array is null: %i;\n", id, variable->getNDArray() == nullptr);

            _varmap.lock();

            _handles->emplace_back(variable);

            if (_auto_counter >= id)
                _auto_counter = id - 1;

            variable->setId(id);

            if (variable->getName() != nullptr && variable->getName()->length() != 0) {
                //std::pair<std::string, nd4j::graph::Variable *> pair(*(variable->getName()), variable);
                _symbolic[*(variable->getName())] = variable;
            }

            // we have special list for external variables to ensure graph completeness

            if (id < 0) {
                //if (variable->isExternal())
                _external.push_back(variable);

                _variables[id] = variable;
            } else {
                _internal.push_back(variable);

                _temporary[id] = variable;
            }

            _varmap.unlock();

            std::pair<int,int> pair(id, 0);
            if (!hasVariable(pair)) {
                this->silentPutVariable(pair, variable);

                if (variable->isPlaceholder())
                    _placeholders.push_back(variable);
            }
        }

        void nd4j::graph::VariableSpace::putVariable(int id, NDArray *array) {
            auto *var = new nd4j::graph::Variable(array);
            this->putVariable(id, var);
        }

        nd4j::graph::Variable * nd4j::graph::VariableSpace::getVariable(int id) {
//            _varmap.lock();

            if (id < 0) {
                auto  v = _variables.at(id);
   //             _varmap.unlock();

                return v;
            } else {
                auto v = _temporary.at(id);
    //            _varmap.unlock();

                return v;
            }
        }

        nd4j::memory::Workspace * nd4j::graph::VariableSpace::workspace() {
            return &_workspace;
        }

        std::vector<Variable*>* nd4j::graph::VariableSpace::handles() {
            return _handles;
        }

/*
 * FIXME: this thing have nice chances to become backend-specific!
 */
        nd4j::graph::VariableSpace::~VariableSpace() {
            // loop through variables and release them
            for (auto p: *_handles) {
                delete p;
            }

            delete _handles;

            //_internal.clear();
            //_external.clear();
            //_temporary.clear();

            //nd4j_printf("Number of NDArrayLists in this space: [%i]\n", _lists.size())
            for (auto p: _lists)
                delete p;

            _lists.clear();

            if (_rng != nullptr) {
                delete[] _rng->getBuffer();
                NativeOps nativeOps;
                nativeOps.destroyRandom(_rng);
            }
        }

        VariableSpace& VariableSpace::operator=(const VariableSpace& other) {
            if (this == &other) return *this;

            for (auto const& x : other._paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                Variable* clonedVar = x.second->clone();

                if (pair.second == 0) {
                    if (pair.first < 0)
                        this->_variables[pair.first] = clonedVar;
                    else
                        this->_temporary[pair.first] = clonedVar;
                }

                if (clonedVar->getName() != nullptr && clonedVar->getName()->length() > 0)
                    this->_symbolic[*(clonedVar->getName())] = clonedVar;

                this->_paired[pair] = clonedVar;

                this->_handles->push_back(clonedVar);
            }

            return *this;
        }

        void VariableSpace::replaceVariable(Variable *variable) {
            bool replaced = false;
            // trying name first
            if (variable->getName() != nullptr && !variable->getName()->empty()) {
                nd4j_printf("Trying to replace variable by name: [%s]\n", variable->getName()->c_str());
                if (hasVariable(variable->getName())) {
                    nd4j_printf("Replacing by name: [%s]\n", variable->getName()->c_str());
                    auto vs = getVariable(variable->getName());
                    dropVariable(vs->id(), vs->index());
                    putVariable(vs->id(), vs->index(), variable);
                    //delete vs;
                    replaced = true;
                }
            } else {
                nd4j_printf("Trying to replace variable by id: [%i:%i]\n", variable->id(), variable->index());
                if (hasVariable(variable->id(), variable->index())) {
                    nd4j_printf("Replacing by id: [%i:%i]\n", variable->id(), variable->index());
                    auto vs = getVariable(variable->id(), variable->index());
                    dropVariable(variable->id(), variable->index());
                    putVariable(vs->id(), vs->index(), variable);
                    //delete vs;
                    replaced = true;
                }
            }

            if (!replaced) {
                nd4j_printf("wasn't able to replace variable, putting\n", "");
                putVariable(variable->id(), variable->index(), variable);
            }
        }

        void VariableSpace::dropVariable(std::pair<int,int> &pair) {
            dropVariable(pair.first, pair.second);
        }

        void VariableSpace::dropVariable(int id, int idx) {

        }

        void VariableSpace::setRNG(nd4j::random::RandomBuffer* rng) {
            _rng = rng;
        }

        nd4j::random::RandomBuffer* VariableSpace::getRNG() {
            return _rng;
        }

        void VariableSpace::setFlowPath(FlowPath* flow) {
            _flow = flow;
        }

        FlowPath* VariableSpace::flowPath() {
            return _flow;
        }

        VariableSpace::VariableSpace() {
            _handles = new std::vector<Variable *>;
        }
    }
}