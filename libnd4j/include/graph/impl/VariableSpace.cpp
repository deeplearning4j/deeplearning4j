/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <legacy/NativeOps.h>

namespace sd {
    namespace graph {
        std::vector<sd::graph::Variable *> * sd::graph::VariableSpace::getExternalVariables() {
            return &_external;
        }

        sd::graph::Stash* sd::graph::VariableSpace::getStash() {
            return &_stash;
        }

        sd::graph::VariableSpace* sd::graph::VariableSpace::clone() {
            auto result = new VariableSpace();

            for (auto const& x : _paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                Variable* clonedVar = x.second->clone();

                result->injectVariable(pair, clonedVar);
            }

            return result;
        }

        void VariableSpace::setWorkspace(sd::memory::Workspace *workspace) {
            //_workspace = *workspace;
        }

        
        sd::graph::VariableSpace* sd::graph::VariableSpace::asT() {
            auto result = new VariableSpace();

            for (auto const& x : _paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                //Variable* clonedVar = x.second->template asT<N>();

                //result->injectVariable(pair, clonedVar);
            }

            return result;
        }

        
        void sd::graph::VariableSpace::injectVariable(std::pair<int, int> &pair, Variable* variable) {
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

        std::vector<sd::graph::Variable*> * sd::graph::VariableSpace::getPlaceholders() {
            return &_placeholders;
        }

        int sd::graph::VariableSpace ::numberOfPlaceholders() {
            return _placeholders.size();
        }

        bool sd::graph::VariableSpace::hasVariable(std::string *symbol) {
            return _symbolic.count(*symbol) == 1;
        }

        sd::graph::Variable * sd::graph::VariableSpace::getVariable(std::string *symbol) {
            return _symbolic.at(*symbol);
        }

        bool sd::graph::VariableSpace::hasVariable(int id, int index) {
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

        sd::graph::Variable * sd::graph::VariableSpace::getVariable(int id, int index) {
            std::pair<int, int> pair(id, index);
            return getVariable(pair);
        }

        sd::graph::Variable * sd::graph::VariableSpace::getVariable(std::pair<int, int>& pair) {
            if (pair.first < 0)
                return getVariable(pair.first);
            else
                return _paired.at(pair);

            nd4j_printf("Unknown variable requested: [%i,%i]\n", pair.first, pair.second);
            throw std::runtime_error("Unknown variable requested");
        }

        bool sd::graph::VariableSpace::hasVariable(int id) {
            return _variables.count(id) == 1 || _temporary.count(id) == 1;
        }

        bool sd::graph::VariableSpace::hasVariable(std::pair<int,int>& id) {
            return _paired.count(id) > 0;
        }

        void sd::graph::VariableSpace::putOutputVariable(Variable *variable) {
            //putVariable(_auto_counter--, variable);
            putVariable(variable->id(), variable);
        }

        int sd::graph::VariableSpace::externalEntries() {
            return _external.size();
        }

        int sd::graph::VariableSpace::internalEntries() {
            return _internal.size();
        }

        int sd::graph::VariableSpace::totalEntries() {
            return externalEntries() + internalEntries();
        }

        Nd4jLong sd::graph::VariableSpace::externalMemory() {
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

        Nd4jLong sd::graph::VariableSpace::internalMemory() {
            Nd4jLong size = 0;
            for (auto n: _internal) {
                size += n->getNDArray()->memoryFootprint();
            }

            return size;
        }

        Nd4jLong sd::graph::VariableSpace::totalMemory() {
            return externalMemory() + internalMemory();
        }

        Variable* sd::graph::VariableSpace::putVariable(std::pair<int,int>& pair, NDArray *array) {
            auto variable = new Variable(array, nullptr, pair.first, pair.second);
            this->putVariable(pair, variable);
            return variable;
        }

        Variable* sd::graph::VariableSpace::putVariable(int node, int idx, NDArray *array) {
            std::pair<int, int> pair(node, idx);
            return this->putVariable(pair, array);
        }

        void sd::graph::VariableSpace::putVariable(int node, int idx, Variable *variable) {
            std::pair<int, int> pair(node, idx);
            this->putVariable(pair, variable);
        }

        void sd::graph::VariableSpace::silentPutVariable(std::pair<int,int>& pair, Variable *variable) {
            _varmap.lock();

            //std::pair<std::pair<int, int>, sd::graph::Variable *> p(pair, variable);
            _paired[pair] = variable;

            _varmap.unlock();
        }

        void sd::graph::VariableSpace::putVariable(std::pair<int,int>& pair, Variable *variable) {
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

        void VariableSpace::trackList(sd::NDArrayList* list) {
            _lists.emplace_back(list);
        }

        void sd::graph::VariableSpace::putVariable(int id, Variable *variable) {
            // we don't want to add variables more then once
            if (_variables.count(id) > 0 || _temporary.count(id) > 0) {
                auto local = id < 0 ? _variables.at(id) : _temporary.at(id);

                if (!local->hasNDArray() && variable->hasNDArray()) {
                    local->setNDArray(variable->getNDArray());

                    // we're inheriting this from Variable
                    local->markReadOnly(variable->isReadOnly());
                    local->markRemovable(variable->isRemovable());
                }

                return;
            }

            _varmap.lock();

            _handles->emplace_back(variable);

            if (_auto_counter >= id)
                _auto_counter = id - 1;

            variable->setId(id);

            if (variable->getName() != nullptr && variable->getName()->length() != 0) {
                //std::pair<std::string, sd::graph::Variable *> pair(*(variable->getName()), variable);
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

        void sd::graph::VariableSpace::putVariable(int id, int idx, NDArray &array) {
            auto *var = new sd::graph::Variable(&array, "", id, idx);
            var->markRemovable(false);
            var->markReadOnly(true);

            // let's see if this op needs
            bool d = this->hasVariable(id, idx);

            this->putVariable(id, var);

            // if var for this nodeid already exists - we'll just delete variable
            if (d)
                delete var;
        }

        void sd::graph::VariableSpace::putVariable(int id, NDArray *array) {
            auto *var = new sd::graph::Variable(array);
            this->putVariable(id, var);
        }

        sd::graph::Variable * sd::graph::VariableSpace::getVariable(int id) {
            if (id < 0) {
                return _variables.at(id);
            } else {
                return _temporary.at(id);
            }
        }

        LaunchContext* sd::graph::VariableSpace::launchContext() {
            return LaunchContext::defaultContext();
        }

        std::vector<Variable*>* sd::graph::VariableSpace::handles() {
            return _handles;
        }

/*
 * FIXME: this thing have nice chances to become backend-specific!
 */
        sd::graph::VariableSpace::~VariableSpace() {
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