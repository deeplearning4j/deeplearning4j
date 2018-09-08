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

#include <Context.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace graph {
        Context::Context(ContextPrototype* prototype, VariableSpace* variableSpace) {
            _variableSpace = variableSpace;

            if (prototype != nullptr) {
                for (const auto &v: *(prototype->inputs())) {
                    this->_inputs.push_back(v);
                }

                for (const auto &v: *(prototype->getTArguments())) {
                    this->_tArgs.push_back(v);
                }

                for (const auto &v: *(prototype->getIArguments())) {
                    this->_iArgs.push_back(v);
                }

                this->_opNum = prototype->opNum();
                this->_isInplace = prototype->isInplace();
                this->_nodeId = prototype->nodeId();
            }


            if (variableSpace != nullptr && variableSpace->workspace() != nullptr)
                    this->_workspace = variableSpace->workspace();
        }

        Context::Context(int nodeId, VariableSpace *variableSpace) {
            this->_nodeId = nodeId;
            this->_variableSpace = variableSpace;
            this->_isInplace = false;
            this->_workspace = nullptr;

            this->_executionTime.first = 0;
            this->_executionTime.second = 0;

            if (variableSpace != nullptr)
                this->_rng = variableSpace->getRNG();

            if (variableSpace != nullptr && variableSpace->workspace() != nullptr)
                this->_workspace = variableSpace->workspace();
        }

        Context::Context(int nodeId, VariableSpace *variableSpace, bool isInplace) : Context(nodeId, variableSpace) {
            this->_isInplace = isInplace;
        }

        Context::~Context() {
            this->_iArgs.clear();
            this->_tArgs.clear();
            this->_inputs.clear();
        }

        bool Context::hasWorkspaceProvided() {
            return this->_workspace != nullptr;
        }

        void Context::attachWorkspace(nd4j::memory::Workspace* workspace) {
            this->_workspace = workspace;
        }

        void Context::setVariableSpace(VariableSpace *variableSpace) {
            this->_variableSpace = variableSpace;

            if (variableSpace != nullptr)
                this->_rng = variableSpace->getRNG();
        }

        void Context::forgetWorkspace() {
            _workspace = nullptr;
        }

        VariableSpace *Context::getVariableSpace() {
            return _variableSpace;
        }

        nd4j::memory::Workspace* Context::getWorkspace() {
            return _workspace;
        }

        nd4j::memory::Workspace* Context::workspace() {
            return _workspace;
        }

        nd4j::random::RandomBuffer* Context::getRNG() {
            return _rng;
        }

        void Context::setRNG(nd4j::random::RandomBuffer* rng) {
            _rng = rng;
        }

        /**
         * This method returns variableSpace used in this block
         * @return
         */
    /*
        VariableSpace* Context::getVariableSpace() {
            return _variableSpace;
        }
*/

        Stash* Context::getStash() {
            return _variableSpace->getStash();
        }

        void Context::trackList(NDArrayList* list) {
            _variableSpace->trackList(list);
        }

/*
        void Block::updateVariables() {
            _variables.clear();
            auto x = _inputs.size();
            for (auto &v:_inputs) {
                auto var = _variableSpace->getVariable(v);
                _variables.emplace_back(var);
            }
        }
*/
        int Context::getBranch() {
            return _variableSpace->flowPath()->branch(this->nodeId());
        }

        void Context::setBranch(int branch) {
            //_branch = branch;
            if (_variableSpace->flowPath() != nullptr)
                _variableSpace->flowPath()->markBranch(this->nodeId(), branch);
        }

        Nd4jLong nd4j::graph::Context::getOuterTime(){
            return this->_executionTime.first;
        }

        Nd4jLong nd4j::graph::Context::getInnerTime(){
            return this->_executionTime.second;
        }

        void nd4j::graph::Context::setOuterTime(Nd4jLong time){
            this->_executionTime.first = time;
        }

        void nd4j::graph::Context::setInnerTime(Nd4jLong time){
            this->_executionTime.second = time;
        }


        Variable* Context::getVariable(int idx) {
            if (idx >= this->_inputs.size()) {
                nd4j_printf("Node %i; Variable [%i] requested, but only %i inputs available\n", this->_nodeId, idx, this->_inputs.size());
                throw std::runtime_error("Bad index");
            }

            auto p = this->_inputs[idx];

            auto v = variable(p);

            if (Environment::getInstance()->isDebugAndVerbose() && v != nullptr &&  v->getNDArray() != nullptr) {
                auto array = v->getNDArray();
                std::string shape_ = ShapeUtils::shapeAsString(array);

                float m = std::numeric_limits<float>::quiet_NaN();
                if (!array->isEmpty()) {
                    auto values = array->asIndexedString(16);
                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: %i; order: %i; first values: %s\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), values.c_str());
                } else {
                    nd4j_printf("Debug info for node_%i input[%i]; shape: %s; ews: %i; order: %i; mean value: [%f]\n", this->_nodeId, idx, shape_.c_str(), array->ews(), array->ordering(), m);
                }
            }

            return v;
        }

        Variable* Context::variable(int idx) {
            return getVariable(idx);
        }

        Variable* Context::variable(std::initializer_list<int> p) {
            if (p.size() != 2)
                throw std::runtime_error("Variable address should have size of 2");

            // FIXME: lol
            std::vector<int> vec(p);
            std::pair<int, int> pair(vec[0], vec[1]);
            return variable(pair);
        }

        Variable* Context::variable(int node, int idx) {
            std::pair<int, int> pair(node, idx);
            return variable(pair);
        }

        Variable* Context::variable(std::pair<int,int>& p) {
            if (!_variableSpace->hasVariable(p)) {
                nd4j_printf("Node %i; Non-existent variable requested: [%i:%i]\n", this->_nodeId, p.first, p.second);
                throw std::runtime_error("Bad variable");
            }

            return _variableSpace->getVariable(p);
        }

        void Context::pushNDArrayToVariableSpace(int nodeId, int index, NDArray *array, bool removable) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayToVariableSpace(pair, array, removable);
        }

        void Context::pushNDArrayToVariableSpace(std::pair<int, int> &pair, NDArray *array, bool removable) {
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable(array, nullptr, pair.first, pair.second);
                _variableSpace->putVariable(pair, var);
                var->markRemovable(removable);
            } else {
                auto var = _variableSpace->getVariable(pair);
                if (var->getNDArray() != array) {
                    if (var->isRemovable() && var->getNDArray() != nullptr)
                        delete var->getNDArray();

                    var->setNDArray(array);
                    var->markRemovable(removable);
                }
            }
        }

        void Context::pushNDArrayListToVariableSpace(int nodeId, int index, NDArrayList* list, bool track) {
            std::pair<int,int> pair(nodeId, index);
            pushNDArrayListToVariableSpace(pair, list, track);
        }
        
        void Context::pushNDArrayListToVariableSpace(std::pair<int, int>& pair, NDArrayList* list, bool track) {
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable(nullptr, nullptr, pair.first, pair.second);
                var->setNDArrayList(list);
                _variableSpace->putVariable(pair, var);
            } else {
                auto var = _variableSpace->getVariable(pair);
                var->setNDArrayList(list);
            }

            if (track)
                _variableSpace->trackList(list);
        }

        Variable* Context::ensureVariable(int idx) {
            std::pair<int, int> pair(this->nodeId(), idx);
            if (!_variableSpace->hasVariable(pair)) {
                auto var = new Variable(nullptr, nullptr, this->nodeId(), idx);
                _variableSpace->putVariable(pair, var);
                return var;
            } else {
                return _variableSpace->getVariable(pair);
            }
        }

        bool Context::isValueAvailable(int idx) {
            auto var = ensureVariable(idx);

            if (var->variableType() == VariableType::NDARRAY) {
                return var->getNDArray() != nullptr;
            } else if (var->variableType() == VariableType::ARRAY_LIST) {
                return var->getNDArrayList() != nullptr;
            }

            return false;
        }

        nd4j::memory::Workspace *Context::fWorkspace() {
            return workspace();
        }

        nd4j::memory::Workspace *Context::tWorkspace() {
            return nullptr;
        }

        nd4j::memory::Workspace *Context::oWorkspace() {
            return nullptr;
        }
    }
}

