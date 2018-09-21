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
//  @author raver119@gmail.com
//

#include <pointercast.h>
#include <dll.h>
#include <types/float16.h>
#include <graph/ContextPrototype.h>

namespace nd4j {
    namespace graph {
        ContextPrototype::ContextPrototype(int nodeId, bool inPlace) {
            _nodeId = nodeId;
            _isInplace = inPlace;
        }

        void ContextPrototype::pickInput(std::pair<int, int>& p) {
            this->_inputs.emplace_back(p);
        }

        void ContextPrototype::pickInput(int input, int index) {
            std::pair<int, int> pair(input, index);
            pickInput(pair);
        }

        int ContextPrototype::opNum() {
            return this->_opNum;
        }

        void ContextPrototype::setOpNum(int opNum) {
            this->_opNum = opNum;
        }

        std::vector<std::pair<int, int>>* ContextPrototype::inputs() {
            return &_inputs;
        }

        void ContextPrototype::fillInputs(std::vector<int>& inputs) {
            for (int e = 0; e < inputs.size(); e++) {
                auto v = inputs.at(e);
                pickInput(v);
            }
        }

        bool ContextPrototype::hasVariablesFilled() {
            return this->_inputs.size() > 0;
        }

        bool ContextPrototype::isInplace() {
            return this->_isInplace;
        }

        std::vector<double>* ContextPrototype::getTArguments() {
            return &(this->_tArgs);
        }

        std::vector<int>* ContextPrototype::getIArguments() {
            return &(this->_iArgs);
        }

        void ContextPrototype::pickInput(int input) {
            std::pair<int, int> pair(input, 0);
            this->_inputs.emplace_back(pair);
        }

        std::pair<int, int>* ContextPrototype::input(int idx) {
            return &(this->_inputs.at(idx));
        }

        void ContextPrototype::fillInputs(std::initializer_list<int> inputs) {
            for (auto v: inputs) {
                pickInput(v);
            }
        }

        int ContextPrototype::nodeId() {
            return getNodeId();
        }

        nd4j::DataType ContextPrototype::dataType() {
            return _dataType;
        }

        int ContextPrototype::numT() {
            return (int) _tArgs.size();
        }

        int ContextPrototype::numI() {
            return (int) _iArgs.size();
        }

        int ContextPrototype::getNodeId() {
            return this->_nodeId;
        }

        /**
         * This method returns number of inputs available in this block
         * @return
         */
        unsigned long ContextPrototype::width() {
            return this->_inputs.size();
        };

        void ContextPrototype::markInplace(bool reallyInplace) {
            this->_isInplace = reallyInplace;
        }

        template <typename N>
        ContextPrototype* ContextPrototype::asT() {
            auto clone = new ContextPrototype(_nodeId, _isInplace);

            return clone;
        }

        ContextPrototype* ContextPrototype::clone() {
            auto clone = new ContextPrototype(_nodeId, _isInplace);
            clone->_opNum = _opNum;
            
            for (auto v: _inputs)
                clone->_inputs.emplace_back(v);

            for (auto v: _tArgs)
                clone->_tArgs.emplace_back(v);

            for (auto v: _iArgs)
                clone->_iArgs.emplace_back(v);

            return clone;
        }
    }
}