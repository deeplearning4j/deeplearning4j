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
//  @author raver119@gmail.com
//

#include <system/pointercast.h>
#include <system/dll.h>
#include <types/float16.h>
#include <graph/ContextPrototype.h>

namespace sd {
    namespace graph {
        ContextPrototype::ContextPrototype(sd::ops::OpDescriptor* opDescriptor, int nodeId, bool inPlace) {
            _nodeId = nodeId;
            _isInplace = inPlace;
            _opDescriptor = opDescriptor;
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

        samediff::Engine ContextPrototype::engine() {
            return _engine;
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

        std::vector<bool>* ContextPrototype::getBArguments() {
            return &(this->_bArgs);
        }

        std::vector<int>* ContextPrototype::getAxis() {
            return &(this->_axis);
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

        sd::DataType ContextPrototype::dataType() {
            return dataType(0);
        }

        sd::DataType ContextPrototype::dataType(int index) {
            return _dataType;
        }

        void ContextPrototype::setDataType(int index, sd::DataType type) {
            // if (_outputs->size() == 0)
            _dataType = type;
        }

        size_t ContextPrototype::numT() {
            return (int) _tArgs.size();
        }

        size_t ContextPrototype::numI() {
            return (int) _iArgs.size();
        }

        size_t ContextPrototype::numB() {
            return (int) _bArgs.size();
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
            auto clone = new ContextPrototype(_opDescriptor, _nodeId, _isInplace);

            return clone;
        }

        void ContextPrototype::setOpDescriptor(sd::ops::OpDescriptor* opDescriptor) {
            _opDescriptor = opDescriptor;
        }

        ContextPrototype* ContextPrototype::clone() {
            auto clone = new ContextPrototype(_opDescriptor, _nodeId, _isInplace);
            clone->_opNum = _opNum;
            
            for (auto v: _inputs)
                clone->_inputs.emplace_back(v);

            for (auto v: _tArgs)
                clone->_tArgs.emplace_back(v);

            for (auto v: _iArgs)
                clone->_iArgs.emplace_back(v);

            return clone;
        }

        std::vector<sd::DataType> *ContextPrototype::getDArguments() {
            return &_dArgs;
        }

        size_t ContextPrototype::numD() {
            return _dArgs.size();
        }
    }
}