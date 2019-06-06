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
// Created by raver119 on 16/11/17.
//

#include <graph/NodeState.h>

namespace nd4j {
    namespace graph {
        NodeState::NodeState(int id) {
            _id = id;
        }

        void NodeState::setInnerTime(Nd4jLong time) {
            _inner = time;
        }

        void NodeState::setOuterTime(Nd4jLong time) {
            _outer = time;
        }

        Nd4jLong NodeState::innerTime() {
            return _inner;
        }

        Nd4jLong NodeState::outerTime() {
            return _outer;
        }

        void NodeState::markActive(bool isActive) {
            _active = isActive;
        }

        bool NodeState::isActive() {
            return _active;
        }

        int NodeState::branch() {
            return _branch;
        }

        void NodeState::markBranch(int index) {
            _branch = index;
        }

        bool NodeState::wasExecuted() {
            return _executed;
        }

        void NodeState::markExecuted(bool wasExecuted) {
            _executed = wasExecuted;
        }
    }
}