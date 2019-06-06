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
// Created by raver119 on 06.02.2018.
//

#include <graph/FrameState.h>


namespace nd4j {
    namespace graph {
        FrameState::FrameState(Nd4jLong id) {
            this->_id = id;
        }

        int FrameState::getNumberOfCycles() {
            return _numberOfCycles;
        }

        void FrameState::incrementNumberOfCycles() {
            ++_numberOfCycles;
        }

        bool FrameState::wasActivated() {
            return _activated;
        }

        void FrameState::markActivated(bool reallyActivated) {
            _activated = reallyActivated;
        }

        std::string &FrameState::getFrameName() {
            return _name;
        }

        bool FrameState::isRewindPlanned() {
            return _rewindPlanned;
        }

        int FrameState::getRewindPosition() {
            return _rewindPosition;
        }

        void FrameState::setRewindPosition(int pos) {
            _rewindPosition = pos;
        }

        void FrameState::setRewindPositionOnce(int pos) {
            if (_rewindPosition < 0)
                _rewindPosition = pos;
        }

        void FrameState::planRewind(bool reallyPlanning) {
            _rewindPlanned = reallyPlanning;
        }
    }
}