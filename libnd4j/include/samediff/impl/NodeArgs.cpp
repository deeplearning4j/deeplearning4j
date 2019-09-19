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

#include "../NodeArgs.h"

namespace samediff {
    void NodeArgs::addInput(int index, int nodeId, int oIndex) {
        // FIXME: ignoring index here
        _inputs.emplace_back(std::pair<int, int>({nodeId, oIndex}));
    }

    void NodeArgs::addIArg(int index, Nd4jLong value) {
        // FIXME: ignoring index here
        _iArgs.emplace_back(value);
    }

    void NodeArgs::addTArg(int index, double value) {
        // FIXME: ignoring index here
        _tArgs.emplace_back(value);
    }

    void NodeArgs::addBArg(int index, bool value) {
        // FIXME: ignoring index here
        _bArgs.emplace_back(value);
    }

    std::vector<std::pair<int, int>>& NodeArgs::inputs() {
        return _inputs;
    }

    std::vector<Nd4jLong>& NodeArgs::iargs() {
        return _iArgs;
    }

    std::vector<double>& NodeArgs::targs() {
        return _tArgs;
    }

    std::vector<bool>& NodeArgs::bargs() {
        return _bArgs;
    }
}