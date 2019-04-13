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

#include <graph/Stash.h>

namespace nd4j {
    namespace graph {
        nd4j::graph::KeyPair::KeyPair(int node, const char * name) {
            _node = node;
            _name = std::string(name);
        }

        bool nd4j::graph::KeyPair::operator<(const KeyPair& other) const {
            if (_node < other._node)
                return true;
            else if (_node > other._node)
                return false;
            else
                return _name < other._name;
        }

        nd4j::graph::Stash::Stash() {
            //
        }

        nd4j::graph::Stash::~Stash() {
            if (_handles.size() > 0)
                this->clear();
        }

/*
bool nd4j::graph::Stash::checkStash(nd4j::graph::Block& block, const char *name) {
    return checkStash(block.getNodeId(), name);
}
 */

        bool nd4j::graph::Stash::checkStash(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash.count(kp) > 0;
        }

/*
nd4j::NDArray* nd4j::graph::Stash::extractArray(nd4j::graph::Block& block, const char *name) {
    return extractArray(block.getNodeId(), name);
}
*/
        nd4j::NDArray* nd4j::graph::Stash::extractArray(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash[kp];
        }
/*
void nd4j::graph::Stash::storeArray(nd4j::graph::Block& block, const char *name, nd4j::NDArray *array) {
    storeArray(block.getNodeId(), name, array);
}
*/

        void nd4j::graph::Stash::storeArray(int nodeId, const char *name, nd4j::NDArray *array) {
            KeyPair kp(nodeId, name);
            _stash[kp] = array;

            // storing reference to delete it once it's not needed anymore
            _handles.push_back(array);
        }

        void nd4j::graph::Stash::clear() {
            for (auto v: _handles)
                delete v;

            _handles.clear();
            _stash.clear();
        }
    }
}