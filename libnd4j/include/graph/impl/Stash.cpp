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


namespace std {
    size_t hash<sd::graph::KeyPair>::operator()(const sd::graph::KeyPair& k) const {
        using std::hash;
        auto res = std::hash<std::string>()(k.name());
        res ^= std::hash<int>()(k.key()) + 0x9e3779b9 + (res << 6) + (res >> 2);
        return res;
    }
}

namespace sd {
    namespace graph {
        sd::graph::KeyPair::KeyPair(int node, const char * name) {
            _node = node;
            _name = std::string(name);
        }

        bool sd::graph::KeyPair::operator<(const KeyPair& other) const {
            if (_node < other._node)
                return true;
            else if (_node > other._node)
                return false;
            else
                return _name < other._name;
        }

        sd::graph::Stash::Stash() {
            //
        }

        sd::graph::Stash::~Stash() {
            if (_handles.size() > 0)
                this->clear();
        }

/*
bool sd::graph::Stash::checkStash(sd::graph::Block& block, const char *name) {
    return checkStash(block.getNodeId(), name);
}
 */

        bool sd::graph::Stash::checkStash(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash.count(kp) > 0;
        }

/*
sd::NDArray* sd::graph::Stash::extractArray(sd::graph::Block& block, const char *name) {
    return extractArray(block.getNodeId(), name);
}
*/
        sd::NDArray* sd::graph::Stash::extractArray(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash[kp];
        }
/*
void sd::graph::Stash::storeArray(sd::graph::Block& block, const char *name, sd::NDArray *array) {
    storeArray(block.getNodeId(), name, array);
}
*/

        void sd::graph::Stash::storeArray(int nodeId, const char *name, sd::NDArray *array) {
            KeyPair kp(nodeId, name);
            _stash[kp] = array;

            // storing reference to delete it once it's not needed anymore
            _handles.push_back(array);
        }

        void sd::graph::Stash::clear() {
            for (auto v: _handles)
                delete v;

            _handles.clear();
            _stash.clear();
        }
    }
}