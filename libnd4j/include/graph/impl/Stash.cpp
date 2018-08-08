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


        template <typename T>
        nd4j::graph::Stash<T>::Stash() {
            //
        }

        template <typename T>
        nd4j::graph::Stash<T>::~Stash() {
            if (_handles.size() > 0)
                this->clear();
        }

/*
template <typename T>
bool nd4j::graph::Stash<T>::checkStash(nd4j::graph::Block<T>& block, const char *name) {
    return checkStash(block.getNodeId(), name);
}
 */

        template <typename T>
        bool nd4j::graph::Stash<T>::checkStash(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash.count(kp) > 0;
        }

/*
template <typename T>
nd4j::NDArray<T>* nd4j::graph::Stash<T>::extractArray(nd4j::graph::Block<T>& block, const char *name) {
    return extractArray(block.getNodeId(), name);
}
*/

        template <typename T>
        nd4j::NDArray<T>* nd4j::graph::Stash<T>::extractArray(int nodeId, const char *name) {
            KeyPair kp(nodeId, name);
            return _stash[kp];
        }
/*
template <typename T>
void nd4j::graph::Stash<T>::storeArray(nd4j::graph::Block<T>& block, const char *name, nd4j::NDArray<T> *array) {
    storeArray(block.getNodeId(), name, array);
}
*/

        template <typename T>
        void nd4j::graph::Stash<T>::storeArray(int nodeId, const char *name, nd4j::NDArray<T> *array) {
            KeyPair kp(nodeId, name);
            _stash[kp] = array;

            // storing reference to delete it once it's not needed anymore
            _handles.push_back(array);
        }

        template <typename T>
        void nd4j::graph::Stash<T>::clear() {
            for (auto v: _handles)
                delete v;

            _handles.clear();
            _stash.clear();
        }


        template class ND4J_EXPORT Stash<float>;
        template class ND4J_EXPORT Stash<float16>;
        template class ND4J_EXPORT Stash<double>;
        template class ND4J_EXPORT Stash<int>;
        template class ND4J_EXPORT Stash<Nd4jLong>;
    }
}