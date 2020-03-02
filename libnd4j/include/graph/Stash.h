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

#ifndef LIBND4J_STASH_H
#define LIBND4J_STASH_H

//#include <graph/Block.h>
#include <array/NDArray.h>
#include <map>
#include <vector>
#include <string>
#include <atomic>
#include <functional>
#include <system/pointercast.h>

namespace sd {
    namespace graph {
        class ND4J_EXPORT KeyPair {
            int _node;
            std::string _name;
        public:
            KeyPair(int node = 0, const char *name = nullptr);

            bool operator<(const KeyPair &other) const;

            bool operator==(const KeyPair &other) const {
                return _node == other._node;
            }

            int key() const { return _node; }
            std::string name() const { return _name; }
        };
    }
}

#ifndef __JAVACPP_HACK__

namespace std {
    template <>
    class ND4J_EXPORT hash<sd::graph::KeyPair> {
    public:
        size_t operator()(const sd::graph::KeyPair& k) const;
    };
};

#endif

namespace sd {
    namespace graph {
        class ND4J_EXPORT Stash {
        protected:
            std::map<sd::graph::KeyPair, sd::NDArray*> _stash;
            std::vector<sd::NDArray*> _handles;

        public:
            Stash();
            ~Stash();

            //void storeArray(sd::graph::Block<T>& block, const char *name, sd::NDArray<T> *array);
            void storeArray(int nodeId, const char *name, sd::NDArray *array);

            //bool checkStash(sd::graph::Block<T>& block, const char *name);
            bool checkStash(int nodeId, const char *name);

            //sd::NDArray<T>* extractArray(sd::graph::Block<T>& block, const char *name);
            sd::NDArray* extractArray(int nodeId, const char *name);

            void clear();
        };
    }

}




#endif //LIBND4J_STASH_H
