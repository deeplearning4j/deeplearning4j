//
// @author raver119@gmail.com
//

#ifndef LIBND4J_STASH_H
#define LIBND4J_STASH_H

//#include <graph/Block.h>
#include <NDArray.h>
#include <map>
#include <string>
#include <atomic>
#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class KeyPair {
            int _node;
            std::string _name;
        public:
            KeyPair(int node = 0, const char * name = nullptr) {
                _node = node;
                _name = std::string(name);
            }

            bool operator<(const KeyPair& other) const {
                if (_node < other._node)
                    return true;
                else if (_node > other._node)
                    return false;
                else
                    return _name < other._name;
            }
        };

        template <typename T>
        class Stash {
        protected:
            std::map<nd4j::graph::KeyPair, nd4j::NDArray<T>*> _stash;
            std::vector<nd4j::NDArray<T>*> _handles;

        public:
            Stash() {
                //
            }

            ~Stash() {
                if (_handles.size() > 0)
                    this->clear();
            }

            //void storeArray(nd4j::graph::Block<T>& block, const char *name, nd4j::NDArray<T> *array);
            void storeArray(int nodeId, const char *name, nd4j::NDArray<T> *array);

            //bool checkStash(nd4j::graph::Block<T>& block, const char *name);
            bool checkStash(int nodeId, const char *name);

            //nd4j::NDArray<T>* extractArray(nd4j::graph::Block<T>& block, const char *name);
            nd4j::NDArray<T>* extractArray(int nodeId, const char *name);

            void clear();
        };
    }
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



#endif //LIBND4J_STASH_H
