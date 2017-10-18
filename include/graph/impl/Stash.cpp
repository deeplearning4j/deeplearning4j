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
    }
}