//
// Created by raver119 on 15/11/17.
//

#include <graph/VariablesSet.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        Nd4jStatus VariablesSet<T>::status() {
            return _status;
        }

        template <typename T>
        int VariablesSet<T>::size() {
            return _holder.size();
        }

        template <typename T>
        void VariablesSet<T>::push_back(Variable<T> *variable) {
            _holder.push_back(variable);
        }

        template <typename T>
        Variable<T> *VariablesSet<T>::at(int index) {
            return _holder.at(index);
        }

        template<typename T>
        VariablesSet<T>::VariablesSet(Nd4jStatus status) {
            _status = status;
        }

        template<typename T>
        VariablesSet<T>::~VariablesSet() {
            for (auto v: _holder)
                delete v;
        }

        template class ND4J_EXPORT VariablesSet<float>;
        template class ND4J_EXPORT VariablesSet<float16>;
        template class ND4J_EXPORT VariablesSet<double>;
    }
}
