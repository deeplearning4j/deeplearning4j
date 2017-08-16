//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <string>
#include <list>
#include <map>
#include <mutex>
#include <NDArray.h>
#include <graph/Variable.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        class VariableSpace {
        protected:
            std::map<const int32_t, nd4j::graph::Variable<T> *> _variables;
            std::list<nd4j::graph::Variable<T> *> _external;
            std::list<nd4j::graph::Variable<T> *> _internal;

            std::mutex _varmap;

            std::map<const int32_t, nd4j::graph::Variable<T> *> _temporary;
        public:
            VariableSpace();
            ~VariableSpace();

            nd4j::graph::Variable<T> *getVariable(const int32_t id);
            void putVariable(int32_t id, Variable<T> *variable);
            void putVariable(int32_t id, NDArray<T> *array);

            // memory-related statistics
            Nd4jIndex externalMemory();
            Nd4jIndex internalMemory();
            Nd4jIndex totalMemory();

            int externalEntries();
            int internalEntries();
            int totalEntries();
        };
    }
}

template <typename T>
int nd4j::graph::VariableSpace<T>::externalEntries() {
    return _external.size();
}

template <typename T>
int nd4j::graph::VariableSpace<T>::internalEntries() {
    return _internal.size();
}

template <typename T>
int nd4j::graph::VariableSpace<T>::totalEntries() {
    return externalEntries() + internalEntries();
}

template <typename T>
Nd4jIndex nd4j::graph::VariableSpace<T>::externalMemory() {
    Nd4jIndex size = 0;
    for (auto n: _external) {
        size += n->getNDArray()->memoryFootprint();
    }

    return size;
}

template <typename T>
Nd4jIndex nd4j::graph::VariableSpace<T>::internalMemory() {
    Nd4jIndex size = 0;
    for (auto n: _internal) {
        size += n->getNDArray()->memoryFootprint();
    }

    return size;
}

template <typename T>
Nd4jIndex nd4j::graph::VariableSpace<T>::totalMemory() {
    return externalMemory() + internalMemory();
}

template <typename T>
void nd4j::graph::VariableSpace<T>::putVariable(const int32_t id, Variable<T> *variable) {

    _varmap.lock();

    // we have special list for external variables to ensure graph completeness
    if (id < 0) {
        if (variable->isExternal())
            _external.push_back(variable);

        std::pair<const int32_t, nd4j::graph::Variable<T> *> pair(id, variable);
        _variables.insert(pair);
    } else {
        _internal.push_back(variable);

        std::pair<const int32_t, nd4j::graph::Variable<T> *> pair(id, variable);
        _temporary.insert(pair);
    }

    _varmap.unlock();
}

template <typename T>
void nd4j::graph::VariableSpace<T>::putVariable(const int32_t id, NDArray<T> *array) {
    nd4j::graph::Variable<T> *var = new nd4j::graph::Variable<T>(array);
    this->putVariable(id, var);
}

template <typename T>
nd4j::graph::Variable<T> * nd4j::graph::VariableSpace<T>::getVariable(const int32_t id) {
    _varmap.lock();

    if (id < 0) {
        auto  v = _variables.at(id);
        _varmap.unlock();

        return v;
    } else {
        auto v = _temporary.at(id);
        _varmap.unlock();

        return v;
    }
}

/*
 * FIXME: this thing have nice chances to become backend-specific!
 */
template <typename T>
nd4j::graph::VariableSpace<T>::~VariableSpace() {
    // loop through variables and release them
    for (auto p: _variables) {
        delete p.second;
    }
}


template <typename T>
nd4j::graph::VariableSpace<T>::VariableSpace() {

}



#endif //LIBND4J_VARIABLESPACE_H
