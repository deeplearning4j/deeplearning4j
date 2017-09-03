//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <string>
#include <list>
#include <map>
#include <mutex>
//#include <NDArray.h>
#include <graph/Variable.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        class VariableSpace {
        protected:
            std::map<std::pair<int, int>, nd4j::graph::Variable<T> *> _paired;
            std::map<std::string, nd4j::graph::Variable<T> *> _symbolic;
            std::map<const int32_t, nd4j::graph::Variable<T> *> _variables;
            std::vector<nd4j::graph::Variable<T> *> _external;
            std::vector<nd4j::graph::Variable<T> *> _internal;

            int _auto_counter = -1;

            std::mutex _varmap;

            std::map<const int32_t, nd4j::graph::Variable<T> *> _temporary;
        public:
            VariableSpace();
            ~VariableSpace();

            bool hasVariable(int32_t id);
            bool hasVariable(std::pair<int,int>& pair);
            bool hasVariable(std::string *symbol);

            nd4j::graph::Variable<T> *getVariable(const int32_t id);
            nd4j::graph::Variable<T> *getVariable(std::pair<int,int>& pair);
            nd4j::graph::Variable<T> *getVariable(std::string *symbol);

            void putVariable(std::pair<int,int>& pair, Variable<T> *variable);
            void putVariable(int32_t id, Variable<T> *variable);
            void putVariable(int32_t id, NDArray<T> *array);

            void putOutputVariable(Variable<T> *variable);

            // memory-related statistics
            Nd4jIndex externalMemory();
            Nd4jIndex internalMemory();
            Nd4jIndex totalMemory();

            int externalEntries();
            int internalEntries();
            int totalEntries();

            std::vector<nd4j::graph::Variable<T> *> * getExternalVariables() {
                return &_external;
            }
        };
    }
}

template <typename T>
bool nd4j::graph::VariableSpace<T>::hasVariable(std::string *symbol) {
    return _symbolic.count(*symbol) == 1;
}

template <typename T>
nd4j::graph::Variable<T> * nd4j::graph::VariableSpace<T>::getVariable(std::string *symbol) {
    return _symbolic.at(*symbol);
}

template <typename T>
nd4j::graph::Variable<T> * nd4j::graph::VariableSpace<T>::getVariable(std::pair<int, int>& pair) {
    if (_paired.count(pair) > 0)
        return _paired.at(pair);
    else {
        if (hasVariable(pair.first) && pair.second == 0)
            return getVariable(pair.first);
    }

    return nullptr;
}

template <typename T>
bool nd4j::graph::VariableSpace<T>::hasVariable(int32_t id) {
    return _variables.count(id) == 1 || _temporary.count(id) == 1;
}

template <typename T>
bool nd4j::graph::VariableSpace<T>::hasVariable(std::pair<int,int>& id) {
    return _paired.count(id) > 0;
}

template <typename T>
void nd4j::graph::VariableSpace<T>::putOutputVariable(Variable<T> *variable) {
    //putVariable(_auto_counter--, variable);
    putVariable(variable->id(), variable);
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
void nd4j::graph::VariableSpace<T>::putVariable(std::pair<int,int>& pair, Variable<T> *variable) {
    _varmap.lock();

    std::pair<std::pair<int, int>, nd4j::graph::Variable<T> *> p(pair, variable);
    _paired.insert(p);

    _varmap.unlock();

    // copying duplicate for compatibility
    if (pair.second == 0 && !this->hasVariable(pair.first)) {
        this->putVariable(pair.first, variable);
    }


}

template <typename T>
void nd4j::graph::VariableSpace<T>::putVariable(const int32_t id, Variable<T> *variable) {

    // we don't want to add variables more then once
    if (_variables.count(id) > 0 || _temporary.count(id) > 0) {
        nd4j_verbose("Trying to update variable for node_%i\n", id);

        auto local = id < 0 ? _variables.at(id) : _temporary.at(id);

        if (local->getNDArray() == nullptr && variable->getNDArray() != nullptr) {
            nd4j_verbose("Saving variable for node_%i\n", id);
            local->setNDArray(variable->getNDArray());
        }
        return;
    }

    nd4j_verbose("Adding Variable to Space: id: %i; Array is null: %i;\n", id, variable->getNDArray() == nullptr);

    _varmap.lock();

    if (_auto_counter >= id)
        _auto_counter = id - 1;

    variable->setId(id);

    if (variable->getName() != nullptr && variable->getName()->length() != 0) {
        std::pair<std::string, nd4j::graph::Variable<T> *> pair(*(variable->getName()), variable);
        _symbolic.insert(pair);
    }

    // we have special list for external variables to ensure graph completeness
    if (id < 0) {
        //if (variable->isExternal())
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
