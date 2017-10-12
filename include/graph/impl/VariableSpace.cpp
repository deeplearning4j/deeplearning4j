//
// @author raver119@gmail.com
//

#include <graph/VariableSpace.h>

namespace nd4j {
    namespace graph {
        template<typename T>
        std::vector<nd4j::graph::Variable<T> *> * nd4j::graph::VariableSpace<T>::getExternalVariables() {
            return &_external;
        }

        template <typename T>
        nd4j::graph::Stash<T>* nd4j::graph::VariableSpace<T>::getStash() {
            return &_stash;
        }

        template <typename T>
        nd4j::graph::VariableSpace<T>* nd4j::graph::VariableSpace<T>::clone() {
            auto result = new VariableSpace<T>();

            for (auto const& x : _paired) {
                std::pair<int, int> pair(x.first.first, x.first.second);

                Variable<T>* clonedVar = x.second->clone();

                if (pair.second == 0) {
                    if (pair.first < 0)
                        result->_variables[pair.first] = clonedVar;
                    else
                        result->_temporary[pair.first] = clonedVar;
                }

                if (clonedVar->getName() != nullptr && clonedVar->getName()->length() > 0)
                    result->_symbolic[*(clonedVar->getName())] = clonedVar;

                result->_paired[pair] = clonedVar;

                result->_handles->push_back(clonedVar);
            }

            return result;
        }

        template <typename T>
        std::vector<nd4j::graph::Variable<T>*> * nd4j::graph::VariableSpace<T>::getPlaceholders() {
            return &_placeholders;
        }

        template <typename T>
        int nd4j::graph::VariableSpace<T>::numberOfPlaceholders() {
            return _placeholders.size();
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
        bool nd4j::graph::VariableSpace<T>::hasVariable(int id) {
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
        void nd4j::graph::VariableSpace<T>::putVariable(std::pair<int,int>& pair, NDArray<T> *array) {
            auto variable = new Variable<T>(array);
            this->putVariable(pair, variable);
        }


        template <typename T>
        void nd4j::graph::VariableSpace<T>::silentPutVariable(std::pair<int,int>& pair, Variable<T> *variable) {
            _varmap.lock();

            //std::pair<std::pair<int, int>, nd4j::graph::Variable<T> *> p(pair, variable);
            _paired[pair] = variable;

            _varmap.unlock();
        }

        template <typename T>
        void nd4j::graph::VariableSpace<T>::putVariable(std::pair<int,int>& pair, Variable<T> *variable) {

            silentPutVariable(pair, variable);


            if (variable->isPlaceholder())
                _placeholders.push_back(variable);

            // copying duplicate for compatibility
            if (pair.second == 0 && !this->hasVariable(pair.first)) {
                this->putVariable(pair.first, variable);
            } else {
                if (variable->getName() != nullptr && variable->getName()->length() != 0) {
                    _symbolic[*(variable->getName())] = variable;
                }

                _varmap.lock();

                _handles->push_back(variable);

                _varmap.unlock();
            }
        }

        template <typename T>
        void nd4j::graph::VariableSpace<T>::putVariable(int id, Variable<T> *variable) {
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

            _handles->emplace_back(variable);

            if (_auto_counter >= id)
                _auto_counter = id - 1;

            variable->setId(id);

            if (variable->getName() != nullptr && variable->getName()->length() != 0) {
                //std::pair<std::string, nd4j::graph::Variable<T> *> pair(*(variable->getName()), variable);
                _symbolic[*(variable->getName())] = variable;
            }

            // we have special list for external variables to ensure graph completeness

            if (id < 0) {
                //if (variable->isExternal())
                _external.push_back(variable);

                _variables[id] = variable;
            } else {
                _internal.push_back(variable);

                _temporary[id] = variable;
            }

            _varmap.unlock();

            std::pair<int,int> pair(id, 0);
            if (!hasVariable(pair)) {
                this->silentPutVariable(pair, variable);

                if (variable->isPlaceholder())
                    _placeholders.push_back(variable);
            }
        }

        template <typename T>
        void nd4j::graph::VariableSpace<T>::putVariable(int id, NDArray<T> *array) {
            auto *var = new nd4j::graph::Variable<T>(array);
            this->putVariable(id, var);
        }

        template <typename T>
        nd4j::graph::Variable<T> * nd4j::graph::VariableSpace<T>::getVariable(int id) {
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
            for (auto p: *_handles) {
                delete p;
            }

            delete _handles;

            //_internal.clear();
            //_external.clear();
            //_temporary.clear();
        }


        template <typename T>
        nd4j::graph::VariableSpace<T>::VariableSpace() {
            _handles = new std::vector<Variable<T> *>;
        }

        template class VariableSpace<float>;
        template class VariableSpace<float16>;
        template class VariableSpace<double>;
    }
}