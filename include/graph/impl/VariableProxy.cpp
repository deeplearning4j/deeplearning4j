//
//  @author raver119@gmail.com
//

#include <dll.h>
#include <graph/VariableProxy.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        VariableProxy<T>::VariableProxy(VariableSpace<T>* ref) {
            if (ref == nullptr)
                _backed = new VariableSpace<T>();

            _backed = ref;
            _current = new VariableSpace<T>();
        }

        template <typename T>
        VariableProxy<T>::~VariableProxy() {
            delete _current;
        }

        template <typename T>
        int VariableProxy<T>::numberOfPlaceholders() {
            return _backed->numberOfPlaceholders();
        }

        template <typename T>
        std::vector<Variable<T>*>* VariableProxy<T>::getPlaceholders() {
            return _backed->getPlaceholders();
        }

        template <typename T>
        nd4j::random::RandomBuffer* VariableProxy<T>::getRNG() {
            return _current->getRNG();
        }

        template <typename T>
        void VariableProxy<T>::setRNG(nd4j::random::RandomBuffer* rng) {
            _current->setRNG(rng);
        }
        
        template <typename T>
        bool VariableProxy<T>::hasExternalVariable(int it) {
            return _backed->hasExternalVariable(it);
        }

        template <typename T>
        bool VariableProxy<T>::hasExternalVariable(std::pair<int,int>& pair) {
            return _backed->hasExternalVariable(pair);
        }

        template <typename T>
        bool VariableProxy<T>::hasExternalVariable(std::string *symbol) {
            return _backed->hasExternalVariable(symbol);
        }

        template <typename T>
        bool VariableProxy<T>::hasVariable(int id) {
            return _current->hasVariable(id) || _backed->hasVariable(id);
        }
        
        template <typename T>
        bool VariableProxy<T>::hasVariable(int id, int idx) {
            return _current->hasVariable(id, idx) || _backed->hasVariable(id, idx);
        }
        
        template <typename T>
        bool VariableProxy<T>::hasVariable(std::pair<int,int>& pair) {
            return _current->hasVariable(pair) || _backed->hasVariable(pair);
        }

        template <typename T>
        void VariableProxy<T>::dropVariable(std::pair<int,int> &pair) {
            dropVariable(pair.first, pair.second);
        }

        template <typename T>
        void VariableProxy<T>::dropVariable(int id, int idx) {

        }

        template <typename T>
        bool VariableProxy<T>::hasVariable(std::string *symbol) {
            return _current->hasVariable(symbol) || _backed->hasVariable(symbol);
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(int id) {
            if (_current->hasVariable(id))
                return _current->getVariable(id);
            
            if (_backed->hasVariable(id))
                return _backed->getVariable(id);

            nd4j_printf("Unable to get Variable to proxy: [%i]\n", id);
            throw "Bad arguments";
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(int id, int idx) {
            if (_current->hasVariable(id, idx))
                return _current->getVariable(id, idx);
            
            if (_backed->hasVariable(id, idx))
                return _backed->getVariable(id, idx);

            nd4j_printf("Unable to get Variable to proxy: [%i:%i]\n", id, idx);
            throw "Bad arguments";
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(std::pair<int,int>& pair) {
            if (_current->hasVariable(pair))
                return _current->getVariable(pair);
            
            if (_backed->hasVariable(pair))
                return _backed->getVariable(pair);

            nd4j_printf("Unable to get Variable to proxy: [%i:%i]\n", pair.first, pair.second);
            throw "Bad arguments";
        }

        template <typename T>
        nd4j::graph::Variable<T> *VariableProxy<T>::getVariable(std::string *symbol) {
            if (_current->hasVariable(symbol))
                return _current->getVariable(symbol);
            
            if (_backed->hasVariable(symbol))
                return _backed->getVariable(symbol);

            nd4j_printf("Unable to get Variable to proxy: [%s]\n", symbol->c_str());
            throw "Bad arguments";
        }

        template <typename T>
        void VariableProxy<T>::putVariable(std::pair<int,int>& pair, NDArray<T> *array) {
            _current->putVariable(pair, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(std::pair<int,int>& pair, Variable<T> *variable) {
            _current->putVariable(pair, variable);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, Variable<T> *variable) {
            _current->putVariable(id, variable);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, NDArray<T> *array) {
            _current->putVariable(id, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, int idx, NDArray<T> *array) {
            _current->putVariable(id, idx, array);
        }

        template <typename T>
        void VariableProxy<T>::putVariable(int id, int idx, Variable<T> *array) {
            _current->putVariable(id, idx, array);
        }

        template <typename T>
        void VariableProxy<T>::trackList(nd4j::NDArrayList<T>* list) {
            _current->trackList(list);
        }

        template <typename T>
        nd4j::graph::Stash<T>* VariableProxy<T>::getStash() {
            return _current->getStash();
        }

        template <typename T>
        void VariableProxy<T>::setFlowPath(FlowPath* timers) {
            _current->setFlowPath(timers);
        }

        template <typename T>
        FlowPath* VariableProxy<T>::flowPath() {
            return _current->flowPath();
        }

        template <typename T>
        void VariableProxy<T>::putOutputVariable(Variable<T> *variable) {
            _current->putOutputVariable(variable);
        }

        template <typename T>
        Nd4jLong VariableProxy<T>::externalMemory() {
            return _backed->externalMemory() + _current->externalMemory();
        }

        template <typename T>
        Nd4jLong VariableProxy<T>::internalMemory() {
            return _backed->internalMemory() + _current->internalMemory();
        }

        template <typename T>
        Nd4jLong VariableProxy<T>::totalMemory() {
            return _backed->totalMemory() + _current->totalMemory();
        }

        template <typename T>
        int VariableProxy<T>::externalEntries() {
            return _backed->externalEntries() + _current->externalEntries();
        }

        template <typename T>
        int VariableProxy<T>::internalEntries() {
            return _backed->internalEntries() + _current->internalEntries();
        }

        template <typename T>
        int VariableProxy<T>::totalEntries() {
            return _backed->totalEntries() + _current->totalEntries();
        }

        template <typename T>
        nd4j::graph::VariableSpace<T>* VariableProxy<T>::clone() {
            auto clone = new VariableProxy(_backed);

            delete clone->_current;
            clone->_current = _current->clone();

            return clone;
        }

        template <typename T>
        VariableSpace<T>& VariableProxy<T>::operator=(const VariableSpace<T>& other) {
            if (this == &other) return *this;

            nd4j_printf("VariableProxy = not implemented\n","");

            return *this;
        }  

        template <typename T>
        nd4j::memory::Workspace * nd4j::graph::VariableProxy<T>::workspace() {
            return &this->_workspace;
        }

        template class ND4J_EXPORT VariableProxy<float>;
        template class ND4J_EXPORT VariableProxy<float16>;
        template class ND4J_EXPORT VariableProxy<double>;
    }
}
