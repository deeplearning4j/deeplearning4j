//
//  @author raver119@gmail.com
//

#include <pointercast.h>
#include <dll.h>
#include <types/float16.h>
#include <graph/ContextPrototype.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        ContextPrototype<T>::ContextPrototype(int nodeId, bool inPlace) {
            _nodeId = nodeId;
            _isInplace = inPlace;
        }

        template <typename T>
        void ContextPrototype<T>::pickInput(std::pair<int, int>& p) {
            this->_inputs.emplace_back(p);
        }

        template <typename T>
        void ContextPrototype<T>::pickInput(int input, int index) {
            std::pair<int, int> pair(input, index);
            pickInput(pair);
        }

        template <typename T>
        int ContextPrototype<T>::opNum() {
            return this->_opNum;
        }

        template <typename T>
        void ContextPrototype<T>::setOpNum(int opNum) {
            this->_opNum = opNum;
        }

        template <typename T>
        std::vector<std::pair<int, int>>* ContextPrototype<T>::inputs() {
            return &_inputs;
        }

        template <typename T>
        void ContextPrototype<T>::fillInputs(std::vector<int>& inputs) {
            for (int e = 0; e < inputs.size(); e++) {
                auto v = inputs.at(e);
                pickInput(v);
            }
        }

        template <typename T>
        bool ContextPrototype<T>::hasVariablesFilled() {
            return this->_inputs.size() > 0;
        }

        template <typename T>
        bool ContextPrototype<T>::isInplace() {
            return this->_isInplace;
        }

        template <typename T>
        std::vector<T>* ContextPrototype<T>::getTArguments() {
            return &(this->_tArgs);
        }

        template <typename T>
        std::vector<int>* ContextPrototype<T>::getIArguments() {
            return &(this->_iArgs);
        }

        template <typename T>
        void ContextPrototype<T>::pickInput(int input) {
            std::pair<int, int> pair(input, 0);
            this->_inputs.emplace_back(pair);
        }

        template <typename T>
        std::pair<int, int>* ContextPrototype<T>::input(int idx) {
            return &(this->_inputs.at(idx));
        }

        template <typename T>
        void ContextPrototype<T>::fillInputs(std::initializer_list<int> inputs) {
            for (auto v: inputs) {
                pickInput(v);
            }
        }

        template <typename T>
        int ContextPrototype<T>::nodeId() {
            return getNodeId();
        }

        template <typename T>
        int ContextPrototype<T>::numT() {
            return (int) _tArgs.size();
        }

        template <typename T>
        int ContextPrototype<T>::numI() {
            return (int) _iArgs.size();
        }

        template <typename T>
        int ContextPrototype<T>::getNodeId() {
            return this->_nodeId;
        }

        /**
         * This method returns number of inputs available in this block
         * @return
         */
        template <typename T>
        unsigned long ContextPrototype<T>::width() {
            return this->_inputs.size();
        };

        template <typename T>
        void ContextPrototype<T>::markInplace(bool reallyInplace) {
            this->_isInplace = reallyInplace;
        }

        template <typename T>
        template <typename N>
        ContextPrototype<N>* ContextPrototype<T>::asT() {
            auto clone = new ContextPrototype<N>(_nodeId, _isInplace);

            return clone;
        }

        template <typename T>
        ContextPrototype<T>* ContextPrototype<T>::clone() {
            auto clone = new ContextPrototype<T>(_nodeId, _isInplace);
            clone->_opNum = _opNum;
            
            for (auto v: _inputs)
                clone->_inputs.emplace_back(v);

            for (auto v: _tArgs)
                clone->_tArgs.emplace_back(v);

            for (auto v: _iArgs)
                clone->_iArgs.emplace_back(v);

            return clone;
        }


        template class ND4J_EXPORT ContextPrototype<float>;
        template class ND4J_EXPORT ContextPrototype<float16>;
        template class ND4J_EXPORT ContextPrototype<double>;


        template ContextPrototype<float>* ContextPrototype<float>::asT<float>();
        template ContextPrototype<float16>* ContextPrototype<float>::asT<float16>();
        template ContextPrototype<double>* ContextPrototype<float>::asT<double>();

        template ContextPrototype<float>* ContextPrototype<float16>::asT<float>();
        template ContextPrototype<float16>* ContextPrototype<float16>::asT<float16>();
        template ContextPrototype<double>* ContextPrototype<float16>::asT<double>();

        template ContextPrototype<float>* ContextPrototype<double>::asT<float>();
        template ContextPrototype<float16>* ContextPrototype<double>::asT<float16>();
        template ContextPrototype<double>* ContextPrototype<double>::asT<double>();
    }
}