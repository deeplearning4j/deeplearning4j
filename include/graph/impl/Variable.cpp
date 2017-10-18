//
// @author raver119@gmail.com
//

#include <graph/Variable.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        nd4j::graph::Variable<T>* nd4j::graph::Variable<T>::clone() {
            auto result = new Variable<T>(this->isPlaceholder());
            result->_external = this->_external;
            result->_id = this->_id;
            result->_readOnly = this->_readOnly;
            result->_name = this->_name;

            if (this->_ndarray != nullptr)
                result->_ndarray = this->_ndarray->dup(this->_ndarray->ordering());

            return result;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::hasNDArray() {
            return _ndarray != nullptr;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::isPlaceholder() {
            return _placeholder;
        }

        template <typename T>
        std::string * nd4j::graph::Variable<T>::getName() {
            return &_name;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setName(std::string *name) {
            _name = *name;
        }

        template <typename T>
        int32_t nd4j::graph::Variable<T>::id() {
            return _id;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setId(int32_t id) {
            _id = id;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::isEmpty() {
            return _ndarray == nullptr || !_ndarray->nonNull();
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::isExternal() {
            return _external;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::isReadOnly() {
            return _readOnly;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::markExternal(bool reallyExternal) {
            this->_external = reallyExternal;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::markRemovable(bool reallyRemovable) {
            this->_removable = reallyRemovable;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::markReadOnly(bool reallyReadOnly) {
            this->_readOnly = reallyReadOnly;
        }

        template <typename T>
        nd4j::NDArray<T> * nd4j::graph::Variable<T>::getNDArray() {
            return this->_ndarray;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setNDArray(nd4j::NDArray<T> * array) {
            this->_ndarray = array;
        }

        template <typename T>
        nd4j::graph::Variable<T>::Variable(const nd4j::graph::FlatVariable *flatVariable) {
            int shapeLen = flatVariable->shape()->Length();
            int *shape = new int[shapeLen];
            this->_id = flatVariable->id();

            if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
                this->_name = flatVariable->name()->str();

            _external = true;
            _readOnly = false;

#pragma omp simd
            for (int e = 0; e < shapeLen; e++) {
                shape[e] = flatVariable->shape()->Get(e);
            }

            int bufLen = flatVariable->values()->Length();
            T *buffer = new T[bufLen];

#pragma omp simd
            for (int e = 0; e < bufLen; e++) {
                buffer[e] = (T) flatVariable->values()->Get(e);
            }

            _ndarray = new NDArray<T>(buffer, shape);
            _ndarray->triggerAllocationFlag(true, true);
        }

        template <typename T>
        nd4j::graph::Variable<T>::Variable(bool placeholder) {
            _placeholder = placeholder;
        }

        template <typename T>
        nd4j::graph::Variable<T>::Variable(NDArray<T> *array, const char *name ) {
            _ndarray = array;

            _external = false;
            _readOnly = false;
            _id = 0;

            if (name != nullptr)
                _name = std::string(name);
        }

        template <typename T>
        nd4j::graph::Variable<T>::~Variable() {
            if (_ndarray != nullptr && _removable)
                delete _ndarray;
        }


        template class ND4J_EXPORT Variable<float>;
        template class ND4J_EXPORT Variable<float16>;
        template class ND4J_EXPORT Variable<double>;
    }
}