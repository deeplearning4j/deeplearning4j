//
// @author raver119@gmail.com
//

#include <helpers/EnumUtils.h>
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

            // TODO: clone NDArrayList
            if (this->_list != nullptr)
                result->_list = this->_list->clone();

            return result;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::hasNDArray() {
            return _variableType == VariableType::NDARRAY && _ndarray != nullptr;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setVariableType(VariableType variableType) {
            _variableType = variableType;
        }

        template <typename T>
        bool nd4j::graph::Variable<T>::hasNDArrayList() {
            return _variableType == VariableType::ARRAY_LIST && _list != nullptr;
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
            if (_variableType == VariableType::NDARRAY) 
                return _ndarray == nullptr || !_ndarray->nonNull();
            else if (_variableType == VariableType::ARRAY_LIST)
                return _list == nullptr;

            return false;
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
            if (!reallyRemovable)
                nd4j_debug("","");
            this->_removable = reallyRemovable;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::markReadOnly(bool reallyReadOnly) {
            this->_readOnly = reallyReadOnly;
        }

        template <typename T>
        nd4j::NDArray<T> * nd4j::graph::Variable<T>::getNDArray() {
            if (_variableType != VariableType::NDARRAY) {
                nd4j_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArray was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }

            return this->_ndarray;
        }

        template <typename T>
        nd4j::NDArrayList<T> * nd4j::graph::Variable<T>::getNDArrayList() {
            if (_variableType != VariableType::ARRAY_LIST) {
                nd4j_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArrayList was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }
            return this->_list;
        }

        template <typename T>
        bool Variable<T>::isRemovable() {
            return _removable;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setNDArrayList(nd4j::NDArrayList<T> * list) {
            this->_variableType = VariableType::ARRAY_LIST;
            this->_list = list;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setNDArray(nd4j::NDArray<T> * array) {
            this->_variableType = VariableType::NDARRAY;
            this->_ndarray = array;
        }

        template <typename T>
        VariableType nd4j::graph::Variable<T>::variableType() {
            return _variableType;
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
            _variableType = VariableType::NDARRAY;
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

            if (name != nullptr)
                _name = std::string(name);

            if (_ndarray != nullptr)
                _variableType = VariableType::NDARRAY;
        }

        template <typename T>
        nd4j::graph::Variable<T>::Variable(NDArray<T> *array, const char *name, int id, int idx) : Variable(array, name) {
            _id = id;
            _index = idx;
        }

        template <typename T>
        nd4j::graph::Variable<T>::~Variable() {
            //nd4j_printf("Removing variable [%i:%i]\n", _id, _index);
            if (_variableType == VariableType::NDARRAY)
                if (_ndarray != nullptr && _removable)
                    delete _ndarray;
        }

        template <typename T>
        void Variable<T>::setId(int id, int idx) {
            _id = id;
            _index = idx;
        }

        template class ND4J_EXPORT Variable<float>;
        template class ND4J_EXPORT Variable<float16>;
        template class ND4J_EXPORT Variable<double>;
    }
}