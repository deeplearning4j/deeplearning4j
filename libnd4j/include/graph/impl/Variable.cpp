/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include <helpers/EnumUtils.h>
#include <graph/Variable.h>
#include <array/DataTypeUtils.h>
#include <array/ByteOrderUtils.h>
#include <array/DataTypeConversions.h>
#include <graph/FlatUtils.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        template <typename N>
        Variable<N>* Variable<T>::asT() {
            auto result = new Variable<N>(this->isPlaceholder());

            result->markExternal(this->_external);
            result->setId(this->_id);
            result->markReadOnly(this->_readOnly);
            result->setName(&this->_name);
            result->setIndex(this->_index);

            if (this->_ndarray != nullptr)
                result->setNDArray(this->_ndarray->template asT<N>());

            // FIXME: add support for ArrayList
            if (this->_list != nullptr) {
                nd4j_printf("ArrayList not supported yet\n", "");
                throw std::runtime_error("ArrayList not supported yet");
            }

            return result;
        }

        template <typename T>
        nd4j::graph::Variable<T>* nd4j::graph::Variable<T>::clone() {
            auto result = new Variable<T>(this->isPlaceholder());
            result->_external = this->_external;
            result->_id = this->_id;
            result->_readOnly = this->_readOnly;
            result->_name = this->_name;
            result->_index = this->_index;

            if (this->_ndarray != nullptr)
                result->_ndarray = this->_ndarray->dup(this->_ndarray->ordering());

            if (this->_list != nullptr)
                result->_list = this->_list->clone();

            return result;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setIndex(int index) {
            _index = index;
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
        int nd4j::graph::Variable<T>::id() {
            return _id;
        }

        template <typename T>
        int nd4j::graph::Variable<T>::index() {
            return _index;
        }

        template <typename T>
        void nd4j::graph::Variable<T>::setId(int id) {
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
            auto vid = flatVariable->id();
            this->_id = vid->first();
            this->_index = vid->second();

            if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
                this->_name = flatVariable->name()->str();

            _external = true;
            _readOnly = false;

            T *buffer = nullptr;

            if (flatVariable->ndarray() != nullptr) {
                 auto ar = flatVariable->ndarray();
                _ndarray = nd4j::graph::FlatUtils::fromFlatArray<T>(ar);
                _ndarray->triggerAllocationFlag(true, true);
            } else if (flatVariable->shape() != nullptr) {
                int shapeLen = flatVariable->shape()->Length();
                //int *shape = new int[shapeLen];

                std::vector<Nd4jLong> shapeInfo(flatVariable->shape()->size());
                for (int i = 0; i < flatVariable->shape()->size(); i++) {
                    shapeInfo[i] = flatVariable->shape()->Get(i);
                }

                // we just create empty array here
                std::vector<Nd4jLong> shape(shapeInfo.at(0));
                for (int i = 0; i < shapeInfo.at(0); i++) {
                    shape[i] = shapeInfo.at(i + 1);
                }

                _ndarray = new NDArray<T>((char) shapeInfo.at(shapeInfo.size() - 1), shape);
            } else {
                nd4j_printf("Either shape or NDArray should be defined in FlatResult variable\n","");
                throw std::runtime_error("Empty variable");
            }

            /*
            if (flatVariable->values() != nullptr && flatVariable->values()->Length() > 0) {
                int bufLen = (int) flatVariable->values()->Length();
                 buffer = new T[bufLen];

#pragma omp parallel for simd
                for (int e = 0; e < bufLen; e++) {
                    buffer[e] = (T) flatVariable->values()->Get(e);
                }
            }

            if (flatVariable->buffer() != nullptr && flatVariable->buffer()->size() > 0) {
                auto dtype = DataTypeUtils::fromFlatDataType(flatVariable->dataType());
                auto bo = ByteOrderUtils::fromFlatByteOrder(flatVariable->order());

                auto bufLen = shape::length(shape);
                buffer = new T[bufLen];

                // TODO: byteorder should be honored here

                // TODO: we want to have variable datatype, so in future we should replace explicit conversion with simple migration
                auto flatBuf = (void *) flatVariable->buffer()->data();

                DataTypeConversions<T>::convertType(buffer, flatBuf, dtype, bufLen);
            }
            */

            //_ndarray = new NDArray<T>(buffer, shape);
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
        template class ND4J_EXPORT Variable<int>;
        template class ND4J_EXPORT Variable<Nd4jLong>;


        template Variable<float>* Variable<float>::asT<float>();
        template Variable<float16>* Variable<float>::asT<float16>();
        template Variable<double>* Variable<float>::asT<double>();
        template Variable<int>* Variable<float>::asT<int>();
        template Variable<Nd4jLong>* Variable<float>::asT<Nd4jLong>();

        template Variable<float>* Variable<float16>::asT<float>();
        template Variable<float16>* Variable<float16>::asT<float16>();
        template Variable<double>* Variable<float16>::asT<double>();
        template Variable<int>* Variable<float16>::asT<int>();
        template Variable<Nd4jLong>* Variable<float16>::asT<Nd4jLong>();

        template Variable<float>* Variable<double>::asT<float>();
        template Variable<float16>* Variable<double>::asT<float16>();
        template Variable<double>* Variable<double>::asT<double>();
        template Variable<int>* Variable<double>::asT<int>();
        template Variable<Nd4jLong>* Variable<double>::asT<Nd4jLong>();

        template Variable<float>* Variable<int>::asT<float>();
        template Variable<float16>* Variable<int>::asT<float16>();
        template Variable<double>* Variable<int>::asT<double>();
        template Variable<int>* Variable<int>::asT<int>();
        template Variable<Nd4jLong>* Variable<int>::asT<Nd4jLong>();

        template Variable<float>* Variable<Nd4jLong>::asT<float>();
        template Variable<float16>* Variable<Nd4jLong>::asT<float16>();
        template Variable<double>* Variable<Nd4jLong>::asT<double>();
        template Variable<int>* Variable<Nd4jLong>::asT<int>();
        template Variable<Nd4jLong>* Variable<Nd4jLong>::asT<Nd4jLong>();
    }
}