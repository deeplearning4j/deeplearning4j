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

        template <typename N>
        Variable* Variable::asT() {
            auto result = new Variable(this->isPlaceholder());

            result->markExternal(this->_external);
            result->setId(this->_id);
            result->markReadOnly(this->_readOnly);
            result->setName(&this->_name);
            result->setIndex(this->_index);

            if (this->_ndarray != nullptr)
                result->setNDArray(this->_ndarray->template asT());

            // FIXME: add support for ArrayList
            if (this->_list != nullptr) {
                nd4j_printf("ArrayList not supported yet\n", "");
                throw std::runtime_error("ArrayList not supported yet");
            }

            return result;
        }

        nd4j::graph::Variable* nd4j::graph::Variable::clone() {
            auto result = new Variable(this->isPlaceholder());
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

        void nd4j::graph::Variable::setIndex(int index) {
            _index = index;
        }

        bool nd4j::graph::Variable::hasNDArray() {
            return _variableType == VariableType::NDARRAY && _ndarray != nullptr;
        }

        void nd4j::graph::Variable::setVariableType(VariableType variableType) {
            _variableType = variableType;
        }

        bool nd4j::graph::Variable::hasNDArrayList() {
            return _variableType == VariableType::ARRAY_LIST && _list != nullptr;
        }

        bool nd4j::graph::Variable::isPlaceholder() {
            return _placeholder;
        }

        std::string * nd4j::graph::Variable::getName() {
            return &_name;
        }

        void nd4j::graph::Variable::setName(std::string *name) {
            _name = *name;
        }

        int nd4j::graph::Variable::id() {
            return _id;
        }

        int nd4j::graph::Variable::index() {
            return _index;
        }

        void nd4j::graph::Variable::setId(int id) {
            _id = id;
        }

        bool nd4j::graph::Variable::isEmpty() {
            if (_variableType == VariableType::NDARRAY) 
                return _ndarray == nullptr || !_ndarray->nonNull();
            else if (_variableType == VariableType::ARRAY_LIST)
                return _list == nullptr;

            return false;
        }

        bool nd4j::graph::Variable::isExternal() {
            return _external;
        }

        bool nd4j::graph::Variable::isReadOnly() {
            return _readOnly;
        }

        void nd4j::graph::Variable::markExternal(bool reallyExternal) {
            this->_external = reallyExternal;
        }

        void nd4j::graph::Variable::markRemovable(bool reallyRemovable) {
            if (!reallyRemovable)
                nd4j_debug("","");
            this->_removable = reallyRemovable;
        }

        void nd4j::graph::Variable::markReadOnly(bool reallyReadOnly) {
            this->_readOnly = reallyReadOnly;
        }

        nd4j::NDArray * nd4j::graph::Variable::getNDArray() {
            if (_variableType != VariableType::NDARRAY) {
                nd4j_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArray was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }

            return this->_ndarray;
        }

        nd4j::NDArrayList * nd4j::graph::Variable::getNDArrayList() {
            if (_variableType != VariableType::ARRAY_LIST) {
                nd4j_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArrayList was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }
            return this->_list;
        }

        
        bool Variable::isRemovable() {
            return _removable;
        }

        
        void nd4j::graph::Variable::setNDArrayList(nd4j::NDArrayList * list) {
            this->_variableType = VariableType::ARRAY_LIST;
            this->_list = list;
        }

        
        void nd4j::graph::Variable::setNDArray(nd4j::NDArray * array) {
            this->_variableType = VariableType::NDARRAY;
            this->_ndarray = array;
        }

        
        VariableType nd4j::graph::Variable::variableType() {
            return _variableType;
        }

        
        nd4j::graph::Variable::Variable(const nd4j::graph::FlatVariable *flatVariable) {
            auto vid = flatVariable->id();
            this->_id = vid->first();
            this->_index = vid->second();

            if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
                this->_name = flatVariable->name()->str();

            _external = true;
            _readOnly = false;

            int8_t *buffer = nullptr;

            if (flatVariable->ndarray() != nullptr) {
                 auto ar = flatVariable->ndarray();
                _ndarray = nd4j::graph::FlatUtils::fromFlatArray(ar);
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

                _ndarray = new NDArray((char) shapeInfo.at(shapeInfo.size() - 1), shape);
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

                DataTypeConversions::convertType(buffer, flatBuf, dtype, bufLen);
            }
            */

            //_ndarray = new NDArray(buffer, shape);
            _variableType = VariableType::NDARRAY;
        }

        
        nd4j::graph::Variable::Variable(bool placeholder) {
            _placeholder = placeholder;
        }

        
        nd4j::graph::Variable::Variable(NDArray *array, const char *name ) {
            _ndarray = array;

            _external = false;
            _readOnly = false;

            if (name != nullptr)
                _name = std::string(name);

            if (_ndarray != nullptr)
                _variableType = VariableType::NDARRAY;
        }

        
        nd4j::graph::Variable::Variable(NDArray *array, const char *name, int id, int idx) : Variable(array, name) {
            _id = id;
            _index = idx;
        }

        
        nd4j::graph::Variable::~Variable() {
            //nd4j_printf("Removing variable [%i:%i]\n", _id, _index);
            if (_variableType == VariableType::NDARRAY)
                if (_ndarray != nullptr && _removable)
                    delete _ndarray;
        }

        
        void Variable::setId(int id, int idx) {
            _id = id;
            _index = idx;
        }

        
        flatbuffers::Offset<FlatVariable> Variable::asFlatVariable(flatbuffers::FlatBufferBuilder &builder) {
            if (this->hasNDArray()) {
                auto array = this->getNDArray();
                auto fShape = builder.CreateVector(array->getShapeInfoAsFlatVector());
                auto fBuffer = builder.CreateVector(array->asByteVector());

                // packing array
                auto fArray = CreateFlatArray(builder, fShape, fBuffer, nd4j::graph::DataType::DataType_FLOAT);

                // packing id/index of this var
                auto fVid = CreateIntPair(builder, this->_id, this->_index);

                // name is still optional
                flatbuffers::Offset<flatbuffers::String> stringId = 0;
                if (!this->_name.empty())
                    stringId = builder.CreateString(this->_name);

                // returning array
                return CreateFlatVariable(builder, fVid, stringId, 0, fArray);
            }
        }
    }
}