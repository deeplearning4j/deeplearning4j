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
#include <helpers/StringUtils.h>

namespace sd {
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
                result->setNDArray(new NDArray(this->_ndarray->template asT<N>()));

            // FIXME: add support for ArrayList
            if (this->_list != nullptr) {
                nd4j_printf("ArrayList not supported yet\n", "");
                throw std::runtime_error("ArrayList not supported yet for asT");
            }

            return result;
        }
        BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT Variable* Variable::asT, (), LIBND4J_TYPES);

        sd::graph::Variable* sd::graph::Variable::clone() {
            auto result = new Variable(this->isPlaceholder());
            result->_external = this->_external;
            result->_id = this->_id;
            result->_readOnly = this->_readOnly;
            result->_name = this->_name;
            result->_index = this->_index;

            if (this->_ndarray != nullptr) {
                result->_ndarray = new NDArray(this->_ndarray->dup(this->_ndarray->ordering()));
                result->_readOnly = false;
                result->_removable = true;
            }

            if (this->_list != nullptr)
                result->_list = this->_list->clone();

            return result;
        }

        void sd::graph::Variable::setIndex(int index) {
            _index = index;
        }

        bool sd::graph::Variable::hasNDArray() {
            return _ndarray != nullptr;
        }

        void sd::graph::Variable::setVariableType(VariableType variableType) {
            _variableType = variableType;
        }

        bool sd::graph::Variable::hasNDArrayList() {
            return _list != nullptr;
        }

        bool sd::graph::Variable::isPlaceholder() {
            return _placeholder;
        }

        std::string * sd::graph::Variable::getName() {
            return &_name;
        }

        void sd::graph::Variable::setName(std::string *name) {
            _name = *name;
        }

        int sd::graph::Variable::id() {
            return _id;
        }

        int sd::graph::Variable::index() {
            return _index;
        }

        void sd::graph::Variable::setId(int id) {
            _id = id;
        }

        bool sd::graph::Variable::isEmpty() {
            if (_variableType == VariableType::NDARRAY)
                return _ndarray == nullptr || !_ndarray->nonNull();
            else if (_variableType == VariableType::ARRAY_LIST)
                return _list == nullptr;

            return false;
        }

        bool sd::graph::Variable::isExternal() {
            return _external;
        }

        bool sd::graph::Variable::isReadOnly() {
            return _readOnly;
        }

        void sd::graph::Variable::markExternal(bool reallyExternal) {
            this->_external = reallyExternal;
        }

        void sd::graph::Variable::markRemovable(bool reallyRemovable) {
            if (!reallyRemovable)
                nd4j_debug("","");
            this->_removable = reallyRemovable;
        }

        void sd::graph::Variable::markReadOnly(bool reallyReadOnly) {
            this->_readOnly = reallyReadOnly;
        }

        sd::NDArray * sd::graph::Variable::getNDArray() {
            if (_variableType != VariableType::NDARRAY) {
                nd4j_printf("Variable[%i:%i/<%s>] is has [%s] type, but NDArray was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }

            if (this->_ndarray == nullptr) {
                if (_name.empty()) {
                    auto nodeId = StringUtils::valueToString<int>(this->id());
                    auto outputIndex = StringUtils::valueToString<int>(this->index());
                    throw std::runtime_error("Array doesn't exist for Variable <" + nodeId + ":" + outputIndex + ">");
                } else {
                    auto outputIndex = StringUtils::valueToString<int>(this->index());
                    throw std::runtime_error("Array doesn't exist for Variable <" + this->_name + ":" + outputIndex+ ">");
                }
            }

            return this->_ndarray;
        }

        sd::NDArrayList * sd::graph::Variable::getNDArrayList() {
            if (_variableType != VariableType::ARRAY_LIST) {
                nd4j_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArrayList was requested\n", this->_id, this->_index, this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
            }
            return this->_list;
        }


        bool Variable::isRemovable() {
            return _removable;
        }


        void sd::graph::Variable::setNDArrayList(sd::NDArrayList * list) {
            this->_variableType = VariableType::ARRAY_LIST;
            this->_list = list;
        }


        void sd::graph::Variable::setNDArray(sd::NDArray * array) {
            this->_variableType = VariableType::NDARRAY;
            this->_ndarray = array;
        }


        VariableType sd::graph::Variable::variableType() {
            return _variableType;
        }


        sd::graph::Variable::Variable(const sd::graph::FlatVariable *flatVariable) {
            auto vid = flatVariable->id();
            this->_id = vid->first();
            this->_index = vid->second();

            if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
                this->_name = flatVariable->name()->str();

            _external = true;
            _readOnly = false;

            int8_t *buffer = nullptr;

            switch (flatVariable->variabletype()) {
                case VarType_VARIABLE: {

                        // ?????
                        if (flatVariable->ndarray() != nullptr) {
                            auto ar = flatVariable->ndarray();
                            _ndarray = sd::graph::FlatUtils::fromFlatArray(ar);
                        }

                        _variableType = VariableType::NDARRAY;
                    }
                    break;
                case VarType_CONSTANT: {
                        if (flatVariable->ndarray() == nullptr)
                            throw std::runtime_error("CONSTANT variable must have NDArray bundled");

                        auto ar = flatVariable->ndarray();
                        if (ar->dtype() == DType_UTF8) {
                            _ndarray = sd::graph::FlatUtils::fromFlatArray(ar);
                        } else {
                            _ndarray = sd::graph::FlatUtils::fromFlatArray(ar);
                        }

                        _variableType = VariableType::NDARRAY;
                    }
                    break;
                case VarType_ARRAY: {

                        // ?????
                        if (flatVariable->ndarray() != nullptr) {
                            auto ar = flatVariable->ndarray();
                            _ndarray = sd::graph::FlatUtils::fromFlatArray(ar);
                            // _ndarray->triggerAllocationFlag(true);
                        }

                        _variableType = VariableType::NDARRAY;
                    }
                    break;
                case VarType_PLACEHOLDER: {
                        if (flatVariable->shape() == nullptr && flatVariable->ndarray() == nullptr)
                            throw std::runtime_error("PLACEHOLDER variable must have shape defined");

                        if (flatVariable->ndarray() != nullptr) {
                            auto ar = flatVariable->ndarray();
                            _ndarray = sd::graph::FlatUtils::fromFlatArray(ar);
                            // _ndarray->triggerAllocationFlag(true);

                            _variableType = VariableType::NDARRAY;
                        }

                        if (flatVariable->shape() != nullptr) {
                            int shapeLen = flatVariable->shape()->Length();
                            for (int i = 0; i < flatVariable->shape()->size(); i++)
                                _shape.emplace_back(flatVariable->shape()->Get(i));

                            if (_ndarray == nullptr)
                                _variableType = VariableType::PLACEHOLDER;
                        }
                    }
                    break;
                default:
                    throw std::runtime_error("Unknown variable type used");
            }
        }

        std::vector<Nd4jLong>& sd::graph::Variable::shape() {
            return _shape;
        }

        sd::graph::Variable::Variable(bool placeholder) {
            _placeholder = placeholder;
        }


        sd::graph::Variable::Variable(NDArray *array, const char *name ) {
            _ndarray = array;

            _external = false;
            _readOnly = false;

            if (name != nullptr)
                _name = std::string(name);

            if (_ndarray != nullptr)
                _variableType = VariableType::NDARRAY;
        }


        sd::graph::Variable::Variable(NDArray *array, const char *name, int id, int idx) : Variable(array, name) {
            _id = id;
            _index = idx;
        }


        sd::graph::Variable::~Variable() {
            //nd4j_printf("Removing variable [%i:%i]\n", _id, _index);
            if (_variableType == VariableType::NDARRAY) {
                nd4j_debug("Removing variable <%i:%i>\n", _id, _index);
                if (_ndarray != nullptr && _removable && !_readOnly)
                    delete _ndarray;
            }
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
                auto fArray = CreateFlatArray(builder, fShape, fBuffer, (sd::graph::DType) array->dataType());

                // packing id/index of this var
                auto fVid = CreateIntPair(builder, this->_id, this->_index);

                // name is still optional
                flatbuffers::Offset<flatbuffers::String> stringId = 0;
                if (!this->_name.empty())
                    stringId = builder.CreateString(this->_name);

                // returning array
                return CreateFlatVariable(builder, fVid, stringId, static_cast<sd::graph::DType>(array->dataType()), 0, fArray);
            } else {
                throw std::runtime_error("Variable::asFlatVariable isn't possible for NDArrayList");
            }
        }
    }
}

namespace std {

    size_t hash<std::pair<int, int>>::operator()(const std::pair<int,int>& k) const {
        auto v = std::hash<int>()(k.first);
        v ^= std::hash<int>()(k.second) + 0x9e3779b9 + (v << 6) + (v >> 2);
        return v;
    }

    size_t hash<bfloat16>::operator()(const bfloat16& k) const {
        return std::hash<float>()((float)k);
    }

    size_t hash<float16>::operator()(const float16& k) const {
        return std::hash<float>()((float)k);
    }
}