/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/ByteOrderUtils.h>
#include <array/DataTypeConversions.h>
#include <array/DataTypeUtils.h>
#include <graph/FlatUtils.h>
#include <graph/Variable.h>
#include <helpers/EnumUtils.h>
#include <helpers/StringUtils.h>

namespace sd {
namespace graph {

template <typename N>
Variable *Variable::asT() {
  auto result = new Variable(this->isPlaceholder());

  result->markExternal(this->_external);
  result->setId(this->_id);
  result->markReadOnly(this->_readOnly);
  result->setName(&this->_name);
  result->setIndex(this->_index);

  if (this->_ndarray != nullptr) result->setNDArray(new NDArray(this->_ndarray->template asT<N>()));

  // FIXME: add support for ArrayList
  if (this->_list != nullptr) {
    sd_printf("ArrayList not supported yet\n", "");
    THROW_EXCEPTION("ArrayList not supported yet for asT");
  }

  return result;
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT Variable *Variable::asT, (), SD_COMMON_TYPES);

Variable *Variable::clone() {
  auto result = new Variable(this->isPlaceholder());
  result->_external = this->_external;
  result->_id = this->_id;
  result->_readOnly = this->_readOnly;
  result->_name = this->_name;
  result->_index = this->_index;

  if (this->_ndarray != nullptr) {
    result->_ndarray = this->_ndarray->dup(this->_ndarray->ordering(), false);
    result->_readOnly = false;
    result->_removable = true;
  }

  if (this->_list != nullptr) result->_list = this->_list->clone();

  return result;
}

void Variable::setIndex(int index) { _index = index; }

bool Variable::hasNDArray() { return _ndarray != nullptr; }

void Variable::setVariableType(VariableType variableType) { _variableType = variableType; }

bool Variable::hasNDArrayList() { return _list != nullptr; }

bool Variable::isPlaceholder() { return _placeholder; }

std::string *Variable::getName() { return &_name; }

void Variable::setName(std::string *name) { _name = *name; }

int Variable::id() { return _id; }

int Variable::index() { return _index; }

void Variable::setId(int id) { _id = id; }

bool Variable::isEmpty() {
  if (_variableType == NDARRAY)
    return _ndarray == nullptr || !_ndarray->nonNull();
  else if (_variableType == ARRAY_LIST)
    return _list == nullptr;

  return false;
}

bool Variable::isExternal() { return _external; }

bool Variable::isReadOnly() { return _readOnly; }

void Variable::markExternal(bool reallyExternal) { this->_external = reallyExternal; }

void Variable::markRemovable(bool reallyRemovable) {
  if (!reallyRemovable) sd_debug("", "");
  this->_removable = reallyRemovable;
}

void Variable::markReadOnly(bool reallyReadOnly) { this->_readOnly = reallyReadOnly; }

NDArray *Variable::getNDArray() {
  if (_variableType != NDARRAY) {
    sd_printf("Variable[%i:%i/<%s>] is has [%s] type, but NDArray was requested\n", this->_id, this->_index,
              this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
  }

  if (this->_ndarray == nullptr) {
    if (_name.empty()) {
      auto nodeId = StringUtils::valueToString<int>(this->id());
      auto outputIndex = StringUtils::valueToString<int>(this->index());
      auto msg = "Array doesn't exist for Variable <" + nodeId + ":" + outputIndex + ">";
      THROW_EXCEPTION(msg.c_str());
    } else {
      auto outputIndex = StringUtils::valueToString<int>(this->index());
      auto msg = "Array doesn't exist for Variable <" + this->_name + ":" + outputIndex + ">";
      THROW_EXCEPTION(msg.c_str());
    }
  }

  return this->_ndarray;
}

NDArrayList *Variable::getNDArrayList() {
  if (_variableType != ARRAY_LIST) {
    sd_debug("Variable[%i:%i/<%s>] is has [%s] type, but NDArrayList was requested\n", this->_id, this->_index,
             this->_name.c_str(), EnumUtils::_VariableTypeToString(_variableType));
  }
  return this->_list;
}

bool Variable::isRemovable() { return _removable; }

void Variable::setNDArrayList(NDArrayList *list) {
  this->_variableType = ARRAY_LIST;
  this->_list = list;
}

void Variable::setNDArray(NDArray *array) {
  this->_variableType = NDARRAY;
  this->_ndarray = array;
}

VariableType Variable::variableType() { return _variableType; }

Variable::Variable(const ::graph::FlatVariable *flatVariable) {
  auto vid = flatVariable->id();
  this->_id = vid->first();
  this->_index = vid->second();

  if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0) this->_name = flatVariable->name()->str();

  _external = true;
  _readOnly = false;

  int8_t *buffer = nullptr;

  switch (flatVariable->variabletype()) {
    case ::graph::VarType_VARIABLE: {
      // ?????
      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = FlatUtils::fromFlatArray(ar);
      }

      _variableType = NDARRAY;
    } break;
    case ::graph::VarType_CONSTANT: {
      if (flatVariable->ndarray() == nullptr) THROW_EXCEPTION("CONSTANT variable must have NDArray bundled");

      auto ar = flatVariable->ndarray();
      if (ar->dtype() == ::graph::DType_UTF8) {
        _ndarray = FlatUtils::fromFlatArray(ar);
      } else {
        _ndarray = FlatUtils::fromFlatArray(ar);
      }

      _variableType = NDARRAY;
    } break;
    case ::graph::VarType_ARRAY: {
      // ?????
      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = FlatUtils::fromFlatArray(ar);
        // _ndarray->triggerAllocationFlag(true);
      }

      _variableType = NDARRAY;
    } break;
    case ::graph::VarType_PLACEHOLDER: {
      if (flatVariable->shape() == nullptr && flatVariable->ndarray() == nullptr)
        THROW_EXCEPTION("PLACEHOLDER variable must have shape defined");

      if (flatVariable->ndarray() != nullptr) {
        auto ar = flatVariable->ndarray();
        _ndarray = FlatUtils::fromFlatArray(ar);

        _variableType = NDARRAY;
      }

      if (flatVariable->shape() != nullptr) {
        int shapeLen = flatVariable->shape()->size();
        for (size_t i = 0; i < flatVariable->shape()->size(); i++) _shape.emplace_back(flatVariable->shape()->Get(i));

        if (_ndarray == nullptr) _variableType = PLACEHOLDER;
      }
    } break;
    default:
      THROW_EXCEPTION("Unknown variable type used");
  }
}

std::vector<LongType> &Variable::shape() { return _shape; }

Variable::Variable(bool placeholder) { _placeholder = placeholder; }

Variable::Variable(NDArray *array, const char *name) {
  _ndarray = array;

  _external = false;
  _readOnly = false;

  if (name != nullptr) _name = std::string(name);

  if (_ndarray != nullptr) _variableType = NDARRAY;
}

Variable::Variable(NDArray *array, const char *name, int id, int idx) : Variable(array, name) {
  _id = id;
  _index = idx;
}

Variable::~Variable() {
  if (_variableType == NDARRAY) {
    sd_debug("Removing variable <%i:%i>\n", _id, _index);
    if (_ndarray != nullptr && _removable && !_readOnly) delete _ndarray;
  }
}

void Variable::setId(int id, int idx) {
  _id = id;
  _index = idx;
}

flatbuffers::Offset<::graph::FlatVariable> Variable::asFlatVariable(flatbuffers::FlatBufferBuilder &builder) {
  if (this->hasNDArray()) {
    auto array = this->getNDArray();
    auto vec = array->getShapeInfoAsFlatVector();
    auto fShape = builder.CreateVector(*vec);
    delete vec;
    auto fBuffer = builder.CreateVector(array->asByteVector());

    // packing array
    auto fArray = CreateFlatArray(builder, fShape, fBuffer, (::graph::DType)array->dataType());

    // packing id/index of this var
    auto fVid = ::graph::CreateIntPair(builder, this->_id, this->_index);

    // name is still optional
    flatbuffers::Offset<flatbuffers::String> stringId = 0;
    if (!this->_name.empty()) stringId = builder.CreateString(this->_name);

    // returning array
    return CreateFlatVariable(builder, fVid, stringId, static_cast<::graph::DType>(array->dataType()), 0, fArray);
  } else {
    THROW_EXCEPTION("Variable::asFlatVariable isn't possible for NDArrayList");
  }

  return CreateFlatVariable(builder, 0, 0, static_cast<::graph::DType>(0), 0, 0);
}
}  // namespace graph
}  // namespace sd

namespace std {

size_t hash<std::pair<int, int>>::operator()(const std::pair<int, int> &k) const {
  auto v = std::hash<int>()(k.first);
  v ^= std::hash<int>()(k.second) + 0x9e3779b9 + (v << 6) + (v >> 2);
  return v;
}

size_t hash<bfloat16>::operator()(const bfloat16 &k) const { return std::hash<float>()((float)k); }

size_t hash<float16>::operator()(const float16 &k) const { return std::hash<float>()((float)k); }
}  // namespace std
