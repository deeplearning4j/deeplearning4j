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
//  @author raver119@gmail.com
//  @author AbdelRauf

#include <array/ShapeDescriptor.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/shape.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////
// equal to operator
bool ShapeDescriptor::operator==(const ShapeDescriptor &other) const {
  if (_extraProperties != other._extraProperties) return false;
  if (_rank != other._rank) return false;
  if (_order != other._order) return false;
  if (_dataType != other._dataType) return false;
  if (_ews != other._ews) return false;

  if (_shape_strides != other._shape_strides) return false;

  return true;
}

//////////////////////////////////////////////////////////////////////////
// less than operator
bool ShapeDescriptor::operator<(const ShapeDescriptor &other) const {
  return std::tie(_extraProperties, _rank, _dataType, _ews, _order, _shape_strides) <
         std::tie(other._extraProperties, other._rank, other._dataType, other._ews, other._order, other._shape_strides);
}

sd::LongType *ShapeDescriptor::toShapeInfo() const {
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  // for empty array use original
  if (isEmpty()) {
    if (_rank == 0)
      return ShapeBuilders::emptyShapeInfo(_dataType);
    else {
      return ShapeBuilders::emptyShapeInfo(_dataType, _order, _rank, _shape);
    }
  }

  sd::LongType *shapeInfo;
  switch (_rank) {
    case 0: {
      shapeInfo = ShapeBuilders::createScalarShapeInfo(_dataType);
      shapeInfo[2] = _ews;
    } break;
    case 1: {
      shapeInfo = ShapeBuilders::createVectorShapeInfo(_dataType, _shape[0]);
      shapeInfo[2 + _rank * 2] = _ews;
      shapeInfo[2] = _strides[0];
      shapeInfo[2 + _rank * 2 + 1] = _order;
    } break;
    default: {
      shapeInfo = ShapeBuilders::createShapeInfo(_dataType, _order, _rank, _shape);
      for (int e = 0; e < _rank; e++) shapeInfo[e + 1 + _rank] = _strides[e];
      shapeInfo[2 + _rank * 2] = _ews;
    }
  }


  ArrayOptions::setPropertyBit(shapeInfo, _extraProperties);
  return shapeInfo;
}

ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape, const int rank)
    : _dataType(type), _order(order), _rank(rank), _ews(1) {
  _shape_strides.resize(2 * rank);
  auto _shape = _shape_strides.data();
  for (int i = 0; i < _rank; i++) {
    _shape[i] = shape[i];
  }

  fillStrides();
}

ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape,
                                 const sd::LongType *strides, const int rank, sd::LongType ews, sd::LongType extras) {
  _shape_strides.resize(2 * rank);
  _dataType = type;
  _order = order;
  _rank = rank;
  _extraProperties = extras;
  _ews = ews;
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + rank;
  for (int e = 0; e < rank; e++) {
    _shape[e] = shape[e];
    _strides[e] = strides[e];
    if (shape[e] == 0) _extraProperties |= ARRAY_EMPTY;
  }
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape)
    : _dataType(type), _order(order) {
  _rank = shape.size();
  _ews = 1;
  _shape_strides.resize(2 * _rank);
  auto _shape = _shape_strides.data();
  for (int i = 0; i < _rank; i++) {
    _shape[i] = shape[i];
  }
  fillStrides();
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order,
                                 const std::initializer_list<sd::LongType> &shape)
    : _dataType(type), _order(order) {
  _rank = shape.size();
  _ews = 1;
  _shape_strides.resize(2 * _rank);
  auto _shape = _shape_strides.data();
  int i = 0;
  for (auto x : shape) {
    _shape[i] = x;
    ++i;
  }
  fillStrides();
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape,
                                 const std::vector<sd::LongType> &strides, const sd::LongType ews)
    : ShapeDescriptor(type, order, shape, strides) {
  _ews = ews;
}

ShapeDescriptor::ShapeDescriptor(const DataType type, const sd::LongType length)
    : _dataType(type), _ews(1), _order('c'), _rank(1), _extraProperties(0) {
  _shape_strides = {length, 1};  //{shape, stride}
}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, bool inheritDtype) {
  if(shapeInfo == nullptr) {
    throw std::runtime_error("ShapeDescriptor constructor: Shape info can not be null!");
  }

  if(shape::rank(shapeInfo) < 0 || shape::rank(shapeInfo) > SD_MAX_RANK) {
    throw std::runtime_error("ShapeDescriptor constructor: Corrupt shape buffer found. Likely was deallocated. Please ensure proper usage of the buffer\n");
  }

  _order = shape::order(shapeInfo);
  sd_printf("ShapeDescriptor constructor: Determined order\n",0);
  _ews = shape::elementWiseStride(shapeInfo);
  sd_printf("ShapeDescriptor constructor: Determined ews\n",0);
  _rank = shape::rank(shapeInfo);
  sd_printf("ShapeDescriptor constructor: Determined rank\n",0);
  _extraProperties = ArrayOptions::propertyWithoutDataType(shapeInfo);
  sd_printf("ShapeDescriptor constructor: Determined extra properties\n",0);
  if (inheritDtype) _dataType = ArrayOptions::dataType(shapeInfo);
  sd_printf("ShapeDescriptor constructor: Determined dtype\n",0);

  _shape_strides.resize(2 * _rank);

  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  auto shapePtr = shape::shapeOf(shapeInfo);
  auto stridePtr = shape::stride(shapeInfo);

  for (int e = 0; e < _rank; e++) {
    _shape[e] = shapePtr[e];
    _strides[e] = stridePtr[e];
    if (shapePtr[e] == 0) _extraProperties |= ARRAY_EMPTY;
  }

}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::DataType dtypeOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, false) {
  _dataType = dtypeOverride;
  sd_printf("Invoking with data type override 2\n",0);
}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, ArrayOptions::dataType(dtypeOverride)) {
          sd_printf("Invoking with data type override\n",0);
}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride,
                                 const sd::LongType *orderOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, ArrayOptions::dataType(dtypeOverride)) {
  _order = shape::order(orderOverride);
  sd_printf("Invoking with order override 2\n",0);
}

int ShapeDescriptor::rank() const { return _rank; }

sd::LongType ShapeDescriptor::ews() const { return _ews; }

sd::LongType ShapeDescriptor::arrLength() const {
  // when _ews == 1 allocation length is also array length
  sd::LongType len = 1;
  for (int i = 0; i < _rank; i++) len *= _shape_strides[i];
  return len;
}

sd::LongType ShapeDescriptor::allocLength() const {
  if (_paddedAllocSize > 0) return _paddedAllocSize;
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  sd::LongType len = 1;
  if (_ews == 1 && _rank > 1) {
    // calculate using max stride
    int ind = _order == 'c' ? 0 : _rank - 1;
    return _shape[ind] * _strides[ind];
  }
  for (int i = 0; i < _rank; i++) {
    len += (_shape[i] - 1) * _strides[i];
  }
  return len;
}

sd::LongType ShapeDescriptor::validate() const {
  auto status = SHAPE_DESC_OK;
  bool is_continous = true;
  if (_rank != _shape_strides.size() / 2 || _rank > SD_MAX_RANK) status |= SHAPE_DESC_INCORRECT_RANK;
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  if (_rank > 0) {
    if (_order == 'c') {
      for (int j = _rank - 2; j >= 0; j--) {
        sd::LongType currentStride = _strides[j];
        sd::LongType allowedStride = _strides[j + 1] * _shape[j + 1];
        if (currentStride < allowedStride) {
          status = status | SHAPE_DESC_INCORRECT_STRIDES;
          break;
        }
        is_continous = is_continous & (currentStride == allowedStride);
      }
    } else {
      for (int j = 1; j < _rank; j++) {
        sd::LongType currentStride = _strides[j];
        sd::LongType allowedStride = _strides[j - 1] * _shape[j - 1];
        if (currentStride < allowedStride) {
          status = status | SHAPE_DESC_INCORRECT_STRIDES;
          break;
        }
        is_continous = is_continous & (currentStride == allowedStride);
      }
    }

    int index = (_order == 'c') ? _rank - 1 : 0;
    auto correctEws = is_continous ? _strides[index] : 0;
    if (correctEws != _ews) status = status | SHAPE_DESC_INCORRECT_EWS;
  }
  return status;
}

char ShapeDescriptor::order() const { return _order; }

DataType ShapeDescriptor::dataType() const { return _dataType; }

bool ShapeDescriptor::isEmpty() const { return _extraProperties & ARRAY_EMPTY; }

std::vector<sd::LongType> &ShapeDescriptor::shape_strides() { return _shape_strides; }

const sd::LongType *ShapeDescriptor::stridesPtr() const {
  return _shape_strides.size() == 2 * _rank ? _shape_strides.data() + _rank : nullptr;
}

ShapeDescriptor::ShapeDescriptor(const ShapeDescriptor &other) {
  _rank = other._rank;
  _ews = other._ews;
  _extraProperties = other._extraProperties;
  _dataType = other._dataType;
  _order = other._order;
  _shape_strides = other._shape_strides;
  _paddedAllocSize = other._paddedAllocSize;
}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape,
                                 const std::vector<sd::LongType> &strides)
    : _dataType(type), _order(order) {
  _rank = shape.size();

  _shape_strides.resize(2 * _rank);
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  if (!shape.empty() && strides.size() != shape.size() ) {
    for (int i = 0; i < _rank; i++) {
      _shape[i] = shape[i];
    }
    fillStrides();
  } else {
    for (int i = 0; i < _rank; i++) {
      _shape[i] = shape[i];
      _strides[i] = strides[i];
      if (shape[i] == 0) {
        _extraProperties |= ARRAY_EMPTY;
      }
    }
  }
}

ShapeDescriptor  * ShapeDescriptor::emptyDescriptor(const DataType type) {
  ShapeDescriptor *descriptor = new ShapeDescriptor();
  descriptor->_dataType = type;
  descriptor->_extraProperties = ARRAY_EMPTY;
  descriptor->_rank = 0;
  descriptor->_order = 'c';
  descriptor->_ews = 1;

  return descriptor;
}

ShapeDescriptor * ShapeDescriptor::scalarDescriptor(const DataType type) {
  ShapeDescriptor *descriptor = new ShapeDescriptor();
  descriptor->_dataType = type;
  descriptor->_extraProperties = 0;
  descriptor->_rank = 0;
  descriptor->_order = 'c';
  descriptor->_ews = 1;

  return descriptor;
}

ShapeDescriptor * ShapeDescriptor::vectorDescriptor(const sd::LongType length, const DataType type) {
  ShapeDescriptor *descriptor = new ShapeDescriptor();
  descriptor->_dataType = type;
  descriptor->_shape_strides = {length, 0};

  if (length > 0)
    descriptor->_shape_strides[1] = 1;
  else {
    descriptor->_shape_strides[1] = 0;
    descriptor->_extraProperties = ARRAY_EMPTY;
  }

  descriptor->_order = 'c';
  descriptor->_ews = 1;
  descriptor->_rank = 1;

  return descriptor;
}

ShapeDescriptor  * ShapeDescriptor::paddedBufferDescriptor(const DataType type, const char order,
                                                        const std::vector<sd::LongType> &shape,
                                                        const std::vector<sd::LongType> &paddings) {
  ShapeDescriptor *descriptor = new ShapeDescriptor();
  descriptor->_dataType = type;
  descriptor->_order = order;
  descriptor->_rank = shape.size();
  descriptor->_extraProperties = 0;
  if (descriptor->_rank < 1) {
    descriptor->_ews = 1;
    return descriptor;
  }

  descriptor->_shape_strides.resize(descriptor->_rank * 2);
  auto _shape = descriptor->_shape_strides.data();
  auto _strides = descriptor->_shape_strides.data() + descriptor->_rank;
  for (int i = 0; i < shape.size(); i++) {
    _shape[i] = shape[i];
  }
  // calculate strides with paddings
  int min_rank = descriptor->_rank > paddings.size() ? paddings.size() : descriptor->_rank;
  bool is_continous = true;
  if (order == 'c') {
    _strides[descriptor->_rank - 1] = 1L;
    for (int j = descriptor->_rank - 2; j >= 0; j--) {
      sd::LongType pad = (j + 1 < min_rank) ? paddings[j + 1] : 0;
      _strides[j] = _strides[j + 1] * (_shape[j + 1] + pad);
      descriptor->_extraProperties = descriptor->_extraProperties | (_shape[j + 1] == 0);
      if (pad != 0) is_continous = false;
    }
    if (!is_continous && descriptor->_rank > 0) {
      sd::LongType size_pad = paddings.size() > 0 ? paddings[0] : 0;
      // alloc size should be supplied manually as we dont have place to store it
      descriptor->_paddedAllocSize = _strides[0] * (_shape[0] + size_pad);
    }
  } else {
    _strides[0] = 1L;
    for (int j = 1; j < descriptor->_rank; j++) {
      sd::LongType pad = (j - 1 < min_rank) ? paddings[j - 1] : 0;
      _strides[j] = _strides[j - 1] * (_shape[j - 1] + pad);
      descriptor->_extraProperties = descriptor->_extraProperties | (_shape[j - 1] == 0);
      if (pad != 0) is_continous = false;
    }
    if (!is_continous && descriptor->_rank > 0) {
      sd::LongType size_pad = paddings.size() >= descriptor->_rank ? paddings[descriptor->_rank - 1] : 0;
      // alloc size should be supplied manually as we dont have place to store it
      descriptor->_paddedAllocSize = _strides[descriptor->_rank - 1] * (_shape[descriptor->_rank - 1] + size_pad);
    }
  }

  descriptor->_ews = is_continous ? 1 : 0;
  if (!is_continous) descriptor->_extraProperties |= ARRAY_HAS_PADDED_BUFFER;
  return descriptor;
}

}  // namespace sd

namespace std {
size_t hash<sd::ShapeDescriptor>::operator()(const sd::ShapeDescriptor &k) const {
#if defined(__NEC__)
 //simplified
   auto res = std::hash<char>()(k.order());
  res ^= std::hash<int>()((int)k.dataType()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  // res ^= std::hash<int>()(k.rank()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  // res ^= std::hash<sd::LongType>()(k.ews()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  auto shape_strides = const_cast<sd::ShapeDescriptor &>(k).shape_strides();
  auto ptr = shape_strides.data();
  // auto strides = const_cast<sd::ShapeDescriptor &>(k).strides();
  //dont include strides if its' ews==1
  int stop = k.ews()==1? shape_strides.size()/2 : shape_strides.size() ;
  for (int j=0; j < stop; j++) {
    res ^= std::hash<sd::LongType>()(ptr[j]) + 0x9e3779b9 + (res << 6) + (res >> 2);
  }

  return res;

#else
  auto res = std::hash<sd::LongType>()(k.arrLength());
  res ^= std::hash<char>()(k.order()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  res ^= k.dataType() + 0x9e3779b9 + (res << 6) + (res >> 2);
  res ^= std::hash<int>()(k.rank()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  res ^= std::hash<sd::LongType>()(k.ews()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  auto shape_strides = const_cast<sd::ShapeDescriptor &>(k).shape_strides();
  // auto strides = const_cast<sd::ShapeDescriptor &>(k).strides();
  for (auto s : shape_strides) {
    res ^= std::hash<sd::LongType>()(s) + 0x9e3779b9 + (res << 6) + (res >> 2);
  }

  return res;
#endif
}
}  // namespace std
