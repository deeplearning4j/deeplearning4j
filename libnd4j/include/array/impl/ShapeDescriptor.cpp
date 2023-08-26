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
  // for empty array use original
  return ShapeBuilders::createShapeInfoFrom(const_cast<ShapeDescriptor *>(this));
}

ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape, const LongType rank)
    : _dataType(type), _order(order), _rank(rank), _ews(1) {
  int rank2 = rank < 1 ? 1 : rank;
  _shape_strides.resize(2 * rank2);
  auto _shape = _shape_strides.data();
  for (int i = 0; i < rank2; i++) {
    _shape[i] = shape[i];
  }

  fillStrides();
}

ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape,
                                 const sd::LongType *strides, const LongType rank, sd::LongType ews, sd::LongType extras) {
  if(shape == nullptr)
    THROW_EXCEPTION("ShapeDescriptor constructor: Shape can not be null!");

  if(strides == nullptr)
    THROW_EXCEPTION("ShapeDescriptor constructor: Strides can not be null!");

  //note this used to operate directly on the vector buffer
  //it now does manual copies with more checks.
  //this is to handle the 0 length case.
  if(rank < 1) {
    _dataType = type;
    _order = order;
    _rank = rank;
    //_extraProperties |= ARRAY_EMPTY;
  } else {
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


}

//////////////////////////////////////////////////////////////////////////
ShapeDescriptor::ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape)
    : _dataType(type), _order(order) {
  _rank = shape.size();
  printf("Set rank to %d\n",_rank);
  int rank2 = shape.size() < 1 ? 1 : shape.size();
  _ews = 1;
  _shape_strides.resize(2 * rank2);
  printf("After resize\n");
  if(_rank > 0) {
    auto _shape = _shape_strides.data();
    for (int i = 0; i < rank2; i++) {
      _shape[i] = shape[i];
    }
    printf("About to fill in strides\n");
    _order = order;
    fillStrides();
  }

  printf("Created shape descriptor object\n");
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
    THROW_EXCEPTION("ShapeDescriptor constructor: Shape info cannot be null!");
  }

  int rankVal = shape::rank(shapeInfo);

  if(rankVal < 0 || rankVal > SD_MAX_RANK) {
    THROW_EXCEPTION("ShapeDescriptor constructor: Corrupt shape buffer found. Likely was deallocated. Please ensure proper usage of the buffer\n");
  }

  _order = shape::order(shapeInfo);
  _ews = shape::elementWiseStride(shapeInfo);
  _rank = rankVal;

  _extraProperties = ArrayOptions::extra(const_cast<LongType *>(shapeInfo));
  ArrayOptions::unsetAllFlags(_extraProperties);
  if(ArrayOptions::hasPropertyBitSet(shapeInfo, ARRAY_EMPTY) && inheritDtype) {
    printf("ShapeDescriptor constructor: Empty array\n");

    _dataType = ArrayOptions::dataType(shapeInfo);
    _extraProperties = ARRAY_EMPTY | _dataType;
  } else {
    printf("ShapeDescriptor constructor: Not Empty array\n");
    _extraProperties = ArrayOptions::propertyWithoutDataType(shapeInfo);
    _dataType = ArrayOptions::dataType(shapeInfo);  // Ensure datatype is set even when array is not empty
  }

  if (_rank > 0) {
    _shape_strides.resize(2 * _rank);
    auto _shape = _shape_strides.data();
    auto _strides = _shape_strides.data() + _rank;
    auto shapePtr = shape::shapeOf(shapeInfo);
    auto stridePtr = shape::stride(shapeInfo);

    for (sd::LongType e = 0; e < _rank; e++) {
      _shape[e] = shapePtr[e];
      _strides[e] = stridePtr[e];
      if (shapePtr[e] == 0 && ArrayOptions::hasPropertyBitSet(shapeInfo, ARRAY_EMPTY)) {
        _extraProperties |= ARRAY_EMPTY;
      }
    }
  } else {  // Handle scalar case
    _shape_strides.resize(2); // Since we're setting shape and stride
    _shape_strides[0] = 0;    // Shape for scalar
    _shape_strides[1] = 1;    // Stride for scalar
  }
}
ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::DataType dtypeOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, false) {
  _dataType = dtypeOverride;
}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, ArrayOptions::dataType(dtypeOverride)) {
}

ShapeDescriptor::ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride,
                                 const sd::LongType *orderOverride)
    : ShapeDescriptor::ShapeDescriptor(shapeInfo, ArrayOptions::dataType(dtypeOverride)) {
  _order = shape::order(orderOverride);
}

int ShapeDescriptor::rank() const { return _rank; }

sd::LongType ShapeDescriptor::ews() const { return _ews; }

sd::LongType ShapeDescriptor::arrLength() const {
  if(_shape_strides.empty()) {
    return 0;
  }

  // when _ews == 1 allocation length is also array length
  sd::LongType len = 1;
  for (int i = 0; i < _rank; i++) len *= _shape_strides[i];

  return len;
}

sd::LongType ShapeDescriptor::allocLength() const {
  if (_paddedAllocSize > 0) return _paddedAllocSize;
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + _rank;
  int rank2 = _rank < 1 ? 1 : _rank;

  sd::LongType len = 1;
  if (_ews == 1 && _rank > 1) {
    // calculate using max stride
    int ind = _order == 'c' ? 0 : rank2 - 1;
    return _shape[ind] * _strides[ind];
  }
  for (int i = 0; i < rank2; i++) {
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

bool ShapeDescriptor::isEmpty() const { return (_extraProperties & ARRAY_EMPTY) == ARRAY_EMPTY; }
bool ShapeDescriptor::isScalar() const { return !isEmpty() && rank() == 0 || rank() == 1 && arrLength() == 1; }

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
  int rank2 = _rank < 1 ? 1 : _rank;

  _shape_strides.resize(2 * rank2);
  auto _shape = _shape_strides.data();
  auto _strides = _shape_strides.data() + rank2;
  if (!shape.empty() && strides.size() != shape.size() ) {
    for (int i = 0; i < rank2; i++) {
      _shape[i] = shape[i];
    }
    fillStrides();
  } else {
    for (int i = 0; i < rank2; i++) {
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

  int rank2 = descriptor->_rank < 1 ? 1 : descriptor->_rank;

  descriptor->_shape_strides.resize(rank2 * 2);
  auto _shape = descriptor->_shape_strides.data();
  auto _strides = descriptor->_shape_strides.data() + rank2;
  for (int i = 0; i < shape.size(); i++) {
    _shape[i] = shape[i];
  }
  // calculate strides with paddings
  int min_rank = descriptor->_rank > paddings.size() ? paddings.size() : rank2;
  bool is_continous = true;
  if (order == 'c') {
    _strides[rank2 - 1] = 1L;
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
    for (int j = 1; j < rank2; j++) {
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
  auto res = std::hash<char>()(k.order());
  res ^= std::hash<int>()((int)k.dataType()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  auto shape_strides = const_cast<sd::ShapeDescriptor &>(k).shape_strides();
  auto ptr = shape_strides.data();
  //dont include strides if its' ews==1
  int stop = k.ews()==1? shape_strides.size()/2 : shape_strides.size() ;
  for (int j=0; j < stop; j++) {
    res ^= std::hash<sd::LongType>()(ptr[j]) + 0x9e3779b9 + (res << 6) + (res >> 2);
  }

  return res;
}
}  // namespace std
