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

#ifndef DEV_TESTS_SHAPEDESCRIPTOR_H
#define DEV_TESTS_SHAPEDESCRIPTOR_H
#include <array/ArrayOptions.h>
#include <array/DataType.h>
#include <helpers/shape.h>
#include <system/common.h>

#include <initializer_list>
#include <unordered_map>
#include <vector>
namespace sd {

#define SHAPE_DESC_OK 0
#define SHAPE_DESC_INCORRECT_STRIDES 1  // strides does not match shapes
#define SHAPE_DESC_INCORRECT_EWS 2      // ews neither matches stride nor continuity
#define SHAPE_DESC_INCORRECT_RANK 4     // rank > 32 or shape size and rank does not match
#define SHAPE_DESC_INVALID_EMPTY 5     // rank > 32 or shape size and rank does not match

class SD_LIB_EXPORT ShapeDescriptor {


 private:
  int _rank = 0;
  LongType * _shape_strides;
  LongType _ews = 1;
  char _order = 'c';
  DataType _dataType;
  LongType _extraProperties = 0;
  LongType _paddedAllocSize = 0;
  LongType _offset = 0;

 public:
  bool ownsShapeStrides = false;

#ifndef __JAVACPP_HACK__
#if defined(SD_GCC_FUNCTRACE)
  StackTrace st;
  //stack trace when stored in cache.
  StackTrace storeStackTrace;
#endif
  ShapeDescriptor(const DataType type, const char order, const std::vector<LongType> &shape, LongType extras);
  ShapeDescriptor(const ShapeDescriptor &other);
  ShapeDescriptor(const LongType *shapeInfo, bool validateDataType = true, bool overrideStrides = false);
  explicit ShapeDescriptor(const LongType *shapeInfo, const DataType dtypeOverride, const bool overrideStrides);
  explicit ShapeDescriptor(const LongType *shapeInfo, const LongType *dtypeOverride);
  explicit ShapeDescriptor(const LongType *shapeInfo, const LongType *dtypeOverride,
                           const LongType *orderOverride);
  explicit ShapeDescriptor(const DataType type, const LongType length);
  explicit ShapeDescriptor(const DataType type, const char order, const LongType *shape, const LongType rank);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<LongType> &shape);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<LongType> &shape,
                           const std::vector<LongType> &strides);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<LongType> &shape,
                           const std::vector<LongType> &strides, const LongType ews);
  explicit ShapeDescriptor(const DataType type, const char order, const LongType *shape,
                           const LongType *strides, const LongType rank, LongType extras);

  ShapeDescriptor() = default;
  ~ShapeDescriptor();
#endif
  int rank() const;
  LongType ews() const;
  LongType arrLength() const;
  LongType offset();
  char order() const;
  DataType dataType() const;
  bool isEmpty() const;
  sd::LongType * shape_strides();
  const LongType *stridesPtr() const;
  LongType extra() const {
    return _extraProperties;
  }


  void collectStoreStackTrace();
  void print() const;
  // returns minimal allocation length
  LongType allocLength() const;

  // returns Status for the correctness
  LongType validate() const;

  // we use default copy assignment operator
  ShapeDescriptor &operator=(const ShapeDescriptor &other) = default;

  // we use default move assignment operator
  ShapeDescriptor &operator=(ShapeDescriptor &&other) noexcept = default;

  // equal to operator
  bool operator==(const ShapeDescriptor &other) const;

  // less than operator
  bool operator<(const ShapeDescriptor &other) const;

  LongType *toShapeInfo() const;

  const char * toString() {
    std::string message;
    message += " Rank:" ;
    message += std::to_string(_rank);
    message += " Shape and Strides:";
    if(_shape_strides == nullptr) {
      message += " Null";
    } else {
      for (int i = 0; i < _rank * 2; i++) {
        message += " ";
        message += std::to_string(_shape_strides[i]);
      }

    }
    message += "Data type:";
    message += std::to_string(_dataType);
    message += " EWS:";
    message += std::to_string(_ews);
    message += " Order:";
    message += std::to_string(_order);
    message += " Extra Properties:";
    message += std::to_string(_extraProperties);
    message += " Padded Alloc Size: ";
    message += std::to_string(_paddedAllocSize);
    message += " Offset: ";
    message += std::to_string(_offset);
    //need this in order to avoid deallocation
    std::string *ret = new std::string(message.c_str());
    return ret->c_str();
  }
  static ShapeDescriptor * emptyDescriptor(const DataType type);
  static ShapeDescriptor  * scalarDescriptor(const DataType type);
  static ShapeDescriptor * vectorDescriptor(const LongType length, const DataType type);

  // create Descriptor with padded buffer.
  static ShapeDescriptor * paddedBufferDescriptor(const DataType type, const char order,
                                                  const std::vector<LongType> &shape,
                                                  const std::vector<LongType> &paddings);

  static  const char *messageForShapeDescriptorError(const int errorCode) {
    switch (errorCode) {
      case SHAPE_DESC_OK:
        return "OK";
      case SHAPE_DESC_INCORRECT_STRIDES:
        return "Incorrect strides";
      case SHAPE_DESC_INCORRECT_EWS:
        return "Incorrect ews";
      case SHAPE_DESC_INCORRECT_RANK:
        return "Incorrect rank";
      case SHAPE_DESC_INVALID_EMPTY:
        return "Invalid empty";
      default:
        return "Unknown error";
    }
  }
  bool isScalar() const;

  SD_INLINE void fillStrides() {
    if(_rank == 0) {
      printf("No strides to be filled for rank 0\n");
      return;
    }

    if(_shape_strides == nullptr) {
      return;
    }

    // double checks if the _rank and _shape_strides are set correctly before filling strides
    auto _shape = _shape_strides;
    auto _strides = _shape_strides + _rank;
    if (_rank > 0) {
      if (_order == 'c')
        shape::calcStrides(_shape, _rank, _strides);
      else
        shape::calcStridesFortran(_shape, _rank, _strides);

    } else {
      for (int i = 0; i < _rank; i++) {
        _strides[i] = 0;
      }
    }


  }

};
}  // namespace sd

#ifndef __JAVACPP_HACK__

namespace std {
template <>
class SD_LIB_EXPORT hash<sd::ShapeDescriptor> {
 public:
  size_t operator()(sd::ShapeDescriptor k) const;
};
}  // namespace std

#endif

#endif  // DEV_TESTS_SHAPEDESCRIPTOR_H
