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
#include <system/common.h>

#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace sd {

#define SHAPE_DESC_OK 0
#define SHAPE_DESC_INCORRECT_STRIDES 1  // strides does not match shapes
#define SHAPE_DESC_INCORRECT_EWS 2      // ews neither matches stride nor continuity
#define SHAPE_DESC_INCORRECT_RANK 4     // rank > 32 or shape size and rank does not match

class SD_LIB_EXPORT ShapeDescriptor {
 private:
  int _rank = 0;
  std::vector<sd::LongType> _shape;
  std::vector<sd::LongType> _strides;
  sd::LongType _ews = 1;
  char _order = 'c';
  DataType _dataType;
  sd::LongType _extraProperties = 0;
  sd::LongType _paddedAllocSize = 0;

 public:
  ShapeDescriptor(const ShapeDescriptor &other);
  ShapeDescriptor(const sd::LongType *shapeInfo, bool inheritDtype = true);
  explicit ShapeDescriptor(const sd::LongType *shapeInfo, const sd::DataType dtypeOverride);
  explicit ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride);
  explicit ShapeDescriptor(const sd::LongType *shapeInfo, const sd::LongType *dtypeOverride,
                           const sd::LongType *orderOverride);
  explicit ShapeDescriptor(const DataType type, const sd::LongType length);
  explicit ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape, const int rank);
  explicit ShapeDescriptor(const DataType type, const char order, const std::initializer_list<sd::LongType> &shape);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape,
                           const std::vector<sd::LongType> &strides);
  explicit ShapeDescriptor(const DataType type, const char order, const std::vector<sd::LongType> &shape,
                           const std::vector<sd::LongType> &strides, const sd::LongType ews);
  explicit ShapeDescriptor(const DataType type, const char order, const sd::LongType *shape,
                           const sd::LongType *strides, const int rank, sd::LongType ews, sd::LongType extras);

  ShapeDescriptor() = default;
  ~ShapeDescriptor() = default;

  int rank() const;
  sd::LongType ews() const;
  sd::LongType arrLength() const;
  char order() const;
  DataType dataType() const;
  bool isEmpty() const;
  std::vector<sd::LongType> &shape();
  std::vector<sd::LongType> &strides();

  // returns minimal allocation length
  sd::LongType allocLength() const;

  // returns Status for the correctness
  sd::LongType validate() const;

  // we use default copy assignment operator
  ShapeDescriptor &operator=(const ShapeDescriptor &other) = default;

  // we use default move assignment operator
  ShapeDescriptor &operator=(ShapeDescriptor &&other) noexcept = default;

  // equal to operator
  bool operator==(const ShapeDescriptor &other) const;

  // less than operator
  bool operator<(const ShapeDescriptor &other) const;

  sd::LongType *toShapeInfo() const;

  static ShapeDescriptor emptyDescriptor(const DataType type);
  static ShapeDescriptor scalarDescriptor(const DataType type);
  static ShapeDescriptor vectorDescriptor(const sd::LongType length, const DataType type);

  // create Descriptor with padded buffer.
  static ShapeDescriptor paddedBufferDescriptor(const DataType type, const char order,
                                                const std::vector<sd::LongType> &shape,
                                                const std::vector<sd::LongType> &paddings);
};
}  // namespace sd

#ifndef __JAVACPP_HACK__

namespace std {
template <>
class SD_LIB_EXPORT hash<sd::ShapeDescriptor> {
 public:
  size_t operator()(const sd::ShapeDescriptor &k) const;
};
}  // namespace std

#endif

#endif  // DEV_TESTS_SHAPEDESCRIPTOR_H
