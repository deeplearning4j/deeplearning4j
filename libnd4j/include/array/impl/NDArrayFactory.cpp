/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Created by GS <sgazeos@gmail.com> on 2018-12-20.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <graph/GraphExecutioner.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/LoopsCoordsHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/StringUtils.h>
#include <legacy/NativeOps.h>

#include <type_traits>

namespace sd {

SD_LIB_EXPORT NDArray* NDArrayFactory::create(ShapeDescriptor *shapeDescriptor, LaunchContext* context) {
  auto status = shapeDescriptor->validate();
  if (status != SHAPE_DESC_OK) {
    THROW_EXCEPTION("NDArrayFactory::create: invalid ShapeDescriptor ");
  }
  LongType allocSize = shapeDescriptor->allocLength() * DataTypeUtils::sizeOfElement(shapeDescriptor->dataType());
  DataBuffer *  buffer =
      new DataBuffer(allocSize, shapeDescriptor->dataType(), context->getWorkspace());
  NDArray *result = new NDArray(buffer, shapeDescriptor, context);
  result->nullify();
  return result;
}


////////////////////////////////////////////////////////////////////////
template <>
SD_LIB_EXPORT NDArray* NDArrayFactory::create<bool>(const char order, const std::vector<LongType>& shape,
                                                   const std::vector<bool>& data, LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    THROW_EXCEPTION("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

  ShapeDescriptor *descriptor = new ShapeDescriptor(BOOL, order, shape);

  if (static_cast<size_t>(descriptor->arrLength()) != data.size()) {
    sd_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(),
              descriptor->arrLength());
    THROW_EXCEPTION("NDArrayFactory::create: data size doesn't match shape");
  }

  bool* hostBuffer = nullptr;
  ALLOCATE(hostBuffer, context->getWorkspace(), data.size(), bool);
  std::copy(data.begin(), data.end(), hostBuffer);

  DataBuffer * buffer = new DataBuffer(hostBuffer, data.size() * sizeof(bool), BOOL, true, context->getWorkspace());

  NDArray *result = new NDArray(buffer, descriptor, context);
  delete descriptor;
  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create(const char order,
                                const std::vector<LongType>& shape,
                                const std::vector<T>& data,
                               LaunchContext* context) {
  if (shape.size() > SD_MAX_RANK)
    THROW_EXCEPTION("NDArrayFactory::create: rank of NDArray can't exceed 32 !");
  ShapeDescriptor *descriptor = new ShapeDescriptor(DataTypeUtils::fromT<T>(), order, shape);

  //scalars can be created with zero length
  if (descriptor->arrLength() != 0 && data.size() != 1 && static_cast<size_t>(descriptor->arrLength()) != data.size()) {
    delete descriptor;
    sd_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(),
              descriptor->arrLength());
    THROW_EXCEPTION("NDArrayFactory::create: data size doesn't match shape");
  }

  T *hostData = nullptr;
  ALLOCATE(hostData, context->getWorkspace(), data.size(), T);
  std::copy(data.begin(), data.end(), hostData);

  //note here we use data.size() to work around the scalar case. If the shape is zero but the data is actually length 1 we need this reflected
  //to create a correct length data buffer
  auto dtypeString = DataTypeUtils::asString(descriptor->dataType());
  DataBuffer *  buffer = new DataBuffer(
      hostData, DataTypeUtils::fromT<T>(), data.size() * sizeof(T), context->getWorkspace());

  NDArray *result = new NDArray(buffer, descriptor, context);
  delete descriptor;
  return result;
}

// Update the instantiation macro to use the expanded type pattern
#define TMPL_INSTANTIATE_CREATE_A_TYPE(TYPE) \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create<TYPE>(const char order, const std::vector<sd::LongType>& shape, \
                                                      const std::vector<TYPE>& data, sd::LaunchContext* context);

#define TMPL_INSTANTIATE_CREATE_A(T) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(T), \
    CONCAT(EXPAND_TYPE_APPLY_, GET_SECOND(T))(TMPL_INSTANTIATE_CREATE_A_TYPE) \
))

ITERATE_LIST((SD_NUMERIC_TYPES), TMPL_INSTANTIATE_CREATE_A)

#undef TMPL_INSTANTIATE_CREATE_A_TYPE
#undef TMPL_INSTANTIATE_CREATE_A
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const char order, std::vector<LongType>& shape, LaunchContext* context) {
  return create_(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT NDArray* NDArrayFactory::create_,
                      (const char order,  std::vector<sd::LongType>& shape, sd::LaunchContext* context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<T>& vector) {
  memcpy(ptr, vector.data(), vector.size() * sizeof(T));
}

template <>
void SD_LIB_EXPORT NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<bool>& vector) {
  auto p = reinterpret_cast<bool*>(ptr);
  for (size_t e = 0; e < vector.size(); e++) p[e] = vector[e];
}


#define TMPL_INSTANTIATE_MEMCPY(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT void NDArrayFactory::memcpyFromVector<GET_SECOND(TYPE)>(void* ptr, const std::vector<GET_SECOND(TYPE)>& vector); \
))

ITERATE_LIST((SD_NUMERIC_TYPES), TMPL_INSTANTIATE_MEMCPY)
#undef TMPL_INSTANTIATE_MEMCPY

#ifndef __JAVACPP_HACK__
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::valueOf(const std::initializer_list<LongType>& shape, const T value, const char order,
                                 LaunchContext* context) {
  std::vector<sd::LongType> shape2 = shape;
  return valueOf(shape2, value, order);
}

#define TMPL_INSTANTIATE_VALUEOF_A(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf<GET_SECOND(TYPE)>(const std::initializer_list<sd::LongType>& shape, \
                                                        const GET_SECOND(TYPE) value, const char order, \
                                                        sd::LaunchContext* context); \
))
ITERATE_LIST((SD_NUMERIC_TYPES), TMPL_INSTANTIATE_VALUEOF_A)

#undef TMPL_INSTANTIATE_VALUEOF_A



#endif

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const T scalar, LaunchContext* context) {
  sd::LongType  size = DataTypeUtils::sizeOfElement(DataTypeUtils::fromT<T>());
  DataBuffer *  buffer =
      new DataBuffer(size,
                     DataTypeUtils::fromT<T>(),
                     context->getWorkspace(),
                     true);

  auto desc = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>());
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  auto recast = const_cast<LongType*>(constDesc->primary());
  NDArray* res = new NDArray(buffer, recast, context);
  res->p<T>(0,scalar);

  res->tickWriteHost();
  res->syncToDevice();

  delete[] desc;  // Free allocated shape info

  return res;
}

#define TMPL_INSTANTIATE_CREATE_C(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create_<GET_SECOND(TYPE)>(const GET_SECOND(TYPE) scalar, sd::LaunchContext* context); \
))
ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_CREATE_C)

#undef TMPL_INSTANTIATE_CREATE_C

NDArray* NDArrayFactory::create(DataType dtype, LaunchContext *context) {
  return create(dtype,0, context);
}

template <typename T>
NDArray* NDArrayFactory::create(DataType type, const T scalar, LaunchContext* context) {
  if (type == DataTypeUtils::fromT<T>()) return NDArrayFactory::create(scalar, context);

  NDArray *res = new NDArray(type, context);
  res->p(0, scalar);
  res->syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_D(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create<GET_SECOND(TYPE)>(DataType type, const GET_SECOND(TYPE) scalar, sd::LaunchContext* context); \
))

ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_CREATE_D)

#undef TMPL_INSTANTIATE_CREATE_D

template <typename T>
NDArray* NDArrayFactory::create(const T scalar, LaunchContext* context) {
  DataBuffer *  buffer =
      new DataBuffer(1 * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  auto desc = ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>());
  NDArray *res = new NDArray(buffer,desc , context);
  res->bufferAsT<T>()[0] = scalar;

  res->tickWriteHost();
  res->syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_E(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create<GET_SECOND(TYPE)>(const GET_SECOND(TYPE) scalar, sd::LaunchContext* context); \
))
ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_CREATE_E)
#undef TMPL_INSTANTIATE_CREATE_E

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<LongType>& shape, const std::vector<T>& data,
                                 LaunchContext* context) {
  return NDArrayFactory::create<T>(order, shape, data, context);
}

#define TMPL_INSTANTIATE_CREATE_F(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create_<GET_SECOND(TYPE)>(const char order, const std::vector<sd::LongType>& shape, \
                                                        const std::vector<GET_SECOND(TYPE)>& data, sd::LaunchContext* context); \
))

ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_CREATE_F)


#undef TMPL_INSTANTIATE_CREATE_F

////////////////////////////////////////////////////////////////////////
template <>
SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf(std::vector<LongType>& shape, NDArray* value, const char order,
                                               LaunchContext* context) {
  auto result = create_(order, shape, value->dataType(), context);
  result->assign(value);
  return result;
}

template <>
SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf(std::vector<LongType>& shape, NDArray& value, const char order,
                                               LaunchContext* context) {
  auto result = create_(order, shape, value.dataType(), context);
  result->assign(&value);
  return result;
}

template <typename T>
NDArray* NDArrayFactory::valueOf(std::vector<LongType>& shape,  T value, const char order,
                                 LaunchContext* context) {
  auto result = create_(order, shape, DataTypeUtils::fromT<T>());
  result->assign(value);
  return result;
}

// Replace TMPL_INSTANTIATE_VALUEOF
#define TMPL_INSTANTIATE_VALUEOF(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* \
    NDArrayFactory::valueOf<GET_SECOND(TYPE)>(std::vector<sd::LongType>& shape,  GET_SECOND(TYPE) value, \
                                                        const char order, sd::LaunchContext* context); \
))

ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_VALUEOF)


#undef TMPL_INSTANTIATE_VALUEOF

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::linspace(const T from, const T to, const LongType numElements) {
  NDArray* result = NDArrayFactory::vector<T>(numElements);
  // TO DO: linspace should be executed on DEVICE, but only CPU version implemnted!
  for (LongType e = 0; e < numElements; e++) {
    T step = (T)e / ((T)numElements - (T)1);
    result->p<T>(e, (from * ((T)1 - step) + step * to));
  }
  result->syncToDevice();

  return result;
}

#define TMPL_INSTANTIATE_LINSPACE(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::linspace<GET_SECOND(TYPE)>(const GET_SECOND(TYPE) from, const GET_SECOND(TYPE) to, \
                                                         const sd::LongType numElements); \
))
ITERATE_LIST((SD_NUMERIC_TYPES), TMPL_INSTANTIATE_LINSPACE)



#undef TMPL_INSTANTIATE_LINSPACE
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::vector(LongType length,  T value, LaunchContext* context) {
  DataBuffer *  buffer =
      new DataBuffer(length * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);
  auto desc = ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(),length);
  auto constDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  auto recast = const_cast<LongType*>(constDesc->primary());
  auto res = new NDArray(buffer, recast, context);
  if (value == (T)0.0f)
    res->nullify();
  else
    res->assign(value);

  delete[] desc;  // Free allocated shape info

  return res;
}

#define TMPL_INSTANTIATE_VECTOR(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::vector<GET_SECOND(TYPE)>(sd::LongType length, const GET_SECOND(TYPE) startingValue, \
                                                       sd::LaunchContext* context); \
))

ITERATE_LIST((SD_COMMON_TYPES_ALL), TMPL_INSTANTIATE_VECTOR)




////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray *NDArrayFactory::create(const char order, const std::vector<LongType>& shape, LaunchContext* context) {
  return create(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT NDArray *NDArrayFactory::create,
                      (const char order, const std::vector<sd::LongType>& shape, sd::LaunchContext* context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::create(const char order, const std::vector<LongType>& shape, DataType dtype,
                               LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    THROW_EXCEPTION("NDArrayFactory::create: rank of NDArray can't exceed 32");


  ShapeDescriptor *descriptor = new ShapeDescriptor(dtype, order, shape);

  DataBuffer *  buffer = new DataBuffer(
      descriptor->arrLength() * DataTypeUtils::sizeOfElement(dtype), dtype, context->getWorkspace());

  NDArray *result = new NDArray(buffer, descriptor, context);
  delete descriptor;
  result->nullify();

  return result;
}

NDArray* NDArrayFactory::create_(DataType dtype, LaunchContext* context) {
  auto result = create(dtype, context);
  return result;
}

template <typename T>
static NDArray *create(DataType type, const std::vector<LongType>& shape, LaunchContext* context) {
  auto buffer = new DataBuffer(DataTypeUtils::sizeOfElement(type) * shape::prodLong(shape.data(),shape.size()), type, context->getWorkspace());
  auto desc = ShapeBuilders::createShapeInfo(type,'c',shape);
  auto cachedDesc = ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
  NDArray *result = new NDArray(buffer, cachedDesc->primary(), context);
  delete[] desc;
  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray *NDArrayFactory::create(const std::vector<T>& values, LaunchContext* context) {
  DataBuffer *  buffer =
      new DataBuffer(values.size() * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  auto desc = ShapeDescriptor::vectorDescriptor(values.size(), DataTypeUtils::fromT<T>());
  NDArray *res = new NDArray(buffer, desc, context);
  memcpyFromVector<T>(res->buffer(), values);

  res->tickWriteHost();
  res->syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_G(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray* NDArrayFactory::create<GET_SECOND(TYPE)>(const std::vector<GET_SECOND(TYPE)>& values, sd::LaunchContext* context); \
))

ITERATE_LIST((SD_NUMERIC_TYPES), TMPL_INSTANTIATE_CREATE_G)


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::empty_(LaunchContext* context) {
  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  auto result = new NDArray(nullptr, shapeInfo, context, false, 0);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT NDArray* NDArrayFactory::empty_, (sd::LaunchContext * context),
                      SD_COMMON_TYPES_ALL);

NDArray* NDArrayFactory::empty_(DataType dataType, LaunchContext* context) {
  if (context == nullptr) context = LaunchContext ::defaultContext();

  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  auto result = new NDArray(nullptr, shapeInfo, context, false, 0);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray *NDArrayFactory::empty(LaunchContext* context) {
  return empty(DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE( SD_LIB_EXPORT NDArray *NDArrayFactory::empty, (sd::LaunchContext * context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
SD_LIB_EXPORT NDArray* NDArrayFactory::empty(DataType dataType, LaunchContext* context) {
  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  NDArray *result= new NDArray(nullptr, shapeInfo, context, false, 0);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::valueOf(std::vector<LongType>& shape, NDArray& value, const char order,
                                 LaunchContext* context) {
  auto res = create_(order, shape, value.dataType(), context);
  res->assign(&value);
  return res;
}

////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::create_(const char order, std::vector<LongType>& shape, DataType dataType,
                                 LaunchContext* context) {
  return new NDArray(order, shape, dataType, context);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray *NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<LongType>& shape,
                               LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    THROW_EXCEPTION("NDArrayFactory::create: Rank of NDArray can't exceed 32");

  std::vector<LongType> shp(shape);
  ShapeDescriptor *descriptor = new ShapeDescriptor(DataTypeUtils::fromT<T>(), order, shp);

  DataBuffer *  pBuffer = new DataBuffer(
      buffer, descriptor->arrLength() * sizeof(T), descriptor->dataType(), false, context->getWorkspace());

  NDArray *result = new NDArray(pBuffer, descriptor, context);
  delete descriptor;
  return result;
}

// Replace TMPL_INSTANTIATE_CREATE_H
#define TMPL_INSTANTIATE_CREATE_H(TYPE) \
EVAL(SD_IF_SINGLE_ALIAS_COMPILED_DECL( \
    GET_FIRST(TYPE), \
    template SD_LIB_EXPORT NDArray *NDArrayFactory::create<GET_SECOND(TYPE)>(GET_SECOND(TYPE)* buffer, const char order, \
                                                      const std::initializer_list<sd::LongType>& shape,  \
                                                      sd::LaunchContext* context); \
))
ITERATE_LIST((SD_COMMON_TYPES),TMPL_INSTANTIATE_CREATE_H)




#if defined(HAS_UTF16)
/////////////////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(const char16_t* u16string, DataType dtype, LaunchContext* context) {
  return new NDArray(u16string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char16_t* u16string, DataType dtype, LaunchContext* context) {
  return string_(std::u16string(u16string), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::u16string& u16string, DataType dtype, LaunchContext* context) {
  auto res = new NDArray(u16string, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string(const std::u16string& u16string, DataType dtype, LaunchContext* context) {
  return new NDArray(u16string, dtype, context);
}
#endif
#if defined(HAS_UTF32)
#if defined(HAS_UTF32)
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(const char32_t* u32string, DataType dtype, LaunchContext* context) {
  return new NDArray(u32string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char32_t* u32string, DataType dtype, LaunchContext* context) {
  return string_(std::u32string(u32string), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::u32string& u32string, DataType dtype, LaunchContext* context) {
  auto res = new NDArray(u32string, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray * NDArrayFactory::string(const std::u32string& u32string, DataType dtype, LaunchContext* context) {
  return new NDArray(u32string, dtype, context);
}
#endif
#endif
#if defined(HAS_UTF8)
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(const char* str, DataType dtype, LaunchContext* context) {
  return new NDArray(str, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char* str, DataType dtype, LaunchContext* context) {
  return string_(std::string(str), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::string& str, DataType dtype, LaunchContext* context) {
  auto res = new NDArray(str, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(const std::string& str, DataType dtype, LaunchContext* context) {
  return new NDArray(str, dtype, context);
}
#endif
#if defined(HAS_UTF8)
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<const char*>& strings,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<const char*>& strings,
                                 DataType dataType, LaunchContext* context) {
  std::vector<std::string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::string(s);

  return string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<std::string>& string,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, string, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<std::string>& string,
                                 DataType dataType, LaunchContext* context) {
  auto res = new NDArray(shape, string, dataType, context);
  return res;
}
#endif
#if defined(HAS_UTF16)
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape,
                               const std::initializer_list<const char16_t*>& strings, DataType dataType,
                               LaunchContext* context) {
  return new NDArray(shape, std::vector<const char16_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<const char16_t*>& strings,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string(std::vector<LongType>& shape,
                               const std::initializer_list<std::u16string>& string,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, std::vector<std::u16string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape,
                                 const std::initializer_list<const char16_t*>& strings, DataType dataType,
                                 LaunchContext* context) {
  return string_(shape, std::vector<const char16_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<const char16_t*>& strings,
                                 DataType dataType, LaunchContext* context) {
  std::vector<std::u16string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::u16string(s);

  return string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape,
                                 const std::initializer_list<std::u16string>& string, DataType dataType,
                                 LaunchContext* context) {
  return string_(shape, std::vector<std::u16string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<std::u16string>& string,
                                 DataType dataType, LaunchContext* context) {
  auto res = new NDArray(shape, string, dataType, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray * NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<std::u16string>& string,
                               DataType dtype, LaunchContext* context) {
  return new NDArray(shape, string, dtype, context);
}
#endif
#if defined(HAS_UTF32)
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape,
                               const std::initializer_list<const char32_t*>& strings, DataType dataType,
                               LaunchContext* context) {
  return new NDArray(shape, std::vector<const char32_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray * NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<const char32_t*>& strings,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray *NDArrayFactory::string(std::vector<LongType>& shape,
                               const std::initializer_list<std::u32string>& string,
                               DataType dataType, LaunchContext* context) {
  return new NDArray(shape, std::vector<std::u32string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape,
                                 const std::initializer_list<const char32_t*>& strings, DataType dataType,
                                 LaunchContext* context) {
  return string_(shape, std::vector<const char32_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<const char32_t*>& strings,
                                 DataType dataType, LaunchContext* context) {
  std::vector<std::u32string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::u32string(s);
  return string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape,
                                 const std::initializer_list<std::u32string>& string, DataType dataType,
                                 LaunchContext* context) {
  return string_(shape, std::vector<std::u32string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(std::vector<LongType>& shape, const std::vector<std::u32string>& string,
                                 DataType dataType, LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(shape, string, dataType, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray * NDArrayFactory::string(std::vector<LongType>& shape, const std::vector<std::u32string>& string,
                               DataType dtype, LaunchContext* context) {
  return new NDArray(shape, string, dtype, context);
}
#endif

NDArray NDArrayFactory::fromNpyFile(const char* fileName) {
  auto size = getFileSize(fileName);
  if (size < 0) THROW_EXCEPTION("File doesn't exit");

  auto pNPY = reinterpret_cast<char*>(numpyFromFile(std::string(fileName)));

  auto nBuffer = reinterpret_cast<void*>(dataPointForNumpy(pNPY));
  auto shape = reinterpret_cast<LongType*>(shapeBufferForNumpy(pNPY));

  auto length = shape::length(shape);
  int8_t* buffer = nullptr;
  memory::Workspace* workspace = nullptr;
  auto byteLen = length * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape));

  ALLOCATE(buffer, workspace, byteLen, int8_t);
  memcpy(buffer, nBuffer, byteLen);

  free(pNPY);

  return NDArray(buffer, shape, LaunchContext::defaultContext(), true, 0);
}
}  // namespace sd
