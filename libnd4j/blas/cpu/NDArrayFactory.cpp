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
// Created by remote on 2018-09-16.
//

#include <NDArrayFactory.h>
#include <ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <openmp_pragmas.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {


    NDArray NDArrayFactory::string(const char *str, nd4j::graph::LaunchContext* context) {
        std::string s(str);
        return string(s, context);
    }

    NDArray* NDArrayFactory::string_(const char *str, nd4j::graph::LaunchContext* context) {
        return string_(std::string(str), context);
    }

    NDArray NDArrayFactory::string(const std::string &str, nd4j::graph::LaunchContext* context) {
        NDArray res;

        int8_t *buffer = nullptr;
        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);
        ALLOCATE(buffer, context->getWorkspace(), headerLength + str.length(), int8_t);
        auto offsets = reinterpret_cast<Nd4jLong *>(buffer);
        auto data = buffer + headerLength;

        offsets[0] = 0;
        offsets[1] = str.length();

        memcpy(data, str.c_str(), str.length());

        res.setBuffer(buffer);
        res.setContext(context);
        res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataType::UTF8, context->getWorkspace()));

        res.triggerAllocationFlag(true);

        return res;
    }

    NDArray* NDArrayFactory::string_(const std::string &str, nd4j::graph::LaunchContext* context) {
        auto res = new NDArray();

        int8_t *buffer = nullptr;
        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);
        ALLOCATE(buffer, context->getWorkspace(), headerLength + str.length(), int8_t);
        auto offsets = reinterpret_cast<Nd4jLong *>(buffer);
        auto data = buffer + headerLength;

        offsets[0] = 0;
        offsets[1] = str.length();

        memcpy(data, str.c_str(), str.length());

        res->setBuffer(buffer);
        res->setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataType::UTF8, context->getWorkspace()));
        res->setContext(context);

        res->triggerAllocationFlag(true);

        return res;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, nd4j::graph::LaunchContext* context) {
    return create_(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::create_, (const char order, const std::vector<Nd4jLong> &shape, nd4j::graph::LaunchContext* context), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<T> &vector) {
    memcpy(ptr, vector.data(), vector.size() * sizeof(T));
}

template <>
void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<bool> &vector) {
    auto p = reinterpret_cast<bool *>(ptr);
    Nd4jLong vectorSize = vector.size();
    for (Nd4jLong e = 0; e < vectorSize; e++)
        p[e] = vector[e];       
}

template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<double> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<float> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<float16> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<Nd4jLong> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<int> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<int16_t> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<uint8_t> &vector);
template void NDArrayFactory::memcpyFromVector(void *ptr, const std::vector<int8_t> &vector);


#ifndef __JAVACPP_HACK__
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order, nd4j::graph::LaunchContext* context) {
        return valueOf(std::vector<Nd4jLong>(shape), value, order);
    }
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const double value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float16 value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int8_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int16_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bool value, const char order, nd4j::graph::LaunchContext* context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<T>& data, nd4j::graph::LaunchContext* context) {
        std::vector<T> vec(data);
        return create<T>(order, shape, vec, context);
    }
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<double>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float16>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bfloat16>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<Nd4jLong>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int16_t>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int8_t>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<uint8_t>& data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bool>& data, nd4j::graph::LaunchContext* context);

#endif

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::create_(const T scalar, nd4j::graph::LaunchContext* context) {
        
        auto res = new NDArray();

        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        res->setAttached(context->getWorkspace() != nullptr);

        int8_t *buffer;
        ALLOCATE(buffer, context->getWorkspace(), 1 * sizeof(T), int8_t);

        auto shapeinfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());

        res->setContext(context);
        res->setShapeInfo(shapeinfo);
        res->setBuffer(buffer);
        res->triggerAllocationFlag(true);

        res->assign(scalar);

        RELEASE(shapeinfo, context->getWorkspace());

        return res;
    }
    template NDArray* NDArrayFactory::create_(const double scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const float scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const float16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const bfloat16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const Nd4jLong scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const int scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const bool scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const int8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const uint8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::create_(const int16_t scalar, nd4j::graph::LaunchContext* context);

    template <typename T>
    NDArray NDArrayFactory::create(DataType type, const T scalar, nd4j::graph::LaunchContext* context) {

        NDArray res(type, context);

        //int8_t *buffer;
        //ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);

        //res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace));
        //res.setBuffer(buffer);
        //res.triggerAllocationFlag(true);
        //res.setWorkspace(workspace);

        res.p(0, scalar);

        return res;
    }
//    BUILD_DOUBLE_TEMPLATE(template NDArray NDArrayFactory::create, (DataType type, const T scalar, nd4j::graph::LaunchContext* context), LIBND4J_TYPES);
    template NDArray NDArrayFactory::create(DataType type, const double scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const float scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const float16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const bfloat16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const Nd4jLong scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const int scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const int8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const uint8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const int16_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(DataType type, const bool scalar, nd4j::graph::LaunchContext* context);

    template <typename T>
    NDArray NDArrayFactory::create(const T scalar, nd4j::graph::LaunchContext* context) {

        NDArray res;

        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        res.setAttached(context->getWorkspace() != nullptr);

        int8_t *buffer;
        ALLOCATE(buffer, context->getWorkspace(), 1 * sizeof(T), int8_t);

        auto shape = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());

        res.setContext(context);
        res.setShapeInfo(shape);
        res.setBuffer(buffer);
        res.triggerAllocationFlag(true);

        res.bufferAsT<T>()[0] = scalar;

        RELEASE(shape, context->getWorkspace());

        return res;
    }
    template NDArray NDArrayFactory::create(const double scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const float scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const float16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const bfloat16 scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const Nd4jLong scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const int scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const int8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const uint8_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const int16_t scalar, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const bool scalar, nd4j::graph::LaunchContext* context);


////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::graph::LaunchContext* context) {
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    auto result = new NDArray();

    if (context == nullptr)
        context = nd4j::graph::LaunchContext::defaultContext();

    result->setAttached(context->getWorkspace() != nullptr);

    result->setShapeInfo(ShapeBuilders::createShapeInfo(DataTypeUtils::fromT<T>(), order, shape, context->getWorkspace()));

    if (result->lengthOf() != data.size()) {
        nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(result->shapeInfo()));
        throw std::runtime_error("Data size doesn't match shape");
    }
        
    int8_t* buffer(nullptr);
    ALLOCATE(buffer, context->getWorkspace(), result->lengthOf() * DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>()), int8_t);
    result->setBuffer(buffer);
    result->setContext(context);
    result->triggerAllocationFlag(true);
    memcpyFromVector(result->getBuffer(), data);        // old memcpy_

    return result;
}
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bfloat16> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::graph::LaunchContext* context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::graph::LaunchContext* context);


    ////////////////////////////////////////////////////////////////////////
    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray* value, const char order, nd4j::graph::LaunchContext* context) {
        auto result = create_(order, shape, value->dataType());
        result->assign(*value);
        return result;
    }

    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray& value, const char order, nd4j::graph::LaunchContext* context) {
        auto result = create_(order, shape, value.dataType());
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order, nd4j::graph::LaunchContext* context) {
        auto result = create_(order, shape, DataTypeUtils::fromT<T>());
        result->assign(value);
        return result;
    }
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const double value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float16 value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int16_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int8_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bool value, const char order, nd4j::graph::LaunchContext* context);


    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::linspace(const T from, const T to, const Nd4jLong numElements) {
        auto result = NDArrayFactory::vector<T>(numElements);

        auto b = reinterpret_cast<T *>(result->getBuffer());
        auto t1 = static_cast<T>(1.0f);
        auto tn = static_cast<T>(numElements);

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (Nd4jLong e = 0; e < numElements; e++) {
            T step = (T) e / (tn - t1);
            b[e] = (from * (t1 - step) + step * to);
        }
        return result;
    }
    template NDArray* NDArrayFactory::linspace(const double from, const double to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const float from, const float to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const float16 from, const float16 to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const bfloat16 from, const bfloat16 to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const Nd4jLong from, const Nd4jLong to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const int from, const int to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const int16_t from, const int16_t to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const uint8_t from, const uint8_t to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const int8_t from, const int8_t to, const Nd4jLong numElements);
    template NDArray* NDArrayFactory::linspace(const bool from, const bool to, const Nd4jLong numElements);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::vector(Nd4jLong length, const T startingValue, nd4j::graph::LaunchContext* context) {

        auto res = new NDArray();

        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        res->setAttached(context->getWorkspace() != nullptr);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), length * sizeof(T), int8_t);

        //ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), length, context->getWorkspace())
        res->setShapeInfo(ShapeDescriptor(DataTypeUtils::fromT<T>(), length));
        res->setBuffer(buffer);
        res->setContext(context);
        res->triggerAllocationFlag(true);

        if (startingValue == (T)0.0f) {
            memset(buffer, 0, length);
        } else {
            res->assign(startingValue);
        }

        return res;
    }
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const double startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float16 startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bfloat16 startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const Nd4jLong startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const uint8_t startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int8_t startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int16_t startingValue, nd4j::graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bool startingValue, nd4j::graph::LaunchContext* context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context) {
        std::vector<Nd4jLong> vec(shape);
        return create<T>(order, vec, context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char, const std::initializer_list<Nd4jLong>&, graph::LaunchContext*), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::graph::LaunchContext* context) {
        return create(order, shape, DataTypeUtils::fromT<T>(), context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char order, const std::vector<Nd4jLong> &shape, nd4j::graph::LaunchContext* context), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {
  
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    NDArray res;

    if (context == nullptr)
        context = nd4j::graph::LaunchContext::defaultContext();

    res.setAttached(context->getWorkspace() != nullptr);
    ShapeDescriptor descriptor(dtype, order, shape);
    auto constantBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    res.setShapeInfo(constantBuffer.primaryAsT<Nd4jLong>());
    
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, context->getWorkspace(), res.lengthOf() * DataTypeUtils::sizeOfElement(dtype), int8_t);
    memset(buffer, 0, res.lengthOf() * res.sizeOfT());

    res.setBuffer(buffer);
    res.setContext(context);
    res.triggerAllocationFlag(true);

    return res;
}


////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {
    
    NDArray res;

    if (context == nullptr)
        context = nd4j::graph::LaunchContext::defaultContext();

    res.setAttached(context->getWorkspace() != nullptr);

    int8_t *buffer = nullptr;
    ALLOCATE(buffer, context->getWorkspace(), DataTypeUtils::sizeOfElement(dtype), int8_t);
    memset(buffer, 0, DataTypeUtils::sizeOfElement(dtype));
    res.setBuffer(buffer);
    res.setContext(context);
    res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(dtype, context->getWorkspace()));
    res.triggerAllocationFlag(true);
    
    return res;
}


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const std::vector<T> &values, nd4j::graph::LaunchContext* context) {
        
    NDArray res;

    if (context == nullptr)
        context = nd4j::graph::LaunchContext::defaultContext();

    res.setAttached(context->getWorkspace() != nullptr);

    Nd4jLong* shapeInfo = ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), values.size(), context->getWorkspace());
    res.setShapeInfo(shapeInfo);
    RELEASE(shapeInfo, context->getWorkspace());
        
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, context->getWorkspace(), values.size() * sizeof(T), int8_t);
    memcpyFromVector<T>(buffer, values);
        
    res.setBuffer(buffer);        
    res.triggerAllocationFlag(true);
    res.setContext(context);
        
    return res;
}
template NDArray NDArrayFactory::create(const std::vector<double> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<float> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<float16> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<bfloat16> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<Nd4jLong> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<int> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<int16_t> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<int8_t> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<uint8_t> &values, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(const std::vector<bool> &values, nd4j::graph::LaunchContext* context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::empty_(nd4j::graph::LaunchContext* context) {
        return empty_(DataTypeUtils::fromT<T>(), context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::empty_, (nd4j::graph::LaunchContext* context), LIBND4J_TYPES);

    NDArray* NDArrayFactory::empty_(nd4j::DataType dataType, nd4j::graph::LaunchContext* context) {
        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, context, false);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::empty(nd4j::graph::LaunchContext* context) {
        return empty(DataTypeUtils::fromT<T>(), context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::empty, (nd4j::graph::LaunchContext* context), LIBND4J_TYPES);

    NDArray NDArrayFactory::empty(nd4j::DataType dataType, nd4j::graph::LaunchContext* context) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        NDArray result(nullptr, shapeInfo, context);
        result.triggerAllocationFlag(false);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order, nd4j::graph::LaunchContext* context) {
        auto res = NDArrayFactory::create_(order, shape, value.dataType(), context);
        res->assign(const_cast<NDArray&>(value));
        return res;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::create_( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::graph::LaunchContext* context) {
        
        return new NDArray(order, shape, dataType, context);
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::graph::LaunchContext* context) {
        auto res = create<T>(order, shape, context);
        //memcpy(res.buffer(), data.data(), res.lengthOf() * res.sizeOfT());
        memcpyFromVector<T>(res.getBuffer(), data);
        return res;
    }
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bfloat16> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::graph::LaunchContext* context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::graph::LaunchContext* context);


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    NDArray result;

    if (context == nullptr)
        context = nd4j::graph::LaunchContext::defaultContext();

    result.setAttached(context->getWorkspace() != nullptr);

    result.setBuffer(reinterpret_cast<uint8_t*>(buffer));
    Nd4jLong* shapeInfo = ShapeBuilders::createShapeInfo(DataTypeUtils::fromT<T>(), order, shape, context->getWorkspace());
    result.setShapeInfo(shapeInfo);
    result.setContext(context);
    result.triggerAllocationFlag(false);

    RELEASE(shapeInfo, context->getWorkspace())
    
    return result;
}

template NDArray NDArrayFactory::create(double* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(float* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(float16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(bfloat16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(Nd4jLong * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(int* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(bool* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(uint8_t * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(int8_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);
template NDArray NDArrayFactory::create(int16_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context);


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::graph::LaunchContext* context) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string(order, shape, vec, context);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::graph::LaunchContext* context) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string(order, shape, vec, context);
    }


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::graph::LaunchContext* context) {
        std::vector<std::string> vec(string);
        return NDArrayFactory::string(order, shape, vec, context);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::graph::LaunchContext* context) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string_(order, shape, vec, context);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::graph::LaunchContext* context) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string_(order, shape, vec, context);
    }


    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::graph::LaunchContext* context) {
        std::vector<std::string> vec(string);

        return NDArrayFactory::string_(order, shape, vec, context);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::graph::LaunchContext* context) {
        NDArray res;

        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        res.setAttached(context->getWorkspace() != nullptr);

        auto newShape = ShapeBuilders::createShapeInfo(DataType::UTF8, order, shape, context->getWorkspace());
        res.setShapeInfo(newShape);
        delete[] newShape;

        if (res.lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());
        std::vector<Nd4jLong> offsets(string.size() + 1);
        Nd4jLong dataLength = 0;
        for (int e = 0; e < string.size(); e++) {
            offsets[e] = dataLength;
            dataLength += string[e].length();
        }
        offsets[string.size()] = dataLength;

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), headerLength + dataLength, int8_t);

        memcpy(buffer, offsets.data(), offsets.size() * sizeof(Nd4jLong));

        auto data = buffer + headerLength;
        int resLen = res.lengthOf();
        for (int e = 0; e < resLen; e++) {
            auto length = offsets[e+1] - offsets[e];
            auto cdata = data + offsets[e];
            memcpy(cdata, string[e].c_str(), string[e].length());
        }


        res.setBuffer(buffer);
        res.setContext(context);

        res.triggerAllocationFlag(true);

        return res;
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::graph::LaunchContext* context) {
        auto res = new NDArray();
        if (context == nullptr)
            context = nd4j::graph::LaunchContext::defaultContext();

        res->setAttached(context->getWorkspace() != nullptr);

        auto newShape = ShapeBuilders::createShapeInfo(DataType::UTF8, order, shape, context->getWorkspace());
        res->setShapeInfo(newShape);
        delete[] newShape;

        if (res->lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());
        std::vector<Nd4jLong> offsets(string.size() + 1);
        Nd4jLong dataLength = 0;
        for (int e = 0; e < string.size(); e++) {
            offsets[e] = dataLength;
            dataLength += string[e].length();
        }
        offsets[string.size()] = dataLength;

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), headerLength + dataLength, int8_t);

        memcpy(buffer, offsets.data(), offsets.size() * sizeof(Nd4jLong));

        auto data = buffer + headerLength;
        int resLen = res->lengthOf();
        for (int e = 0; e < resLen; e++) {
            auto length = offsets[e+1] - offsets[e];
            auto cdata = data + offsets[e];
            memcpy(cdata, string[e].c_str(), string[e].length());
        }


        res->setBuffer(buffer);
        res->setContext(context);

        res->triggerAllocationFlag(true);

        return res;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet NDArrayFactory::createSetOfArrs(const Nd4jLong numOfArrs, const void* buffer,  const Nd4jLong* shapeInfo,  const Nd4jLong* offsets, nd4j::graph::LaunchContext* launchContext) {

        ResultSet result;

        const auto sizeOfT = DataTypeUtils::sizeOf(shapeInfo);

        for (int idx = 0; idx < numOfArrs; idx++ ) {
            auto array = new NDArray(reinterpret_cast<int8_t*>(const_cast<void*>(buffer)) + offsets[idx] * sizeOfT, const_cast<Nd4jLong*>(shapeInfo), launchContext);
            result.push_back(array);
        }

        return result;
    }



}
