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
// Created by GS <sgazeos@gmail.com> on 2018-12-20.
//

#include <NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <ConstantHelper.h>
#include <ConstantShapeHelper.h>

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

        utf8string **us = nullptr;
        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), sizeof(utf8string*), int8_t);
        us = reinterpret_cast<utf8string**>(buffer);
        us[0] = new utf8string(str);
        int8_t* specialBuffer = nullptr;//res.specialBuffer();

        ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), sizeof(utf8string*), int8_t);
        Nd4jLong* specialShape = nullptr;
        //int8_t* specialBuffer = nullptr;
        res.setBuffer(buffer);
        res.setShapeInfo(ShapeDescriptor::scalarDescriptor(DataType::UTF8));
        //ALLOCATE_SPECIAL(specialShape, context->getWorkspace(), shape::shapeInfoLength(res.shapeInfo()), Nd4jLong);

        //cudaMemcpy(specialShape, res.shapeInfo(), shape::shapeInfoByteLength(res.shapeInfo()), cudaMemcpyHostToDevice);
        cudaMemcpy(specialBuffer, res.buffer(), sizeof(utf8string*), cudaMemcpyHostToDevice);
        res.setSpecialBuffer(specialBuffer);//, specialShape);
        res.setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);

        res.triggerAllocationFlag(true);

        return res;
    }

    NDArray* NDArrayFactory::string_(const std::string &str, nd4j::graph::LaunchContext* context) {
        auto res = new NDArray();

        utf8string **us = nullptr;
        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), sizeof(utf8string*), int8_t);
        us = reinterpret_cast<utf8string**>(buffer);
        us[0] = new utf8string(str);

        res->setBuffer(buffer);
        res->setShapeInfo(ShapeDescriptor::scalarDescriptor(nd4j::DataType::UTF8));
        //Nd4jLong* specialShape = nullptr;
        int8_t* specialBuffer = nullptr;

        //ALLOCATE_SPECIAL(specialShape, context->getWorkspace(), shape::shapeInfoLength(res->shapeInfo()), Nd4jLong);
        ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), sizeof(utf8string*), int8_t);

        //cudaMemcpy(specialShape, res->shapeInfo(), shape::shapeInfoByteLength(res->shapeInfo()), cudaMemcpyHostToDevice);
        cudaMemcpy(specialBuffer, res->buffer(), sizeof(utf8string*), cudaMemcpyHostToDevice);
        res->setSpecialBuffer(specialBuffer);//, specialShape);
        res->setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);

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
    for (Nd4jLong e = 0; e < vector.size(); e++) 
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

        int8_t *buffer;
        ALLOCATE(buffer, context->getWorkspace(), 1 * sizeof(T), int8_t);

        res->setShapeInfo(ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>())); //ShapeBuilders::createScalarShapeInfo(, context->getWorkspace()));
        res->setBuffer(buffer);
        res->triggerAllocationFlag(true);
        res->setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
        int8_t* specialBuffer = nullptr;
        //Nd4jLong* specialShape = nullptr;
        //ALLOCATE_SPECIAL(specialShape, context->getWorkspace(), shape::shapeInfoLength(res->shapeInfo()), Nd4jLong);
        ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), sizeof(DataTypeUtils::fromT<T>()), int8_t);
        res->setSpecialBuffer(specialBuffer);//, specialShape);

        res->bufferAsT<T>()[0] = scalar;

        //cudaMemcpyAsync(specialShape, res->shapeInfo(), shape::shapeInfoByteLength(res->shapeInfo()), cudaMemcpyHostToDevice, *context->getCudaStream());
        cudaMemcpyAsync(specialBuffer, res->buffer(), sizeof(T), cudaMemcpyHostToDevice, *context->getCudaStream()); // only one element
        cudaStreamSynchronize(*context->getCudaStream());

        res->tickWriteDevice();
        res->tickReadHost();

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
    NDArray NDArrayFactory::create(nd4j::DataType type, const T scalar, nd4j::graph::LaunchContext* context) {
        if (type == DataTypeUtils::fromT<T>())
            return NDArrayFactory::create(scalar,  context);
        NDArray res(type, context);

        int8_t *buffer;
        ALLOCATE(buffer, context->getWorkspace(), 1 * sizeof(T), int8_t);

        //res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace));
        res.setBuffer(buffer);
        res.triggerAllocationFlag(true);
        res.setContext(context);

        res.p(0, scalar);
        res.syncShape();
        res.syncToDevice();

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

        int8_t *buffer;
        size_t bufferSize = sizeof(T);
        ALLOCATE(buffer, context->getWorkspace(), bufferSize, int8_t);

        res.setShapeInfo(ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()));
        size_t shapeSize = shape::shapeInfoLength(res.shapeInfo());
        res.setBuffer(buffer);
        res.triggerAllocationFlag(true);
        res.setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
        int8_t* specialBuffer = nullptr;
        Nd4jLong* specialShape = nullptr;

        //ALLOCATE_SPECIAL(specialShape, context->getWorkspace(), shapeSize, Nd4jLong);
        ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), bufferSize, int8_t);

        res.bufferAsT<T>()[0] = scalar;
        //cudaMemcpyAsync(specialShape, res.shapeInfo(), shapeSize * sizeof(Nd4jLong), cudaMemcpyHostToDevice, *context->getCudaStream());
        cudaMemcpyAsync(specialBuffer, res.buffer(), bufferSize, cudaMemcpyHostToDevice, *context->getCudaStream()); // only one element
        cudaStreamSynchronize(*context->getCudaStream());

        res.setSpecialBuffer(specialBuffer);//.., specialShape);
        res.tickWriteDevice();
        res.tickReadHost();

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

    result->setShapeInfo(ShapeDescriptor(DataTypeUtils::fromT<T>(), order, shape));

    if (result->lengthOf() != data.size()) {
        nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(result->shapeInfo()));
        throw std::runtime_error("Data size doesn't match shape");
    }
    const size_t bufferSize = result->lengthOf() * sizeof(T);
    const size_t shapeSize = shape::shapeInfoLength(result->shapeInfo());
    int8_t* buffer(nullptr);
    ALLOCATE(buffer, context->getWorkspace(), bufferSize, int8_t);
    result->setBuffer(buffer);
    result->setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
    result->triggerAllocationFlag(true);
    memcpyFromVector(result->getBuffer(), data);        // old memcpy_

    int8_t* specialBuffer = nullptr;
    Nd4jLong* specialShape = nullptr;

    //ALLOCATE_SPECIAL(specialShape, context->getWorkspace(), shapeSize, Nd4jLong);
    ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), bufferSize, int8_t);

    //cudaMemcpy(specialShape, result->shapeInfo(), shapeSize * sizeof(Nd4jLong), cudaMemcpyHostToDevice);
    cudaMemcpy(specialBuffer, result->buffer(), bufferSize, cudaMemcpyHostToDevice); // only one element
    result->setSpecialBuffer(specialBuffer);//, specialShape);
    result->tickWriteDevice();
    result->tickReadHost();

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
        auto result = create_(order, shape, value->dataType(), context);
        result->assign(*value);
        return result;
    }

    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray& value, const char order, nd4j::graph::LaunchContext* context) {
        auto result = create_(order, shape, value.dataType(), context);
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
//TO DO: linspace should be executed on DEVICE, but only CPU version implemnted!
        for (Nd4jLong e = 0; e < numElements; e++) {
            T step = (T) e / ((T) numElements - (T) 1);
            reinterpret_cast<T *>(result->getBuffer())[e] = (from * ((T) 1 - step) + step * to);            
        }
        cudaMemcpy(result->specialBuffer(), result->getBuffer(), numElements * sizeof(T), cudaMemcpyHostToDevice);
        result->tickWriteDevice();
        result->tickReadHost();
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
    NDArray* NDArrayFactory::vector(Nd4jLong length, const T startingValue, graph::LaunchContext* context) {

        auto res = new NDArray(context);

        int8_t *buffer = nullptr;
        int8_t* specialBuffer = nullptr;
        Nd4jLong* specialShapeInfo = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), length * sizeof(T), int8_t);

        ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), length * sizeof(T), int8_t);

        res->setShapeInfo(ShapeDescriptor::vectorDescriptor(length, DataTypeUtils::fromT<T>()));

        //ALLOCATE_SPECIAL(specialShapeInfo, context->getWorkspace(), shape::shapeInfoLength(res->shapeInfo()), Nd4jLong);

        //cudaMemcpy(specialShapeInfo, res->shapeInfo(), shape::shapeInfoByteLength(res->shapeInfo()), cudaMemcpyHostToDevice);
        res->setBuffer(buffer);
        res->setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
        res->setSpecialBuffer(specialBuffer);
        res->triggerAllocationFlag(true);

        // tick this before potential assign call, to avoid additional sync before assign
        res->tickWriteDevice();

        if (startingValue == (T)0.0f) {
            memset(buffer, 0, length);
            res->syncToDevice();
            res->tickReadHost();
        } else {
            res->assign(startingValue);
        }
        return res;
    }
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const double startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float16 startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bfloat16 startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const Nd4jLong startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const uint8_t startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int8_t startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int16_t startingValue, graph::LaunchContext* context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bool startingValue, graph::LaunchContext* context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::graph::LaunchContext* context) {
        std::vector<Nd4jLong> vec(shape);
        return create<T>(order, vec, context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char, const std::initializer_list<Nd4jLong>&, nd4j::graph::LaunchContext* context), LIBND4J_TYPES);

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

    //......;
    //auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    res.setShapeInfo(ShapeDescriptor(dtype, order, shape)); //reinterpret_cast<Nd4jLong *>(shapeBuffer.primary()));

    int8_t* specialBuffer = nullptr;
    size_t bufferSize = res.lengthOf() * res.sizeOfT();

    ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), bufferSize, int8_t);

    cudaMemset(specialBuffer, 0, bufferSize);

    res.tickWriteDevice();
    res.setContext(context);
    res.setSpecialBuffer(specialBuffer);//, reinterpret_cast<Nd4jLong *>(shapeBuffer.special()));
    res.triggerAllocationFlag(true);

    nd4j_printf("Created NdArray with shape %lu and shape rank %lld.\n", shape.size(), res.getShapeInfo()[0]);

    return res;
}


////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(nd4j::DataType dtype, nd4j::graph::LaunchContext* context) {
    
    NDArray res;
    
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, context->getWorkspace(), DataTypeUtils::sizeOfElement(dtype), int8_t);
    size_t bufferSize = DataTypeUtils::sizeOfElement(dtype);
    memset(buffer, 0, bufferSize);
    res.setBuffer(buffer);
    res.setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
    res.setShapeInfo(ShapeDescriptor::scalarDescriptor(dtype));
    res.triggerAllocationFlag(true);
    int8_t* specialBuffer = nullptr;
    //Nd4jLong* specialShapeInfo = nullptr;
    size_t shapeSize = shape::shapeInfoLength(res.shapeInfo());

    //ALLOCATE_SPECIAL(specialShapeInfo, context->getWorkspace(), shapeSize, Nd4jLong);
    ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), DataTypeUtils::sizeOfElement(dtype), int8_t);

    cudaMemcpy(specialBuffer, res.buffer(), bufferSize, cudaMemcpyHostToDevice);
    //cudaMemcpy(specialShapeInfo, res.shapeInfo(), shapeSize * sizeof(Nd4jLong), cudaMemcpyHostToDevice);
    res.setSpecialBuffer(specialBuffer);//, specialShapeInfo);

    return res;
}


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const std::vector<T> &values, nd4j::graph::LaunchContext* context) {
        
    NDArray res;        

    res.setShapeInfo(ShapeDescriptor::vectorDescriptor(values.size(), DataTypeUtils::fromT<T>()));
        
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, context->getWorkspace(), values.size() * sizeof(T), int8_t);
    memcpyFromVector<T>(buffer, values);
    int8_t* specialBuffer = nullptr;
    Nd4jLong* specialShapeInfo = nullptr;

    size_t bufferSize = res.lengthOf() * sizeof(T);
    size_t shapeSize = shape::shapeInfoLength(res.shapeInfo());
    res.setBuffer(buffer);

    //ALLOCATE_SPECIAL(specialShapeInfo, context->getWorkspace(), shapeSize, Nd4jLong);
    ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), bufferSize, int8_t);

    cudaMemcpy(specialBuffer, res.buffer(), bufferSize, cudaMemcpyHostToDevice);
    //cudaMemcpy(specialShapeInfo, res.shapeInfo(), shapeSize * sizeof(Nd4jLong), cudaMemcpyHostToDevice);

    res.setSpecialBuffer(specialBuffer);//, specialShapeInfo);

    res.triggerAllocationFlag(true);
    res.setContext(context);
    res.tickWriteDevice();
    res.tickReadHost();
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
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, context);
        result->triggerAllocationFlag(false);

        return result;
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
        int8_t* buffer = nullptr;
        size_t allocationSize = res.lengthOf() * res.sizeOfT();
        ALLOCATE(buffer, context->getWorkspace(), allocationSize, int8_t);
        res.setBuffer(buffer);
        //res.setShapeInfo(ShapeDescriptor( order, DataTypeUtils::fromT<T>(), shape));
        //res.lazyAllocateBuffer();
        //memcpy(res.buffer(), data.data(), res.lengthOf() * res.sizeOfT());
        memcpyFromVector<T>(res.buffer(), data);
        size_t bufferSize = data.size() * sizeof(T);
        //size_t shapeSize = shape::shapeInfoByteLength(res.shapeInfo());//shape.size() * sizeof(Nd4jLong);
        cudaMemcpy(res.specialBuffer(), res.buffer(), bufferSize, cudaMemcpyHostToDevice);
        //cudaMemcpy(res.specialShapeInfo(), res.shapeInfo(), shapeSize, cudaMemcpyHostToDevice);
        res.tickWriteDevice();
        res.tickReadHost();
        res.triggerAllocationFlag(true);
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

    std::vector<Nd4jLong> shp(shape);
    ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shp);
    auto shapeBuffer = ConstantShapeHelper::getInstance()->bufferForShapeInfo(descriptor);

    result.setBuffer(reinterpret_cast<int8_t*>(buffer));
    result.setShapeInfo(descriptor);//reinterpret_cast<Nd4jLong *>(shapeBuffer.primary()));
    result.setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);
    result.triggerAllocationFlag(false);

    int8_t* specialBuffer = nullptr;
    ALLOCATE_SPECIAL(specialBuffer, context->getWorkspace(), shape::length(result.shapeInfo()) * sizeof(T), int8_t);

    cudaMemcpy(specialBuffer, result.buffer(), result.lengthOf() * sizeof(T), cudaMemcpyHostToDevice);

    result.setSpecialBuffer(specialBuffer);//, reinterpret_cast<Nd4jLong *>(shapeBuffer.special()));
    result.tickWriteDevice();
    result.tickReadHost();

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

        res.setShapeInfo(ShapeDescriptor(DataType::UTF8, order, shape));

        if (res.lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), sizeof(utf8string*) * res.lengthOf(), int8_t);

        auto us = reinterpret_cast<utf8string**>(buffer);
        for (int e = 0; e < res.lengthOf(); e++)
            us[e] = new utf8string(string[e]);

        res.setBuffer(buffer);
        res.setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);

        res.triggerAllocationFlag(true);

        return res;
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::graph::LaunchContext* context) {
        auto res = new NDArray();

        res->setShapeInfo(ShapeDescriptor(DataType::UTF8, order, shape));

        if (res->lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, context->getWorkspace(), sizeof(utf8string*) * res->lengthOf(), int8_t);

        auto us = reinterpret_cast<utf8string**>(buffer);
        for (int e = 0; e < res->lengthOf(); e++)
            us[e] = new utf8string(string[e]);

        res->setBuffer(buffer);
        res->setContext(context == nullptr ? nd4j::graph::LaunchContext::defaultContext() : context);

        res->triggerAllocationFlag(true);

        return res;
    }

}
