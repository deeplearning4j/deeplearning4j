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
#include <ShapeUtils.h>
#include <type_traits>

namespace nd4j {


    NDArray NDArrayFactory::string(const char *str, nd4j::LaunchContext * context) {
        std::string s(str);
        return string(s, context);
    }

    NDArray* NDArrayFactory::string_(const char *str, nd4j::LaunchContext * context) {
        return string_(std::string(str), context);
    }

    NDArray NDArrayFactory::string(const std::string &str, nd4j::LaunchContext * context) {

        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);

        std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(headerLength + str.length(), DataType::UTF8, context->getWorkspace(), true);

        NDArray res(pBuffer, ShapeDescriptor::scalarDescriptor(DataType::UTF8), context);

        int8_t* buffer = reinterpret_cast<int8_t*>(res.getBuffer());

        auto offsets = reinterpret_cast<Nd4jLong *>(buffer);
        offsets[0] = 0;
        offsets[1] = str.length();

        auto data = buffer + headerLength;

        memcpy(data, str.c_str(), str.length());

        res.tickWriteHost();
        res.syncToDevice();

        return res;
    }

    NDArray* NDArrayFactory::string_(const std::string &str, nd4j::LaunchContext * context) {
        auto res = new NDArray();
        *res = NDArrayFactory::string(str, context);
        return res;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, nd4j::LaunchContext * context) {
    return create_(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::create_, (const char order, const std::vector<Nd4jLong> &shape, nd4j::LaunchContext * context), LIBND4J_TYPES);

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
    NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order, nd4j::LaunchContext * context) {
        return valueOf(std::vector<Nd4jLong>(shape), value, order);
    }
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const double value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float16 value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int8_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int16_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bool value, const char order, nd4j::LaunchContext * context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<T>& data, nd4j::LaunchContext * context) {
        std::vector<T> vec(data);
        return create<T>(order, shape, vec, context);
    }
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<double>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float16>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bfloat16>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<Nd4jLong>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int16_t>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int8_t>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<uint8_t>& data, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bool>& data, nd4j::LaunchContext * context);

#endif

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::create_(const T scalar, nd4j::LaunchContext * context) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(1 * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

        NDArray* res = new NDArray(buffer, ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()), context);

        res->bufferAsT<T>()[0] = scalar;

        res->tickWriteHost();
        res->syncToDevice();

        return res;
    }
    template NDArray* NDArrayFactory::create_(const double scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const float scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const float16 scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const bfloat16 scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const Nd4jLong scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const int scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const bool scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const int8_t scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const uint8_t scalar, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::create_(const int16_t scalar, nd4j::LaunchContext * context);

    template <typename T>
    NDArray NDArrayFactory::create(nd4j::DataType type, const T scalar, nd4j::LaunchContext * context) {

        if (type == DataTypeUtils::fromT<T>())
            return NDArrayFactory::create(scalar,  context);

        NDArray res(type, context);
        res.p(0, scalar);
        res.syncToDevice();

        return res;
    }
//    BUILD_DOUBLE_TEMPLATE(template NDArray NDArrayFactory::create, (DataType type, const T scalar, nd4j::LaunchContext * context), LIBND4J_TYPES);
    template NDArray NDArrayFactory::create(DataType type, const double scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const float scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const float16 scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const bfloat16 scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const Nd4jLong scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const int scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const int8_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const uint8_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const int16_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(DataType type, const bool scalar, nd4j::LaunchContext * context);

    template <typename T>
    NDArray NDArrayFactory::create(const T scalar, nd4j::LaunchContext * context) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(1 * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

        NDArray res(buffer, ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()), context);

        res.bufferAsT<T>()[0] = scalar;

        res.tickWriteHost();
        res.syncToDevice();

        return res;
    }
    template NDArray NDArrayFactory::create(const double scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const float scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const float16 scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const bfloat16 scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const Nd4jLong scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const int scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const int8_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const uint8_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const int16_t scalar, nd4j::LaunchContext * context);
    template NDArray NDArrayFactory::create(const bool scalar, nd4j::LaunchContext * context);


////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::LaunchContext * context) {

   return new NDArray(NDArrayFactory::create<T>(order, shape, data, context));
}
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bfloat16> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::LaunchContext * context);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::LaunchContext * context);


    ////////////////////////////////////////////////////////////////////////
    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray* value, const char order, nd4j::LaunchContext * context) {
        auto result = create_(order, shape, value->dataType(), context);
        result->assign(*value);
        return result;
    }

    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray& value, const char order, nd4j::LaunchContext * context) {
        auto result = create_(order, shape, value.dataType(), context);
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order, nd4j::LaunchContext * context) {
        auto result = create_(order, shape, DataTypeUtils::fromT<T>());
        result->assign(value);
        return result;
    }
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const double value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float16 value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int16_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int8_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bool value, const char order, nd4j::LaunchContext * context);


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
        result->tickBothActual();

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
    NDArray* NDArrayFactory::vector(Nd4jLong length, const T value, nd4j::LaunchContext * context) {

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(length * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

        auto res = new NDArray(buffer, ShapeDescriptor::vectorDescriptor(length, DataTypeUtils::fromT<T>()), context);

        if (value == (T)0.0f)
            res->nullify();
        else
            res->assign(value);

        return res;
    }
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const double startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float16 startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bfloat16 startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const Nd4jLong startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const uint8_t startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int8_t startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int16_t startingValue, nd4j::LaunchContext * context);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bool startingValue, nd4j::LaunchContext * context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context) {
        std::vector<Nd4jLong> vec(shape);
        return create<T>(order, vec, context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char, const std::initializer_list<Nd4jLong>&, nd4j::LaunchContext * context), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::LaunchContext * context) {
        return create(order, shape, DataTypeUtils::fromT<T>(), context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char order, const std::vector<Nd4jLong> &shape, nd4j::LaunchContext * context), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::LaunchContext* context) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32");

    ShapeDescriptor descriptor(dtype, order, shape);

    std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(descriptor.arrLength() * DataTypeUtils::sizeOfElement(dtype), dtype, context->getWorkspace());

    NDArray result(buffer, descriptor, context);

    result.nullify();

    return result;
}


////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(nd4j::DataType dtype, nd4j::LaunchContext * context) {

    std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(DataTypeUtils::sizeOfElement(dtype), dtype, context->getWorkspace(), true);

    NDArray res(buffer, ShapeDescriptor::scalarDescriptor(dtype), context);

    res.nullify();

    return res;
}


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const std::vector<T> &values, nd4j::LaunchContext * context) {

    std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(values.size() * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

    NDArray res(buffer, ShapeDescriptor::vectorDescriptor(values.size(), DataTypeUtils::fromT<T>()), context);

    memcpyFromVector<T>(res.getBuffer(), values);

    res.tickWriteHost();
    res.syncToDevice();

    return res;
}
template NDArray NDArrayFactory::create(const std::vector<double> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<float> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<float16> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<bfloat16> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<Nd4jLong> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<int> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<int16_t> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<int8_t> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<uint8_t> &values, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(const std::vector<bool> &values, nd4j::LaunchContext * context);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::empty_(nd4j::LaunchContext * context) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, context, false);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::empty_, (nd4j::LaunchContext * context), LIBND4J_TYPES);

    NDArray* NDArrayFactory::empty_(nd4j::DataType dataType, nd4j::LaunchContext * context) {
        if (context == nullptr)
            context = nd4j::LaunchContext ::defaultContext();

        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, context, false);

        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::empty(nd4j::LaunchContext * context) {
        return empty(DataTypeUtils::fromT<T>(), context);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::empty, (nd4j::LaunchContext * context), LIBND4J_TYPES);

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArrayFactory::empty(nd4j::DataType dataType, nd4j::LaunchContext * context) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        NDArray result(nullptr, shapeInfo, context, false);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order, nd4j::LaunchContext * context) {
        auto res = NDArrayFactory::create_(order, shape, value.dataType(), context);
        res->assign(const_cast<NDArray&>(value));
        return res;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::create_( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::LaunchContext * context) {

        return new NDArray(order, shape, dataType, context);
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::LaunchContext * context) {

        if ((int) shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

        ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shape);

        if (descriptor.arrLength() != data.size()) {
            nd4j_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(), descriptor.arrLength());
            throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
        }

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(data.data(), DataTypeUtils::fromT<T>(), descriptor.arrLength() * sizeof(T), context->getWorkspace());

        NDArray result(buffer, descriptor, context);

        return result;

    }
    ////////////////////////////////////////////////////////////////////////
    template <>
    NDArray NDArrayFactory::create<bool>(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::LaunchContext * context) {

        if ((int) shape.size() > MAX_RANK)
            throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

        ShapeDescriptor descriptor(nd4j::DataType::BOOL, order, shape);

        if (descriptor.arrLength() != data.size()) {
            nd4j_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(), descriptor.arrLength());
            throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
        }

        bool* hostBuffer = nullptr;
        ALLOCATE(hostBuffer, context->getWorkspace(), data.size(), bool);
        std::copy(data.begin(), data.end(), hostBuffer);

        std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(hostBuffer, data.size() * sizeof(bool), nd4j::DataType::BOOL, true, context->getWorkspace());

        NDArray result(buffer, descriptor, context);

        return result;

    }

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("NDArrayFactory::create: Rank of NDArray can't exceed 32");

    std::vector<Nd4jLong> shp(shape);
    ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shp);

    std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(buffer, descriptor.arrLength() * sizeof(T), descriptor.dataType(), false, context->getWorkspace());

    NDArray result(pBuffer, descriptor, context);

    return result;
}

template NDArray NDArrayFactory::create(double* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(float* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(float16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(bfloat16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(Nd4jLong * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(int* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(bool* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(uint8_t * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(int8_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);
template NDArray NDArrayFactory::create(int16_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::LaunchContext * context);


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::LaunchContext * context) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string(order, shape, vec, context);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::LaunchContext * context) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string(order, shape, vec, context);
    }


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::LaunchContext * context) {
        std::vector<std::string> vec(string);
        return NDArrayFactory::string(order, shape, vec, context);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::LaunchContext * context) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string_(order, shape, vec, context);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::LaunchContext * context) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string_(order, shape, vec, context);
    }


    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::LaunchContext * context) {
        std::vector<std::string> vec(string);
        return NDArrayFactory::string_(order, shape, vec, context);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::LaunchContext * context) {

          if (context == nullptr)
            context = nd4j::LaunchContext ::defaultContext();

        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(string.size());

        std::vector<Nd4jLong> offsets(string.size() + 1);
        Nd4jLong dataLength = 0;
        for (int e = 0; e < string.size(); e++) {
            offsets[e] = dataLength;
            dataLength += string[e].length();
        }
        offsets[string.size()] = dataLength;

        std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(headerLength + dataLength, DataType::UTF8, context->getWorkspace(), true);

        NDArray res(pBuffer, ShapeDescriptor(DataType::UTF8, order, shape), context);
        res.setAttached(context->getWorkspace() != nullptr);

        if (res.lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        memcpy(res.buffer(), offsets.data(), offsets.size() * sizeof(Nd4jLong));

        auto data = static_cast<int8_t*>(res.buffer()) + headerLength;
        int resLen = res.lengthOf();
        for (int e = 0; e < resLen; e++) {
            auto length = offsets[e+1] - offsets[e];
            auto cdata = data + offsets[e];
            memcpy(cdata, string[e].c_str(), string[e].length());
        }

        res.tickWriteHost();
        res.syncToDevice();

        return res;
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::LaunchContext * context) {
        auto res = new NDArray();
        *res = NDArrayFactory::string(order, shape, string, context);
        return res;
    }


}
