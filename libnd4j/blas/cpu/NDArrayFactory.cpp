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
#include <openmp_pragmas.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {


    NDArray NDArrayFactory::string(const char *str, nd4j::memory::Workspace* workspace) {
        std::string s(str);
        return string(s, workspace);
    }

    NDArray* NDArrayFactory::string_(const char *str, nd4j::memory::Workspace* workspace) {
        return string_(std::string(str), workspace);
    }

    NDArray NDArrayFactory::string(const std::string &str, nd4j::memory::Workspace* workspace) {
        NDArray res;

        int8_t *buffer = nullptr;
        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);
        ALLOCATE(buffer, workspace, headerLength + str.length(), int8_t);
        auto offsets = reinterpret_cast<Nd4jLong *>(buffer);
        auto data = buffer + headerLength;

        offsets[0] = 0;
        offsets[1] = str.length();

        memcpy(data, str.c_str(), str.length());

        res.setBuffer(buffer);
        res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataType::UTF8, workspace));
        res.setWorkspace(workspace);

        res.triggerAllocationFlag(true, true);

        return res;
    }

    NDArray* NDArrayFactory::string_(const std::string &str, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        int8_t *buffer = nullptr;
        auto headerLength = ShapeUtils::stringBufferHeaderRequirements(1);
        ALLOCATE(buffer, workspace, headerLength + str.length(), int8_t);
        auto offsets = reinterpret_cast<Nd4jLong *>(buffer);
        auto data = buffer + headerLength;

        offsets[0] = 0;
        offsets[1] = str.length();

        memcpy(data, str.c_str(), str.length());

        res->setBuffer(buffer);
        res->setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataType::UTF8, workspace));
        res->setWorkspace(workspace);

        res->triggerAllocationFlag(true, true);

        return res;
    }

////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {
    return create_(order, shape, DataTypeUtils::fromT<T>(), workspace);
}
BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::create_, (const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

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
    NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order, nd4j::memory::Workspace* workspace) {
        return valueOf(std::vector<Nd4jLong>(shape), value, order);
    }
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const double value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float16 value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int16_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bool value, const char order, nd4j::memory::Workspace* workspace);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<T>& data, nd4j::memory::Workspace* workspace) {
        std::vector<T> vec(data);
        return create<T>(order, shape, vec, workspace);
    }
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<double>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<float16>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bfloat16>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<Nd4jLong>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int16_t>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<int8_t>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<uint8_t>& data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<bool>& data, nd4j::memory::Workspace* workspace);

#endif

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::create_(const T scalar, nd4j::memory::Workspace* workspace) {
        
        auto res = new NDArray();        

        int8_t *buffer;
        ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);        

        res->setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace));
        res->setBuffer(buffer);
        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);

        res->assign(scalar);

        return res;
    }
    template NDArray* NDArrayFactory::create_(const double scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const float scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const bfloat16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const int scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const bool scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create_(const int16_t scalar, nd4j::memory::Workspace* workspace);

    template <typename T>
    NDArray NDArrayFactory::create(DataType type, const T scalar, nd4j::memory::Workspace* workspace) {

        NDArray res(type, workspace);

        //int8_t *buffer;
        //ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);

        //res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace));
        //res.setBuffer(buffer);
        //res.triggerAllocationFlag(true, true);
        //res.setWorkspace(workspace);

        res.p(0, scalar);

        return res;
    }
//    BUILD_DOUBLE_TEMPLATE(template NDArray NDArrayFactory::create, (DataType type, const T scalar, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);
    template NDArray NDArrayFactory::create(DataType type, const double scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const float scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const bfloat16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const int scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const int16_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(DataType type, const bool scalar, nd4j::memory::Workspace* workspace);

    template <typename T>
    NDArray NDArrayFactory::create(const T scalar, nd4j::memory::Workspace* workspace) {

        NDArray res;

        int8_t *buffer;
        ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);

        res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace));
        res.setBuffer(buffer);
        res.triggerAllocationFlag(true, true);
        res.setWorkspace(workspace);

        res.bufferAsT<T>()[0] = scalar;

        return res;
    }
    template NDArray NDArrayFactory::create(const double scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const float scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const bfloat16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const int scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const int16_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const bool scalar, nd4j::memory::Workspace* workspace);


////////////////////////////////////////////////////////////////////////
template<typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    auto result = new NDArray();

    result->setShapeInfo(ShapeBuilders::createShapeInfo(DataTypeUtils::fromT<T>(), order, shape, workspace));

    if (result->lengthOf() != data.size()) {
        nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(result->shapeInfo()));
        throw std::runtime_error("Data size doesn't match shape");
    }
        
    int8_t* buffer(nullptr);
    ALLOCATE(buffer, workspace, result->lengthOf() * DataTypeUtils::sizeOf(DataTypeUtils::fromT<T>()), int8_t);        
    result->setBuffer(buffer);
    result->setWorkspace(workspace);
    result->triggerAllocationFlag(true, true);
    memcpyFromVector(result->getBuffer(), data);        // old memcpy_

    return result;
}
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bfloat16> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::memory::Workspace* workspace);
template NDArray* NDArrayFactory::create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::memory::Workspace* workspace);


    ////////////////////////////////////////////////////////////////////////
    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray* value, const char order, nd4j::memory::Workspace* workspace) {
        auto result = create_(order, shape, value->dataType());
        result->assign(*value);
        return result;
    }

    template <>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, NDArray& value, const char order, nd4j::memory::Workspace* workspace) {
        auto result = create_(order, shape, value.dataType());
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order, nd4j::memory::Workspace* workspace) {
        auto result = create_(order, shape, DataTypeUtils::fromT<T>());
        result->assign(value);
        return result;
    }
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const double value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float16 value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bfloat16 value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int16_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const int8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const bool value, const char order, nd4j::memory::Workspace* workspace);


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
    NDArray* NDArrayFactory::vector(Nd4jLong length, const T startingValue, nd4j::memory::Workspace *workspace) {

        auto res = new NDArray();

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, length * sizeof(T), int8_t);

        res->setShapeInfo(ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), length, workspace));
        res->setBuffer(buffer);
        res->setWorkspace(workspace);
        res->triggerAllocationFlag(true, true);

        if (startingValue == (T)0.0f) {
            memset(buffer, 0, length);
        } else {
            res->assign(startingValue);
        }

        return res;
    }
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const double startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const float16 startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bfloat16 startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const Nd4jLong startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const uint8_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int8_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int16_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bool startingValue, nd4j::memory::Workspace *workspace);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace) {
        std::vector<Nd4jLong> vec(shape);
        return create<T>(order, vec, workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char, const std::initializer_list<Nd4jLong>&, nd4j::memory::Workspace*), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {
        return create(order, shape, DataTypeUtils::fromT<T>(), workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::create, (const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
  
    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    NDArray res;        

    res.setShapeInfo(ShapeBuilders::createShapeInfo(dtype, order, shape, workspace));
    
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, workspace, res.lengthOf() * DataTypeUtils::sizeOfElement(dtype), int8_t);
    memset(buffer, 0, res.lengthOf() * res.sizeOfT());

    res.setBuffer(buffer);
    res.setWorkspace(workspace);    
    res.triggerAllocationFlag(true, true);

    return res;
}


////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
    
    NDArray res;
    
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, workspace, DataTypeUtils::sizeOfElement(dtype), int8_t);
    memset(buffer, 0, DataTypeUtils::sizeOfElement(dtype));
    res.setBuffer(buffer);
    res.setWorkspace(workspace);
    res.setShapeInfo(ShapeBuilders::createScalarShapeInfo(dtype, workspace));
    res.triggerAllocationFlag(true, true);
    
    return res;
}


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        
    NDArray res;        

    res.setShapeInfo(ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), values.size(), workspace));
        
    int8_t *buffer = nullptr;
    ALLOCATE(buffer, workspace, values.size() * sizeof(T), int8_t);
    memcpyFromVector<T>(buffer, values);
        
    res.setBuffer(buffer);        
    res.triggerAllocationFlag(true, true);
    res.setWorkspace(workspace);
        
    return res;
}
template NDArray NDArrayFactory::create(const std::vector<double> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<float> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<float16> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<bfloat16> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<Nd4jLong> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<int> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<int16_t> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<int8_t> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<uint8_t> &values, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(const std::vector<bool> &values, nd4j::memory::Workspace* workspace);

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::empty_(nd4j::memory::Workspace* workspace) {
        return empty_(DataTypeUtils::fromT<T>(), workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::empty_, (nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

    NDArray* NDArrayFactory::empty_(nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace);
        result->triggerAllocationFlag(false, true);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::empty(nd4j::memory::Workspace* workspace) {
        return empty(DataTypeUtils::fromT<T>(), workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::empty, (nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

    NDArray NDArrayFactory::empty(nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        NDArray result(nullptr, shapeInfo, workspace);
        result.triggerAllocationFlag(false, true);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order, nd4j::memory::Workspace* workspace) {
        auto res = NDArrayFactory::create_(order, shape, value.dataType(), workspace);
        res->assign(const_cast<NDArray&>(value));
        return res;
    }

////////////////////////////////////////////////////////////////////////
    NDArray* NDArrayFactory::create_( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        
        return new NDArray(order, shape, dataType, workspace);
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        auto res = create<T>(order, shape, workspace);
        //memcpy(res.buffer(), data.data(), res.lengthOf() * res.sizeOfT());
        memcpyFromVector<T>(res.getBuffer(), data);
        return res;
    }
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bfloat16> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::memory::Workspace* workspace);


////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace) {

    if ((int) shape.size() > MAX_RANK)
        throw std::invalid_argument("Rank of NDArray can't exceed 32");

    NDArray result;

    result.setBuffer(reinterpret_cast<uint8_t*>(buffer));
    result.setShapeInfo(ShapeBuilders::createShapeInfo(DataTypeUtils::fromT<T>(), order, shape, workspace));
    result.setWorkspace(workspace);
    result.triggerAllocationFlag(false, true);
    
    return result;
}

template NDArray NDArrayFactory::create(double* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(float* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(float16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(bfloat16* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(Nd4jLong * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(int* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(bool* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(uint8_t * buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(int8_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);
template NDArray NDArrayFactory::create(int16_t* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace);


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::memory::Workspace* workspace) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string(order, shape, vec, workspace);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::memory::Workspace* workspace) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string(order, shape, vec, workspace);
    }


    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::memory::Workspace* workspace) {
        std::vector<std::string> vec(string);
        return NDArrayFactory::string(order, shape, vec, workspace);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<const char *> &strings, nd4j::memory::Workspace* workspace) {
        std::vector<const char*> vec(strings);
        return NDArrayFactory::string_(order, shape, vec, workspace);
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<const char *> &strings, nd4j::memory::Workspace* workspace) {
        std::vector<std::string> vec(strings.size());
        int cnt = 0;
        for (auto s:strings)
            vec[cnt++] = std::string(s);

        return NDArrayFactory::string_(order, shape, vec, workspace);
    }


    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<std::string> &string, nd4j::memory::Workspace* workspace) {
        std::vector<std::string> vec(string);
        return NDArrayFactory::string_(order, shape, vec, workspace);
    }

    NDArray NDArrayFactory::string(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::memory::Workspace* workspace) {
        NDArray res;

        res.setShapeInfo(ShapeBuilders::createShapeInfo(DataType::UTF8, order, shape, workspace));

        if (res.lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, sizeof(utf8string*) * res.lengthOf(), int8_t);

        auto us = reinterpret_cast<utf8string**>(buffer);
        int resLen = res.lengthOf();
        for (int e = 0; e < resLen; e++)
            us[e] = new utf8string(string[e]);

        res.setBuffer(buffer);
        res.setWorkspace(workspace);

        res.triggerAllocationFlag(true, true);

        return res;
    }

    NDArray* NDArrayFactory::string_(char order, const std::vector<Nd4jLong> &shape, const std::vector<std::string> &string, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        res->setShapeInfo(ShapeBuilders::createShapeInfo(DataType::UTF8, order, shape, workspace));

        if (res->lengthOf() != string.size())
            throw std::invalid_argument("Number of strings should match length of array");

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, sizeof(utf8string*) * res->lengthOf(), int8_t);

        auto us = reinterpret_cast<utf8string**>(buffer);
        for (int e = 0; e < res->lengthOf(); e++)
            us[e] = new utf8string(string[e]);

        res->setBuffer(buffer);
        res->setWorkspace(workspace);

        res->triggerAllocationFlag(true, true);

        return res;
    }

    ////////////////////////////////////////////////////////////////////////
    ResultSet NDArrayFactory::createSetOfArrs(const Nd4jLong numOfArrs, const void* buffer,  const Nd4jLong* shapeInfo,  const Nd4jLong* offsets, memory::Workspace* workspace) {

        ResultSet result;

        const auto sizeOfT = DataTypeUtils::sizeOf(shapeInfo);

        for (int idx = 0; idx < numOfArrs; idx++ ) {
            auto array = new NDArray(reinterpret_cast<int8_t*>(const_cast<void*>(buffer)) + offsets[idx] * sizeOfT, const_cast<Nd4jLong*>(shapeInfo), workspace);
            result.push_back(array);
        }        

        return result;
    }



}