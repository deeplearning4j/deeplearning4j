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

namespace nd4j {
    template<typename T>
    NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {
        return create(order, shape, DataTypeUtils::fromT<T>(), workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::create, (const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);


    template <>
    void NDArrayFactory::_memcpy(void *ptr, std::vector<bool> &vector) {
            auto p = reinterpret_cast<bool *>(ptr);
            for (Nd4jLong e = 0; e < vector.size(); e++) {
                p[e] = vector[e];
            }
    }

    template <typename T>
    void NDArrayFactory::_memcpy(void *ptr, std::vector<T> &vector) {
        memcpy(ptr, vector.data(), vector.size() * sizeof(T));
    }
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<double> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<float> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<float16> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<Nd4jLong> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<int> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<int16_t> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<uint8_t> &vector);
    template void NDArrayFactory::_memcpy(void *ptr, std::vector<int8_t> &vector);


#ifndef __JAVACPP_HACK__
    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order, nd4j::memory::Workspace* workspace) {
        return valueOf(std::vector<Nd4jLong>(shape), value, order);
    }
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const double value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const float16 value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const Nd4jLong value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const uint8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int8_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const int16_t value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const bool value, const char order, nd4j::memory::Workspace* workspace);


    template <typename T>
    NDArray* NDArrayFactory::create(std::initializer_list<T> v, nd4j::memory::Workspace* workspace) {
        std::vector<T> vector(v);
        return create(vector, workspace);
    }
    template NDArray* NDArrayFactory::create(std::initializer_list<float> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<double> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<float16> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<Nd4jLong> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<int> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<bool> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<uint8_t> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<int8_t> v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::initializer_list<int16_t> v, nd4j::memory::Workspace* workspace);

    template <typename T>
    NDArray NDArrayFactory::_create(std::initializer_list<T> values, nd4j::memory::Workspace* workspace) {
        std::vector<T> vec(values);
        return _create<T>(vec, workspace);
    }
    template NDArray NDArrayFactory::_create(std::initializer_list<double> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<float> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<float16> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<Nd4jLong> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<int> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<int16_t> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<int8_t> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<uint8_t> values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::initializer_list<bool> values, nd4j::memory::Workspace* workspace);


    template <typename T>
    NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<T> data, nd4j::memory::Workspace* workspace) {
        std::vector<T> vec(data);
        return _create<T>(order, shape, vec, workspace);
    }
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<double> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<float> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<float16> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<Nd4jLong> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<int> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<int16_t> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<int8_t> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<uint8_t> data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::initializer_list<bool> data, nd4j::memory::Workspace* workspace);


    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::create(std::initializer_list<Nd4jLong> s, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        std::vector<Nd4jLong> shape(s);
        int rank = (int) shape.size();

        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(rank), Nd4jLong);
        shape::shapeBuffer(rank, shape.data(), shapeInfo);
        ArrayOptions::setDataType(shapeInfo, DataTypeUtils::fromT<T>());
        res->setShapeInfo(shapeInfo);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, res->lengthOf() * sizeof(T), int8_t);
        res->setBuffer(buffer);

        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);
        return res;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::create, (std::initializer_list<Nd4jLong> s, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);


    template <typename T>
    NDArray* NDArrayFactory::create(std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        int8_t *buffer = nullptr;
        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(buffer, workspace, values.size() * sizeof(T), int8_t);
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(1), Nd4jLong);
        shape::shapeVector(values.size(), shapeInfo);

        ArrayOptions::setDataType(shapeInfo, DataTypeUtils::fromT<T>());
        res->setBuffer(buffer);
        res->setShapeInfo(shapeInfo);
        //T *t = values.data();
        //std::memcpy(res->getBuffer(), values.data(), values.size() * sizeof(T));

        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);
        return res;
    }
    template NDArray* NDArrayFactory::create(std::vector<float> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<double> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<float16> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<Nd4jLong> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<int> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<bool> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<uint8_t> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<int8_t> &v, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(std::vector<int16_t> &v, nd4j::memory::Workspace* workspace);

    template <typename T>
    NDArray NDArrayFactory::_create(T* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace) {
        NDArray res;

        return res;
    }
    template NDArray NDArrayFactory::_create(double* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(float* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(float16* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(Nd4jLong * buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(bool* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(uint8_t * buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int8_t* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int16_t* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);

    template <typename T>
    NDArray* NDArrayFactory::create(T* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace) {
        return nullptr;
    }
    template NDArray* NDArrayFactory::create(double* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(float* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(float16* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(Nd4jLong * buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(int* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(bool* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(uint8_t * buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(int8_t* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(int16_t* buffer, char order, std::initializer_list<Nd4jLong> shape, nd4j::memory::Workspace* workspace);


#endif

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::scalar(T scalar, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        auto zType = DataTypeUtils::fromT<T>();

        int8_t *buffer;
        ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);

        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(zType, workspace);

        res->setShapeInfo(shapeInfo);
        res->setBuffer(buffer);
        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);

        res->assign(scalar);

        return res;
    }
    template NDArray* NDArrayFactory::scalar(double scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(float scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(int scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(bool scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::scalar(int16_t scalar, nd4j::memory::Workspace* workspace);

    ////////////////////////////////////////////////////////////////////////
    NDArray NDArrayFactory::_create(const NDArray *other, const bool copyStrides, nd4j::memory::Workspace* workspace) {
        NDArray result;
        //this->_length = shape::length(other->_shapeInfo);
        auto shapeLength = shape::shapeInfoByteLength(other->getShapeInfo());

        result.setWorkspace(workspace);
        auto tLen = nd4j::DataTypeUtils::sizeOf(ArrayOptions::dataType(other->getShapeInfo()));

        int8_t *buffer = nullptr;
        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(buffer, workspace, other->lengthOf() * tLen, int8_t);
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(other->getShapeInfo()), Nd4jLong);
        // FIXME: memcpy should be removed
        // memcpy(_buffer, other->_buffer, arrLength*sizeOfT());      // copy other._buffer information into new array

        memcpy(shapeInfo, other->getShapeInfo(), shapeLength);     // copy shape information into new array

        if(!copyStrides)
            shape::updateStrides(shapeInfo, other->ordering());

        ArrayOptions::setDataType(shapeInfo, other->dataType());
        result.setBuffer(buffer);
        result.setShapeInfo(shapeInfo);
        result.triggerAllocationFlag(true, true);

        return result;
    }

////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray NDArrayFactory::_scalar(T scalar, nd4j::memory::Workspace* workspace) {
        NDArray res;
        auto zType = DataTypeUtils::fromT<T>();

        int8_t *buffer;
        ALLOCATE(buffer, workspace, 1 * sizeof(T), int8_t);

        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(zType, workspace);

        res.setShapeInfo(shapeInfo);
        res.setBuffer(buffer);
        res.triggerAllocationFlag(true, true);
        res.setWorkspace(workspace);

        res.assign(scalar);

        return res;
    }
    template NDArray NDArrayFactory::_scalar(double scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(float scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(int scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(int16_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_scalar(bool scalar, nd4j::memory::Workspace* workspace);


    template <typename T>
    NDArray NDArrayFactory::_create(T scalar, nd4j::memory::Workspace* workspace) {
        return _scalar<T>(scalar, workspace);
    }
    template NDArray NDArrayFactory::_create(double scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(float scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(float16 scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(Nd4jLong scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(uint8_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(int16_t scalar, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(bool scalar, nd4j::memory::Workspace* workspace);


    template<typename T>
    NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        auto result = new NDArray();
        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        Nd4jLong shapeOf[MAX_RANK];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        result->setWorkspace(workspace);

        if (workspace == nullptr) {
            if (order == 'f')
                result->setShapeInfo(shape::shapeBufferFortran(rank, shapeOf));
            else
                result->setShapeInfo(shape::shapeBuffer(rank, shapeOf));

            result->setBuffer(new int8_t[result->lengthOf() * sizeof(T)]);
        } else {
            result->setBuffer(reinterpret_cast<int8_t *>(workspace->allocateBytes(data.size() * sizeof(T))));
            result->setShapeInfo(reinterpret_cast<Nd4jLong*>(workspace->allocateBytes(shape::shapeInfoByteLength(rank))));
            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, result->shapeInfo());
            else
                shape::shapeBuffer(rank, shapeOf, result->shapeInfo());

            // we're triggering length recalculation
            result->setShapeInfo(result->shapeInfo());
        }

        ArrayOptions::setDataType(result->shapeInfo(), DataTypeUtils::fromT<T>());

        if (result->lengthOf() != data.size()) {
            nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(result->shapeInfo()));
            throw std::runtime_error("Data size doesn't match shape");
        }

        //memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        //memcpy(result->buffer(), data.data(), sizeof(T) * result->lengthOf());

        result->triggerAllocationFlag(true, true);
        shape::updateStrides(result->shapeInfo(), order);

        return result;
    }
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<double> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<float16> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<Nd4jLong> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<uint8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<int16_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<bool> &data, nd4j::memory::Workspace* workspace);


        ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order, nd4j::memory::Workspace* workspace) {
        auto result = new NDArray(order, shape, DataTypeUtils::fromT<T>());
        result->assign(value);
        return result;
    }
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const double value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float value, const char order, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const float16 value, const char order, nd4j::memory::Workspace* workspace);
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

        for (Nd4jLong e = 0; e < numElements; e++) {
            T step = (T) e / ((T) numElements - (T) 1.0f);
            reinterpret_cast<T *>(result->getBuffer())[e] = (from * ((T) 1.0f - step) + step * to);
        }
        return result;
    }

    template <typename T>
    NDArray* NDArrayFactory::vector(Nd4jLong length, const T startingValue, nd4j::memory::Workspace *workspace) {
        auto res = new NDArray();
        Nd4jLong *shapeInfo = nullptr;
        ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), length, workspace);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, length * sizeof(T), int8_t);

        res->setShapeInfo(shapeInfo);
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
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const Nd4jLong startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const uint8_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int8_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const int16_t startingValue, nd4j::memory::Workspace *workspace);
    template NDArray* NDArrayFactory::vector(Nd4jLong length, const bool startingValue, nd4j::memory::Workspace *workspace);



    template <typename T>
    NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace) {
        return _create(order, shape, DataTypeUtils::fromT<T>(), workspace);
    }
    BUILD_SINGLE_TEMPLATE(template NDArray NDArrayFactory::_create, (const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

    NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
        NDArray res;
        int rank = (int) shape.size();

        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(rank), Nd4jLong);
        if (order == 'c')
            shape::shapeBuffer(rank, (const_cast<std::vector<Nd4jLong>&>(shape)).data(), shapeInfo);
        else
            shape::shapeBufferFortran(rank, (const_cast<std::vector<Nd4jLong>&>(shape)).data(), shapeInfo);

        ArrayOptions::setDataType(shapeInfo, dtype);
        res.setShapeInfo(shapeInfo);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, res.lengthOf() * DataTypeUtils::sizeOfElement(dtype), int8_t);
        res.setBuffer(buffer);
        res.setWorkspace(workspace);

        return res;
    }

    NDArray NDArrayFactory::_create(nd4j::DataType dtype, nd4j::memory::Workspace* workspace) {
        NDArray res;
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dtype, workspace);
        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, DataTypeUtils::sizeOfElement(dtype), int8_t);
        memset(buffer, 0, DataTypeUtils::sizeOfElement(dtype));
        res.setBuffer(buffer);
        res.setWorkspace(workspace);
        res.setShapeInfo(shapeInfo);
        res.triggerAllocationFlag(true, true);
        return res;
    }


    template <typename T>
    NDArray* NDArrayFactory::create(const T value, nd4j::memory::Workspace* workspace) {
        return scalar<T>(value, workspace);
    }
    template NDArray* NDArrayFactory::create(const double value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const float value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const float16 value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const Nd4jLong value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const int value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const int16_t value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const int8_t value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const uint8_t value, nd4j::memory::Workspace* workspace);
    template NDArray* NDArrayFactory::create(const bool value, nd4j::memory::Workspace* workspace);


    template <typename T>
    NDArray NDArrayFactory::_create(std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        NDArray res;
        auto shapeInfo = ShapeBuilders::createVectorShapeInfo(DataTypeUtils::fromT<T>(), values.size(), workspace);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, values.size() * sizeof(T), int8_t);
        res.setBuffer(buffer);
        res.setShapeInfo(shapeInfo);

        _memcpy<T>(buffer, values);

        res.triggerAllocationFlag(true, true);
        res.setWorkspace(workspace);
        return res;
    }
    template NDArray NDArrayFactory::_create(std::vector<double> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<float> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<float16> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<Nd4jLong> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<int> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<int16_t> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<int8_t> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<uint8_t> &values, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(std::vector<bool> &values, nd4j::memory::Workspace* workspace);


    template <typename T>
    NDArray* NDArrayFactory::empty(nd4j::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace);
        result->triggerAllocationFlag(false, true);

        return result;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray* NDArrayFactory::empty, (nd4j::memory::Workspace* workspace), LIBND4J_TYPES);

    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray(order, shape, value.dataType(), workspace);
        res->assign(value);
        return res;
    }

    NDArray* NDArrayFactory::create( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        int rank = (int) shape.size();

        if (rank > MAX_RANK)
            throw std::invalid_argument("Rank of NDArray can't exceed 32");

        auto shapeOf = new Nd4jLong[rank];
        int cnt = 0;

        for (auto &item: shape)
            shapeOf[cnt++] = item;

        res->setWorkspace(workspace);
        if (workspace == nullptr) {
            if (order == 'f')
                res->setShapeInfo(shape::shapeBufferFortran(rank, shapeOf));
            else
                res->setShapeInfo(shape::shapeBuffer(rank, shapeOf));

            res->setBuffer(new int8_t[res->lengthOf() * res->sizeOfT()]);
        } else {
            if (order == 'f')
                shape::shapeBufferFortran(rank, shapeOf, res->shapeInfo());
            else
                shape::shapeBuffer(rank, shapeOf, res->shapeInfo());


            res->setShapeInfo(reinterpret_cast<Nd4jLong*>(workspace->allocateBytes(shape::shapeInfoByteLength(rank))));
            res->setBuffer(reinterpret_cast<int8_t *>(workspace->allocateBytes(res->lengthOf() * DataTypeUtils::sizeOfElement(dataType))));
        }

        ArrayOptions::setDataType(res->shapeInfo(), dataType);
        memset(res->buffer(), 0, DataTypeUtils::sizeOfElement(dataType) * res->lengthOf());

        res->triggerAllocationFlag(true, true);

        shape::updateStrides(res->shapeInfo(), order);

        delete[] shapeOf;
        return res;
    }

    template <typename T>
    NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<T> &data, nd4j::memory::Workspace* workspace) {
        auto res = _create<T>(order, shape, workspace);
        //memcpy(res.buffer(), data.data(), res.lengthOf() * res.sizeOfT());
        _memcpy<T>(res.getBuffer(), data);
        return res;
    }
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<double> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<float> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<float16> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<Nd4jLong> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<int> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<int16_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<int8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<uint8_t> &data, nd4j::memory::Workspace* workspace);
    template NDArray NDArrayFactory::_create(const char order, const std::vector<Nd4jLong> &shape, std::vector<bool> &data, nd4j::memory::Workspace* workspace);
}

