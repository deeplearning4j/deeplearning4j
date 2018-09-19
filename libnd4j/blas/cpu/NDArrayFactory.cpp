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
            res->setBuffer(reinterpret_cast<int8_t *>(workspace->allocateBytes(res->lengthOf() * res->sizeOfT())));
        }

        memset(res->buffer(), 0, res->sizeOfT() * res->lengthOf());

        res->triggerAllocationFlag(true, true);

        shape::updateStrides(res->shapeInfo(), order);

        delete[] shapeOf;
        return res;
    }


#ifndef __JAVACPP_HACK__
    template <typename T>
    NDArray* NDArrayFactory::create(std::initializer_list<T> v, nd4j::memory::Workspace* workspace) {
        std::vector<T> vector(v);
        return create(vector, workspace);
    }

    template <typename T>
    NDArray* NDArrayFactory::create(std::vector<T> &values, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();

        int8_t *buffer = nullptr;
        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(buffer, workspace, values.size() * sizeof(T), int8_t);
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(1), Nd4jLong);
        shape::shapeVector(values.size(), shapeInfo);

        res->setBuffer(buffer);
        res->setShapeInfo(shapeInfo);
        memcpy(res->buffer(), values.data(), values.size() * sizeof(T));

        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);
        return res;
    }
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

        if (result->lengthOf() != data.size()) {
            nd4j_printf("Data size [%i] doesn't match shape length [%i]\n", data.size(), shape::length(result->shapeInfo()));
            throw std::runtime_error("Data size doesn't match shape");
        }

        //memset(_buffer, 0, sizeOfT() * shape::length(_shapeInfo));
        memcpy(result->buffer(), data.data(), sizeof(T) * result->lengthOf());

        result->triggerAllocationFlag(true, true);
        shape::updateStrides(result->shapeInfo(), order);

        return result;
    }


    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order) {
        auto result = new NDArray(order, shape);
        result->assign(value);
        return result;
    }

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order) {
        return valueOf(std::vector<Nd4jLong>(shape), value, order);
    }

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

    ////////////////////////////////////////////////////////////////////////
    template <typename T>
    NDArray* NDArrayFactory::create(std::initializer_list<Nd4jLong> s, nd4j::memory::Workspace* workspace) {
        auto res = new NDArray();
        std::vector<Nd4jLong> shape(s);
        int rank = (int) shape.size();

        Nd4jLong *shapeInfo = nullptr;
        ALLOCATE(shapeInfo, workspace, shape::shapeInfoByteLength(rank), Nd4jLong);
        shape::shapeBuffer(rank, shape.data(), res->shapeInfo());
        res->setShapeInfo(shapeInfo);

        int8_t *buffer = nullptr;
        ALLOCATE(buffer, workspace, res->lengthOf() * sizeof(T), int8_t);
        res->setBuffer(buffer);

        res->triggerAllocationFlag(true, true);
        res->setWorkspace(workspace);
        return res;
    }

    template <typename T>
    NDArray* NDArrayFactory::empty(nd4j::memory::Workspace* workspace) {
        auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), workspace);
        ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
        auto result = new NDArray(nullptr, shapeInfo, workspace);
        result->triggerAllocationFlag(false, true);

        return result;
    }
}

