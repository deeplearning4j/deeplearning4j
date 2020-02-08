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
// @author raver119@gmail.com
//

#include "../ConstantShapeHelper.h"
#include <exceptions/cuda_exception.h>
#include <ShapeDescriptor.h>
#include <ShapeBuilders.h>
#include <AffinityManager.h>
#include <ConstantHelper.h>

namespace nd4j {

    ConstantShapeHelper::ConstantShapeHelper() {
        auto numDevices = AffinityManager::numberOfDevices();

        _cache.resize(numDevices);
        for (int e = 0; e < numDevices; e++) {
            std::map<ShapeDescriptor, ConstantDataBuffer> cache;
            _cache[e] = cache;
        }
    }

    ConstantShapeHelper* ConstantShapeHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new ConstantShapeHelper();

        return _INSTANCE;
    }

    ConstantDataBuffer ConstantShapeHelper::bufferForShapeInfo(nd4j::DataType dataType, char order, const std::vector<Nd4jLong> &shape) {
        ShapeDescriptor descriptor(dataType, order, shape);
        return bufferForShapeInfo(descriptor);
    }

    ConstantDataBuffer ConstantShapeHelper::bufferForShapeInfo(const nd4j::DataType dataType, const char order, const int rank, const Nd4jLong* shape) {
        ShapeDescriptor descriptor(dataType, order, shape, rank);
        return bufferForShapeInfo(descriptor);
    }

    ConstantDataBuffer ConstantShapeHelper::bufferForShapeInfo(const ShapeDescriptor &descriptor) {
        int deviceId = AffinityManager::currentDeviceId();

        std::lock_guard<std::mutex> lock(_mutex);

        if (_cache[deviceId].count(descriptor) == 0) {
            auto hPtr = descriptor.toShapeInfo();
            auto dPtr = ConstantHelper::getInstance()->replicatePointer(hPtr, shape::shapeInfoByteLength(hPtr));
            ConstantDataBuffer buffer(hPtr, dPtr, shape::shapeInfoLength(hPtr) * sizeof(Nd4jLong), DataType::INT64);
            ShapeDescriptor descriptor1(descriptor);
            _cache[deviceId][descriptor1] = buffer;
            return _cache[deviceId][descriptor1];
        } else {
            return _cache[deviceId].at(descriptor);
        }
    }

    ConstantDataBuffer ConstantShapeHelper::bufferForShapeInfo(const Nd4jLong *shapeInfo) {
        ShapeDescriptor descriptor(shapeInfo);
        return bufferForShapeInfo(descriptor);
    }

    bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor &descriptor) {
        auto deviceId = AffinityManager::currentDeviceId();
        std::lock_guard<std::mutex> lock(_mutex);

        return _cache[deviceId].count(descriptor) != 0;
    }

    Nd4jLong* ConstantShapeHelper::createShapeInfo(const nd4j::DataType dataType, const char order, const int rank, const Nd4jLong* shape) {
        ShapeDescriptor descriptor(dataType, order, shape, rank);
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::createShapeInfo(const nd4j::DataType dataType, const Nd4jLong* shapeInfo) {
        return ConstantShapeHelper::createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo), shape::shapeOf(const_cast<Nd4jLong*>(shapeInfo)));
    }

    Nd4jLong* ConstantShapeHelper::emptyShapeInfo(const nd4j::DataType dataType) {
        auto descriptor = ShapeDescriptor::emptyDescriptor(dataType);
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::scalarShapeInfo(const nd4j::DataType dataType) {
        auto descriptor = ShapeDescriptor::scalarDescriptor(dataType);
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::vectorShapeInfo(const Nd4jLong length, const nd4j::DataType dataType) {
        auto descriptor = ShapeDescriptor::vectorDescriptor(length, dataType);
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::createShapeInfo(const nd4j::DataType dataType, const char order, const std::vector<Nd4jLong> &shape) {
        ShapeDescriptor descriptor(dataType, order, shape);
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::createShapeInfo(const ShapeDescriptor &descriptor) {
        return bufferForShapeInfo(descriptor).primaryAsT<Nd4jLong>();
    }

    Nd4jLong* ConstantShapeHelper::createFromExisting(Nd4jLong *shapeInfo, bool destroyOriginal) {
        ShapeDescriptor descriptor(shapeInfo);
        auto result = createShapeInfo(descriptor);

        if (destroyOriginal)
            RELEASE(shapeInfo, nullptr);

        return result;
    }

    Nd4jLong* ConstantShapeHelper::createFromExisting(Nd4jLong *shapeInfo, nd4j::memory::Workspace *workspace) {
        ShapeDescriptor descriptor(shapeInfo);
        auto result = createShapeInfo(descriptor);

        RELEASE(shapeInfo, workspace);

        return result;
    }

    nd4j::ConstantShapeHelper* nd4j::ConstantShapeHelper::_INSTANCE = 0;
}