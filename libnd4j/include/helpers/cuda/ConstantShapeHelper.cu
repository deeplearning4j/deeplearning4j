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
#include <ConstantHelper.h>

namespace nd4j {
    static int getCurrentDevice() {
        int dev = 0;
        auto res = cudaGetDevice(&dev);

        if (res != 0)
            throw cuda_exception::build("cudaGetDevice failed", res);

        return dev;
    }

    ConstantShapeHelper::ConstantShapeHelper() {
        for (int e = 0; e < 32; e++) {
            std::map<ShapeDescriptor, DataBuffer> cache;
            _cache[e] = cache;
        }
    }

    ConstantShapeHelper* ConstantShapeHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new ConstantShapeHelper();

        return _INSTANCE;
    }

    DataBuffer& ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor &descriptor) {
        int deviceId = 0;

        _mutex.lock();

        if (_cache[deviceId].count(descriptor) == 0) {
            switch (descriptor.rank()) {
                case 0: {
                    auto hPtr = descriptor.isEmpty() ? ShapeBuilders::emptyShapeInfo(descriptor.dataType()) : ShapeBuilders::createScalarShapeInfo(descriptor.dataType());
                    auto dPtr = ConstantHelper::getInstance()->replicatePointer(hPtr, shape::shapeInfoByteLength(hPtr));
                    DataBuffer buffer(hPtr, dPtr);
                    ShapeDescriptor descriptor1(descriptor);
                    _cache[deviceId][descriptor1] = buffer;

                    _mutex.unlock();

                    return _cache[deviceId][descriptor1];
                }
                case 1: {
                    auto hPtr = ShapeBuilders::createVectorShapeInfo(descriptor.dataType(), descriptor.shape()[0]);
                    auto dPtr = ConstantHelper::getInstance()->replicatePointer(hPtr, shape::shapeInfoByteLength(hPtr));
                    DataBuffer buffer(hPtr, dPtr);
                    ShapeDescriptor descriptor1(descriptor);
                    _cache[deviceId][descriptor1] = buffer;

                    _mutex.unlock();

                    return _cache[deviceId][descriptor1];
                }
                case 2:
                default: {
                    auto hPtr = ShapeBuilders::createShapeInfo(descriptor.dataType(), descriptor.order(), descriptor.shape());
                    auto dPtr = ConstantHelper::getInstance()->replicatePointer(hPtr, shape::shapeInfoByteLength(hPtr));
                    DataBuffer buffer(hPtr, dPtr);
                    ShapeDescriptor descriptor1(descriptor);
                    _cache[deviceId][descriptor1] = buffer;

                    _mutex.unlock();

                    return _cache[deviceId][descriptor1];
                }
            }
        } else {
            _mutex.unlock();

            return _cache[deviceId].at(descriptor);
        }
    }

    DataBuffer& ConstantShapeHelper::bufferForShapeInfo(const Nd4jLong *shapeInfo) {
        ShapeDescriptor descriptor(shapeInfo);
        return bufferForShapeInfo(descriptor);
    }

    bool ConstantShapeHelper::checkBufferExistanceForShapeInfo(ShapeDescriptor &descriptor) {
        bool result;
        int deviceId = getCurrentDevice();
        _mutex.lock();

        if (_cache[deviceId].count(descriptor) == 0)
            result = false;
        else
            result = true;

        _mutex.unlock();

        return result;
    }

    nd4j::ConstantShapeHelper* nd4j::ConstantShapeHelper::_INSTANCE = 0;
}