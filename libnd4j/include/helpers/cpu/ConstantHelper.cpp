/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
//  @author raver119@gmail.com
//

#ifndef __CUDABLAS__

#include <ConstantHelper.h>
#include <execution/AffinityManager.h>
#include <types/types.h>
#include <loops/type_conversions.h>
#include <type_boilerplate.h>
#include <cstring>

namespace nd4j {
    ConstantHelper::ConstantHelper() {
        int numDevices = getNumberOfDevices();
        _cache.resize(numDevices);
        _counters.resize(numDevices);
        for (int e = 0; e < numDevices; e++) {
            std::map<ConstantDescriptor, ConstantHolder*> map;

            _cache[e] = map;
            _counters[e] = 0L;
        }
    }

    ConstantHelper* ConstantHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new nd4j::ConstantHelper();

        return _INSTANCE;
    }

    void* ConstantHelper::replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace) {
        if (workspace == nullptr) {
            auto deviceId = getCurrentDevice();
            _counters[deviceId] += numBytes;
        }

        int8_t *ptr = nullptr;
        ALLOCATE(ptr, workspace, numBytes, int8_t);

        std::memcpy(ptr, src, numBytes);
        return ptr;
    }

    int ConstantHelper::getCurrentDevice() {
        return AffinityManager::currentDeviceId();
    }

    int ConstantHelper::getNumberOfDevices() {
        return AffinityManager::numberOfDevices();
    }

    ConstantDataBuffer* ConstantHelper::constantBuffer(const ConstantDescriptor &descriptor, nd4j::DataType dataType) {
        const auto deviceId = getCurrentDevice();

        // we're locking away cache modification
        _mutexHolder.lock();

        if (_cache[deviceId].count(descriptor) == 0) {
            _cache[deviceId][descriptor] = new ConstantHolder();
        }

        auto holder = _cache[deviceId][descriptor];

        // releasing cache lock
        _mutexHolder.unlock();


        ConstantDataBuffer* result;

        // access to this holder instance is synchronous
        holder->mutex()->lock();

        if (holder->hasBuffer(dataType))
            result = holder->getConstantDataBuffer(dataType);
        else {
            auto size = descriptor.length() * DataTypeUtils::sizeOf(dataType);
            auto cbuff = new int8_t[size];
            _counters[deviceId] += size;

            // create buffer with this dtype
            if (descriptor.isFloat()) {
                BUILD_DOUBLE_SELECTOR(nd4j::DataType::DOUBLE, dataType, nd4j::TypeCast::convertGeneric, (nullptr, const_cast<double *>(descriptor.floatValues().data()), descriptor.length(), cbuff), (nd4j::DataType::DOUBLE, double), LIBND4J_TYPES);
            } else if (descriptor.isInteger()) {
                BUILD_DOUBLE_SELECTOR(nd4j::DataType::INT64, dataType, nd4j::TypeCast::convertGeneric, (nullptr, const_cast<Nd4jLong *>(descriptor.integerValues().data()), descriptor.length(), cbuff), (nd4j::DataType::INT64, Nd4jLong), LIBND4J_TYPES);
            }

            ConstantDataBuffer dataBuffer(cbuff, nullptr, descriptor.length(), DataTypeUtils::sizeOf(dataType));
            holder->addBuffer(dataBuffer, dataType);

            result = holder->getConstantDataBuffer(dataType);
        }
        holder->mutex()->unlock();

        return result;
    }

    Nd4jLong ConstantHelper::getCachedAmount(int deviceId) {
        int numDevices = getNumberOfDevices();
        if (deviceId > numDevices || deviceId < 0)
            return 0L;
        else
            return _counters[deviceId];
    }

    nd4j::ConstantHelper* nd4j::ConstantHelper::_INSTANCE = 0;
}

#endif