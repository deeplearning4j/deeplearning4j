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

#include <exceptions/cuda_exception.h>
#include <ConstantHelper.h>
#include <DataTypeUtils.h>
#include <shape.h>
#include <execution/LaunchContext.h>
#include <specials.h>
#include <logger.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <execution/AffinityManager.h>

#define CONSTANT_LIMIT 49152

__constant__ char deviceConstantMemory[CONSTANT_LIMIT];

namespace nd4j {
    static void* getConstantSpace() {
        Nd4jPointer dConstAddr;
        auto dZ = cudaGetSymbolAddress(reinterpret_cast<void **>(&dConstAddr), deviceConstantMemory);

        if (dZ != 0)
            throw cuda_exception::build("cudaGetSymbolAddress(...) failed", dZ);

        return dConstAddr;
    }

    int ConstantHelper::getCurrentDevice() {
        return AffinityManager::currentDeviceId();
    }

    int ConstantHelper::getNumberOfDevices() {
        return AffinityManager::numberOfDevices();
    }


    ConstantHelper::ConstantHelper() {
        auto initialDevice = getCurrentDevice();

        auto numDevices = getNumberOfDevices();
        _devicePointers.resize(numDevices);
        _deviceOffsets.resize(numDevices);
        _cache.resize(numDevices);
        _counters.resize(numDevices);

        // filling all pointers
        for (int e = 0; e < numDevices; e++) {
            auto res = cudaSetDevice(e);
            if (res != 0)
                throw cuda_exception::build("cudaSetDevice failed", res);
             auto constant = getConstantSpace();

            std::map<ConstantDescriptor, ConstantHolder*> devCache;

            _devicePointers[e] = constant;
            _deviceOffsets[e] = 0;
            _cache[e] = devCache;
            _counters[e] = 0L;
        }

        //
        auto res = cudaSetDevice(initialDevice);
        if (res != 0)
            throw cuda_exception::build("Final cudaSetDevice failed", res);
    }

    ConstantHelper* ConstantHelper::getInstance() {
        if (!_INSTANCE)
            _INSTANCE = new nd4j::ConstantHelper();

        return _INSTANCE;
    }

    void* ConstantHelper::replicatePointer(void *src, size_t numBytes, memory::Workspace *workspace) {
        _mutex.lock();

        auto deviceId = getCurrentDevice();
        Nd4jPointer constantPtr = nullptr;
        Nd4jLong constantOffset = 0L;
        if (_devicePointers[deviceId] == 0) {
            auto constant = getConstantSpace();

            // filling default ptr, which will be 0 probably
            _devicePointers[deviceId] = constant;
            _deviceOffsets[deviceId] = 0;
            constantPtr = constant;
        } else {
            constantPtr = _devicePointers[deviceId];
            constantOffset = _deviceOffsets[deviceId];
        }

        if (constantOffset + numBytes >= CONSTANT_LIMIT) {
            int8_t *ptr = nullptr;
            ALLOCATE_SPECIAL(ptr, workspace, numBytes, int8_t);
            auto res = cudaMemcpy(ptr, src, numBytes, cudaMemcpyHostToDevice);
            if (res != 0)
                throw cuda_exception::build("cudaMemcpy failed", res);

            _mutex.unlock();
            return ptr;
        } else {
            auto originalBytes = numBytes;
            auto rem = numBytes % 8;
            if (rem != 0)
                numBytes += 8 - rem;

            _deviceOffsets[deviceId] += numBytes;

            auto res = cudaMemcpyToSymbol(deviceConstantMemory, const_cast<const void *>(src), originalBytes, constantOffset, cudaMemcpyHostToDevice);
            if (res != 0)
                throw cuda_exception::build("cudaMemcpyToSymbol failed", res);

            _mutex.unlock();
            return reinterpret_cast<int8_t *>(constantPtr) + constantOffset;
        }
    }

    ConstantDataBuffer* ConstantHelper::constantBuffer(const ConstantDescriptor &descriptor, nd4j::DataType dataType) {
        const auto deviceId = getCurrentDevice();

        // all cache modifications are synchronous
        _mutexHolder.lock();

        if (_cache[deviceId].count(descriptor) == 0) {
            _cache[deviceId][descriptor] = new ConstantHolder();
        }
        auto holder = _cache[deviceId][descriptor];

        // release cache lock
        _mutexHolder.unlock();

        ConstantDataBuffer* result;

        // access to this holder instance is synchronous
        holder->mutex()->lock();

        if (holder->hasBuffer(dataType)) {
             result = holder->getConstantDataBuffer(dataType);
        } else {
            auto numBytes = descriptor.length() * DataTypeUtils::sizeOf(dataType);
            auto cbuff = new int8_t[numBytes];
            _counters[deviceId] += numBytes;

            // create buffer with this dtype
            if (descriptor.isFloat()) {
                BUILD_DOUBLE_SELECTOR(nd4j::DataType::DOUBLE, dataType, nd4j::SpecialTypeConverter::convertGeneric, (nullptr, const_cast<double *>(descriptor.floatValues().data()), descriptor.length(), cbuff), (nd4j::DataType::DOUBLE, double), LIBND4J_TYPES);
            } else if (descriptor.isInteger()) {
                BUILD_DOUBLE_SELECTOR(nd4j::DataType::INT64, dataType, nd4j::SpecialTypeConverter::convertGeneric, (nullptr, const_cast<Nd4jLong *>(descriptor.integerValues().data()), descriptor.length(), cbuff), (nd4j::DataType::INT64, Nd4jLong), LIBND4J_TYPES);
            }

            auto dbuff = replicatePointer(cbuff, descriptor.length() * DataTypeUtils::sizeOf(dataType));

            ConstantDataBuffer dataBuffer(cbuff, dbuff, descriptor.length(), DataTypeUtils::sizeOf(dataType));

            holder->addBuffer(dataBuffer, dataType);
            result = holder->getConstantDataBuffer(dataType);
        }
        // release holder lock
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