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
//  @author raver119@gmail.com
//

#include <exceptions/cuda_exception.h>
#include <ConstantHelper.h>
#include <execution/LaunchContext.h>
#include <logger.h>
#include <cuda_runtime.h>
#include <cuda.h>

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
        int dev = 0;
        auto res = cudaGetDevice(&dev);

        if (res != 0)
            throw cuda_exception::build("cudaGetDevice failed", res);

        return dev;
    }

    int ConstantHelper::getNumberOfDevices() {
        int dev = 0;
        auto res = cudaGetDeviceCount(&dev);

        if (res != 0)
            throw cuda_exception::build("cudaGetDeviceCount failed", res);

        return dev;
    }


    ConstantHelper::ConstantHelper() {
        auto initialDevice = getCurrentDevice();

        auto numDevices = getNumberOfDevices();

        // filling all pointers
        for (int e = 0; e < numDevices; e++) {
            auto res = cudaSetDevice(e);
            if (res != 0)
                throw cuda_exception::build("cudaSetDevice failed", res);
             auto constant = getConstantSpace();

            _devicePointers[e] = constant;
            _deviceOffsets[e] = 0;
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
        if (_devicePointers.count(deviceId) == 0) {
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
            auto res = cudaMemcpyAsync(ptr, src, numBytes, cudaMemcpyHostToDevice, *nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("cudaMemcpyToSymbolAsync failed", res);

            res = cudaStreamSynchronize(*nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("cudaStreamSynchronize failed", res);

            _mutex.unlock();
            return ptr;
        } else {
            auto originalBytes = numBytes;
            auto rem = numBytes % 8;
            if (rem != 0)
                numBytes += 8 - rem;

            _deviceOffsets[deviceId] += numBytes;

            auto res = cudaMemcpyToSymbolAsync(deviceConstantMemory, const_cast<const void *>(src), originalBytes, constantOffset, cudaMemcpyHostToDevice, *nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("cudaMemcpyToSymbolAsync failed", res);

            res = cudaStreamSynchronize(*nd4j::LaunchContext ::defaultContext()->getCudaSpecialStream());
            if (res != 0)
                throw cuda_exception::build("cudaStreamSynchronize failed", res);

            _mutex.unlock();
            return reinterpret_cast<int8_t *>(constantPtr) + constantOffset;
        }
    }

    nd4j::ConstantHelper* nd4j::ConstantHelper::_INSTANCE = 0;
}