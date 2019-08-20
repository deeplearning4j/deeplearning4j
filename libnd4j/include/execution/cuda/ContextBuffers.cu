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

#include <execution/ContextBuffers.h>
#include <exceptions/cuda_exception.h>
#include <logger.h>
#include <AffinityManager.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace nd4j {
    ContextBuffers::ContextBuffers() {
        nd4j_printf("Creating ContextBuffers for device [%i]\n", AffinityManager::currentDeviceId());
        _deviceId = AffinityManager::currentDeviceId();
    }

    ContextBuffers::~ContextBuffers() {
        if (_allocated) {
            nd4j_printf("Releasing ContextBuffers\n","");

            if (_allocationPointer != nullptr)
                cudaFree(_allocationPointer);

            if (_scalarPointer != nullptr)
                cudaFree(_scalarPointer);

            if (_allocationPointer != nullptr)
                cudaFree(_reductionPointer);

            auto _cudaStream = reinterpret_cast<cudaStream_t*>(_execStream);
            auto _cudaSpecialStream = reinterpret_cast<cudaStream_t*>(_specialStream);

            cudaStreamSynchronize(*_cudaStream);
            cudaStreamSynchronize(*_cudaSpecialStream);

            cudaStreamDestroy(*_cudaStream);
            cudaStreamDestroy(*_cudaSpecialStream);

            delete _cudaStream;
            delete _cudaSpecialStream;
        }
    }

    ContextBuffers::ContextBuffers(void* rPointer, void* sPointer, void* aPointer, bool isOwner) {
        _reductionPointer = rPointer;
        _scalarPointer = sPointer;
        _allocationPointer = aPointer;
        _allocated = isOwner;
    }

    void ContextBuffers::initialize() {
        nd4j_printf("Initializing buffers on deviceId [%i]\n", AffinityManager::currentNativeDeviceId());

        auto res = cudaMalloc(reinterpret_cast<void**>(&_reductionPointer), 1024 * 1024 * 8);
        if (res != 0)
            throw std::runtime_error("_reductionPointer allocation failed");

        res = cudaMalloc(reinterpret_cast<void**>(&_scalarPointer), 16);
        if (res != 0)
            throw std::runtime_error("_scalarPointer allocation failed");

        res = cudaMalloc(reinterpret_cast<void**>(&_allocationPointer), 1024 * 1024 * 8);
        if (res != 0)
            throw std::runtime_error("_allocationPointer allocation failed");

        _execStream  = new cudaStream_t();
        _specialStream = new cudaStream_t();
        if (nullptr == _execStream || nullptr == _specialStream)
            throw std::runtime_error("Failed to allocate memory for new CUDA stream");

        res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_execStream));
        if (res != 0)
            throw cuda_exception::build("Failed to create default CUDA stream with launch context", res);

        res = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(_specialStream));
        if (res != 0)
            throw cuda_exception::build("Failed to create special CUDA stream with launch context", res);

        _allocated = true;
    }

    void* ContextBuffers::reductionBuffer() {
        if (_reductionPointer == nullptr)
            initialize();

        return _reductionPointer;
    }

    void* ContextBuffers::scalarBuffer() {
        if (_scalarPointer == nullptr)
            initialize();

        return _scalarPointer;
    }

    void* ContextBuffers::allocationBuffer() {
        if (_allocationPointer == nullptr)
            initialize();

        return _allocationPointer;
    }

    void ContextBuffers::setReductionBuffer(void* pointer) {
        _reductionPointer = pointer;
    }

    void ContextBuffers::setScalarBuffer(void* pointer) {
        _scalarPointer = pointer;
    }

    void ContextBuffers::setAllocationBuffer(void* pointer) {
        _allocationPointer = pointer;
    }

    void ContextBuffers::triggerOwnership(bool isOwner) {
        _allocated = isOwner;
    }

    int ContextBuffers::deviceId() {
        return _deviceId;
    }

    void* ContextBuffers::execStream() {
        if (_execStream == nullptr)
            initialize();

        return _execStream;
    }

    void* ContextBuffers::specialStream() {
        if (_specialStream == nullptr)
            initialize();

        return _specialStream;
    }
}
