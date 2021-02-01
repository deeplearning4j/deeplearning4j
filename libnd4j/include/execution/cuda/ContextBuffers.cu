/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <helpers/logger.h>
#include <execution/AffinityManager.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace sd {
    ContextBuffers::ContextBuffers() {
        //nd4j_printf("Creating ContextBuffers for device [%i]\n", AffinityManager::currentDeviceId());
        _deviceId = AffinityManager::currentDeviceId();
    }

    ContextBuffers::ContextBuffers(const ContextBuffers &other) {
        release();

        this->_initialized = other._initialized;
        this->_allocated = other._allocated;
        this->_deviceId = other._deviceId;

        this->_specialStream = other._specialStream;
        this->_execStream = other._execStream;
        this->_allocationPointer = other._allocationPointer;
        this->_reductionPointer = other._reductionPointer;
        this->_scalarPointer = other._scalarPointer;
    }

    ContextBuffers& ContextBuffers::operator=(const ContextBuffers& other) {
        release();

        this->_initialized = other._initialized;
        this->_allocated = other._allocated;
        this->_deviceId = other._deviceId;

        this->_specialStream = other._specialStream;
        this->_execStream = other._execStream;
        this->_allocationPointer = other._allocationPointer;
        this->_reductionPointer = other._reductionPointer;
        this->_scalarPointer = other._scalarPointer;

        return *this;
    }

    ContextBuffers& ContextBuffers::operator=(ContextBuffers&& other) {
        release();

        this->_initialized = other._initialized;
        this->_allocated = other._allocated;
        this->_deviceId = other._deviceId;

        this->_specialStream = other._specialStream;
        this->_execStream = other._execStream;
        this->_allocationPointer = other._allocationPointer;
        this->_reductionPointer = other._reductionPointer;
        this->_scalarPointer = other._scalarPointer;

        return *this;
    }

    void ContextBuffers::release() {
        if (_allocated) {
            //nd4j_printf("Releasing ContextBuffers on device [%i]\n", _deviceId);

            if (_allocationPointer != nullptr)
                cudaFree(_allocationPointer);

            if (_scalarPointer != nullptr)
                cudaFreeHost(_scalarPointer);

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

            //////
            _allocated = false;
            _deviceId = -1;

            this->_specialStream = nullptr;
            this->_execStream = nullptr;
            this->_allocationPointer = nullptr;
            this->_reductionPointer = nullptr;
            this->_scalarPointer = nullptr;
        }

        _initialized = false;
    }

    ContextBuffers::~ContextBuffers() {
        release();
    }

    ContextBuffers::ContextBuffers(void* rPointer, void* sPointer, void* aPointer, bool isOwner) {
        _reductionPointer = rPointer;
        _scalarPointer = sPointer;
        _allocationPointer = aPointer;
        _allocated = isOwner;
    }

    void ContextBuffers::initialize() {
        _deviceId = AffinityManager::currentNativeDeviceId();
        //nd4j_printf("Initializing buffers on deviceId [%i]\n", _deviceId);

        auto res = cudaMalloc(reinterpret_cast<void**>(&_reductionPointer), 1024 * 1024 * 8);
        if (res != 0)
            throw cuda_exception::build("_reductionPointer allocation failed", res);

        res = cudaHostAlloc(reinterpret_cast<void**>(&_scalarPointer), 16, cudaHostAllocDefault);
        if (res != 0)
            throw cuda_exception::build("_scalarPointer allocation failed", res);

        res = cudaMalloc(reinterpret_cast<void**>(&_allocationPointer), 1024 * 1024 * 8);
        if (res != 0)
            throw cuda_exception::build("_allocationPointer allocation failed", res);

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
        _initialized = true;
    }

    void* ContextBuffers::reductionBuffer() {
        if (!_initialized)
            initialize();

        return _reductionPointer;
    }

    void* ContextBuffers::scalarBuffer() {
        if (!_initialized)
            initialize();

        return _scalarPointer;
    }

    void* ContextBuffers::allocationBuffer() {
        if (!_initialized)
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
        if (!_initialized) {
            //nd4j_printf("execStream not initialized\n", "");
            initialize();
        } else {
            //nd4j_printf("execStream is initialized\n", "");
        }

        return _execStream;
    }

    void* ContextBuffers::specialStream() {
        if (!_initialized) {
            //nd4j_printf("specialStream not initialized\n", "");
            initialize();
        } else {
            //nd4j_printf("specialStream is initialized\n", "");
        }

        return _specialStream;
    }

    bool ContextBuffers::isInitialized() {
        return _initialized;
    }

    sd::ErrorReference* ContextBuffers::errorReference() {
        return &_errorReference;
    }
}

