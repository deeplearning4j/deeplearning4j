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
// Created by raver119 on 30.11.17.
//

#include <execution/LaunchContext.h>
#include <logger.h>
#include <exceptions/cuda_exception.h>
#include <helpers/cublasHelper.h>
#include <thread>
#include <execution/AffinityManager.h>

thread_local nd4j::ContextBuffers contextBuffers = nd4j::ContextBuffers();

namespace nd4j {

    std::vector<std::shared_ptr<LaunchContext>> LaunchContext::_contexts = std::vector<std::shared_ptr<LaunchContext>>();
    std::mutex LaunchContext::_mutex;

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext(cudaStream_t *cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer, void* scalarPointer, int* allocationPointer)  {

	//_cudaStream 	   = cudaStream;
	//_cudaSpecialStream = &specialCudaStream; // ideal is = new cudaStream_t; *_cudaSpecialStream = specialCudaStream;
	//_reductionPointer  = reductionPointer;
	//_scalarPointer     = scalarPointer;
	//_allocationPointer = allocationPointer;
	_workspace = nullptr;
	_isAllocated = false;
}

LaunchContext::~LaunchContext() {
    if (_isAllocated) {

    }
}

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
            // default constructor, just to make clang/ranlib happy
    _workspace = nullptr;
    _deviceID = 0;

    _isAllocated = true;
}

    LaunchContext::LaunchContext(Nd4jPointer cudaStream, Nd4jPointer reductionPointer, Nd4jPointer scalarPointer, Nd4jPointer allocationPointer) {
        _isAllocated = false;
        //_cudaStream = reinterpret_cast<cudaStream_t*>(cudaStream);
       // _cudaSpecialStream = reinterpret_cast<cudaStream_t*>(cudaStream);
        //_reductionPointer = reductionPointer;
        //_scalarPointer = scalarPointer;
        //_allocationPointer = reinterpret_cast<int *>(allocationPointer);
    }

    LaunchContext* LaunchContext::defaultContext() {
        /**
        * This method returns LaunchContext, that has multiple entities within:
        * 1) temporary buffers. they must be per-thread
        * 2) CUDA stream. it must be either per-thread or per-device
        * 3) cuBLAS handle. it must be per-device
        */
        auto deviceId = AffinityManager::currentDeviceId();

        // we need this block synchronous, to avoid double initialization etc
        _mutex.lock();
        if (LaunchContext::_contexts.empty()) {
            // create one context per device
            auto numDevices = AffinityManager::numberOfDevices();

            _contexts.resize(numDevices);
            for (int e = 0; e < numDevices; e++) {
                AffinityManager::setCurrentNativeDevice(e);

                LaunchContext::_contexts[e] = std::make_shared<LaunchContext>();
            }

            // don't forget to restore device back again
            AffinityManager::setCurrentNativeDevice(deviceId);
        }
        _mutex.unlock();

        // return context for current device
        return LaunchContext::_contexts[deviceId].get();
    }


    void* LaunchContext::getReductionPointer () const {
        return contextBuffers.reductionBuffer();
    };

    void* LaunchContext::getScalarPointer() const {
        return contextBuffers.scalarBuffer();
    };

    int* LaunchContext::getAllocationPointer() const {
        return reinterpret_cast<int*>(contextBuffers.allocationBuffer());
    };

    void* LaunchContext::getCublasHandle() const {
        return CublasHelper::getInstance()->handle();
    };

    void* LaunchContext::getCusolverHandle() const {
        return CublasHelper::getInstance()->solver();
    };

    cudaStream_t* LaunchContext::getCudaStream() const {
        return reinterpret_cast<cudaStream_t*>(contextBuffers.execStream());
    };

    cudaStream_t* LaunchContext::getCudaSpecialStream() const {
        return reinterpret_cast<cudaStream_t*>(contextBuffers.specialStream());;
    };


    void LaunchContext::setReductionPointer (void* reductionPointer) {
        contextBuffers.setReductionBuffer(reductionPointer);
    };

    void LaunchContext::setScalarPointer(void* scalarPointer) {
        contextBuffers.setScalarBuffer(scalarPointer);
    };

    void LaunchContext::setAllocationPointer(int* allocationPointer) {
        contextBuffers.setAllocationBuffer(allocationPointer);
    };

    void LaunchContext::setCudaStream(cudaStream_t* cudaStream)  {
        //_cudaStream = cudaStream;
    };

    void LaunchContext::setCudaSpecialStream(cudaStream_t* cudaStream)  {
        //_cudaSpecialStream = cudaStream;
    };

    void LaunchContext::setCublasHandle(void *handle) {
        _cublasHandle = handle;
    };

    void LaunchContext::swapContextBuffers(ContextBuffers &buffers) {
        contextBuffers = buffers;
    };

    void LaunchContext::releaseBuffers() {
        nd4j_printf("LaunchContext::releaseBuffers() was invoked\n", "");
        contextBuffers.release();
    }

    bool LaunchContext::isInitialized() {
        return contextBuffers.isInitialized();
    }

    sd::ErrorReference* LaunchContext::errorReference() {
        return contextBuffers.errorReference();
    }
}