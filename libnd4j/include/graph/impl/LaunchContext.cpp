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

#include <graph/LaunchContext.h>
#include <logger.h>
#include <exceptions/cuda_exception.h>
#include <helpers/cublasHelper.h>

namespace nd4j {
namespace graph {

LaunchContext* LaunchContext::sDefaultContext = nullptr;

#ifdef __CUDABLAS__

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext(cudaStream_t *cudaStream, void* reductionPointer, void* scalarPointer, int* allocationPointer)  {

	_cudaStream 	   = cudaStream;
	_cudaSpecialStream = nullptr;
	_reductionPointer  = reductionPointer;
	_scalarPointer     = scalarPointer;
	_allocationPointer = allocationPointer;
	_workspace = nullptr;
}
////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext(cudaStream_t *cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer, void* scalarPointer, int* allocationPointer)  {

	_cudaStream 	   = cudaStream;
	_cudaSpecialStream = &specialCudaStream; // ideal is = new cudaStream_t; *_cudaSpecialStream = specialCudaStream;
	_reductionPointer  = reductionPointer;
	_scalarPointer     = scalarPointer;
	_allocationPointer = allocationPointer;
	_workspace = nullptr;
}
#endif

LaunchContext::~LaunchContext() {
#ifdef __CUDABLAS__
    if (_isAllocated) {
        cudaStreamSynchronize(*_cudaStream);
        cudaStreamSynchronize(*_cudaSpecialStream);

        cudaStreamDestroy(*_cudaStream);
        cudaStreamDestroy(*_cudaSpecialStream);

        delete _cudaStream;
        delete _cudaSpecialStream;

        cudaFree(_reductionPointer);
        cudaFree(_allocationPointer);
        cudaFree(_scalarPointer);
    }
#endif
}

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
            // default constructor, just to make clang/ranlib happy
    _workspace = nullptr;
    _deviceID = 0;

#ifdef __CUDABLAS__
    _cudaStream  = new cudaStream_t();
    _cudaSpecialStream = new cudaStream_t();
    if (nullptr == _cudaStream || nullptr == _cudaSpecialStream)
        throw std::runtime_error("Failed to allocate memory for new CUDA stream");

    cudaError_t err = cudaStreamCreate(_cudaStream);
    if (err != 0)
        throw cuda_exception::build("Failed to create default CUDA stream with launch context", err);

    err = cudaStreamCreate(_cudaSpecialStream);
    if (err != 0)
        throw cuda_exception::build("Failed to create special CUDA stream with launch context", err);

    _cublasHandle = cublas::handle();

    auto res = cudaStreamSynchronize(*_cudaStream);
    if (res != 0)
        throw cuda_exception::build("Initial sync failed", res);

    res = cudaMalloc(reinterpret_cast<void**>(&_reductionPointer), 1024 * 1024 * 8);
    if (res != 0)
        throw std::runtime_error("_reductionPointer allocation failed");

    res = cudaMalloc(reinterpret_cast<void**>(&_scalarPointer), 8);
    if (res != 0)
        throw std::runtime_error("_scalarPointer allocation failed");

    res = cudaMalloc(reinterpret_cast<void**>(&_allocationPointer), 1024 * 1024 * 8);
    if (res != 0)
        throw std::runtime_error("_allocationPointer allocation failed");
#else
    //
#endif
}

LaunchContext* LaunchContext::defaultContext() {
    if (!LaunchContext::sDefaultContext) {
        LaunchContext::sDefaultContext = new LaunchContext;
    }
    return LaunchContext::sDefaultContext;
}

}
}