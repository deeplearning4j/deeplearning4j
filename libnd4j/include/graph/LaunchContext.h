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

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H


#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#endif


#include <dll.h>
#include <op_boilerplate.h>
#include <memory/Workspace.h>



namespace nd4j  {
namespace graph {

class ND4J_EXPORT LaunchContext {

	private:
#ifdef __CUDABLAS__
		void* _reductionPointer;
		void* _scalarPointer;
		int* _allocationPointer;		
		cudaStream_t *_cudaStream = nullptr;
		cudaStream_t *_cudaSpecialStream = nullptr;
		void *_cublasHandle = nullptr;
#endif		
	nd4j::memory::Workspace* _workspace = nullptr;
    int _deviceID = 0;
	public:
#ifdef __CUDABLAS__
		
		LaunchContext(cudaStream_t* cudaStream, void* reductionPointer = nullptr,  void* scalarPointer = nullptr,  int* allocationPointer = nullptr);
		LaunchContext(cudaStream_t* cudaStream, cudaStream_t& specialCudaStream, void* reductionPointer = nullptr,  void* scalarPointer = nullptr,  int* allocationPointer = nullptr);

		FORCEINLINE void* getReductionPointer () const {return _reductionPointer;};
		
		FORCEINLINE void* getScalarPointer() const {return _scalarPointer;};

		FORCEINLINE int* getAllocationPointer() const {return _allocationPointer;};

		FORCEINLINE void* getCublasHandle() const {return _cublasHandle;};
		FORCEINLINE cudaStream_t* getCudaStream() const {return _cudaStream;};
		FORCEINLINE cudaStream_t* getCudaSpecialStream() const {return _cudaSpecialStream;};

		FORCEINLINE void setReductionPointer (void* reductionPointer) {_reductionPointer = reductionPointer;};
		
		FORCEINLINE void setScalarPointer(void* scalarPointer) {_scalarPointer = scalarPointer;};

		FORCEINLINE void setAllocationPointer(int* allocationPointer) {_allocationPointer = allocationPointer;};

		FORCEINLINE void setCudaStream(cudaStream_t* cudaStream)  {_cudaStream = cudaStream;};
		FORCEINLINE void setCudaSpecialStream(cudaStream_t* cudaStream)  {_cudaSpecialStream = cudaStream;};

#endif

    	LaunchContext();
    	~LaunchContext() = default;
    	nd4j::memory::Workspace* getWorkspace() const { return _workspace; }
    	void setWorkspace(nd4j::memory::Workspace* theWorkspace) {
    	    _workspace = theWorkspace;
    	}

    	int getDeviceID() const {return _deviceID;}
    	void setDeviceID(int deviceID) { _deviceID = deviceID; }

public:
	static LaunchContext* defaultContext();
private:
	static LaunchContext* sDefaultContext;
};

}
}


#endif //LIBND4J_CUDACONTEXT_H
