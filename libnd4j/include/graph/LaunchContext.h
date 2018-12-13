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

#include <dll.h>

#ifdef __CUDABLAS__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#endif

namespace nd4j  {
namespace graph {

class ND4J_EXPORT LaunchContext {

	private:
#ifdef __CUDABLAS__
		void* _reductionPointer;
		void* _scalarPointer;
		int* _allocationPointer;		
		cudaStream_t *_cudaStream;
#endif		
	
	public:
#ifdef __CUDABLAS__
		
		LaunchContext(cudaStream_t* _cudaStream, void* reductionPointer = nullptr,  void* scalarPointer = nullptr,  int* allocationPointer = nullptr);
		
		inline void* getReductionPointer () const {return _reductionPointer;};			
		
		inline void* getScalarPointer() const {return _scalarPointer;};

		inline int* getAllocationPointer() const {return _allocationPointer;};

		inline cudaStream_t* getCudaStream() const {return _cudaStream;};
		
#endif

    	LaunchContext();
    	~LaunchContext() = default;
};

}
}


#endif //LIBND4J_CUDACONTEXT_H
