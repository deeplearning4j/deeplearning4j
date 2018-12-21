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
LaunchContext::LaunchContext(cudaStream_t *cudaStream, cudaStream* specialCudaStream, void* reductionPointer, void* scalarPointer, int* allocationPointer)  {

	_cudaStream 	   = cudaStream;
	_cudaSpecialStream = specialCudaStream;
	_reductionPointer  = reductionPointer;
	_scalarPointer     = scalarPointer;
	_allocationPointer = allocationPointer;
	_workspace = nullptr;
}
#endif

////////////////////////////////////////////////////////////////////////
LaunchContext::LaunchContext() {
            // default constructor, just to make clang/ranlib happy
            _workspace = nullptr;
}


}
}