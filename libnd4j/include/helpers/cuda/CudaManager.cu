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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.02.2019
//

#include <CudaManager.h>
#include <exceptions/cuda_exception.h>

namespace nd4j {

//////////////////////////////////////////////////////////////////////////
CudaManager::CudaManager(cudaStream_t stream): _stream(stream) { }

//////////////////////////////////////////////////////////////////////////
void* CudaManager::allocGlobMemAndCopy(const void* src, const size_t size, const std::string& message) {

	void* dst = nullptr;
    cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void **>(&dst), size);
    if(cudaResult != 0) 
        throw cuda_exception::build(message + ": cannot allocate global memory on device!", cudaResult);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, _stream);
    _pOnGlobMem.push_back(dst);
    
    return dst;
}

//////////////////////////////////////////////////////////////////////////
void CudaManager::syncStream(const std::string& message) const {
	cudaError_t cudaResult = cudaStreamSynchronize(_stream);
    if (cudaResult != 0) 
    	throw cuda_exception::build(message + ": cuda stream synchronization failed !", cudaResult);
}

//////////////////////////////////////////////////////////////////////////
CudaManager::~CudaManager() {
    
    for (auto& p :_pOnGlobMem)
        cudaFree(p);
}


}
