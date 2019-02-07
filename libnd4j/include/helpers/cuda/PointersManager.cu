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

#include <PointersManager.h>
#include <exceptions/cuda_exception.h>
#include <logger.h>
#include <memory/Workspace.h>

namespace nd4j {

//////////////////////////////////////////////////////////////////////////
PointersManager::PointersManager(nd4j::graph::LaunchContext *context)  {
        _context = context;
}

//////////////////////////////////////////////////////////////////////////
void* PointersManager::replicatePointer(const void* src, const size_t numberOfBytes, const std::string& message) {

	void* dst = nullptr;
	if (_context->getWorkspace() == nullptr) {
        cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void **>(&dst), numberOfBytes);
        if (cudaResult != 0)
            throw cuda_exception::build(message + ": cannot allocate of global memory on device!", cudaResult);
    } else {
	    dst = _context->getWorkspace()->allocateBytes(nd4j::memory::MemoryType::DEVICE, numberOfBytes);
	}

    if (_context != nullptr)
        cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, *_context->getCudaStream());
    else
        cudaMemcpy(dst, src, numberOfBytes, cudaMemcpyHostToDevice);

    _pOnGlobMem.emplace_back(dst);
    
    return dst;
}

//////////////////////////////////////////////////////////////////////////
void PointersManager::synchronize(const std::string& message) const {
    if (_context != nullptr) {
        cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build(message + ": cuda stream synchronization failed !", cudaResult);
    } else {
        nd4j_printf("<%s> syncStream isn't possible: no stream set!", message.c_str());
    }
}

//////////////////////////////////////////////////////////////////////////
PointersManager::~PointersManager() {
    
    for (auto& p :_pOnGlobMem)
        cudaFree(p);
}


}
