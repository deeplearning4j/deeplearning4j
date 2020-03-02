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
// @author raver119@gmail.com
//

#include <helpers/PointersManager.h>
#include <exceptions/cuda_exception.h>
#include <helpers/StringUtils.h>
#include <helpers/logger.h>
#include <memory/Workspace.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////
PointersManager::PointersManager(const sd::LaunchContext* context, const std::string& funcName)  {
        _context  = const_cast<sd::LaunchContext*>(context);
        _funcName = funcName;
}

//////////////////////////////////////////////////////////////////////////
void* PointersManager::replicatePointer(const void* src, const size_t numberOfBytes) {

	void* dst = nullptr;
	if (_context->getWorkspace() == nullptr) {
        cudaError_t cudaResult = cudaMalloc(reinterpret_cast<void **>(&dst), numberOfBytes);
        if (cudaResult != 0)
            throw cuda_exception::build(_funcName + ": cannot allocate global memory on device!", cudaResult);
    } else {
	    dst = _context->getWorkspace()->allocateBytes(sd::memory::MemoryType::DEVICE, numberOfBytes);
	}

    if (_context != nullptr)
        cudaMemcpyAsync(dst, src, numberOfBytes, cudaMemcpyHostToDevice, *_context->getCudaStream());
    else
        cudaMemcpy(dst, src, numberOfBytes, cudaMemcpyHostToDevice);

    _pOnGlobMem.emplace_back(dst);

    return dst;
}

//////////////////////////////////////////////////////////////////////////
void PointersManager::synchronize() const {
    if (_context != nullptr) {
        cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
        if (cudaResult != 0)
            throw cuda_exception::build(_funcName + ": cuda stream synchronization failed !", cudaResult);
    } else {
        nd4j_printf("<%s> syncStream isn't possible: no stream set!", _funcName.c_str());
    }
}

//////////////////////////////////////////////////////////////////////////
PointersManager::~PointersManager() {

    for (auto& p :_pOnGlobMem)
        cudaFree(p);
}


////////////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void printDevContentOnDev_(const void* pDev, const Nd4jLong len, const int tid) {

    PointersManager::printDevContentOnDev<T>(pDev, len, tid);
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void PointersManager::printDevContentOnDevFromHost(const void* pDev, const Nd4jLong len, const int tid) {
    printDevContentOnDev_<T><<<512, 512, 1024, *sd::LaunchContext ::defaultContext()->getCudaStream()>>>(pDev, len, tid);
    auto res = cudaStreamSynchronize(*sd::LaunchContext ::defaultContext()->getCudaStream());
    if (res != 0)
        throw std::runtime_error("PointersManager::printDevContentOnDevFromHost: cudaStreamSynchronize failed!");
}
template void PointersManager::printDevContentOnDevFromHost<Nd4jLong>(const void* pDev, const Nd4jLong len, const int tid);
template void PointersManager::printDevContentOnDevFromHost<int>(const void* pDev, const Nd4jLong len, const int tid);
template void PointersManager::printDevContentOnDevFromHost<float>(const void* pDev, const Nd4jLong len, const int tid);
template void PointersManager::printDevContentOnDevFromHost<double>(const void* pDev, const Nd4jLong len, const int tid);

//BUILD_SINGLE_TEMPLATE(template void PointersManager::printDevContentOnDevFromHost, (void* pDev, Nd4jLong len, int tid), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
template<typename T>
void PointersManager::printDevContentOnHost(const void* pDev, const Nd4jLong len) const {
    printf("host print out\n");
    void* pHost = operator new(sizeof(T) * len);

    cudaMemcpyAsync(pHost, pDev, sizeof(T) * len, cudaMemcpyDeviceToHost, *_context->getCudaStream());
    cudaError_t cudaResult = cudaStreamSynchronize(*_context->getCudaStream());
    if(cudaResult != 0)
        throw std::runtime_error("PointersManager::printCudaHost: cudaStreamSynchronize failed!");

    for(Nd4jLong i = 0; i < len; ++i)
        printf("%f, ", (double)reinterpret_cast<T*>(pHost)[i]);
    printf("\n");

    operator delete(pHost);
}


template void PointersManager::printDevContentOnHost<Nd4jLong>(const void* pDev, const Nd4jLong len) const;
template void PointersManager::printDevContentOnHost<int>(const void* pDev, const Nd4jLong len) const;
template void PointersManager::printDevContentOnHost<float>(const void* pDev, const Nd4jLong len) const;
template void PointersManager::printDevContentOnHost<double>(const void* pDev, const Nd4jLong len) const;


}
