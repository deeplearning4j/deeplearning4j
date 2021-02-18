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
// CUDA workspaces implementation
//
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include "../Workspace.h"
#include <helpers/logger.h>
#include <math/templatemath.h>
#include <cstring>
#include <exceptions/cuda_exception.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sd {
    namespace memory {
        Workspace::Workspace(ExternalWorkspace *external) {
            if (external->sizeHost() > 0) {
                _ptrHost = (char *) external->pointerHost();
                _ptrDevice = (char *) external->pointerDevice();

                _initialSize = external->sizeDevice();
                _currentSize = external->sizeDevice();
                _initialSizeSecondary = external->sizeHost();
                _currentSizeSecondary = external->sizeHost();
                _offset = 0L;
                _offsetSecondary = 0L;
                this->_cycleAllocations = 0;
                this->_cycleAllocationsSecondary = 0;
                this->_spillsSize = 0;
                this->_spillsSizeSecondary = 0;

                _externalized = true;
            }
        }

        Workspace::Workspace(Nd4jLong primarySize, Nd4jLong secondarySize) {
            if (secondarySize > 0) {
                auto res = cudaHostAlloc(reinterpret_cast<void **>(&_ptrHost), secondarySize, cudaHostAllocDefault);
                if (res != 0)
                    throw cuda_exception::build("Can't allocate [HOST] memory", res);

                cudaMemset(this->_ptrHost, 0, secondarySize);
                this->_allocatedHost = true;
            } else
                this->_allocatedHost = false;

            if (primarySize > 0) {
                auto res = cudaMalloc(reinterpret_cast<void **>(&_ptrDevice), primarySize);
                if (res != 0)
                    throw cuda_exception::build("Can't allocate [DEVICE] memory", res);

                cudaMemset(this->_ptrDevice, 0, primarySize);
                this->_allocatedDevice = true;
            } else
                this->_allocatedDevice = false;

            this->_initialSize = primarySize;
            this->_initialSizeSecondary = secondarySize;
            this->_currentSize = primarySize;
            this->_currentSizeSecondary = secondarySize;
            this->_offset = 0;
            this->_offsetSecondary = 0;
            this->_cycleAllocations = 0;
            this->_spillsSize = 0;
            this->_spillsSizeSecondary = 0;
        }

        void Workspace::init(Nd4jLong primaryBytes, Nd4jLong secondaryBytes) {
            if (this->_currentSize < primaryBytes) {
                if (this->_allocatedDevice && !_externalized)
                    cudaFree((void *)this->_ptrDevice);

                auto res = cudaMalloc(reinterpret_cast<void **>(&_ptrDevice), secondaryBytes);
                if (res != 0)
                    throw cuda_exception::build("Can't allocate [DEVICE] memory", res);

                cudaMemset(this->_ptrDevice, 0, primaryBytes);
                this->_currentSize = primaryBytes;
                this->_allocatedDevice = true;
            }

            if (this->_currentSizeSecondary < secondaryBytes) {
                if (this->_allocatedHost && !_externalized)
                    cudaFreeHost((void *)this->_ptrHost);

                auto res = cudaHostAlloc(reinterpret_cast<void **>(&_ptrHost), secondaryBytes, cudaHostAllocDefault);
                if (res != 0)
                    throw cuda_exception::build("Can't allocate [HOST] memory", res);


                cudaMemset(this->_ptrHost, 0, secondaryBytes);
                this->_currentSizeSecondary = secondaryBytes;
                this->_allocatedHost = true;
            }
        }

        void Workspace::expandBy(Nd4jLong numBytes, Nd4jLong secondaryBytes) {
            this->init(_currentSize + numBytes, _currentSizeSecondary + secondaryBytes);
        }

        void Workspace::expandTo(Nd4jLong numBytes, Nd4jLong secondaryBytes) {
            this->init(numBytes, secondaryBytes);
        }

        void Workspace::freeSpills() {
            _spillsSize = 0;
            _spillsSizeSecondary = 0;

            for (auto v:_spills)
                cudaFree(v);

            for (auto v:_spillsSecondary)
                cudaFreeHost(v);

            _spills.clear();
            _spillsSecondary.clear();
        }

        Workspace::~Workspace() {
            if (this->_allocatedHost && !_externalized)
                cudaFreeHost((void *)this->_ptrHost);

            if (this->_allocatedDevice && !_externalized)
                cudaFree((void *)this->_ptrDevice);

            freeSpills();
        }

        Nd4jLong Workspace::getUsedSize() {
            return getCurrentOffset();
        }

        Nd4jLong Workspace::getCurrentSize() {
            return _currentSize;
        }

        Nd4jLong Workspace::getCurrentOffset() {
            return _offset.load();
        }


        void* Workspace::allocateBytes(Nd4jLong numBytes) {
            return allocateBytes(sd::memory::MemoryType::HOST, numBytes);
        }

        Nd4jLong Workspace::getAllocatedSize() {
            return getCurrentSize() + getSpilledSize();
        }

        void Workspace::scopeIn() {
            freeSpills();
            init(_cycleAllocations.load());
            _cycleAllocations = 0;
        }

        void Workspace::scopeOut() {
            _offset = 0;
        }

        Nd4jLong Workspace::getSpilledSize() {
            return _spillsSize.load();
        }

        void* Workspace::allocateBytes(sd::memory::MemoryType type, Nd4jLong numBytes) {
            switch (type) {
                case HOST: {
                        if (numBytes < 1)
                            throw allocation_exception::build("Number of [HOST] bytes for allocation should be positive", numBytes);


                        //numBytes += 32;
                        void* result = nullptr;
                        this->_cycleAllocationsSecondary += numBytes;
                        this->_mutexAllocation.lock();

                        if (_offsetSecondary.load() + numBytes > _currentSizeSecondary) {
                            nd4j_debug("Allocating %lld [HOST] bytes in spills\n", numBytes);
                            this->_mutexAllocation.unlock();

                            Nd4jPointer p;
                            auto res = cudaHostAlloc(reinterpret_cast<void **>(&p), numBytes, cudaHostAllocDefault);
                            if (res != 0)
                                throw cuda_exception::build("Can't allocate [HOST] memory", res);

                            _mutexSpills.lock();
                            _spillsSecondary.push_back(p);
                            _mutexSpills.unlock();

                            _spillsSizeSecondary += numBytes;

                            return p;
                        }

                        result = (void *)(_ptrHost + _offsetSecondary.load());
                        _offsetSecondary += numBytes;
                        //memset(result, 0, (int) numBytes);

                        nd4j_debug("Allocating %lld bytes from [HOST] workspace; Current PTR: %p; Current offset: %lld\n", numBytes, result, _offset.load());

                        this->_mutexAllocation.unlock();

                        return result;
                    }
                    break;
                case DEVICE: {
                        if (numBytes < 1)
                            throw allocation_exception::build("Number of [DEVICE] bytes for allocation should be positive", numBytes);


                        //numBytes += 32;
                        void* result = nullptr;
                        this->_cycleAllocations += numBytes;
                        this->_mutexAllocation.lock();

                        if (_offset.load() + numBytes > _currentSize) {
                            nd4j_debug("Allocating %lld [DEVICE] bytes in spills\n", numBytes);
                            this->_mutexAllocation.unlock();

                            Nd4jPointer p;
                            auto res = cudaMalloc(reinterpret_cast<void **>(&p), numBytes);
                            if (res != 0)
                                throw cuda_exception::build("Can't allocate [DEVICE] memory", res);

                            _mutexSpills.lock();
                            _spills.push_back(p);
                            _mutexSpills.unlock();

                            _spillsSize += numBytes;

                            return p;
                        }

                        result = (void *)(_ptrDevice + _offset.load());
                        _offset += numBytes;
                        //memset(result, 0, (int) numBytes);

                        nd4j_debug("Allocating %lld bytes from [DEVICE] workspace; Current PTR: %p; Current offset: %lld\n", numBytes, result, _offset.load());

                        this->_mutexAllocation.unlock();

                        return result;
                    }
                    break;
                default:
                    throw std::runtime_error("Unknown MemoryType was passed in");
            }
        }

        Workspace* Workspace::clone() {
            // for clone we take whatever is higher: current allocated size, or allocated size of current loop
            return new Workspace(sd::math::nd4j_max<Nd4jLong >(this->getCurrentSize(), this->_cycleAllocations.load()));
        }

        Nd4jLong Workspace::getAllocatedSecondarySize() {
            return getCurrentSecondarySize() + getSpilledSecondarySize();
        }

        Nd4jLong Workspace::getCurrentSecondarySize() {
            return _currentSizeSecondary;
        }

        Nd4jLong Workspace::getCurrentSecondaryOffset() {
            return _offsetSecondary.load();
        }

        Nd4jLong Workspace::getSpilledSecondarySize() {
            return _spillsSizeSecondary;
        }

        Nd4jLong Workspace::getUsedSecondarySize() {
            return getCurrentSecondaryOffset();
        }

    }
}
