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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include "../DataBuffer.h"
#include <op_boilerplate.h>
#include <exceptions/cuda_exception.h>

namespace nd4j {

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {

    if (_specialBuffer == nullptr && getLenInBytes() > 0) {
        ALLOCATE_SPECIAL(_specialBuffer, _workspace, getLenInBytes(), int8_t);
        _isOwnerSpecial = true;
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {

    if(isPrimaryActual() && !forceSync)
        return;

    allocatePrimary();

    auto res = cudaStreamSynchronize(*context->getCudaStream());
    if (res != 0)
        throw cuda_exception::build("DataBuffer::syncToPrimary failed to to some previous kernel failre", res);

    cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);

    readPrimary();
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {

    if(isSpecialActual() && !forceSync)
        return;

    allocateSpecial();

    cudaMemcpy(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);

    readSpecial();
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {

    if(getLenInBytes() != 0 && _primaryBuffer != nullptr && _isOwnerSpecial) {
        auto p = reinterpret_cast<int8_t*>(_specialBuffer);
        RELEASE_SPECIAL(p, _workspace);
        _specialBuffer = nullptr;
        _isOwnerSpecial = false;
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteBuffers() {

    deleteSpecial();
    deletePrimary();
    _lenInBytes = 0;
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {

    _counter.store(0L);
    _writePrimary.store(0L);
    _writeSpecial.store(0L);
    _readPrimary.store(0L);
    _readSpecial.store(0L);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {

    _counter.store(other._counter);
    _writePrimary.store(other._readSpecial);
    _writeSpecial.store(other._readPrimary);
    _readPrimary.store(other._writeSpecial);
    _readSpecial.store(other._writePrimary);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBuffers(const DataBuffer& other) {     // always copies only to special buffer

    if(other._primaryBuffer == nullptr && other._specialBuffer == nullptr)
        return;

    if(other.isPrimaryActual()) {
        auto res = cudaMemcpy(_specialBuffer, other._primaryBuffer, other._lenInBytes, cudaMemcpyHostToDevice);
        if (res != 0)
            throw cuda_exception::build("DataBuffer::copyBuffers: cudaMemcpy_cudaMemcpyHostToDevice failed!", res);
        other.readPrimary();
    }
    else {
        auto res = cudaMemcpy(_specialBuffer, other._specialBuffer, other._lenInBytes, cudaMemcpyDeviceToDevice);
        if (res != 0)
            throw cuda_exception::build("DataBuffer::copyBuffers: cudaMemcpy_cudaMemcpyDeviceToDevice failed!", res);
        other.readSpecial();
    }

    writeSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecail) {

    deleteSpecial();
    _specialBuffer = special;
    _isOwnerSpecial = isOwnerSpecial;
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {    // always allocate special buffer only (cuda case)

    allocateSpecial();

    if(allocBoth)
        allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setToZeroBuffers(const bool both) {

    cudaMemset(special(), 0, getLenInBytes());
    writeSpecial();

    if(both) {
        memset(primary(), 0, getLenInBytes());
        readPrimary();
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const    { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const    { _writeSpecial = ++_counter; }
void DataBuffer::readPrimary()  const    { _readPrimary  = ++_counter; }
void DataBuffer::readSpecial()  const    { _readSpecial  = ++_counter; }
bool DataBuffer::isPrimaryActual() const { return (_writePrimary.load() > _writeSpecial.load() || _readPrimary.load() > _writeSpecial.load()); }
bool DataBuffer::isSpecialActual() const { return (_writeSpecial.load() > _writePrimary.load() || _readSpecial.load() > _writePrimary.load()); }

}
