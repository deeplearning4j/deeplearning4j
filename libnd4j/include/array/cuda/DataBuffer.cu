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

namespace nd4j {

////////////////////////////////////////////////////////////////////////
// default constructor
DataBuffer::DataBuffer() {

    _primaryBuffer = nullptr;
    _specialBuffer = nullptr;
    _lenInBytes = 0;
    _dataType = INT8;
    _workspace = nullptr;

    _counter.store(0L);
    _writePrimary.store(0L);
    _writeSpecial.store(0L);
    _readPrimary.store(0L);
    _readSpecial.store(0L);
}


////////////////////////////////////////////////////////////////////////
// copy constructor
DataBuffer::DataBuffer(const DataBuffer &other) {

    _lenInBytes    = other._lenInBytes;
    _dataType      = other._dataType;
    _workspace     = other._workspace;

    _primaryBuffer = nullptr;
    _specialBuffer = nullptr;

    if(other._primaryBuffer != nullptr) {
        allocatePrimary();
        memcpy(_primaryBuffer, other._primaryBuffer, _lenInBytes);
    }

    if(other._specialBuffer != nullptr) {
        allocateSpecial();
        cudaMemcpy(_specialBuffer, other._specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice);
    }

    _counter.store(other._counter);
    _writePrimary.store(other._readSpecial);
    _writeSpecial.store(other._readPrimary);
    _readPrimary.store(other._writeSpecial);
    _readSpecial.store(other._writePrimary);
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(Nd4jPointer primary, Nd4jPointer special, const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace) {

    _primaryBuffer = primary;
    _specialBuffer = special;
    _lenInBytes    = lenInBytes;
    _dataType      = dataType;
    _workspace     = workspace;

    _counter.store(0L);
    _writePrimary.store(0L);
    _writeSpecial.store(0L);
    _readPrimary.store(0L);
    _readSpecial.store(0L);
}


////////////////////////////////////////////////////////////////////////
// move constructor
DataBuffer::DataBuffer(DataBuffer&& other) {

    _primaryBuffer = other._primaryBuffer;
    _specialBuffer = other._specialBuffer;
    _lenInBytes    = other._lenInBytes;
    _dataType      = other._dataType;
    _workspace     = other._workspace;

    _counter.store(other._counter);
    _writePrimary.store(other._readSpecial);
    _writeSpecial.store(other._readPrimary);
    _readPrimary.store(other._writeSpecial);
    _readSpecial.store(other._writePrimary);

    other._primaryBuffer = other._specialBuffer = nullptr;
    other._lenInBytes = 0;
}

////////////////////////////////////////////////////////////////////////
// assignment operator
DataBuffer& DataBuffer::operator=(const DataBuffer& other) {

    if (this == &other)
        return *this;

    deleteBuffers();

    _lenInBytes    = other._lenInBytes;
    _dataType      = other._dataType;
    _workspace     = other._workspace;

    if(other._primaryBuffer != nullptr) {
        allocatePrimary();
        memcpy(_primaryBuffer, other._primaryBuffer, _lenInBytes);
    }

    if(other._specialBuffer != nullptr) {
        allocateSpecial();
        cudaMemcpy(_specialBuffer, other._specialBuffer, _lenInBytes, cudaMemcpyDeviceToDevice);
    }

    _counter.store(other._counter);
    _writePrimary.store(other._readSpecial);
    _writeSpecial.store(other._readPrimary);
    _readPrimary.store(other._writeSpecial);
    _readSpecial.store(other._writePrimary);
}


////////////////////////////////////////////////////////////////////////
// move assignment operator
DataBuffer& DataBuffer::operator=(DataBuffer&& other) noexcept {

    if (this == &other)
        return *this;

    deleteBuffers();

    _primaryBuffer = other._primaryBuffer;
    _specialBuffer = other._specialBuffer;
    _lenInBytes    = other._lenInBytes;
    _dataType      = other._dataType;
    _workspace     = other._workspace;

    _counter.store(other._counter);
    _writePrimary.store(other._readSpecial);
    _writeSpecial.store(other._readPrimary);
    _readPrimary.store(other._writeSpecial);
    _readSpecial.store(other._writePrimary);

    other._primaryBuffer = other._specialBuffer = nullptr;
    other._lenInBytes = 0;
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {

    if (_specialBuffer == nullptr && getLenInBytes() > 0)
        ALLOCATE_SPECIAL(_specialBuffer, _workspace, getLenInBytes(), int8_t);
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context) {

    allocatePrimary();

    auto res = cudaStreamSynchronize(*context->getCudaStream());
    if (res != 0)
        throw cuda_exception::build("DataBuffer::syncToPrimary failed to to some previous kernel failre", res);

    cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);

    readPrimary();
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial() {

    allocateSpecial();

    cudaMemcpy(_specialBuffer, _primaryBuffer, getLenInBytes(), cudaMemcpyHostToDevice);

    readSpecial();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteBuffers() {

    if(getLenInBytes() != 0) {
        if(_primaryBuffer != nullptr) {
            auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
            RELEASE(p, _workspace);
            _primaryBuffer = nullptr;
        }
        if(_primaryBuffer != nullptr) {
            auto p = reinterpret_cast<int8_t*>(_specialBuffer);
            RELEASE_SPECIAL(p, _workspace);
            _specialBuffer = nullptr;
        }
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const    { _writePrimary = ++_counter; }
void DataBuffer::writeSpecial() const    { _writeSpecial = ++_counter; }
void DataBuffer::readPrimary()  const    { _readPrimary  = ++_counter; }
void DataBuffer::readSpecial()  const    { _readSpecial  = ++_counter; }
bool DataBuffer::isPrimaryActual() const { return (_writePrimary > _writeSpecial || _readPrimary > _writeSpecial); }
bool DataBuffer::isSpecialActual() const { return (_writeSpecial > _writePrimary || _readSpecial > _writePrimary); }

}
