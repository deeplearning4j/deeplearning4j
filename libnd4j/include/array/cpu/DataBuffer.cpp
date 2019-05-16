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
}

////////////////////////////////////////////////////////////////////////
DataBuffer::DataBuffer(Nd4jPointer primary, Nd4jPointer special, const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace) {

    _primaryBuffer = primary;
    _specialBuffer = special;
    _lenInBytes    = lenInBytes;
    _dataType      = dataType;
    _workspace     = workspace;
}

////////////////////////////////////////////////////////////////////////
// move constructor
DataBuffer::DataBuffer(DataBuffer&& other) {

    _primaryBuffer = other._primaryBuffer;
    _specialBuffer = other._specialBuffer;
    _lenInBytes    = other._lenInBytes;
    _dataType      = other._dataType;
    _workspace     = other._workspace;

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

    other._primaryBuffer = other._specialBuffer = nullptr;
    other._lenInBytes = 0;
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {

}


////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context) {

}


////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial() {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteBuffers() {

    if(getLenInBytes() != 0 && _primaryBuffer != nullptr) {
        auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
        RELEASE(p, _workspace);
        _primaryBuffer = nullptr;
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const    { }
void DataBuffer::writeSpecial() const    { }
void DataBuffer::readPrimary()  const    { }
void DataBuffer::readSpecial()  const    { }
bool DataBuffer::isPrimaryActual() const { }
bool DataBuffer::isSpecialActual() const { }


}
