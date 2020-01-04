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

#include <array/DataBuffer.h>
#include <helpers/logger.h>
#include <array/DataTypeUtils.h>
#include <execution/AffinityManager.h>

namespace nd4j {
    ///// IMLEMENTATION OF COMMON METHODS /////


////////////////////////////////////////////////////////////////////////
// default constructor
    DataBuffer::DataBuffer() {

        _primaryBuffer = nullptr;
        _specialBuffer = nullptr;
        _lenInBytes = 0;
        _dataType = INT8;
        _workspace = nullptr;
        _isOwnerPrimary = false;
        _isOwnerSpecial = false;
        _deviceId = nd4j::AffinityManager::currentDeviceId();

        setCountersToZero();
    }

////////////////////////////////////////////////////////////////////////
// copy constructor
    DataBuffer::DataBuffer(const DataBuffer &other) {

        throw std::runtime_error("DataBuffer copy constructor: we don't expect using of this constructor!");

        _lenInBytes    = other._lenInBytes;
        _dataType      = other._dataType;
        _workspace     = other._workspace;

        _primaryBuffer = nullptr;
        _specialBuffer = nullptr;

        _deviceId.store(other._deviceId.load());

        setCountersToZero();

        allocateBuffers();
        copyBufferFrom(other);
    }

////////////////////////////////////////////////////////////////////////
    DataBuffer::DataBuffer(void* primary, void* special,
                           const size_t lenInBytes, const DataType dataType,
                           const bool isOwnerPrimary, const bool isOwnerSpecial,
                           memory::Workspace* workspace) {

        if (primary == nullptr && special == nullptr)
            throw std::runtime_error("DataBuffer constructor: can't be initialized with both nullptr buffers !");

        _primaryBuffer  = primary;
        _specialBuffer  = special;
        _lenInBytes     = lenInBytes;
        _dataType       = dataType;
        _workspace      = workspace;
        _isOwnerPrimary = isOwnerPrimary;
        _isOwnerSpecial = isOwnerSpecial;
        _deviceId = nd4j::AffinityManager::currentDeviceId();

        setCountersToZero();

        if(primary != nullptr)
            readPrimary();
        if(special != nullptr)
            readSpecial();
    }

////////////////////////////////////////////////////////////////////////
    DataBuffer::DataBuffer(void* primary, const size_t lenInBytes, const DataType dataType, const bool isOwnerPrimary, memory::Workspace* workspace):
            DataBuffer(primary, nullptr, lenInBytes, dataType, isOwnerPrimary, false, workspace) {

        syncToSpecial(true);
    }

////////////////////////////////////////////////////////////////////////
// copies data from hostBuffer to own memory buffer
    DataBuffer::DataBuffer(const void* hostBuffer, const DataType dataType, const size_t lenInBytes, memory::Workspace* workspace) {

        if (hostBuffer == nullptr)
            throw std::runtime_error("DataBuffer constructor: can't be initialized with nullptr host buffer !");
        if (lenInBytes == 0)
            throw std::runtime_error("DataBuffer constructor: can't be initialized with zero length !");

        _primaryBuffer  = nullptr;
        _specialBuffer  = nullptr;
        _lenInBytes     = lenInBytes;
        _dataType       = dataType;
        _workspace      = workspace;

        _deviceId = nd4j::AffinityManager::currentDeviceId();

        setCountersToZero();

        allocateBuffers();

        copyBufferFromHost(hostBuffer, lenInBytes);
    }

////////////////////////////////////////////////////////////////////////
    DataBuffer::DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace, const bool allocBoth) {

        _dataType   = dataType;
        _workspace  = workspace;
        _lenInBytes = lenInBytes;

        _primaryBuffer = nullptr;
        _specialBuffer = nullptr;

        _deviceId = nd4j::AffinityManager::currentDeviceId();

        setCountersToZero();

        if(lenInBytes != 0) {
            allocateBuffers(allocBoth);
            writeSpecial();
        }
    }

////////////////////////////////////////////////////////////////////////
// move constructor
    DataBuffer::DataBuffer(DataBuffer&& other) {

        _primaryBuffer  = other._primaryBuffer;
        _specialBuffer  = other._specialBuffer;
        _lenInBytes     = other._lenInBytes;
        _dataType       = other._dataType;
        _workspace      = other._workspace;
        _isOwnerPrimary = other._isOwnerPrimary;
        _isOwnerSpecial = other._isOwnerSpecial;
        _deviceId.store(other._deviceId);

        copyCounters(other);

        other._primaryBuffer = other._specialBuffer = nullptr;
        other.setAllocFlags(false, false);
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

        allocateBuffers();
        copyBufferFrom(other);

        return *this;
    }

////////////////////////////////////////////////////////////////////////
// move assignment operator
    DataBuffer& DataBuffer::operator=(DataBuffer&& other) noexcept {

        if (this == &other)
            return *this;

        deleteBuffers();

        _primaryBuffer  = other._primaryBuffer;
        _specialBuffer  = other._specialBuffer;
        _lenInBytes     = other._lenInBytes;
        _dataType       = other._dataType;
        _workspace      = other._workspace;
        _isOwnerPrimary = other._isOwnerPrimary;
        _isOwnerSpecial = other._isOwnerSpecial;

        copyCounters(other);

        other._primaryBuffer = other._specialBuffer = nullptr;
        other.setAllocFlags(false, false);
        other._lenInBytes = 0;

        return *this;
    }

////////////////////////////////////////////////////////////////////////
    void* DataBuffer::primary() {
        return _primaryBuffer;
    }

////////////////////////////////////////////////////////////////////////
    void* DataBuffer::special() {
        return _specialBuffer;
    }

////////////////////////////////////////////////////////////////////////
    DataType DataBuffer::getDataType() {
        return _dataType;
    }

////////////////////////////////////////////////////////////////////////
    size_t DataBuffer::getLenInBytes() const {
        return _lenInBytes;
    }


////////////////////////////////////////////////////////////////////////
    void DataBuffer::allocatePrimary() {

        if (_primaryBuffer == nullptr && getLenInBytes() > 0) {
            ALLOCATE(_primaryBuffer, _workspace, getLenInBytes(), int8_t);
            _isOwnerPrimary = true;
        }
    }

////////////////////////////////////////////////////////////////////////
    void DataBuffer::setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial) {

        _isOwnerPrimary = isOwnerPrimary;
        _isOwnerSpecial = isOwnerSpecial;
    }

////////////////////////////////////////////////////////////////////////
    void DataBuffer::deletePrimary() {

        if(_isOwnerPrimary && _primaryBuffer != nullptr && getLenInBytes() != 0) {
            auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
            RELEASE(p, _workspace);
            _primaryBuffer = nullptr;
            _isOwnerPrimary = false;
        }
    }

////////////////////////////////////////////////////////////////////////
    void DataBuffer::deleteBuffers() {

        deletePrimary();
        deleteSpecial();
        _lenInBytes = 0;
    }

////////////////////////////////////////////////////////////////////////
    DataBuffer::~DataBuffer() {

        deleteBuffers();
    }

    void DataBuffer::setPrimaryBuffer(void *buffer, size_t length) {
        if (_primaryBuffer != nullptr && _isOwnerPrimary) {
            deletePrimary();
        }
        _primaryBuffer = buffer;
        _isOwnerPrimary = false;
        _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
    }

    void DataBuffer::setSpecialBuffer(void *buffer, size_t length) {
        this->setSpecial(buffer, false);
        _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
    }

    void DataBuffer::setDataType(DataType dataType) {
        _dataType = dataType;
    }

    int DataBuffer::deviceId() const {
        return _deviceId.load();
    }

    void DataBuffer::close() {
        this->deleteBuffers();
    }

    void DataBuffer::setDeviceId(int deviceId) {
        _deviceId = deviceId;
    }
}
