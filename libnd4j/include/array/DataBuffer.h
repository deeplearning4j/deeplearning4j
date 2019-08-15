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

#ifndef DEV_TESTS_DATABUFFER_H
#define DEV_TESTS_DATABUFFER_H

#include <cstring>
#include <op_boilerplate.h>
#include <dll.h>
#include <pointercast.h>
#include <array/DataType.h>
#include <memory/Workspace.h>
#include <execution/LaunchContext.h>

namespace nd4j {

class ND4J_EXPORT DataBuffer {

    private:

        void* _primaryBuffer;
        void* _specialBuffer;
        size_t _lenInBytes;
        DataType _dataType;
        memory::Workspace* _workspace;
        bool _isOwnerPrimary;
        bool _isOwnerSpecial;

    #ifdef __CUDABLAS__
        mutable std::atomic<Nd4jLong> _counter;
        mutable std::atomic<Nd4jLong> _writePrimary;
        mutable std::atomic<Nd4jLong> _writeSpecial;
        mutable std::atomic<Nd4jLong> _readPrimary;
        mutable std::atomic<Nd4jLong> _readSpecial;
    #endif

                    void setCountersToZero();
                    void copyCounters(const DataBuffer& other);
                    void deleteSpecial();
        FORCEINLINE void deletePrimary();
        FORCEINLINE void deleteBuffers();
        FORCEINLINE void setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial = false);
                    void allocateBuffers(const bool allocBoth = false);
                    void setSpecial(void* special, const bool isOwnerSpecial);
                    void copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes = 0, const Nd4jLong offsetThis = 0, const Nd4jLong offsetHostBuffer = 0);


    public:

        FORCEINLINE DataBuffer(void* primary, void* special,
                               const size_t lenInBytes, const DataType dataType,
                               const bool isOwnerPrimary = false, const bool isOwnerSpecial = false,
                               memory::Workspace* workspace = nullptr);

        FORCEINLINE DataBuffer(void* primary,
                               const size_t lenInBytes, const DataType dataType,
                               const bool isOwnerPrimary = false,
                               memory::Workspace* workspace = nullptr);

        FORCEINLINE DataBuffer(const void* hostBuffer,      // copies data from hostBuffer to own memory buffer
                               const DataType dataType, const size_t lenInBytes,
                               memory::Workspace* workspace = nullptr);

        FORCEINLINE DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace = nullptr, const bool allocBoth = false);

        FORCEINLINE DataBuffer(const DataBuffer& other);
        FORCEINLINE DataBuffer(DataBuffer&& other);
        FORCEINLINE explicit DataBuffer();
        FORCEINLINE ~DataBuffer();

        FORCEINLINE DataBuffer& operator=(const DataBuffer& other);
        FORCEINLINE DataBuffer& operator=(DataBuffer&& other) noexcept;

        FORCEINLINE DataType getDataType();
        FORCEINLINE size_t getLenInBytes() const;

        FORCEINLINE void* primary();
        FORCEINLINE void* special();

        FORCEINLINE void allocatePrimary();
                    void allocateSpecial();

        void writePrimary() const;
        void writeSpecial() const;
        void readPrimary()  const;
        void readSpecial()  const;
        bool isPrimaryActual() const;
        bool isSpecialActual() const;

        template <typename T> FORCEINLINE T* primaryAsT();
        template <typename T> FORCEINLINE T* specialAsT();

        void syncToPrimary(const LaunchContext* context, const bool forceSync = false);
        void syncToSpecial(const bool forceSync = false);

        void setToZeroBuffers(const bool both = false);

        void copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes = 0, const Nd4jLong offsetThis = 0, const Nd4jLong offsetOther = 0);
};



///// IMLEMENTATION OF INLINE METHODS /////


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

    setCountersToZero();

    if(!lenInBytes == 0) {
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
template <typename T>
T* DataBuffer::primaryAsT() {
    return reinterpret_cast<T*>(_primaryBuffer);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
T* DataBuffer::specialAsT() {
    return reinterpret_cast<T*>(_specialBuffer);
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


}


#endif //DEV_TESTS_DATABUFFER_H
