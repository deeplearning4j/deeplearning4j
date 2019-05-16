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

#include <dll.h>
#include <op_boilerplate.h>
#include <pointercast.h>
#include <memory/Workspace.h>
#include <DataType.h>
#include <LaunchContext.h>

namespace nd4j {

class ND4J_EXPORT DataBuffer {

    private:

        Nd4jPointer _primaryBuffer = nullptr;
        Nd4jPointer _specialBuffer = nullptr;
        size_t _lenInBytes = 0;
        DataType _dataType = INT8;
        memory::Workspace* _workspace = nullptr;
        mutable Nd4jLong _counter = 0L;
        mutable Nd4jLong _writePrimary = 0L;
        mutable Nd4jLong _writeSpecial = 0L;
        mutable Nd4jLong _readPrimary = 0L;
        mutable Nd4jLong _readSpecial = 0L;
    
    public:
        
        FORCEINLINE DataBuffer(Nd4jPointer primary, Nd4jPointer special, const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace = nullptr);
        DataBuffer(const DataBuffer &other) = default;
        explicit DataBuffer() = default;
        ~DataBuffer();

        FORCEINLINE DataType getDataType();
        FORCEINLINE size_t getLenInBytes();

        FORCEINLINE Nd4jPointer primary();
        FORCEINLINE Nd4jPointer special();

        DataBuffer& operator=(const DataBuffer& other) = default;
        DataBuffer& operator=(DataBuffer&& other) noexcept = default;

        void writePrimary() const;
        void writeSpecial() const;
        void readPrimary()  const;
        void readSpecial()  const;
        bool isPrimaryActual() const;
        bool isSpecialActual() const;

        template <typename T>
        FORCEINLINE T* primaryAsT();

        template <typename T>
        FORCEINLINE T* specialAsT();

        void syncToPrimary(const LaunchContext* context);
        void syncToSpecial();

        void allocatePrimary();
        void allocateSpecial();
};



///// IMLEMENTATION OF INLINE METHODS ///// 
DataBuffer::DataBuffer(Nd4jPointer primary, Nd4jPointer special, const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace) {
    _primaryBuffer = primary;
    _specialBuffer = special;
    _lenInBytes = lenInBytes;
    _dataType = dataType;
    _workspace = workspace;
}

////////////////////////////////////////////////////////////////////////
Nd4jPointer DataBuffer::primary() {
    return _primaryBuffer;
}

////////////////////////////////////////////////////////////////////////
Nd4jPointer DataBuffer::special() {
    return _specialBuffer;
}

////////////////////////////////////////////////////////////////////////
DataType DataBuffer::getDataType() {
    return _dataType;
}

////////////////////////////////////////////////////////////////////////
size_t DataBuffer::getLenInBytes() {
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



}


#endif //DEV_TESTS_DATABUFFER_H
