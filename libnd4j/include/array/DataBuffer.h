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

        void* _primaryBuffer = nullptr;
        void* _specialBuffer = nullptr;
        size_t _lenInBytes = 0;
        DataType _dataType;
        memory::Workspace* _workspace = nullptr;
        bool _isOwnerPrimary;
        bool _isOwnerSpecial;
        std::atomic<int> _deviceId;

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
        void deletePrimary();
        void deleteBuffers();
        void setAllocFlags(const bool isOwnerPrimary, const bool isOwnerSpecial = false);
        void allocateBuffers(const bool allocBoth = false);
        void setSpecial(void* special, const bool isOwnerSpecial);
        void copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes = 0, const Nd4jLong offsetThis = 0, const Nd4jLong offsetHostBuffer = 0);


    public:

        DataBuffer(void* primary, void* special,
                               const size_t lenInBytes, const DataType dataType,
                               const bool isOwnerPrimary = false, const bool isOwnerSpecial = false,
                               memory::Workspace* workspace = nullptr);

        DataBuffer(void* primary,
                               const size_t lenInBytes, const DataType dataType,
                               const bool isOwnerPrimary = false,
                               memory::Workspace* workspace = nullptr);

        DataBuffer(const void* hostBuffer,      // copies data from hostBuffer to own memory buffer
                               const DataType dataType, const size_t lenInBytes,
                               memory::Workspace* workspace = nullptr);

        DataBuffer(const size_t lenInBytes, const DataType dataType, memory::Workspace* workspace = nullptr, const bool allocBoth = false);

        DataBuffer(const DataBuffer& other);
        DataBuffer(DataBuffer&& other);
        explicit DataBuffer();
        ~DataBuffer();

        DataBuffer& operator=(const DataBuffer& other);
        DataBuffer& operator=(DataBuffer&& other) noexcept;

        DataType getDataType();
        void setDataType(DataType dataType);
        size_t getLenInBytes() const;

        void* primary();
        void* special();

        void allocatePrimary();
        void allocateSpecial();

        void writePrimary() const;
        void writeSpecial() const;
        void readPrimary()  const;
        void readSpecial()  const;
        bool isPrimaryActual() const;
        bool isSpecialActual() const;

        void expand(const uint64_t size);

        int deviceId() const;
        void setDeviceId(int deviceId);
        void migrate();

        template <typename T> FORCEINLINE T* primaryAsT();
        template <typename T> FORCEINLINE T* specialAsT();

        void syncToPrimary(const LaunchContext* context, const bool forceSync = false);
        void syncToSpecial(const bool forceSync = false);

        void setToZeroBuffers(const bool both = false);

        void copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes = 0, const Nd4jLong offsetThis = 0, const Nd4jLong offsetOther = 0);

        static void memcpy(const DataBuffer &dst, const DataBuffer &src);

        void setPrimaryBuffer(void *buffer, size_t length);
        void setSpecialBuffer(void *buffer, size_t length);

        /**
         * This method deletes buffers, if we're owners
         */
        void close();
};
///// IMLEMENTATION OF INLINE METHODS /////

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
