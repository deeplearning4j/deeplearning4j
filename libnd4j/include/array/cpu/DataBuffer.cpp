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
#include <DataTypeUtils.h>

namespace nd4j {

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {

}
////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers(const bool allocBoth) {    // always allocate primary buffer only (cpu case)

    allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFrom(const DataBuffer& other, size_t sizeToCopyinBytes, const Nd4jLong offsetThis, const Nd4jLong offsetOther) {

    if(sizeToCopyinBytes == 0)
        sizeToCopyinBytes = other.getLenInBytes();
    if(sizeToCopyinBytes == 0)
        return;

    if(other._primaryBuffer != nullptr)
        std::memcpy(static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType), static_cast<const int8_t*>(other._primaryBuffer) + offsetOther * DataTypeUtils::sizeOfElement(other._dataType), sizeToCopyinBytes);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBufferFromHost(const void* hostBuffer, size_t sizeToCopyinBytes, const Nd4jLong offsetThis, const Nd4jLong offsetHostBuffer) {

	if(sizeToCopyinBytes == 0)
        sizeToCopyinBytes = getLenInBytes();
    if(sizeToCopyinBytes == 0)
        return;

    if(hostBuffer != nullptr)
        std::memcpy(static_cast<int8_t*>(_primaryBuffer) + offsetThis * DataTypeUtils::sizeOfElement(_dataType), static_cast<const int8_t*>(hostBuffer) + offsetHostBuffer * DataTypeUtils::sizeOfElement(_dataType), sizeToCopyinBytes);
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::deleteSpecial() {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setSpecial(void* special, const bool isOwnerSpecail) {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setToZeroBuffers(const bool both) {

    memset(primary(), 0, getLenInBytes());
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::syncToSpecial(const bool forceSync) {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateSpecial() {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::migrate() {

}
///////////////////////////////////////////////////////////////////////
void DataBuffer::memcpy(const DataBuffer &dst, const DataBuffer &src) {
    if (src._lenInBytes < dst._lenInBytes)
        throw std::runtime_error("DataBuffer::memcpy: Source data buffer is smaller than destination");

    std::memcpy(dst._primaryBuffer, src._primaryBuffer, dst._lenInBytes);
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const    { }
void DataBuffer::writeSpecial() const    { }
void DataBuffer::readPrimary()  const    { }
void DataBuffer::readSpecial()  const    { }
bool DataBuffer::isPrimaryActual() const { return true;}
bool DataBuffer::isSpecialActual() const { return false;}


}
