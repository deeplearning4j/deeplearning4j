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
        _primaryBuffer  = nullptr;
        _isOwnerPrimary = false;
        _lenInBytes     = 0;
    }
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::setCountersToZero() {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyCounters(const DataBuffer& other) {

}

////////////////////////////////////////////////////////////////////////
void DataBuffer::allocateBuffers() {    // always allocate primary buffer only (cpu case)

    allocatePrimary();
}

////////////////////////////////////////////////////////////////////////
void DataBuffer::copyBuffers(const DataBuffer& other) {

    if(other._primaryBuffer != nullptr)
        memcpy(_primaryBuffer, other._primaryBuffer, other._lenInBytes);
}


////////////////////////////////////////////////////////////////////////
void DataBuffer::writePrimary() const    { }
void DataBuffer::writeSpecial() const    { }
void DataBuffer::readPrimary()  const    { }
void DataBuffer::readSpecial()  const    { }
bool DataBuffer::isPrimaryActual() const { return true;}
bool DataBuffer::isSpecialActual() const { return false;}


}
