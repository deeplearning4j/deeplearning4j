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
void DataBuffer::allocatePrimary() {
    
    if (_primaryBuffer == nullptr && getLenInBytes() > 0) 
        ALLOCATE(_primaryBuffer, _workspace, getLenInBytes(), int8_t);    
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
DataBuffer::~DataBuffer() {

    if(getLenInBytes() != 0) {
        if(_primaryBuffer != nullptr) {
            auto p = reinterpret_cast<int8_t*>(_primaryBuffer);
            RELEASE(p, _workspace);
        }
        if(_primaryBuffer != nullptr) {
            auto p = reinterpret_cast<int8_t*>(_specialBuffer);
            RELEASE_SPECIAL(p, _workspace);
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
