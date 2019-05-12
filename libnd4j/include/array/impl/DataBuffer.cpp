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
//

#include "../DataBuffer.h"

namespace nd4j {
    DataBuffer::DataBuffer(Nd4jPointer primary, Nd4jPointer special) {
        _primaryBuffer = primary;
        _specialBuffer = special;
    }

    Nd4jPointer DataBuffer::primary() {
        return _primaryBuffer;
    }

    Nd4jPointer DataBuffer::special() {
        return _specialBuffer;
    }

    DataBuffer::DataBuffer(const DataBuffer &other) {
        _primaryBuffer = other._primaryBuffer;
        _specialBuffer = other._specialBuffer;
    }

    template <typename T>
    T* DataBuffer::primaryAsT() {
        return reinterpret_cast<T*>(_primaryBuffer);
    }
    template ND4J_EXPORT float* DataBuffer::primaryAsT<float>();
    template ND4J_EXPORT double* DataBuffer::primaryAsT<double>();
    template ND4J_EXPORT int* DataBuffer::primaryAsT<int>();
    template ND4J_EXPORT Nd4jLong* DataBuffer::primaryAsT<Nd4jLong>();

    template <typename T>
    T* DataBuffer::specialAsT() {
        return reinterpret_cast<T*>(_specialBuffer);
    }
    template ND4J_EXPORT float* DataBuffer::specialAsT<float>();
    template ND4J_EXPORT double* DataBuffer::specialAsT<double>();
    template ND4J_EXPORT int* DataBuffer::specialAsT<int>();
    template ND4J_EXPORT Nd4jLong* DataBuffer::specialAsT<Nd4jLong>();

}
