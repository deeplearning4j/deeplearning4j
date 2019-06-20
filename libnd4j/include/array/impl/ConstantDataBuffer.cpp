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

#include "../ConstantDataBuffer.h"

namespace nd4j {
    ConstantDataBuffer::ConstantDataBuffer(Nd4jPointer primary, Nd4jPointer special, Nd4jLong numEelements, Nd4jLong sizeOf) {
        _primaryBuffer = primary;
        _specialBuffer = special;
        _length = numEelements;
        _sizeOf = sizeOf;
    }

    Nd4jPointer ConstantDataBuffer::primary() const {
        return _primaryBuffer;
    }

    Nd4jPointer ConstantDataBuffer::special() const {
        return _specialBuffer;
    }

    Nd4jLong ConstantDataBuffer::sizeOf() const {
        return _sizeOf;
    }

    Nd4jLong ConstantDataBuffer::length() const {
        return _length;
    }

    ConstantDataBuffer::ConstantDataBuffer(const ConstantDataBuffer &other) {
        _primaryBuffer = other._primaryBuffer;
        _specialBuffer = other._specialBuffer;
        _length = other._length;
        _sizeOf = other._sizeOf;
    }

    template <typename T>
    T* ConstantDataBuffer::primaryAsT() {
        return reinterpret_cast<T*>(_primaryBuffer);
    }
    template ND4J_EXPORT float* ConstantDataBuffer::primaryAsT<float>();
    template ND4J_EXPORT double* ConstantDataBuffer::primaryAsT<double>();
    template ND4J_EXPORT int* ConstantDataBuffer::primaryAsT<int>();
    template ND4J_EXPORT Nd4jLong* ConstantDataBuffer::primaryAsT<Nd4jLong>();

    template <typename T>
    T* ConstantDataBuffer::specialAsT() {
        return reinterpret_cast<T*>(_specialBuffer);
    }
    template ND4J_EXPORT float* ConstantDataBuffer::specialAsT<float>();
    template ND4J_EXPORT double* ConstantDataBuffer::specialAsT<double>();
    template ND4J_EXPORT int* ConstantDataBuffer::specialAsT<int>();
    template ND4J_EXPORT Nd4jLong* ConstantDataBuffer::specialAsT<Nd4jLong>();

}
