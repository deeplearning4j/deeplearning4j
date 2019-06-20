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

#ifndef LIBND4J_CONSTANTDATABUFFER_H
#define LIBND4J_CONSTANTDATABUFFER_H

#include <dll.h>
#include <pointercast.h>


namespace nd4j {
    class ND4J_EXPORT ConstantDataBuffer {
    private:
        Nd4jPointer _primaryBuffer = nullptr;
        Nd4jPointer _specialBuffer = nullptr;
        Nd4jLong _length = 0;
        Nd4jLong _sizeOf = 0;

    public:
        ConstantDataBuffer(Nd4jPointer primary, Nd4jPointer special, Nd4jLong numEelements, Nd4jLong sizeOf);
        ConstantDataBuffer(const ConstantDataBuffer &other);
        ConstantDataBuffer() = default;
        ~ConstantDataBuffer() = default;

        Nd4jLong sizeOf() const;
        Nd4jLong length() const;

        Nd4jPointer primary() const;
        Nd4jPointer special() const;

        ConstantDataBuffer& operator=(const ConstantDataBuffer& other) = default;
        ConstantDataBuffer& operator=(ConstantDataBuffer&& other) noexcept = default;


        template <typename T>
        T* primaryAsT();

        template <typename T>
        T* specialAsT();
    };
}

#endif //DEV_TESTS_CONSTANTDATABUFFER_H
