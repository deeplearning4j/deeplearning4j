/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#ifndef DEV_TESTS_DATABUFFER_H
#define DEV_TESTS_DATABUFFER_H

#include <dll.h>
#include <pointercast.h>

namespace nd4j {
    class ND4J_EXPORT DataBuffer {
    private:
        Nd4jPointer _primaryBuffer = nullptr;
        Nd4jPointer _specialBuffer = nullptr;
    public:
        DataBuffer(Nd4jPointer primary, Nd4jPointer special = nullptr);
        DataBuffer(const DataBuffer &other);
        explicit DataBuffer() = default;
        ~DataBuffer() = default;

        Nd4jPointer primary();
        Nd4jPointer special();

        DataBuffer& operator=(const DataBuffer& other) = default;
        DataBuffer& operator=(DataBuffer&& other) noexcept = default;


        template <typename T>
        T* primaryAsT();

        template <typename T>
        T* specialAsT();
    };
}


#endif //DEV_TESTS_DATABUFFER_H
