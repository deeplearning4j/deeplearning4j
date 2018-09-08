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
// Created by raver on 9/2/2018.
//

#ifndef LIBND4J_NUMBER_H
#define LIBND4J_NUMBER_H

#include <pointercast.h>
#include <types/u64.h>

#include <array/DataType.h>

namespace nd4j {
    class Number {
    protected:
        nd4j::DataType _type;

        int8_t* _buffer = nullptr;
        Nd4jLong _offset = 0;
        u64 _storage;

        Number() = default;
    public:
        Number(int8_t* _buffer, nd4j::DataType type);

        template <class T>
        Number(const T value);

        ~Number() = default;

        template <typename T>
        T asT();
    };
}


#endif //DEV_TESTS_NUMBER_H
