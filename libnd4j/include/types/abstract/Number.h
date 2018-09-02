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
#include <types/float16.h>

namespace nd4j {
    class Number {
    protected:
        Number() = default;
    public:
        ~Number() = default;

        template <typename T>
        T asT();


        virtual double asDoubleValue() = 0;
        virtual float asFloatValue() = 0;
        virtual float16 asHalfValue() = 0;
        virtual int asInt32Value() = 0;
        virtual int16_t asInt16Value() = 0;
        virtual int8_t asInt8Value() = 0;
        virtual bool asBoolValue() = 0;
    };
}


#endif //DEV_TESTS_NUMBER_H
