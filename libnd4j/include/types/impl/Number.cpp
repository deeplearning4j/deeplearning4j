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

#include <types/Number.h>

namespace nd4j {
    Number::Number(int8_t* buffer, nd4j::DataType type) {
        _buffer = buffer;
        _type = type;
    }

    template <>
    Number::Number(const float value) {
        _storage._float = value;
        _type = DataType_FLOAT;
    }

    template <>
    Number::Number(const double value) {
        _storage._double = value;
        _type = DataType_DOUBLE;
    }

    template <>
    Number::Number(const int value) {
        _storage._int = value;
        _type = DataType_INT32;
    }

    template <>
    Number::Number(const Nd4jLong value) {
        _storage._long = value;
        _type = DataType_INT64;
    }
}