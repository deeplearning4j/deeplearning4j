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

#include <helpers/logger.h>
#include <array/DataTypeUtils.h>
#include <types/float16.h>

namespace nd4j {
    DataType DataTypeUtils::fromInt(int val) {
        return (DataType) val;
    }

    DataType DataTypeUtils::fromFlatDataType(nd4j::graph::DataType dtype) {
        return (DataType) dtype;
    }

    int DataTypeUtils::asInt(DataType type) {
        return (int) type;
    }

    size_t DataTypeUtils::sizeOfElement(DataType type) {
        switch (type) {
            case UINT8:
            case INT8:
            case FLOAT8:
            case QINT8:
            case BOOL: return (size_t) 1;
            
            case HALF:
            case INT16:
            case QINT16:
            case UINT16: return (size_t) 2;

            case INT32:
            case UINT32:
            case HALF2:
            case FLOAT32: return (size_t) 4;

            case UINT64:
            case INT64:
            case DOUBLE: return (size_t) 8;

            default: {
                nd4j_printf("Unknown DataType used: [%i]\n", asInt(type));
                throw std::runtime_error("Unknown DataType requested");
            }
        }
    }

    template <>
    DataType DataTypeUtils::fromT<bool>() {
        return BOOL;
    }

    template <>
    DataType DataTypeUtils::fromT<float>() {
        return FLOAT32;
    }

    template <>
    DataType DataTypeUtils::fromT<float16>() {
        return HALF;
    }

    template <>
    DataType DataTypeUtils::fromT<double>() {
        return DOUBLE;
    }

    template <>
    DataType DataTypeUtils::fromT<int8_t>() {
        return INT8;
    }

    template <>
    DataType DataTypeUtils::fromT<int16_t>() {
        return INT16;
    }

    template <>
    DataType DataTypeUtils::fromT<int>() {
        return INT32;
    }

    template <>
    DataType DataTypeUtils::fromT<Nd4jLong>() {
        return INT64;
    }

    template <>
    DataType DataTypeUtils::fromT<Nd4jULong>() {
        return UINT64;
    }

    template <>
    DataType DataTypeUtils::fromT<uint32_t >() {
        return UINT32;
    }

    template <>
    DataType DataTypeUtils::fromT<uint16_t >() {
        return UINT16;
    }

    template <>
    DataType DataTypeUtils::fromT<uint8_t >() {
        return UINT8;
    }
}