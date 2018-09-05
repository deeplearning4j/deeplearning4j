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
// Created by raver on 6/12/2018.
//

#ifndef LIBND4J_TYPES_H
#define LIBND4J_TYPES_H

#include <pointercast.h>
#include <types/float8.h>
#include <types/float16.h>
#include <types/int8.h>
#include <types/int16.h>
#include <types/uint8.h>
#include <types/uint16.h>

#include <type_boilerplate.h>

#define LIBND4J_TYPES \
        (nd4j::DataType::DataType_HALF, float16), \
        (nd4j::DataType::DataType_FLOAT, float), \
        (nd4j::DataType::DataType_DOUBLE, double), \
        (nd4j::DataType::DataType_BOOL, bool), \
        (nd4j::DataType::DataType_INT8, int8_t), \
        (nd4j::DataType::DataType_UINT8, uint8_t), \
        (nd4j::DataType::DataType_INT16, int16_t), \
        (nd4j::DataType::DataType_UINT16, uint16_t), \
        (nd4j::DataType::DataType_INT32, int32_t), \
        (nd4j::DataType::DataType_UINT32, uint32_t), \
        (nd4j::DataType::DataType_INT64, Nd4jLong), \
        (nd4j::DataType::DataType_UINT64, Nd4jULong)

#define FLOAT_TYPES \
        (nd4j::DataType::DataType_HALF, float16), \
        (nd4j::DataType::DataType_FLOAT, float), \
        (nd4j::DataType::DataType_DOUBLE, double)

#define DECIMAL_TYPES \
        (nd4j::DataType::DataType_INT8, int8_t), \
        (nd4j::DataType::DataType_UINT8, uint8_t), \
        (nd4j::DataType::DataType_INT16, int16_t), \
        (nd4j::DataType::DataType_UINT16, uint16_t), \
        (nd4j::DataType::DataType_INT32, int32_t), \
        (nd4j::DataType::DataType_UINT32, uint32_t), \
        (nd4j::DataType::DataType_UINT64, Nd4jLong), \
        (nd4j::DataType::DataType_HALF, Nd4jULong)



#endif //LIBND4J_TYPES_H
