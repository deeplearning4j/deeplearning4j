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
        (nd4j::DataType::HALF, float16), \
        (nd4j::DataType::FLOAT32, float), \
        (nd4j::DataType::DOUBLE, double), \
        (nd4j::DataType::BOOL, bool), \
        (nd4j::DataType::INT8, int8_t), \
        (nd4j::DataType::UINT8, uint8_t), \
        (nd4j::DataType::INT16, int16_t), \
        (nd4j::DataType::INT32, int32_t), \
        (nd4j::DataType::INT64, Nd4jLong)

        //(nd4j::DataType::UINT16, uint16_t), \
        //(nd4j::DataType::UINT64, Nd4jULong)
        //(nd4j::DataType::UINT32, uint32_t), \

#define BOOL_TYPES \
        (nd4j::DataType::BOOL, bool)

#define FLOAT_TYPES \
        (nd4j::DataType::HALF, float16), \
        (nd4j::DataType::FLOAT32, float), \
        (nd4j::DataType::DOUBLE, double)

#define INTEGER_TYPES \
        (nd4j::DataType::INT8, int8_t), \
        (nd4j::DataType::UINT8, uint8_t), \
        (nd4j::DataType::INT16, int16_t), \
        (nd4j::DataType::INT32, int32_t), \
        (nd4j::DataType::INT64, Nd4jLong)


#define NUMERIC_TYPES \
        (nd4j::DataType::HALF, float16), \
        (nd4j::DataType::FLOAT32, float), \
        (nd4j::DataType::DOUBLE, double), \
        (nd4j::DataType::INT8, int8_t), \
        (nd4j::DataType::UINT8, uint8_t), \
        (nd4j::DataType::INT16, int16_t), \
        (nd4j::DataType::INT32, int32_t), \
        (nd4j::DataType::INT64, Nd4jLong)



///
#define LIBND4J_TYPES_0 \
        (nd4j::DataType::HALF, float16), \
        (nd4j::DataType::FLOAT32, float), \
        (nd4j::DataType::DOUBLE, double)

#define LIBND4J_TYPES_1 \
        (nd4j::DataType::BOOL, bool), \
        (nd4j::DataType::INT8, int8_t), \
        (nd4j::DataType::UINT8, uint8_t)

#define LIBND4J_TYPES_2 \
        (nd4j::DataType::INT16, int16_t), \
        (nd4j::DataType::INT32, int32_t), \
        (nd4j::DataType::INT64, Nd4jLong)



#endif //LIBND4J_TYPES_H
