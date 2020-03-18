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

#include <system/pointercast.h>
#include <types/float8.h>
#include <types/float16.h>
#include <types/int8.h>
#include <types/int16.h>
#include <types/uint8.h>
#include <types/uint16.h>
#include <types/utf8string.h>
#include <types/bfloat16.h>
#include <system/type_boilerplate.h>


#define LIBND4J_STRINGTYPES \
        (sd::DataType::UTF8, std::string),\
        (sd::DataType::UTF16, std::u16string), \
        (sd::DataType::UTF32, std::u32string)

#define LIBND4J_TYPES \
        (sd::DataType::BFLOAT16, bfloat16),\
        (sd::DataType::HALF, float16), \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double), \
        (sd::DataType::BOOL, bool), \
        (sd::DataType::INT8, int8_t), \
        (sd::DataType::UINT8, uint8_t), \
        (sd::DataType::UINT16, uint16_t), \
        (sd::DataType::UINT32, uint32_t), \
        (sd::DataType::UINT64, uint64_t), \
        (sd::DataType::INT16, int16_t), \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong)

#define LIBND4J_TYPES_EXTENDED \
        (sd::DataType::HALF, float16), \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double), \
        (sd::DataType::BOOL, bool), \
        (sd::DataType::INT8, int8_t), \
        (sd::DataType::UINT8, uint8_t), \
        (sd::DataType::INT16, int16_t), \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong), \
        (sd::DataType::UINT16, uint16_t), \
        (sd::DataType::UINT64, Nd4jULong), \
        (sd::DataType::UINT32, uint32_t), \
        (sd::DataType::BFLOAT16, bfloat16)

#define BOOL_TYPES \
        (sd::DataType::BOOL, bool)

#define LONG_TYPES \
        (sd::DataType::INT64, Nd4jLong),\
        (sd::DataType::UINT64, uint64_t)

#define FLOAT_TYPES \
        (sd::DataType::BFLOAT16, bfloat16) ,\
        (sd::DataType::HALF, float16), \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double)

#define INDEXING_TYPES \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong)

#define FLOAT_NATIVE \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double)

#define FLOAT_TYPES_0 \
        (sd::DataType::HALF, float16)

#define FLOAT_TYPES_1 \
        (sd::DataType::FLOAT32, float)

#define FLOAT_TYPES_2 \
        (sd::DataType::DOUBLE, double)

#define FLOAT_TYPES_3 \
        (sd::DataType::BFLOAT16, bfloat16)

#define LIBND4J_TYPES_0 \
        (sd::DataType::HALF, float16)

#define LIBND4J_TYPES_1 \
        (sd::DataType::FLOAT32, float)

#define LIBND4J_TYPES_2 \
        (sd::DataType::DOUBLE, double)

#define LIBND4J_TYPES_3 \
        (sd::DataType::BOOL, bool)

#define LIBND4J_TYPES_4 \
        (sd::DataType::INT8, int8_t)

#define LIBND4J_TYPES_5 \
        (sd::DataType::UINT8, uint8_t)

#define LIBND4J_TYPES_6 \
        (sd::DataType::INT16, int16_t),\
        (sd::DataType::UINT16, uint16_t)

#define LIBND4J_TYPES_7 \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::UINT32, uint32_t)

#define LIBND4J_TYPES_8 \
        (sd::DataType::INT64, Nd4jLong),\
        (sd::DataType::UINT64, uint64_t)

#define LIBND4J_TYPES_9 \
        (sd::DataType::BFLOAT16, bfloat16)

#define INTEGER_TYPES \
        (sd::DataType::INT8, int8_t), \
        (sd::DataType::UINT8, uint8_t), \
        (sd::DataType::UINT16, uint16_t), \
        (sd::DataType::UINT32, uint32_t), \
        (sd::DataType::UINT64, uint64_t), \
        (sd::DataType::INT16, int16_t), \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong)

#define INTEGER_TYPES_0 \
        (sd::DataType::INT8, int8_t)

#define INTEGER_TYPES_1 \
        (sd::DataType::UINT8, uint8_t)

#define INTEGER_TYPES_2 \
        (sd::DataType::UINT16, uint16_t)

#define INTEGER_TYPES_3 \
        (sd::DataType::UINT32, uint32_t)

#define INTEGER_TYPES_4 \
        (sd::DataType::UINT64, uint64_t)

#define INTEGER_TYPES_5 \
        (sd::DataType::INT16, int16_t)

#define INTEGER_TYPES_6 \
        (sd::DataType::INT32, int32_t)

#define INTEGER_TYPES_7 \
        (sd::DataType::INT64, Nd4jLong)


#define NUMERIC_TYPES \
        (sd::DataType::HALF, float16), \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double), \
        (sd::DataType::INT8, int8_t), \
        (sd::DataType::UINT8, uint8_t), \
        (sd::DataType::UINT16, uint16_t), \
        (sd::DataType::UINT32, uint32_t), \
        (sd::DataType::UINT64, uint64_t), \
        (sd::DataType::INT16, int16_t), \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong), \
        (sd::DataType::BFLOAT16, bfloat16)

#define NUMERIC_TYPES_0 \
        (sd::DataType::HALF, float16)

#define NUMERIC_TYPES_1 \
        (sd::DataType::FLOAT32, float)

#define NUMERIC_TYPES_2 \
        (sd::DataType::DOUBLE, double)

#define NUMERIC_TYPES_3 \
        (sd::DataType::INT8, int8_t), \
        (sd::DataType::BFLOAT16, bfloat16)

#define NUMERIC_TYPES_4 \
        (sd::DataType::UINT8, uint8_t)

#define NUMERIC_TYPES_5 \
        (sd::DataType::UINT16, uint16_t)

#define NUMERIC_TYPES_6 \
        (sd::DataType::UINT32, uint32_t)

#define NUMERIC_TYPES_7 \
        (sd::DataType::UINT64, uint64_t)

#define NUMERIC_TYPES_8 \
        (sd::DataType::INT16, int16_t)

#define NUMERIC_TYPES_9 \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong)


#define GENERIC_NUMERIC_TYPES \
        (sd::DataType::HALF, float16), \
        (sd::DataType::FLOAT32, float), \
        (sd::DataType::DOUBLE, double), \
        (sd::DataType::INT32, int32_t), \
        (sd::DataType::INT64, Nd4jLong), \
        (sd::DataType::BFLOAT16, bfloat16)


#ifdef __ND4J_EXPERIMENTAL__
#define PAIRWISE_TYPES_0 \
		(double, double, double), \
		(double, uint8_t, double), \
		(double, uint8_t, uint8_t), \
		(double, float, double), \
		(double, float, float), \
		(double, bfloat16, double), \
		(double, bfloat16, bfloat16), \
		(double, Nd4jLong, double), \
		(double, Nd4jLong, Nd4jLong), \
		(double, int32_t, double), \
		(double, int32_t, int32_t) , \
		(bool, bool, bool), \
		(bool, int8_t, bool), \
		(int8_t, bool, bool), \
		(int8_t, int8_t, int8_t), \
		(int16_t, bool, int16_t), \
		(float16, int8_t, float16), \
		(bfloat16, bool, bool), \
		(double, int8_t, double)

#define PAIRWISE_TYPES_9 \
		(double, int16_t, double), \
		(double, int16_t, int16_t), \
		(double, float16, double), \
		(double, float16, float16), \
		(double, bool, double), \
		(double, bool, bool), \
		(int8_t, double, int8_t), \
		(int8_t, double, double), \
		(int8_t, uint8_t, int8_t), \
		(int8_t, uint8_t, uint8_t), \
		(int8_t, float, int8_t), \
		(double, int8_t, int8_t) , \
		(bool, int8_t, int8_t) ,\
		(float16, int32_t, float16), \
		(float16, int32_t, int32_t), \
		(float16, int16_t, float16), \
		(float16, int16_t, int16_t), \
		(float16, float16, float16), \
		(float16, bool, float16), \
		(float16, bool, bool), \
		(float16, int8_t, int8_t)

#define PAIRWISE_TYPES_1 \
		(uint8_t, double, uint8_t), \
		(uint8_t, double, double), \
		(uint8_t, uint8_t, uint8_t), \
		(uint8_t, float, uint8_t), \
		(uint8_t, float, float), \
		(uint8_t, bfloat16, uint8_t), \
		(uint8_t, bfloat16, bfloat16), \
		(uint8_t, Nd4jLong, uint8_t) , \
		(uint8_t, Nd4jLong, Nd4jLong), \
		(uint8_t, int32_t, uint8_t), \
		(uint8_t, int32_t, int32_t), \
		(uint8_t, int16_t, uint8_t), \
		(uint8_t, int16_t, int16_t), \
		(uint8_t, float16, uint8_t), \
		(uint8_t, float16, float16), \
		(uint8_t, bool, uint8_t), \
		(uint8_t, bool, bool), \
		(uint8_t, int8_t, uint8_t), \
		(uint8_t, int8_t, int8_t)

#define PAIRWISE_TYPES_2 \
		(float, double, float), \
		(float, double, double), \
		(float, uint8_t, float), \
		(float, uint8_t, uint8_t), \
		(float, float, float), \
		(float, bfloat16, float), \
		(float, bfloat16, bfloat16), \
		(float, Nd4jLong, float), \
		(float, Nd4jLong, Nd4jLong) , \
		(float, int32_t, float), \
		(int8_t, int32_t, int8_t), \
		(int8_t, int32_t, int32_t), \
		(float, int32_t, int32_t), \
		(float, int16_t, float), \
		(float, int16_t, int16_t), \
		(float, float16, float), \
		(float, float16, float16), \
		(float, bool, float)

#define PAIRWISE_TYPES_3 \
		(bfloat16, double, bfloat16), \
		(bfloat16, double, double), \
		(bfloat16, uint8_t, bfloat16), \
		(bfloat16, uint8_t, uint8_t), \
		(bfloat16, float, bfloat16), \
		(bfloat16, float, float), \
		(bfloat16, bfloat16, bfloat16), \
		(bfloat16, Nd4jLong, bfloat16), \
		(bfloat16, Nd4jLong, Nd4jLong), \
		(bfloat16, int32_t, bfloat16) , \
		(float, bool, bool), \
		(bfloat16, int32_t, int32_t), \
		(float, int8_t, float), \
		(int8_t, float, float), \
		(int8_t, bfloat16, int8_t), \
		(int8_t, bfloat16, bfloat16), \
		(int8_t, Nd4jLong, int8_t), \
		(int8_t, Nd4jLong, Nd4jLong), \
		(float, int8_t, int8_t)

#define PAIRWISE_TYPES_4 \
		(Nd4jLong, double, Nd4jLong), \
		(Nd4jLong, double, double), \
		(Nd4jLong, uint8_t, Nd4jLong), \
		(Nd4jLong, uint8_t, uint8_t), \
		(Nd4jLong, float, Nd4jLong), \
		(Nd4jLong, float, float), \
		(Nd4jLong, bfloat16, Nd4jLong), \
		(Nd4jLong, bfloat16, bfloat16), \
		(Nd4jLong, Nd4jLong, Nd4jLong), \
		(Nd4jLong, int32_t, Nd4jLong), \
		(bfloat16, int16_t, bfloat16), \
		(bfloat16, int16_t, int16_t), \
		(bfloat16, float16, bfloat16), \
		(bfloat16, float16, float16), \
		(bfloat16, bool, bfloat16), \
		(int8_t, int16_t, int8_t), \
		(int8_t, int16_t, int16_t), \
		(bfloat16, int8_t, bfloat16), \
		(bfloat16, int8_t, int8_t)

#define PAIRWISE_TYPES_5 \
		(int32_t, double, int32_t), \
		(int32_t, double, double), \
		(int32_t, uint8_t, int32_t), \
		(int32_t, uint8_t, uint8_t), \
		(int32_t, float, int32_t), \
		(int32_t, float, float), \
		(int32_t, bfloat16, int32_t), \
		(int32_t, bfloat16, bfloat16), \
		(int32_t, Nd4jLong, int32_t), \
		(Nd4jLong, int32_t, int32_t), \
		(Nd4jLong, int16_t, Nd4jLong), \
		(Nd4jLong, int16_t, int16_t), \
		(Nd4jLong, float16, Nd4jLong), \
		(Nd4jLong, float16, float16), \
		(Nd4jLong, bool, Nd4jLong), \
		(Nd4jLong, bool, bool), \
		(Nd4jLong, int8_t, Nd4jLong), \
		(Nd4jLong, int8_t, int8_t)


#define PAIRWISE_TYPES_6 \
		(int16_t, double, int16_t), \
		(int16_t, double, double), \
		(int16_t, uint8_t, int16_t), \
		(int16_t, uint8_t, uint8_t), \
		(int16_t, float, int16_t), \
		(int16_t, float, float), \
		(int16_t, bfloat16, int16_t), \
		(int16_t, bfloat16, bfloat16), \
		(int16_t, Nd4jLong, int16_t), \
		(float16, bfloat16, bfloat16), \
		(int16_t, Nd4jLong, Nd4jLong), \
		(int32_t, Nd4jLong, Nd4jLong), \
		(int32_t, int32_t, int32_t), \
		(int32_t, int16_t, int32_t), \
		(int32_t, int16_t, int16_t), \
		(int32_t, float16, int32_t), \
		(int32_t, float16, float16), \
		(int32_t, bool, int32_t), \
		(int32_t, bool, bool), \
		(int32_t, int8_t, int32_t), \
		(int32_t, int8_t, int8_t)


#define PAIRWISE_TYPES_7 \
		(float16, double, float16), \
		(float16, double, double), \
		(float16, uint8_t, float16), \
		(float16, uint8_t, uint8_t), \
		(float16, float, float16), \
		(float16, float, float), \
		(float16, bfloat16, float16), \
		(float16, Nd4jLong, float16), \
		(float16, Nd4jLong, Nd4jLong), \
		(int16_t, int32_t, int16_t), \
		(int16_t, int32_t, int32_t), \
		(int16_t, int16_t, int16_t), \
		(int16_t, float16, int16_t), \
		(int16_t, float16, float16), \
		(int8_t, float16, int8_t), \
		(int8_t, float16, float16), \
		(int8_t, bool, int8_t), \
		(int16_t, bool, bool), \
		(int16_t, int8_t, int16_t), \
		(int16_t, int8_t, int8_t)


#define PAIRWISE_TYPES_8 \
		(bool, double, bool), \
		(bool, double, double), \
		(bool, uint8_t, bool), \
		(bool, uint8_t, uint8_t), \
		(bool, float, bool), \
		(bool, float, float), \
		(bool, bfloat16, bool) ,\
		(bool, bfloat16, bfloat16), \
		(bool, Nd4jLong, bool), \
		(bool, Nd4jLong, Nd4jLong), \
		(bool, int32_t, bool), \
		(bool, int32_t, int32_t), \
		(bool, int16_t, bool), \
		(bool, int16_t, int16_t), \
		(bool, float16, bool), \
		(bool, float16, float16)


#else

#define PAIRWISE_TYPES_0 \
(float16, float16, float16) , \
(float16, bool, float16)

#define PAIRWISE_TYPES_1 \
(float, float, float) , \
(float, bool, float)

#define PAIRWISE_TYPES_2 \
(double, double, double) , \
(double, bool, double)

#define PAIRWISE_TYPES_3 \
(int8_t, int8_t, int8_t) , \
(int8_t, bool, int8_t)

#define PAIRWISE_TYPES_4 \
(int16_t, int16_t, int16_t) , \
(int16_t, bool, int16_t)

#define PAIRWISE_TYPES_5 \
(uint8_t, uint8_t, uint8_t) , \
(uint8_t, bool, uint8_t)

#define PAIRWISE_TYPES_6 \
(int, int, int) ,\
(int, bool, int)

#define PAIRWISE_TYPES_7 \
(bool, bool, bool)

#define PAIRWISE_TYPES_8 \
(Nd4jLong, Nd4jLong, Nd4jLong) ,\
(Nd4jLong, bool, Nd4jLong)

#define PAIRWISE_TYPES_9 \
(bfloat16, bfloat16, bfloat16) , \
(bfloat16, bool, bfloat16)

#define PAIRWISE_TYPES_10 \
(uint64_t, uint64_t, uint64_t) ,\
(uint64_t, bool, uint64_t)

#define PAIRWISE_TYPES_11 \
(uint32_t, uint32_t, uint32_t) ,\
(uint32_t, bool, uint32_t)

#define PAIRWISE_TYPES_12 \
(uint16_t, uint16_t, uint16_t) ,\
(uint16_t, bool, uint16_t)

#endif

#endif //LIBND4J_TYPES_H

