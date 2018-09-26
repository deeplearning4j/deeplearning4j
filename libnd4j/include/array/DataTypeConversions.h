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
// Created by raver119 on 21.11.17.
//

#ifndef LIBND4J_DATATYPECONVERSIONS_H
#define LIBND4J_DATATYPECONVERSIONS_H

#include <pointercast.h>
#include <helpers/logger.h>
#include <op_boilerplate.h>
#include <array/DataType.h>
#include <types/float16.h>
#include <helpers/BitwiseUtils.h>
#include <loops/type_conversions.h>

namespace nd4j {
    template <typename T>
    class DataTypeConversions {
    public:
        static FORCEINLINE void convertType(void* vbuffer, void* src, DataType dataType, ByteOrder order, Nd4jLong length) {
            auto buffer = reinterpret_cast<T *>(vbuffer);
            bool isBe = BitwiseUtils::isBE();
            bool canKeep = (isBe && order == ByteOrder::BE) || (!isBe && order == ByteOrder::LE);

            switch (dataType) {
                case FLOAT: {
                        if (std::is_same<T, float>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = new float[length];
                            memcpy(tmp, src, length * sizeof(float));


#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                TypeCast::convertGeneric<float, T>(nullptr, tmp, length, buffer);
#else
#pragma omp parallel for simd schedule(guided)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif

                            delete[] tmp;
                        }
                    }
                    break;
                case DOUBLE: {
                        if (std::is_same<T, double>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = new double[length];
                            memcpy(tmp, src, length * sizeof(double));

#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                TypeCast::convertGeneric<double, T>(nullptr, tmp, length, buffer);


#else
#pragma omp parallel for schedule(static)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif
                            delete[] tmp;
                        }
                    }
                    break;
                case HALF: {

                        if (std::is_same<T, float16>::value && canKeep) {
                            memcpy(buffer, src, length * sizeof(T));
                        } else {
                            auto tmp = new float16[length];
                            memcpy(tmp, src, length * sizeof(float16));

#if __GNUC__ <= 4
                            if (!canKeep)
                                for (Nd4jLong e = 0; e < length; e++)
                                    buffer[e] = BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
                            else
                                TypeCast::convertGeneric<float16, T>(nullptr, tmp, length, buffer);
#else
#pragma omp parallel for schedule(static)
                            for (Nd4jLong e = 0; e < length; e++)
                                buffer[e] = canKeep ? static_cast<T>(tmp[e]) : BitwiseUtils::swap_bytes<T>(static_cast<T>(tmp[e]));
#endif
                            delete[] tmp;
                        }
                    }
                    break;
                default: {
                    nd4j_printf("Unsupported DataType requested: [%i]\n", static_cast<int>(dataType));
                    throw std::runtime_error("Unsupported DataType");
                }
            }
        }
    };
}



#endif //LIBND4J_DATATYPECONVERSIONS_H
