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

namespace nd4j {
    template <typename T>
    class DataTypeConversions {
    public:
        static FORCEINLINE void convertType(T* buffer, void* src, DataType dataType, ByteOrder order, Nd4jLong length) {
            bool isBe = BitwiseUtils::isBE();
            bool canKeep = (isBe && order == ByteOrder::BE) || (!isBe && order == ByteOrder::LE);

            switch (dataType) {
                case DataType_FLOAT: {
                        auto tmp = (float *) src;

                        //#pragma omp parallel for simd schedule(guided)
                        for (Nd4jLong e = 0; e < length; e++) {
                            buffer[e] = canKeep ? (T) tmp[e] : BitwiseUtils::swap_bytes<T>((T) tmp[e]);
                        }
                    }
                    break;
                case DataType_DOUBLE: {
                        auto tmp = (double *) src;

                        //#pragma omp parallel for simd schedule(guided)
                        for (Nd4jLong e = 0; e < length; e++)
                            buffer[e] = canKeep ? (T) tmp[e] : BitwiseUtils::swap_bytes<T>((T) tmp[e]);
                    }
                    break;
                case DataType_HALF: {
                        auto tmp = (float16 *) src;

                        //#pragma omp parallel for simd schedule(guided)
                        for (Nd4jLong e = 0; e < length; e++)
                            buffer[e] = canKeep ? (T) tmp[e] : BitwiseUtils::swap_bytes<T>((T) tmp[e]);
                    }
                    break;
                default: {
                    nd4j_printf("Unsupported DataType requested: [%i]\n", (int) dataType);
                    throw "Unsupported DataType";
                }
            }
        }
    };
}



#endif //LIBND4J_DATATYPECONVERSIONS_H
