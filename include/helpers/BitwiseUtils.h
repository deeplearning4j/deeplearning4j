//
// Created by raver119 on 10.11.2017.
//

#ifndef LIBND4J_BITWISEUTILS_H
#define LIBND4J_BITWISEUTILS_H

#include <vector>
#include <array/ByteOrder.h>
#include <op_boilerplate.h>
#include <climits>

namespace nd4j {
    class BitwiseUtils {
    public:


        /**
         * This method returns first non-zero bit index
         * @param holder
         * @return
         */
        static int valueBit(int holder);

        /**
         *  This method returns vector representation of bits.
         * 
         *  PLEASE NOTE: Result is ALWAYS left-to-right 
         */
        static std::vector<int> valueBits(int holder);

        /**
         *  This method returns TRUE if it's called on Big-Endian system, and false otherwise
         */
        static bool isBE();

        /**
         * This method returns enum
         * @return
         */
        static nd4j::ByteOrder asByteOrder();

        template <typename T>
        static FORCEINLINE T swap_bytes(T v) {
            static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

            union {
                T v = (T) 0;
                unsigned char u8[sizeof(T)];
            } source, dest;

            source.v = v;

            for (size_t k = 0; k < sizeof(T); k++)
                dest.u8[k] = source.u8[sizeof(T) - k - 1];

            return dest.v;
        }
    };
}


#endif //LIBND4J_BITWISEUTILS_H
