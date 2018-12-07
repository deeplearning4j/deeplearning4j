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
// Created by raver119 on 10.11.2017.
//

#ifndef LIBND4J_BITWISEUTILS_H
#define LIBND4J_BITWISEUTILS_H

#include <vector>
#include <array/ByteOrder.h>
#include <op_boilerplate.h>
#include <pointercast.h>
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

        /**
         * This method swaps bytes: LE vs BE
         * @tparam T
         * @param v
         * @return
         */
        template <typename T>
        static FORCEINLINE T swap_bytes(T v) {
            static_assert (CHAR_BIT == 8, "CHAR_BIT != 8");

            union S {
                T v;
                unsigned char u8[sizeof(T)];
                S(T val) {
                    v = val;
                }
            };

            S source(v);
            S dest(v);

            for (size_t k = 0; k < sizeof(T); k++)
                dest.u8[k] = source.u8[sizeof(T) - k - 1];

            return dest.v;
        }

        /**
         * This method flips bits in given value
         *
         * @tparam T
         * @param v
         * @return
         */
        static int FORCEINLINE flip_bits(int v) {
            return ~v;
        }

        static int8_t FORCEINLINE flip_bits(int8_t v) {
            return ~v;
        }

        static int16_t FORCEINLINE flip_bits(int16_t v) {
            return ~v;
        }

        static uint8_t FORCEINLINE flip_bits(uint8_t v) {
            return ~v;
        }

        static Nd4jLong FORCEINLINE flip_bits(Nd4jLong v) {
            return ~v;
        }
    };
}


#endif //LIBND4J_BITWISEUTILS_H
