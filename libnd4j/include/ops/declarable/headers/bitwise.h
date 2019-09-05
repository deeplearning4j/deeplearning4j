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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_BITWISE_H
#define LIBND4J_HEADERS_BITWISE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation toggles individual bits of each element in array
         * 
         * PLEASE NOTE: This operation is possible only on integer data types
         * 
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_toggle_bits)
        DECLARE_OP(toggle_bits, -1, -1, true);
        #endif


        /**
         * This operation shift individual bits of each element in array to the left: <<
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_shift_bits)
        DECLARE_BROADCASTABLE_OP(shift_bits,  0, 0);
        #endif

        /**
         * This operation shift individual bits of each element in array to the right: >>
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_rshift_bits)
        DECLARE_BROADCASTABLE_OP(rshift_bits,  0, 0);
        #endif

        /**
         * This operation shift individual bits of each element in array, shifting to the left
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_cyclic_shift_bits)
        DECLARE_BROADCASTABLE_OP(cyclic_shift_bits,  0, 0);
        #endif

        /**
         * This operation shift individual bits of each element in array, shifting to the right
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_cyclic_rshift_bits)
        DECLARE_BROADCASTABLE_OP(cyclic_rshift_bits,  0, 0);
        #endif

        /**
         * This operation applies bitwise AND
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_bitwise_and)
        DECLARE_BROADCASTABLE_OP(bitwise_and,  0, 0);
        #endif

        /**
         * This operation applies bitwise OR
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_bitwise_or)
        DECLARE_BROADCASTABLE_OP(bitwise_or,  0, 0);
        #endif

        /**
         * This operation applies bitwise XOR
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_bitwise_xor)
        DECLARE_BROADCASTABLE_OP(bitwise_xor,  0, 0);
        #endif

        /**
         * This operation returns hamming distance based on bits
         *
         * PLEASE NOTE: This operation is applicable only to integer data types
         *
         * @tparam T
         */
        #if NOT_EXCLUDED(OP_bits_hamming_distance)
        DECLARE_CUSTOM_OP(bits_hamming_distance, 2, 1, true, 0, 0);
        #endif
    }
}

#endif