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

#include <helpers/logger.h>
#include <helpers/BitwiseUtils.h>
#include <pointercast.h>
#include <types/float16.h>

namespace nd4j {

     bool BitwiseUtils::isBE() {
        short int word = 0x0001;
        char *byte = (char *) &word;
        return(byte[0] ? false : true);
     }

    int BitwiseUtils::valueBit(int holder) {
        if (holder == 0)
            return -1;

#ifdef REVERSE_BITS
        for (int e = 32; e >= 0; e--) {
#else
        for (int e = 0; e < 32; e++) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                return e;
        }

        return -1;
    }


    std::vector<int> BitwiseUtils::valueBits(int holder) {
        std::vector<int> bits;
        if (holder == 0) {
            for (int e = 0; e < 32; e++)
                bits.emplace_back(0);

            return bits;
        }


#ifdef REVERSE_BITS
        for (int e = 32; e >= 0; e--) {
#else
        for (int e = 0; e < 32; e++) {
#endif
            bool isOne = (holder & 1 << e) != 0;

            if (isOne)
                bits.emplace_back(1);
            else
                bits.emplace_back(0);
        }

        return bits;
    }

    nd4j::ByteOrder BitwiseUtils::asByteOrder() {
        return isBE() ? ByteOrder::BE : ByteOrder::LE;
    }
}
