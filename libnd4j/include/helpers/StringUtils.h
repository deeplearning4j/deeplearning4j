/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
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
// Created by raver119 on 20/04/18.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//

#ifndef LIBND4J_STRINGUTILS_H
#define LIBND4J_STRINGUTILS_H

#include <system/pointercast.h>
#include <system/op_boilerplate.h>
#include <string>
#include <sstream>
#include <vector>
#include <array/NDArray.h>
#include <helpers/unicode.h>

namespace sd {
    class ND4J_EXPORT StringUtils {
    public:
        template <typename T>
        static FORCEINLINE std::string valueToString(T value) {
            std::ostringstream os;

            os << value ;

            //convert the string stream into a string and return
            return os.str();
        }

        /**
         * This method just concatenates error message with a given graphId
         * @param message
         * @param graphId
         * @return
         */
        static FORCEINLINE std::string buildGraphErrorMessage(const char *message, Nd4jLong graphId) {
            std::string result(message);
            result += " [";
            result += valueToString<Nd4jLong>(graphId);
            result += "]";

            return result;
        }

        /**
         * This method returns number of needle matches within haystack
         * PLEASE NOTE: this method operates on 8-bit arrays interpreted as uint8
         *
         * @param haystack
         * @param haystackLength
         * @param needle
         * @param needleLength
         * @return
         */
        static uint64_t countSubarrays(const void *haystack, uint64_t haystackLength, const void *needle, uint64_t needleLength);

        /**
         * This method returns number of bytes used for string NDArrays content
         * PLEASE NOTE: this doesn't include header
         *
         * @param array
         * @return
         */
        static uint64_t byteLength(const NDArray &array);

        /**
         * This method splits a string into substring by delimiter
         *
         * @param haystack
         * @param delimiter
         * @return
         */
        static std::vector<std::string> split(const std::string &haystack, const std::string &delimiter);


        /**
         * This method convert u8 string to u16
         * @param const reference to input string
         * @param reference to output u16string
         * @return boolean status
         */
        static bool u8StringToU16String(const std::string& u8, std::u16string& u16);

        /**
         * This method convert u8 string to u32
         * @param const reference to input string
         * @param reference to output u32string
         * @return boolean status
         */
        static bool u8StringToU32String(const std::string& u8, std::u32string& u32);

        /**
         * This method convert u16 string to u32
         * @param const reference to input u16string
         * @param reference to output u32string
         * @return boolean status
         */
        static bool u16StringToU32String(const std::u16string& u16, std::u32string& u32);

        /**
         * This method convert u16 string to u8 string
         * @param const reference to input u16string
         * @param reference to output string
         * @return boolean status
         */
        static bool u16StringToU8String(const std::u16string& u16, std::string& u8);

        /**
         * This method convert u32 string to u16 string
         * @param const reference to input u32string
         * @param reference to output u16string
         * @return boolean status
         */
        static bool u32StringToU16String(const std::u32string& u32, std::u16string& u16);

        /**
        * This method convert u32 string to u8 string
        * @param const reference to input u32string
        * @param reference to output string
        * @return boolean status
        */
        static bool u32StringToU8String(const std::u32string& u32, std::string& u8);
    };
}


#endif //LIBND4J_STRINGUTILS_H
