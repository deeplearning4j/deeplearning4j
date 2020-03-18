/*******************************************************************************
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
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//

#ifndef LIBND4J_UNICODE_H
#define LIBND4J_UNICODE_H

#include <array/NDArray.h>

namespace sd {
namespace unicode {

    /**
        * This method calculate u16 offset based on utf8
        * @param const pointer to the utf8 string start point
        * @param size of the string
        * @return offset of utf16
   */
    Nd4jLong offsetUtf8StringInUtf16(const void* start, const void* end);

    /**
        * This method calculate u8 offset based on utf16
        * @param const pointer to the utf16 string start point
        * @param size of the string
        * @return offset of utf8
   */
    Nd4jLong offsetUtf16StringInUtf8(const void* start, const void* end);

    /**
       * This method calculate u32 offset based on utf16
       * @param const pointer to the utf16 string start point
       * @param size of the string
       * @return offset of utf32
    */
    Nd4jLong offsetUtf32StringInUtf16(const void* start, const void* end);

    /**
       * This method calculate u32 offset based on utf8
       * @param const pointer to the utf16 string start point
       * @param size of the string
       * @return offset of utf8
    */
    Nd4jLong offsetUtf32StringInUtf8(const void* start, const void* end);

    /*
    * This function check is valid charecter in u8 string
    */
    bool isStringValidU8(const void* start, const void* stop);

    /*
    * This function check is valid charecter in u16 string
    */
    bool isStringValidU16(const void* start, const void* stop);

    /*
    * This function check is valid u32 charecter in string
    */
    bool isStringValidU32(const void* start, const void* stop);

    /**
         * This method count offset for utf8 string in utf32
         * @param const pointer to the utf8 string start point
         * @param size of the string
         * @return offset
    */
    Nd4jLong offsetUtf8StringInUtf32(const void* input, uint32_t nInputSize);

    /**
     * This method count offset for utf8 string in utf32
     * @param const pointer to the utf8 string start point
     * @param const end pointer to the utf8 string
     * @return offset
    */
    Nd4jLong offsetUtf8StringInUtf32(const void* input, const void* stop);

    /**
         * This method count offset for utf32 based on utf16 string
         * @param const pointer to the utf16 string start point
         * @param size of the string
         * @return offset
    */
    Nd4jLong offsetUtf16StringInUtf32(const void* input, uint32_t nInputSize);

    /**
         * This method calculate offset of u16 based on utf8
         * @param const pointer to the utf8 string start point
         * @param size of the string
         * @return offset of utf16
    */
    Nd4jLong offsetUtf8StringInUtf16(const void* input, uint32_t nInputSize);

    /**
        * This method calculate offset of u8 based on utf16
        * @param const pointer to the utf16 string start point
        * @param size of the string
        * @return offset of utf8
   */
    Nd4jLong offsetUtf16StringInUtf8(const void* input, uint32_t nInputSize);

    /**
       * This method calculate offset of u32 based on utf8
       * @param const pointer to the utf16 string start point
       * @param size of the string
       * @return offset of utf32
    */
    Nd4jLong offsetUtf32StringInUtf8(const void* input, uint32_t nInputSize);

    /**
       * This method calculate offset of u32 based on utf16
       * @param const pointer to the utf16 string start point
       * @param size of the string
       * @return offset of utf32
    */
    Nd4jLong offsetUtf32StringInUtf16(const void* input, const uint32_t nInputSize);

    /**
         * This method convert utf8 string to utf16 string
         * @param const pointer to the utf8 string start point
         * @param reference to start point to utf16 
         * @param size of input utf8 string
         * @return status of convertion
    */
    bool utf8to16(const void* input, void* output, uint32_t nInputSize);

    /**
         * This method convert utf8 string to utf32 string
         * @param const pointer to the utf8 string start point
         * @param reference to start point to utf32
         * @param size of input utf8 string
         * @return status of convertion
    */
    bool utf8to32(const void* input, void* output, uint32_t nInputSize);

    /**
         * This method convert utf16 string to utf32 string
         * @param const pointer to the utf16 string start point
         * @param reference to start point to utf32
         * @param size of input utf16 string
         * @return status of convertion
    */
    bool utf16to32(const void* input, void* output, uint32_t nInputSize);

    /**
         * This method convert utf16 string to utf8 string
         * @param const pointer to the utf16 string start point
         * @param reference to start point to utf8
         * @param size of input utf16 string
         * @return status of convertion
    */
    bool utf16to8(const void* input, void* output, uint32_t nInputSize);

    /**
         * This method convert utf32 string to utf16 string
         * @param const pointer to the utf32 string start point
         * @param reference to start point to utf16
         * @param size of input utf32 string
         * @return status of convertion
    */
    bool utf32to16(const void* input, void* output, uint32_t nInputSize);

    /**
         * This method convert utf32 string to utf8 string
         * @param const pointer to the utf32 string start point
         * @param reference to start point to utf8
         * @param size of input utf32 string
         * @return status of convertion
    */
    bool utf32to8(const void* input, void* output, const Nd4jLong nInputSize);
}
}


#endif //LIBND4J_UNICODE_H
