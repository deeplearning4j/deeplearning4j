/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Created by raver119 on 20/04/18.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//

#include <helpers/StringUtils.h>
#include <helpers/BitwiseUtils.h>
#include <exceptions/datatype_exception.h>
#include <bitset>

namespace sd {
    static FORCEINLINE bool match(const uint8_t *haystack, const uint8_t *needle, uint64_t length) {
        for (int e = 0; e < length; e++)
            if (haystack[e] != needle[e])
                return false;

        return true;
    }

    template <typename T>
    std::string StringUtils::bitsToString(T value) {
      return std::bitset<sizeof(T) * 8>(value).to_string();
    }

template std::string StringUtils::bitsToString(int value);
template std::string StringUtils::bitsToString(uint32_t value);
template std::string StringUtils::bitsToString(Nd4jLong value);
template std::string StringUtils::bitsToString(uint64_t value);


    uint64_t StringUtils::countSubarrays(const void *vhaystack, uint64_t haystackLength, const void *vneedle, uint64_t needleLength) {
        auto haystack = reinterpret_cast<const uint8_t*>(vhaystack);
        auto needle = reinterpret_cast<const uint8_t*>(vneedle);

        uint64_t number = 0;

        for (uint64_t e = 0; e < haystackLength - needleLength; e++) {
            if (match(&haystack[e], needle, needleLength))
                number++;
        }

        return number;
    }


    uint64_t StringUtils::byteLength(const NDArray &array) {
        if (!array.isS())
            throw sd::datatype_exception::build("StringUtils::byteLength expects one of String types;", array.dataType());

        auto buffer = array.bufferAsT<Nd4jLong>();
        return buffer[array.lengthOf()];
    }

    std::vector<std::string> StringUtils::split(const std::string &haystack, const std::string &delimiter) {
        std::vector<std::string> output;

        std::string::size_type prev_pos = 0, pos = 0;

        // iterating through the haystack till the end
        while((pos = haystack.find(delimiter, pos)) != std::string::npos) {
            output.emplace_back(haystack.substr(prev_pos, pos-prev_pos));
            prev_pos = ++pos;
        }

        output.emplace_back(haystack.substr(prev_pos, pos - prev_pos)); // Last word

        return output;
    }
    
    bool StringUtils::u8StringToU16String(const std::string& u8, std::u16string& u16) {

        if (u8.empty()) 
            return false;

        u16.resize(unicode::offsetUtf8StringInUtf16(u8.data(), u8.size()) / sizeof(char16_t));
        if (u8.size() == u16.size()) 
            u16.assign(u8.begin(), u8.end());
        else
            return unicode::utf8to16(u8.data(), &u16[0], u8.size());
        
        return true;
    }

    bool StringUtils::u8StringToU32String(const std::string& u8, std::u32string& u32) {

        if (u8.empty()) 
            return false;

        u32.resize( unicode::offsetUtf8StringInUtf32(u8.data(), u8.size()) / sizeof(char32_t) );
        if (u8.size() == u32.size()) 
            u32.assign(u8.begin(), u8.end());
        else 
            return unicode::utf8to32(u8.data(), &u32[0], u8.size());
        
        return true;
    }

    bool StringUtils::u16StringToU32String(const std::u16string& u16, std::u32string& u32) {
        
        if (u16.empty()) 
            return false;

        u32.resize(unicode::offsetUtf16StringInUtf32(u16.data(), u16.size()) / sizeof(char32_t));
        if (u16.size() == u32.size()) 
            u32.assign(u16.begin(), u16.end());
        else 
            return unicode::utf16to32(u16.data(), &u32[0], u16.size());
        
        return true;
    }
    
    bool StringUtils::u16StringToU8String(const std::u16string& u16, std::string& u8) {

        if (u16.empty()) 
            return false;
        
        u8.resize(unicode::offsetUtf16StringInUtf8(u16.data(), u16.size()));
        if (u16.size() == u8.size()) 
            u8.assign(u16.begin(), u16.end());
        else
            return unicode::utf16to8(u16.data(), &u8[0], u16.size());
        
        return true;
    }
    
    bool StringUtils::u32StringToU16String(const std::u32string& u32, std::u16string& u16) {
        
        if (u32.empty()) 
            return false;

        u16.resize(unicode::offsetUtf32StringInUtf16(u32.data(), u32.size()) / sizeof(char16_t));
        if (u32.size() == u16.size()) 
            u16.assign(u32.begin(), u32.end());
        else 
            return unicode::utf32to16(u32.data(), &u16[0], u32.size());
        
        return true;
    }
    
    bool StringUtils::u32StringToU8String(const std::u32string& u32, std::string& u8) {
        
        if (u32.empty()) 
            return false;

        u8.resize(unicode::offsetUtf32StringInUtf8(u32.data(), u32.size()));
        if (u32.size() == u8.size()) 
            u8.assign(u32.begin(), u32.end());
        else 
            return unicode::utf32to8(u32.data(), &u8[0], u32.size());

        return true;
    }

  template<typename T>
  std::string StringUtils::vectorToString(const std::vector<T> &vec) {
    std::string result;
    for (auto v:vec)
      result += valueToString<T>(v);

    return result;
  }

  template std::string StringUtils::vectorToString(const std::vector<int> &vec);
  template std::string StringUtils::vectorToString(const std::vector<Nd4jLong> &vec);
  template std::string StringUtils::vectorToString(const std::vector<int16_t> &vec);
  template std::string StringUtils::vectorToString(const std::vector<uint32_t> &vec);
}
