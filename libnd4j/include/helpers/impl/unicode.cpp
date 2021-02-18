/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <helpers/unicode.h>

namespace sd {
namespace unicode {

    constexpr uint32_t ONEBYTEBOUND = 0x00000080;
    constexpr uint32_t TWOBYTEBOUND = 0x00000800;
    constexpr uint32_t THREEBYTEBOUND = 0x00010000;
    constexpr uint16_t HIGHBYTEMIN = 0xd800u;
    constexpr uint16_t HIGHBYTEMAX = 0xdbffu;
    constexpr uint16_t TRAILBYTEMIN = 0xdc00u;
    constexpr uint16_t TRAILBYTEMAX = 0xdfffu;
    constexpr uint16_t HIGHBYTEOFFSET = HIGHBYTEMIN - (0x10000 >> 10);
    constexpr uint32_t BYTEOFFSET = 0x10000u - (HIGHBYTEMIN << 10) - TRAILBYTEMIN;
    // Maximum valid value for a Unicode code point
    constexpr uint32_t CODEPOINTMAX = 0x0010ffffu;

    template<typename T>
    FORCEINLINE uint8_t castToU8(const T cp) {
        return static_cast<uint8_t>(0xff & cp);
    }

    template<typename T>
    FORCEINLINE uint16_t castToU16(const T cp) {
        return static_cast<uint16_t>(0xffff & cp);
    }

    template<typename T>
    FORCEINLINE uint32_t castToU32(const T cp) {
        return static_cast<uint32_t>(0xffffff & cp);
    }

    template<typename T>
    FORCEINLINE bool isTrail(const T cp) {
        return ((castToU8(cp) >> 6) == 0x2);
    }

    template <typename T>
    FORCEINLINE bool isHighSurrogate(const T cp) {
        return (cp & 0xfffffc00) == 0xd800;
    }

    template <typename T>
    bool isLowSurrogate(const T cp) {
        return (cp & 0xfffffc00) == 0xdc00;
    }

    template <typename T>
    FORCEINLINE bool isLeadSurrogate(const T cp) {
        return (cp >= HIGHBYTEMIN && cp <= HIGHBYTEMAX);
    }

    template <typename T>
    FORCEINLINE bool isTrailSurrogate(const T cp) {
        return (cp >= TRAILBYTEMIN && cp <= TRAILBYTEMAX);
    }

    template <typename T>
    FORCEINLINE bool isSurrogateU8(const T cp) {
        return (cp >= HIGHBYTEMIN && cp <= TRAILBYTEMAX);
    }

    template <typename T>
    FORCEINLINE bool isSurrogateU16(const T cp) {
        return ((cp - 0xd800u) < 2048u);
    }

    template <typename T>
    FORCEINLINE bool isSymbolU8Valid(const T cp) {
        return (cp <= CODEPOINTMAX && !isSurrogateU8(cp));
    }

    template <typename T>
    FORCEINLINE bool isSymbolValid(const T cp) {
        return (cp <= CODEPOINTMAX);
    }

    template <typename T>
    FORCEINLINE uint32_t surrogateU32(const T& high, const T& low) {
        return (high << 10) + low - 0x35fdc00;
    }

    template <typename T>
    Nd4jLong symbolLength(const T* it) {
        uint8_t lead = castToU8(*it);
        if (lead < 0x80)
            return 1;
        else if ((lead >> 5) == 0x6)
            return 2;
        else if ((lead >> 4) == 0xe)
            return 3;
        else if ((lead >> 3) == 0x1e)
            return 4;
        else
            return 0;
    }

    template <typename T>
    Nd4jLong symbolLength32(const T* it) {
        auto lead = castToU32(*it);
        if (lead < ONEBYTEBOUND)
            return 1;
        else if (lead < TWOBYTEBOUND)
            return 2;
        else if (lead < THREEBYTEBOUND)
            return 3;
        else if (lead <= CODEPOINTMAX)
            return 4;
        else
            return 0;
    }

    template <typename T>
    Nd4jLong symbolLength16(const T* it) {

        uint32_t lead = castToU16(*it);
        if (!isLeadSurrogate(lead)) {
            if (lead < ONEBYTEBOUND)
                return 1;
            else if (lead < TWOBYTEBOUND)
                return 2;
            else if (lead < THREEBYTEBOUND)
                return 3;
            else
                return 0;
        }
        else {
            return 4;
        }
    }

    Nd4jLong offsetUtf8StringInUtf32(const void* start, const void* end) {
        
        Nd4jLong count = 0;
        for (auto it = static_cast<const int8_t*>(start); it != end; it++) {
            auto length = symbolLength(it);
            it += (length > 0) ? (length - 1) : 0;
            count += 1;
        }
        return static_cast<Nd4jLong>(count * sizeof(char32_t));
    }
    
    Nd4jLong offsetUtf16StringInUtf32(const void* start, const void* end) {

        Nd4jLong count = 0;
        for (auto it = static_cast<const uint16_t*>(start); it != end;) {
            auto length = symbolLength16(it);
            it += (4 == length) ? 2 : 1;
            count += 1;
        }
        return static_cast<Nd4jLong>(count*sizeof(char32_t));
    }
    
    Nd4jLong offsetUtf8StringInUtf16(const void* start, const void* end) {

        Nd4jLong count = 0;
        for (auto it = static_cast<const int8_t*>(start); it != end; it++) {
            auto length = symbolLength(it);
            auto step = ((length > 0) ? (length - 1) : 0);
            it += step;
            count += (4 == length) ? 2 : 1;
        }
        return static_cast<Nd4jLong>(count*sizeof(char16_t));
    }
    
    Nd4jLong offsetUtf16StringInUtf8(const void* start, const void* end) {

        Nd4jLong count = 0;
        for (auto it = static_cast<const uint16_t*>(start); it != end;) {
            auto length = symbolLength16(it);
            it += (4 == length) ? 2 : 1;
            count += length;
        }
        return static_cast<Nd4jLong>(count);
    }
    
    Nd4jLong offsetUtf32StringInUtf16(const void* start, const void* end) {
        
        Nd4jLong count = 0;
        for (auto it = static_cast<const uint32_t*>(start); it != end; it++) {
            auto length = symbolLength32(it);
            count += (4 == length) ? 2 : 1;;
        }
        return static_cast<Nd4jLong>(count*sizeof(char16_t));
    }
    
    Nd4jLong offsetUtf32StringInUtf8(const void* start, const void* end) {

        Nd4jLong count = 0;
        for (auto it = static_cast<const uint32_t*>(start); it != end; it++) {
            count += symbolLength32(it);
        }
        return count;
    }
    
    bool isStringValidU8(const void* start, const void* stop) {
        for (auto it = static_cast<const int8_t*>(start); it != stop; it++) {
            if (!isSymbolU8Valid( castToU8(*it) )) {
                return false;
            }
        }
        return true;
    }
    
    bool isStringValidU16(const void* start, const void* stop) {
        for (auto it = static_cast<const uint16_t*>(start); it != stop; it++) {
            if (!isSymbolValid( castToU32(*it) )) {
                return false;
            }
        }
        return true;
    }
    
    bool isStringValidU32(const void* start, const void* stop) {
        for (auto it = static_cast<const uint32_t*>(start); it != stop; it++) {
            if (!isSymbolValid( castToU32(*it) )) {
                return false;
            }
        }
        return true;
    }

    void* utf16to8Ptr(const void* start, const void* end, void* res) {

        auto result = static_cast<int8_t*>(res);
        // result have to be  pre-allocated
        for (auto it = static_cast<const uint16_t*>(start); it != end;) {
            uint32_t cp = castToU16(*it++);
             if (!isLeadSurrogate(cp)) {
                 if (cp < 0x80) {                        // for one byte
                     *(result++) = static_cast<uint8_t>(cp);
                 }
                 else if (cp < 0x800) {                // for two bytes
                     *(result++) = static_cast<uint8_t>((cp >> 6) | 0xc0);
                     *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
                 }
                 else{              // for three bytes
                     *(result++) = static_cast<uint8_t>((cp >> 12) | 0xe0);
                     *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
                     *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
                 }
             }
             else {
                 if (it != end) {
                     uint32_t trail_surrogate = castToU16(*it++);
                     if (isTrailSurrogate(trail_surrogate))
                         cp = (cp << 10) + trail_surrogate + BYTEOFFSET;
                 }
                  // for four bytes
                 *(result++) = static_cast<uint8_t>((cp >> 18) | 0xf0);
                 *(result++) = static_cast<uint8_t>(((cp >> 12) & 0x3f) | 0x80);
                 *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
                 *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
             }
         }
         return result;
     }
     
     void* utf8to16Ptr(const void* start, const void* end, void* res) {
         
         auto result = static_cast<uint16_t*>(res);
         // result have to be  pre-allocated
         for (auto it = static_cast<const int8_t*>(start); it != end;) {
             
             auto nLength = symbolLength(it);
             uint32_t cp = castToU8(*it++);
             if (4 != nLength) {
                 if (2 == nLength) {
                     cp = ((cp << 6) & 0x7ff) + ((*it++) & 0x3f);
                 }
                 else if (3 == nLength) {
                     cp = ((cp << 12) & 0xffff) + ((castToU8(*it++) << 6) & 0xfff);
                     cp += (*it++) & 0x3f;
                 }
                 *(result++) = static_cast<uint16_t>(cp);
             }
             else {
                 cp = ((cp << 18) & 0x1fffff) + ((castToU8(*it++) << 12) & 0x3ffff);
                 cp += (castToU8(*it++) << 6) & 0xfff;
                 cp += (*it++) & 0x3f;
                 //make a surrogate pair
                 *(result++) = static_cast<uint16_t>((cp >> 10) + HIGHBYTEOFFSET);
                 *(result++) = static_cast<uint16_t>((cp & 0x3ff) + TRAILBYTEMIN);
             }
         }
         return result;
     }

     void* utf32to8Ptr( const void* start, const void* end, void* result) {
          
         auto res = static_cast<uint8_t*>(result);
         // result have to be  pre-allocated
         for (auto it = static_cast<const uint32_t*>(start); it != end; it++) {

            if (*it < 0x80)                        // for one byte
                 *(res++) = static_cast<uint8_t>(*it);
             else if (*it < 0x800) {                // for two bytes
                 *(res++) = static_cast<uint8_t>((*it >> 6) | 0xc0);
                 *(res++) = static_cast<uint8_t>((*it & 0x3f) | 0x80);
             }
             else if (*it < 0x10000) {              // for three bytes
                 *(res++) = static_cast<uint8_t>((*it >> 12) | 0xe0);
                 *(res++) = static_cast<uint8_t>(((*it >> 6) & 0x3f) | 0x80);
                 *(res++) = static_cast<uint8_t>((*it & 0x3f) | 0x80);
             }
             else {                                // for four bytes
                 *(res++) = static_cast<uint8_t>((*it >> 18) | 0xf0);
                 *(res++) = static_cast<uint8_t>(((*it >> 12) & 0x3f) | 0x80);
                 *(res++) = static_cast<uint8_t>(((*it >> 6) & 0x3f) | 0x80);
                 *(res++) = static_cast<uint8_t>((*it & 0x3f) | 0x80);
             }
         }
         return result;
     }

     void* utf8to32Ptr(const void* start, const void* end, void* res) {
         
         auto result = static_cast<uint32_t*>(res);
         // result have to be  pre-allocated
         for (auto it = static_cast<const int8_t*>(start); it != end;) {
             
             auto nLength = symbolLength(it);
             uint32_t cp = castToU8(*it++);
             if (2 == nLength) {
                 cp = ((cp << 6) & 0x7ff) + ((*it++) & 0x3f);
             }
             else if (3 == nLength) {
                 cp = ((cp << 12) & 0xffff) + ((castToU8(*it++) << 6) & 0xfff);
                 cp += (*it++) & 0x3f;
             }
             else if (4 == nLength) {
                 cp = ((cp << 18) & 0x1fffff) + ((castToU8(*it++) << 12) & 0x3ffff);
                 cp += (castToU8(*it++) << 6) & 0xfff;
                 cp += (*it++) & 0x3f;
             }
             (*result++) = cp;
         }
         return result;
     }
    
     void* utf16to32Ptr(const void* start, const void* end, void* res) {

         auto result = static_cast<uint32_t*>(res);
         // result have to be  pre-allocated
         for (auto it = static_cast<const uint16_t*>(start); it != end; it++) {
             
             uint32_t cpHigh = castToU32(*it);
             if (!isSurrogateU16(cpHigh)) {
                 *result++ = cpHigh;
             }
             else {
                 it++;
                 uint32_t cpLow = castToU32(*it);
                 if (isHighSurrogate(cpHigh) && it != end && isLowSurrogate(cpLow)) {
                     *result++ = surrogateU32(cpHigh, cpLow);
                 }
             }
         }
         return result;
     }

     void* utf32to16Ptr(const void* start, const void* end, void* res) {
         
         auto result = static_cast<uint16_t*>(res);
         // result have to be  pre-allocate
         for (auto it = static_cast<const uint32_t*>(start); it != end; it++) {

             uint32_t cpHigh = castToU32(*it);
             // todo check do we need this as we have pre-validation, if yes find out how to check u16
             if (cpHigh < 0 || cpHigh > 0x10FFFF || (cpHigh >= 0xD800 && cpHigh <= 0xDFFF)) {
                 // Invalid code point.  Replace with sentinel, per Unicode standard:
                 *result++ = u'\uFFFD';
             }
             else if (cpHigh < 0x10000UL) { // In the BMP.
                 *result++ = static_cast<char16_t>(cpHigh);
             }
             else {
                 *result++ = static_cast<char16_t>(((cpHigh - 0x10000UL) / 0x400U) + 0xD800U);
                 *result++ = static_cast<char16_t>(((cpHigh - 0x10000UL) % 0x400U) + 0xDC00U);
             }
         }
         return result;
     }

     Nd4jLong offsetUtf8StringInUtf32(const void* input, uint32_t nInputSize) {
         return  offsetUtf8StringInUtf32(input, static_cast<const int8_t*>(input) + nInputSize);
     }

     Nd4jLong offsetUtf16StringInUtf32(const void* input, uint32_t nInputSize) {
         return  offsetUtf16StringInUtf32(input, static_cast<const uint16_t*>(input) + nInputSize);
     }
    
     Nd4jLong offsetUtf8StringInUtf16(const void* input, uint32_t nInputSize) {
         return offsetUtf8StringInUtf16(input, static_cast<const int8_t*>(input) + nInputSize);
     }
 
     Nd4jLong offsetUtf16StringInUtf8(const void* input, uint32_t nInputSize) {
         return offsetUtf16StringInUtf8(input, static_cast<const uint16_t*>(input) + nInputSize);
     }

     Nd4jLong offsetUtf32StringInUtf8(const void* input, uint32_t nInputSize) {
         return offsetUtf32StringInUtf8(input, static_cast<const uint32_t*>(input) + nInputSize);
     }
     
     Nd4jLong offsetUtf32StringInUtf16(const void* input, const uint32_t nInputSize) {
         return offsetUtf32StringInUtf16(input, static_cast<const uint32_t*>(input) + nInputSize);
     }
     
     bool utf8to16(const void* input, void* output, uint32_t nInputSize) {
         return utf8to16Ptr(input, static_cast<const int8_t*>(input) + nInputSize, output);
     }
     
     bool utf8to32(const void* input, void* output, uint32_t nInputSize) {
         return utf8to32Ptr(input, static_cast<const int8_t*>(input) + nInputSize, output);
     }
     
     bool utf16to32(const void* input, void* output, uint32_t nInputSize) {
         return utf16to32Ptr(input, static_cast<const uint16_t*>(input) + nInputSize, output);
     }
     
     bool utf16to8(const void* input, void* output, uint32_t nInputSize) {
         return utf16to8Ptr(input, static_cast<const uint16_t*>(input) + nInputSize, output);
     }
     
     bool utf32to16(const void* input, void* output, uint32_t nInputSize) {
         return utf32to16Ptr(input, static_cast<const uint32_t*>(input) + nInputSize, output);
     }
     
     bool utf32to8(const void* input, void* output, const Nd4jLong nInputSize) {
         return utf32to8Ptr(input, static_cast<const uint32_t*>(input) + nInputSize, output);
     }

 }

}

