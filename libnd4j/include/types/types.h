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
// Created by raver on 6/12/2018.
// Modified by AbdelRauf

#ifndef SD_COMMON_TYPES_HEADER_INCLUDE
#define SD_COMMON_TYPES_HEADER_INCLUDE

#include <system/type_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>
#include <types/float8.h>
#include <types/int16.h>
#include <types/int8.h>
#include <types/uint16.h>
#include <types/uint8.h>
#include <types/utf8string.h>
/**
 * @brief type related Macrosses to help with template instantiations.
 *
 *  To restrict certain types one should define SD_SELECTIVE_TYPES and desired types using HAS_${TYPENAME}
 *  where ${TYPENAME} is a type name written in capital letters:
 *    #define SD_SELECTIVE_TYPES
 *    #define HAS_FLOAT
 *    #define HAS_INT8
 *    // and et cetera
 *    For gcc as compiler args: -DSD_SELECTIVE_TYPES -DHAS_FLOAT -DHAS_INT8
 *
 *  Some types are grouped for the template usages.
 *  For example:
 *    SD_FLOAT_TYPES is a list of floating types.
 *    SD_COMMON_TYPES is a list of commonly used types
 *
 *  To access one of the items of that list one could use GET_ELEMENT(INDEX, TYPE_GROUP)
 *  where is INDEX is number and TYPE_GROUP is the name of that group:
 *
 *  GET_ELEMENT(0, SD_FLOAT_TYPES) //will get the first element of the floating-point type list.
 *  Also, it is preferable to guard against out of boundary access using COUNT_NARG such way:
 *  #if COUNT_NARG(SD_FLOAT_TYPES) >3
 *    #define FL_ITEM_3 GET_ELEMENT(3, SD_FLOAT_TYPES)
 *  #endif
 *
 *  As mandatory we define at least one default type for some groups when no specific type was defined for that list.
 *  Such cases will be informed with a warning message while compilation.
 *
 *  If you want to use a specific type you should better use that type directly this way below:
 *
 *  #define HALFTYPE_SINGLETON SKIP_FIRST_COMMA(TTYPE_HALF)
 *  and use it inside template instantiations for half type only
 *
 *  For some compilers we can split types into singletons or into a small sublists to help with template instantiations
 *  In our case, for our CMAKE generator we pre-defined lists with indices for that purpose.
 *  And also pairwise types are defined that way as well.
 *  For example:
 *  SD_FLOAT_TYPES_0 will be a singleton made of the 1st element
 *  SD_FLOAT_TYPES_1 will be a singleton made of the 2nd element if it is available.
 *  As it could be undefined due to limited types, one should guard it with ifdef
 *  #if defined(SD_FLOAT_TYPES_2)
 *    //use the 3rd singleton from the floating-type lists
 *  #endif
 *
 *  In the same way pairwise types are used:
 *
 *  #if defined(SD_PAIRWISE_TYPES_2)
 *    //use the third sublist from the pairwise_types
 *  #endif
 *
 *  For GCC and Clang we have defined this STRINGIFY_TEST  macros helper to output lists.
 *  For example: STRINGIFY_TEST(SD_FLOAT_TYPES)
 */

#define ND_EXPAND(...) __VA_ARGS__

// Use:     #pragma message WARN("My message")
#if _MSC_VER
#define FILE_LINE_LINK __FILE__ "(" STRINGIZE(__LINE__) ") : "
#define WARN(exp) (FILE_LINE_LINK "WARNING: " exp)

//check if msvc cross-platform compatible preprocessor is not enabled (/Zc:preprocessor)
#if (defined(_MSVC_TRADITIONAL) && _MSVC_TRADITIONAL)
//MSVC_OLD_PREPROCESSOR indicates that we have MSVC old preprocessor
#define MSVC_OLD_PREPROCESSOR
// does not work, but suppress compiler error
#define STRINGIFY_TEST(...) (#__VA_ARGS__)
#endif
#else  //__GNUC__
#define STRINGIFY_TEST(...) STRINGIFY_TEST_INNER(__VA_ARGS__)
#define STRINGIFY_TEST_INNER(...) #__VA_ARGS__
#define WARN(exp) ("WARNING: " exp)
#endif

#if !defined(SD_SELECTIVE_TYPES)
#define HAS_BFLOAT16
#define HAS_BOOL
#define HAS_DOUBLE
#define HAS_FLOAT16
#define HAS_FLOAT32
#define HAS_INT16
#define HAS_INT32
#define HAS_INT8
#define HAS_LONG
#define HAS_UNSIGNEDLONG
#define HAS_UINT16
#define HAS_UINT32
#define HAS_UINT8
#define HAS_UTF16
#define HAS_UTF32
#define HAS_UTF8
#endif

// synonyms
#if defined(HAS_FLOAT64) && !defined(HAS_DOUBLE)
#define HAS_DOUBLE
#endif
#if defined(HAS_FLOAT) && !defined(HAS_FLOAT32)
#define HAS_FLOAT32
#endif
#if defined(HAS_HALF) && !defined(HAS_FLOAT16)
#define HAS_FLOAT16
#endif
#if defined(HAS_INT64) && !defined(HAS_LONG)
#define HAS_LONG
#endif
#if defined(HAS_UINT64) && !defined(HAS_UNSIGNEDLONG)
#define HAS_UNSIGNEDLONG
#endif
#if defined(HAS_INT) && !defined(HAS_INT32)
#define HAS_INT32
#endif
#if defined(HAS_BFLOAT16) && !defined(HAS_BFLOAT)
#define HAS_BFLOAT
#endif

#if defined(HAS_BFLOAT)
#define TTYPE_BFLOAT , (sd::DataType::BFLOAT16, bfloat16)
#else
#define TTYPE_BFLOAT
#endif
#if defined(HAS_BOOL)
#define TTYPE_BOOL , (sd::DataType::BOOL, bool)
#else
#define TTYPE_BOOL
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_DOUBLE , (sd::DataType::DOUBLE, double)
#else
#define TTYPE_DOUBLE
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_FLOAT32 , (sd::DataType::FLOAT32, float)
#else
#define TTYPE_FLOAT32
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_HALF , (sd::DataType::HALF, float16)
#else
#define TTYPE_HALF
#endif
#if defined(HAS_INT16)
#define TTYPE_INT16 , (sd::DataType::INT16, int16_t)
#else
#define TTYPE_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_INT32 , (sd::DataType::INT32, int32_t)
#else
#define TTYPE_INT32
#endif

#if defined(HAS_LONG)
#define TTYPE_INT64 , (sd::DataType::INT64, sd::LongType)
#else
#define TTYPE_INT64
#endif

#if defined(HAS_INT8)
#define TTYPE_INT8 , (sd::DataType::INT8, int8_t)
#else
#define TTYPE_INT8
#endif
#if defined(HAS_UINT16)
#define TTYPE_UINT16 , (sd::DataType::UINT16, uint16_t)
#else
#define TTYPE_UINT16
#endif
#if defined(HAS_UINT32)
#define TTYPE_UINT32 , (sd::DataType::UINT32, uint32_t)
#else
#define TTYPE_UINT32
#endif
#if defined(HAS_UNSIGNEDLONG)
#define TTYPE_UINT64 , (sd::DataType::UINT64, uint64_t)
#else
#define TTYPE_UINT64
#endif
#if defined(HAS_UINT8)
#define TTYPE_UINT8 , (sd::DataType::UINT8, uint8_t)
#else
#define TTYPE_UINT8
#endif
#if defined(HAS_UTF16)
#define TTYPE_UTF16 , (sd::DataType::UTF16, std::u16string)
#else
#define TTYPE_UTF16
#endif
#if defined(HAS_UTF32)
#define TTYPE_UTF32 , (sd::DataType::UTF32, std::u32string)
#else
#define TTYPE_UTF32
#endif
#if defined(HAS_UTF8)
#define TTYPE_UTF8 , (sd::DataType::UTF8, std::string)
#else
#define TTYPE_UTF8
#endif

#define M_CONCAT2(A, B) A##B
#define M_CONCAT1(A, B) M_CONCAT2(A, B)
#define M_CONCAT(A, B) M_CONCAT1(A, B)

#define COUNT_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, \
                _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, N, ...)                                 \
  N

#if defined(MSVC_OLD_PREPROCESSOR)
#pragma message WARN("MSVC old preprocessor")
//workaround the bug in MSVC old preprocessor using ND_CALL macro where it improperly expands __VA_ARGS__ 
#define COUNT_M(...)                                                                                                 \
  ND_EXPAND(COUNT_N(__VA_ARGS__, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, \
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define COUNT_NARG2(...) ND_CALL(COUNT_M, (, __VA_ARGS__))
#define COUNT_NARG(...) ND_EXPAND(COUNT_NARG2(__VA_ARGS__))
#define ND_CALL(X, Y) X Y
#else
#define COUNT_NARG_(...) COUNT_N(__VA_ARGS__)
#define COUNT_NARG2(...)                                                                                            \
  COUNT_NARG_(_, ##__VA_ARGS__, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, \
              12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define COUNT_NARG(...) COUNT_NARG2(__VA_ARGS__)
#endif

#define SKIP_FIRST_COMMA(...) ND_EXPAND(SKIP_FIRST_COMMA_Z(__VA_ARGS__))
#define SKIP_FIRST_COMMA_Z(Z, ...) __VA_ARGS__


#if defined(MSVC_OLD_PREPROCESSOR)
// workaround the bug in MSVC  old preprocessor using ND_CALL macro where it improperly expands __VA_ARGS__  
#define GET_ELEMENT(N, ...) ND_CALL(M_CONCAT(GET_ELEMENT_, N), (__VA_ARGS__))
#else
#define GET_ELEMENT(N, ...) M_CONCAT(GET_ELEMENT_, N)(__VA_ARGS__)
#endif
#define GET_ELEMENT_0(_0, ...) _0
#define GET_ELEMENT_1(_0, _1, ...) _1
#define GET_ELEMENT_2(_0, _1, _2, ...) _2
#define GET_ELEMENT_3(_0, _1, _2, _3, ...) _3
#define GET_ELEMENT_4(_0, _1, _2, _3, _4, ...) _4
#define GET_ELEMENT_5(_0, _1, _2, _3, _4, _5, ...) _5
#define GET_ELEMENT_6(_0, _1, _2, _3, _4, _5, _6, ...) _6
#define GET_ELEMENT_7(_0, _1, _2, _3, _4, _5, _6, _7, ...) _7
#define GET_ELEMENT_8(_0, _1, _2, _3, _4, _5, _6, _7, _8, ...) _8
#define GET_ELEMENT_9(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) _9
#define GET_ELEMENT_10(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, ...) _10
#define GET_ELEMENT_11(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, ...) _11
#define GET_ELEMENT_12(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, ...) _12
#define GET_ELEMENT_13(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, ...) _13
#define GET_ELEMENT_14(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, ...) _14
#define GET_ELEMENT_15(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, ...) _15
#define GET_ELEMENT_16(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, ...) _16
#define GET_ELEMENT_17(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, ...) _17
#define GET_ELEMENT_18(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, ...) _18
#define GET_ELEMENT_19(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, ...) \
  _19
#define GET_ELEMENT_20(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
                       ...)                                                                                           \
  _20

// we have to define bool anyway
#define SD_BOOL_TYPES (sd::DataType::BOOL, bool)
#define SD_LONG_TYPES_L TTYPE_INT64 TTYPE_UINT64
#define SD_STRING_TYPES_L TTYPE_UTF8 TTYPE_UTF16 TTYPE_UTF32
#define SD_INDEXING_TYPES_L TTYPE_INT32 TTYPE_INT64
#define SD_INTEGER_TYPES_L SD_INDEXING_TYPES_L TTYPE_INT8 TTYPE_INT16 TTYPE_UINT8 TTYPE_UINT16 TTYPE_UINT32 TTYPE_UINT64
#define SD_FLOAT_NATIVE_L TTYPE_FLOAT32 TTYPE_DOUBLE TTYPE_HALF
#define SD_FLOAT_TYPES_L TTYPE_BFLOAT SD_FLOAT_NATIVE_L
#define SD_NUMERIC_TYPES_L SD_FLOAT_TYPES_L SD_INTEGER_TYPES_L
#define SD_GENERIC_NUMERIC_TYPES_L SD_FLOAT_TYPES_L SD_INDEXING_TYPES_L
#define SD_COMMON_TYPES_L TTYPE_BOOL SD_NUMERIC_TYPES_L

#if COUNT_NARG(SD_STRING_TYPES_L) < 1
#pragma message WARN("it will use utf8 as SD_STRING_TYPES")
#define SD_STRING_TYPES (sd::DataType::UTF8, std::string)
#else
#define SD_STRING_TYPES SKIP_FIRST_COMMA(SD_STRING_TYPES_L)
#endif

#if COUNT_NARG(SD_LONG_TYPES_L) < 1
#pragma message WARN("it will use int64 as SD_LONG_TYPES")
#define SD_LONG_TYPES (sd::DataType::INT64, int64_t)
#else
#define SD_LONG_TYPES SKIP_FIRST_COMMA(SD_LONG_TYPES_L)
#endif

#if COUNT_NARG(SD_INDEXING_TYPES_L) < 1
#pragma message WARN("it will use int32 as SD_INDEXING_TYPES")
#define SD_INDEXING_TYPES (sd::DataType::INT32, int32_t)
#else
#define SD_INDEXING_TYPES SKIP_FIRST_COMMA(SD_INDEXING_TYPES_L)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES_L) < 1
#pragma message WARN("it will use int32 as SD_INTEGER_TYPES")
#define SD_INTEGER_TYPES (sd::DataType::INT32, int32_t)
#else
#define SD_INTEGER_TYPES SKIP_FIRST_COMMA(SD_INTEGER_TYPES_L)
#endif

#if COUNT_NARG(SD_FLOAT_NATIVE_L) < 1
#pragma message WARN("it will use float32 as SD_FLOAT_NATIVE")
#define SD_FLOAT_NATIVE (sd::DataType::FLOAT32, float)
#else
#define SD_FLOAT_NATIVE SKIP_FIRST_COMMA(SD_FLOAT_NATIVE_L)
#endif

#if COUNT_NARG(SD_FLOAT_TYPES_L) < 1
#pragma message WARN("it will use float32 as SD_FLOAT_TYPES")
#define SD_FLOAT_TYPES (sd::DataType::FLOAT32, float)
#else
#define SD_FLOAT_TYPES SKIP_FIRST_COMMA(SD_FLOAT_TYPES_L)
#endif

#if COUNT_NARG(SD_COMMON_TYPES_L) < 1
#pragma message WARN("it will use float32 as SD_COMMON_TYPES")
#define SD_COMMON_TYPES (sd::DataType::FLOAT32, float)
#define SD_COMMON_TYPES_EXTENDED (sd::DataType::FLOAT32, float)
#else
#define SD_COMMON_TYPES SKIP_FIRST_COMMA(SD_COMMON_TYPES_L)
#define SD_COMMON_TYPES_EXTENDED SKIP_FIRST_COMMA(SD_COMMON_TYPES_L)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES_L) < 1
#pragma message WARN("it will use float32 as SD_NUMERIC_TYPES")
#define SD_NUMERIC_TYPES (sd::DataType::FLOAT32, float)
#else
#define SD_NUMERIC_TYPES SKIP_FIRST_COMMA(SD_NUMERIC_TYPES_L)
#endif

#if COUNT_NARG(SD_GENERIC_NUMERIC_TYPES_L) < 1
#pragma message WARN("it will use float32 as SD_GENERIC_NUMERIC_TYPES")
#define SD_GENERIC_NUMERIC_TYPES (sd::DataType::FLOAT32, float)
#else
#define SD_GENERIC_NUMERIC_TYPES SKIP_FIRST_COMMA(SD_GENERIC_NUMERIC_TYPES_L)
#endif

#define SD_NATIVE_FLOAT_TYPES  (sd::DataType::FLOAT32, float), (sd::DataType::DOUBLE, double)


///////////FULL LIST FOR THE METHODS WHICH SHOULD BE DEFINED FOR GENERAL TYPES///////////////
#define SD_COMMON_TYPES_ALL                                                                                         \
  (sd::DataType::HALF, float16), (sd::DataType::FLOAT32, float), (sd::DataType::DOUBLE, double),                    \
      (sd::DataType::BOOL, bool), (sd::DataType::INT8, int8_t), (sd::DataType::UINT8, uint8_t),                     \
      (sd::DataType::INT16, int16_t), (sd::DataType::INT32, int32_t), (sd::DataType::INT64, sd::LongType),          \
      (sd::DataType::UINT16, uint16_t), (sd::DataType::UINT64, sd::UnsignedLong), (sd::DataType::UINT32, uint32_t), \
      (sd::DataType::BFLOAT16, bfloat16)



///////////TRIPLES GENERATED MANUALLY USING REGEX /////////////////////////
#if defined(HAS_BFLOAT16)
#define TTYPE_BFLOAT16_BFLOAT16_BFLOAT16 , (bfloat16, bfloat16, bfloat16)
#if defined(HAS_BOOL)
#define TTYPE_BFLOAT16_BOOL_BFLOAT16 , (bfloat16, bool, bfloat16)
#define TTYPE_BFLOAT16_BOOL_BOOL , (bfloat16, bool, bool)
#else
#define TTYPE_BFLOAT16_BOOL_BFLOAT16
#define TTYPE_BFLOAT16_BOOL_BOOL
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_BFLOAT16_DOUBLE_BFLOAT16 , (bfloat16, double, bfloat16)
#define TTYPE_BFLOAT16_DOUBLE_DOUBLE , (bfloat16, double, double)
#else
#define TTYPE_BFLOAT16_DOUBLE_BFLOAT16
#define TTYPE_BFLOAT16_DOUBLE_DOUBLE
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_BFLOAT16_FLOAT_BFLOAT16 , (bfloat16, float, bfloat16)
#define TTYPE_BFLOAT16_FLOAT_FLOAT , (bfloat16, float, float)
#else
#define TTYPE_BFLOAT16_FLOAT_BFLOAT16
#define TTYPE_BFLOAT16_FLOAT_FLOAT
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_BFLOAT16_FLOAT16_BFLOAT16 , (bfloat16, float16, bfloat16)
#define TTYPE_BFLOAT16_FLOAT16_FLOAT16 , (bfloat16, float16, float16)
#else
#define TTYPE_BFLOAT16_FLOAT16_BFLOAT16
#define TTYPE_BFLOAT16_FLOAT16_FLOAT16
#endif
#if defined(HAS_INT16)
#define TTYPE_BFLOAT16_INT16_BFLOAT16 , (bfloat16, int16_t, bfloat16)
#define TTYPE_BFLOAT16_INT16_INT16 , (bfloat16, int16_t, int16_t)
#else
#define TTYPE_BFLOAT16_INT16_BFLOAT16
#define TTYPE_BFLOAT16_INT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_BFLOAT16_INT32_BFLOAT16 , (bfloat16, int32_t, bfloat16)
#define TTYPE_BFLOAT16_INT32_INT32 , (bfloat16, int32_t, int32_t)
#else
#define TTYPE_BFLOAT16_INT32_BFLOAT16
#define TTYPE_BFLOAT16_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_BFLOAT16_INT8_BFLOAT16 , (bfloat16, int8_t, bfloat16)
#define TTYPE_BFLOAT16_INT8_INT8 , (bfloat16, int8_t, int8_t)
#else
#define TTYPE_BFLOAT16_INT8_BFLOAT16
#define TTYPE_BFLOAT16_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_BFLOAT16_LONG_BFLOAT16 , (bfloat16, sd::LongType, bfloat16)
#define TTYPE_BFLOAT16_LONG_LONG , (bfloat16, sd::LongType, sd::LongType)
#else
#define TTYPE_BFLOAT16_LONG_BFLOAT16
#define TTYPE_BFLOAT16_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_BFLOAT16_UINT8_BFLOAT16 , (bfloat16, uint8_t, bfloat16)
#define TTYPE_BFLOAT16_UINT8_UINT8 , (bfloat16, uint8_t, uint8_t)
#else
#define TTYPE_BFLOAT16_UINT8_BFLOAT16
#define TTYPE_BFLOAT16_UINT8_UINT8
#endif
#else
#define TTYPE_BFLOAT16_BFLOAT16_BFLOAT16
#endif

#if defined(HAS_BOOL)
#define TTYPE_BOOL_BOOL_BOOL , (bool, bool, bool)
#if defined(HAS_BFLOAT16)
#define TTYPE_BOOL_BFLOAT16_BFLOAT16 , (bool, bfloat16, bfloat16)
#define TTYPE_BOOL_BFLOAT16_BOOL , (bool, bfloat16, bool)
#else
#define TTYPE_BOOL_BFLOAT16_BFLOAT16
#define TTYPE_BOOL_BFLOAT16_BOOL
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_BOOL_DOUBLE_BOOL , (bool, double, bool)
#define TTYPE_BOOL_DOUBLE_DOUBLE , (bool, double, double)
#else
#define TTYPE_BOOL_DOUBLE_BOOL
#define TTYPE_BOOL_DOUBLE_DOUBLE
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_BOOL_FLOAT_BOOL , (bool, float, bool)
#define TTYPE_BOOL_FLOAT_FLOAT , (bool, float, float)
#else
#define TTYPE_BOOL_FLOAT_BOOL
#define TTYPE_BOOL_FLOAT_FLOAT
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_BOOL_FLOAT16_BOOL , (bool, float16, bool)
#define TTYPE_BOOL_FLOAT16_FLOAT16 , (bool, float16, float16)
#else
#define TTYPE_BOOL_FLOAT16_BOOL
#define TTYPE_BOOL_FLOAT16_FLOAT16
#endif
#if defined(HAS_INT16)
#define TTYPE_BOOL_INT16_BOOL , (bool, int16_t, bool)
#define TTYPE_BOOL_INT16_INT16 , (bool, int16_t, int16_t)
#else
#define TTYPE_BOOL_INT16_BOOL
#define TTYPE_BOOL_INT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_BOOL_INT32_BOOL , (bool, int32_t, bool)
#define TTYPE_BOOL_INT32_INT32 , (bool, int32_t, int32_t)
#else
#define TTYPE_BOOL_INT32_BOOL
#define TTYPE_BOOL_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_BOOL_INT8_BOOL , (bool, int8_t, bool)
#define TTYPE_BOOL_INT8_INT8 , (bool, int8_t, int8_t)
#else
#define TTYPE_BOOL_INT8_BOOL
#define TTYPE_BOOL_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_BOOL_LONG_BOOL , (bool, sd::LongType, bool)
#define TTYPE_BOOL_LONG_LONG , (bool, sd::LongType, sd::LongType)
#else
#define TTYPE_BOOL_LONG_BOOL
#define TTYPE_BOOL_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_BOOL_UINT8_BOOL , (bool, uint8_t, bool)
#define TTYPE_BOOL_UINT8_UINT8 , (bool, uint8_t, uint8_t)
#else
#define TTYPE_BOOL_UINT8_BOOL
#define TTYPE_BOOL_UINT8_UINT8
#endif
#else
#define TTYPE_BOOL_BOOL_BOOL
#endif

#if defined(HAS_DOUBLE)
#define TTYPE_DOUBLE_DOUBLE_DOUBLE , (double, double, double)
#if defined(HAS_BFLOAT16)
#define TTYPE_DOUBLE_BFLOAT16_BFLOAT16 , (double, bfloat16, bfloat16)
#define TTYPE_DOUBLE_BFLOAT16_DOUBLE , (double, bfloat16, double)
#else
#define TTYPE_DOUBLE_BFLOAT16_BFLOAT16
#define TTYPE_DOUBLE_BFLOAT16_DOUBLE
#endif
#if defined(HAS_BOOL)
#define TTYPE_DOUBLE_BOOL_BOOL , (double, bool, bool)
#define TTYPE_DOUBLE_BOOL_DOUBLE , (double, bool, double)
#else
#define TTYPE_DOUBLE_BOOL_BOOL
#define TTYPE_DOUBLE_BOOL_DOUBLE
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_DOUBLE_FLOAT_DOUBLE , (double, float, double)
#define TTYPE_DOUBLE_FLOAT_FLOAT , (double, float, float)
#else
#define TTYPE_DOUBLE_FLOAT_DOUBLE
#define TTYPE_DOUBLE_FLOAT_FLOAT
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_DOUBLE_FLOAT16_DOUBLE , (double, float16, double)
#define TTYPE_DOUBLE_FLOAT16_FLOAT16 , (double, float16, float16)
#else
#define TTYPE_DOUBLE_FLOAT16_DOUBLE
#define TTYPE_DOUBLE_FLOAT16_FLOAT16
#endif
#if defined(HAS_INT16)
#define TTYPE_DOUBLE_INT16_DOUBLE , (double, int16_t, double)
#define TTYPE_DOUBLE_INT16_INT16 , (double, int16_t, int16_t)
#else
#define TTYPE_DOUBLE_INT16_DOUBLE
#define TTYPE_DOUBLE_INT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_DOUBLE_INT32_DOUBLE , (double, int32_t, double)
#define TTYPE_DOUBLE_INT32_INT32 , (double, int32_t, int32_t)
#else
#define TTYPE_DOUBLE_INT32_DOUBLE
#define TTYPE_DOUBLE_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_DOUBLE_INT8_DOUBLE , (double, int8_t, double)
#define TTYPE_DOUBLE_INT8_INT8 , (double, int8_t, int8_t)
#else
#define TTYPE_DOUBLE_INT8_DOUBLE
#define TTYPE_DOUBLE_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_DOUBLE_LONG_DOUBLE , (double, sd::LongType, double)
#define TTYPE_DOUBLE_LONG_LONG , (double, sd::LongType, sd::LongType)
#else
#define TTYPE_DOUBLE_LONG_DOUBLE
#define TTYPE_DOUBLE_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_DOUBLE_UINT8_DOUBLE , (double, uint8_t, double)
#define TTYPE_DOUBLE_UINT8_UINT8 , (double, uint8_t, uint8_t)
#else
#define TTYPE_DOUBLE_UINT8_DOUBLE
#define TTYPE_DOUBLE_UINT8_UINT8
#endif
#else
#define TTYPE_DOUBLE_DOUBLE_DOUBLE
#endif

#if defined(HAS_FLOAT32)
#define TTYPE_FLOAT_FLOAT_FLOAT , (float, float, float)
#if defined(HAS_BFLOAT16)
#define TTYPE_FLOAT_BFLOAT16_BFLOAT16 , (float, bfloat16, bfloat16)
#define TTYPE_FLOAT_BFLOAT16_FLOAT , (float, bfloat16, float)
#else
#define TTYPE_FLOAT_BFLOAT16_BFLOAT16
#define TTYPE_FLOAT_BFLOAT16_FLOAT
#endif
#if defined(HAS_BOOL)
#define TTYPE_FLOAT_BOOL_BOOL , (float, bool, bool)
#define TTYPE_FLOAT_BOOL_FLOAT , (float, bool, float)
#else
#define TTYPE_FLOAT_BOOL_BOOL
#define TTYPE_FLOAT_BOOL_FLOAT
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_FLOAT_DOUBLE_DOUBLE , (float, double, double)
#define TTYPE_FLOAT_DOUBLE_FLOAT , (float, double, float)
#else
#define TTYPE_FLOAT_DOUBLE_DOUBLE
#define TTYPE_FLOAT_DOUBLE_FLOAT
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_FLOAT_FLOAT16_FLOAT , (float, float16, float)
#define TTYPE_FLOAT_FLOAT16_FLOAT16 , (float, float16, float16)
#else
#define TTYPE_FLOAT_FLOAT16_FLOAT
#define TTYPE_FLOAT_FLOAT16_FLOAT16
#endif
#if defined(HAS_INT16)
#define TTYPE_FLOAT_INT16_FLOAT , (float, int16_t, float)
#define TTYPE_FLOAT_INT16_INT16 , (float, int16_t, int16_t)
#else
#define TTYPE_FLOAT_INT16_FLOAT
#define TTYPE_FLOAT_INT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_FLOAT_INT32_FLOAT , (float, int32_t, float)
#define TTYPE_FLOAT_INT32_INT32 , (float, int32_t, int32_t)
#else
#define TTYPE_FLOAT_INT32_FLOAT
#define TTYPE_FLOAT_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_FLOAT_INT8_FLOAT , (float, int8_t, float)
#define TTYPE_FLOAT_INT8_INT8 , (float, int8_t, int8_t)
#else
#define TTYPE_FLOAT_INT8_FLOAT
#define TTYPE_FLOAT_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_FLOAT_LONG_FLOAT , (float, sd::LongType, float)
#define TTYPE_FLOAT_LONG_LONG , (float, sd::LongType, sd::LongType)
#else
#define TTYPE_FLOAT_LONG_FLOAT
#define TTYPE_FLOAT_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_FLOAT_UINT8_FLOAT , (float, uint8_t, float)
#define TTYPE_FLOAT_UINT8_UINT8 , (float, uint8_t, uint8_t)
#else
#define TTYPE_FLOAT_UINT8_FLOAT
#define TTYPE_FLOAT_UINT8_UINT8
#endif
#else
#define TTYPE_FLOAT_FLOAT_FLOAT
#endif

#if defined(HAS_FLOAT16)
#define TTYPE_FLOAT16_FLOAT16_FLOAT16 , (float16, float16, float16)
#if defined(HAS_BFLOAT16)
#define TTYPE_FLOAT16_BFLOAT16_BFLOAT16 , (float16, bfloat16, bfloat16)
#define TTYPE_FLOAT16_BFLOAT16_FLOAT16 , (float16, bfloat16, float16)
#else
#define TTYPE_FLOAT16_BFLOAT16_BFLOAT16
#define TTYPE_FLOAT16_BFLOAT16_FLOAT16
#endif
#if defined(HAS_BOOL)
#define TTYPE_FLOAT16_BOOL_BOOL , (float16, bool, bool)
#define TTYPE_FLOAT16_BOOL_FLOAT16 , (float16, bool, float16)
#else
#define TTYPE_FLOAT16_BOOL_BOOL
#define TTYPE_FLOAT16_BOOL_FLOAT16
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_FLOAT16_DOUBLE_DOUBLE , (float16, double, double)
#define TTYPE_FLOAT16_DOUBLE_FLOAT16 , (float16, double, float16)
#else
#define TTYPE_FLOAT16_DOUBLE_DOUBLE
#define TTYPE_FLOAT16_DOUBLE_FLOAT16
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_FLOAT16_FLOAT_FLOAT , (float16, float, float)
#define TTYPE_FLOAT16_FLOAT_FLOAT16 , (float16, float, float16)
#else
#define TTYPE_FLOAT16_FLOAT_FLOAT
#define TTYPE_FLOAT16_FLOAT_FLOAT16
#endif
#if defined(HAS_INT16)
#define TTYPE_FLOAT16_INT16_FLOAT16 , (float16, int16_t, float16)
#define TTYPE_FLOAT16_INT16_INT16 , (float16, int16_t, int16_t)
#else
#define TTYPE_FLOAT16_INT16_FLOAT16
#define TTYPE_FLOAT16_INT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_FLOAT16_INT32_FLOAT16 , (float16, int32_t, float16)
#define TTYPE_FLOAT16_INT32_INT32 , (float16, int32_t, int32_t)
#else
#define TTYPE_FLOAT16_INT32_FLOAT16
#define TTYPE_FLOAT16_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_FLOAT16_INT8_FLOAT16 , (float16, int8_t, float16)
#define TTYPE_FLOAT16_INT8_INT8 , (float16, int8_t, int8_t)
#else
#define TTYPE_FLOAT16_INT8_FLOAT16
#define TTYPE_FLOAT16_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_FLOAT16_LONG_FLOAT16 , (float16, sd::LongType, float16)
#define TTYPE_FLOAT16_LONG_LONG , (float16, sd::LongType, sd::LongType)
#else
#define TTYPE_FLOAT16_LONG_FLOAT16
#define TTYPE_FLOAT16_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_FLOAT16_UINT8_FLOAT16 , (float16, uint8_t, float16)
#define TTYPE_FLOAT16_UINT8_UINT8 , (float16, uint8_t, uint8_t)
#else
#define TTYPE_FLOAT16_UINT8_FLOAT16
#define TTYPE_FLOAT16_UINT8_UINT8
#endif
#else
#define TTYPE_FLOAT16_FLOAT16_FLOAT16
#endif

#if defined(HAS_INT16)
#define TTYPE_INT16_INT16_INT16 , (int16_t, int16_t, int16_t)
#if defined(HAS_BFLOAT16)
#define TTYPE_INT16_BFLOAT16_BFLOAT16 , (int16_t, bfloat16, bfloat16)
#define TTYPE_INT16_BFLOAT16_INT16 , (int16_t, bfloat16, int16_t)
#else
#define TTYPE_INT16_BFLOAT16_BFLOAT16
#define TTYPE_INT16_BFLOAT16_INT16
#endif
#if defined(HAS_BOOL)
#define TTYPE_INT16_BOOL_BOOL , (int16_t, bool, bool)
#define TTYPE_INT16_BOOL_INT16 , (int16_t, bool, int16_t)
#else
#define TTYPE_INT16_BOOL_BOOL
#define TTYPE_INT16_BOOL_INT16
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_INT16_DOUBLE_DOUBLE , (int16_t, double, double)
#define TTYPE_INT16_DOUBLE_INT16 , (int16_t, double, int16_t)
#else
#define TTYPE_INT16_DOUBLE_DOUBLE
#define TTYPE_INT16_DOUBLE_INT16
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_INT16_FLOAT_FLOAT , (int16_t, float, float)
#define TTYPE_INT16_FLOAT_INT16 , (int16_t, float, int16_t)
#else
#define TTYPE_INT16_FLOAT_FLOAT
#define TTYPE_INT16_FLOAT_INT16
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_INT16_FLOAT16_FLOAT16 , (int16_t, float16, float16)
#define TTYPE_INT16_FLOAT16_INT16 , (int16_t, float16, int16_t)
#else
#define TTYPE_INT16_FLOAT16_FLOAT16
#define TTYPE_INT16_FLOAT16_INT16
#endif
#if defined(HAS_INT32)
#define TTYPE_INT16_INT32_INT16 , (int16_t, int32_t, int16_t)
#define TTYPE_INT16_INT32_INT32 , (int16_t, int32_t, int32_t)
#else
#define TTYPE_INT16_INT32_INT16
#define TTYPE_INT16_INT32_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_INT16_INT8_INT16 , (int16_t, int8_t, int16_t)
#define TTYPE_INT16_INT8_INT8 , (int16_t, int8_t, int8_t)
#else
#define TTYPE_INT16_INT8_INT16
#define TTYPE_INT16_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_INT16_LONG_INT16 , (int16_t, sd::LongType, int16_t)
#define TTYPE_INT16_LONG_LONG , (int16_t, sd::LongType, sd::LongType)
#else
#define TTYPE_INT16_LONG_INT16
#define TTYPE_INT16_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_INT16_UINT8_INT16 , (int16_t, uint8_t, int16_t)
#define TTYPE_INT16_UINT8_UINT8 , (int16_t, uint8_t, uint8_t)
#else
#define TTYPE_INT16_UINT8_INT16
#define TTYPE_INT16_UINT8_UINT8
#endif
#else
#define TTYPE_INT16_INT16_INT16
#endif

#if defined(HAS_INT32)
#define TTYPE_INT32_INT32_INT32 , (int32_t, int32_t, int32_t)
#if defined(HAS_BFLOAT16)
#define TTYPE_INT32_BFLOAT16_BFLOAT16 , (int32_t, bfloat16, bfloat16)
#define TTYPE_INT32_BFLOAT16_INT32 , (int32_t, bfloat16, int32_t)
#else
#define TTYPE_INT32_BFLOAT16_BFLOAT16
#define TTYPE_INT32_BFLOAT16_INT32
#endif
#if defined(HAS_BOOL)
#define TTYPE_INT32_BOOL_BOOL , (int32_t, bool, bool)
#define TTYPE_INT32_BOOL_INT32 , (int32_t, bool, int32_t)
#else
#define TTYPE_INT32_BOOL_BOOL
#define TTYPE_INT32_BOOL_INT32
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_INT32_DOUBLE_DOUBLE , (int32_t, double, double)
#define TTYPE_INT32_DOUBLE_INT32 , (int32_t, double, int32_t)
#else
#define TTYPE_INT32_DOUBLE_DOUBLE
#define TTYPE_INT32_DOUBLE_INT32
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_INT32_FLOAT_FLOAT , (int32_t, float, float)
#define TTYPE_INT32_FLOAT_INT32 , (int32_t, float, int32_t)
#else
#define TTYPE_INT32_FLOAT_FLOAT
#define TTYPE_INT32_FLOAT_INT32
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_INT32_FLOAT16_FLOAT16 , (int32_t, float16, float16)
#define TTYPE_INT32_FLOAT16_INT32 , (int32_t, float16, int32_t)
#else
#define TTYPE_INT32_FLOAT16_FLOAT16
#define TTYPE_INT32_FLOAT16_INT32
#endif
#if defined(HAS_INT16)
#define TTYPE_INT32_INT16_INT16 , (int32_t, int16_t, int16_t)
#define TTYPE_INT32_INT16_INT32 , (int32_t, int16_t, int32_t)
#else
#define TTYPE_INT32_INT16_INT16
#define TTYPE_INT32_INT16_INT32
#endif
#if defined(HAS_INT8)
#define TTYPE_INT32_INT8_INT32 , (int32_t, int8_t, int32_t)
#define TTYPE_INT32_INT8_INT8 , (int32_t, int8_t, int8_t)
#else
#define TTYPE_INT32_INT8_INT32
#define TTYPE_INT32_INT8_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_INT32_LONG_INT32 , (int32_t, sd::LongType, int32_t)
#define TTYPE_INT32_LONG_LONG , (int32_t, sd::LongType, sd::LongType)
#else
#define TTYPE_INT32_LONG_INT32
#define TTYPE_INT32_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_INT32_UINT8_INT32 , (int32_t, uint8_t, int32_t)
#define TTYPE_INT32_UINT8_UINT8 , (int32_t, uint8_t, uint8_t)
#else
#define TTYPE_INT32_UINT8_INT32
#define TTYPE_INT32_UINT8_UINT8
#endif
#else
#define TTYPE_INT32_INT32_INT32
#endif

#if defined(HAS_INT8)
#define TTYPE_INT8_INT8_INT8 , (int8_t, int8_t, int8_t)
#if defined(HAS_BFLOAT16)
#define TTYPE_INT8_BFLOAT16_BFLOAT16 , (int8_t, bfloat16, bfloat16)
#define TTYPE_INT8_BFLOAT16_INT8 , (int8_t, bfloat16, int8_t)
#else
#define TTYPE_INT8_BFLOAT16_BFLOAT16
#define TTYPE_INT8_BFLOAT16_INT8
#endif
#if defined(HAS_BOOL)
#define TTYPE_INT8_BOOL_BOOL , (int8_t, bool, bool)
#define TTYPE_INT8_BOOL_INT8 , (int8_t, bool, int8_t)
#else
#define TTYPE_INT8_BOOL_BOOL
#define TTYPE_INT8_BOOL_INT8
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_INT8_DOUBLE_DOUBLE , (int8_t, double, double)
#define TTYPE_INT8_DOUBLE_INT8 , (int8_t, double, int8_t)
#else
#define TTYPE_INT8_DOUBLE_DOUBLE
#define TTYPE_INT8_DOUBLE_INT8
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_INT8_FLOAT_FLOAT , (int8_t, float, float)
#define TTYPE_INT8_FLOAT_INT8 , (int8_t, float, int8_t)
#else
#define TTYPE_INT8_FLOAT_FLOAT
#define TTYPE_INT8_FLOAT_INT8
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_INT8_FLOAT16_FLOAT16 , (int8_t, float16, float16)
#define TTYPE_INT8_FLOAT16_INT8 , (int8_t, float16, int8_t)
#else
#define TTYPE_INT8_FLOAT16_FLOAT16
#define TTYPE_INT8_FLOAT16_INT8
#endif
#if defined(HAS_INT16)
#define TTYPE_INT8_INT16_INT16 , (int8_t, int16_t, int16_t)
#define TTYPE_INT8_INT16_INT8 , (int8_t, int16_t, int8_t)
#else
#define TTYPE_INT8_INT16_INT16
#define TTYPE_INT8_INT16_INT8
#endif
#if defined(HAS_INT32)
#define TTYPE_INT8_INT32_INT32 , (int8_t, int32_t, int32_t)
#define TTYPE_INT8_INT32_INT8 , (int8_t, int32_t, int8_t)
#else
#define TTYPE_INT8_INT32_INT32
#define TTYPE_INT8_INT32_INT8
#endif
#if defined(HAS_LONG)
#define TTYPE_INT8_LONG_INT8 , (int8_t, sd::LongType, int8_t)
#define TTYPE_INT8_LONG_LONG , (int8_t, sd::LongType, sd::LongType)
#else
#define TTYPE_INT8_LONG_INT8
#define TTYPE_INT8_LONG_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_INT8_UINT8_INT8 , (int8_t, uint8_t, int8_t)
#define TTYPE_INT8_UINT8_UINT8 , (int8_t, uint8_t, uint8_t)
#else
#define TTYPE_INT8_UINT8_INT8
#define TTYPE_INT8_UINT8_UINT8
#endif
#else
#define TTYPE_INT8_INT8_INT8
#endif

#if defined(HAS_LONG)
#define TTYPE_LONG_LONG_LONG , (sd::LongType, sd::LongType, sd::LongType)
#if defined(HAS_BFLOAT16)
#define TTYPE_LONG_BFLOAT16_BFLOAT16 , (sd::LongType, bfloat16, bfloat16)
#define TTYPE_LONG_BFLOAT16_LONG , (sd::LongType, bfloat16, sd::LongType)
#else
#define TTYPE_LONG_BFLOAT16_BFLOAT16
#define TTYPE_LONG_BFLOAT16_LONG
#endif
#if defined(HAS_BOOL)
#define TTYPE_LONG_BOOL_BOOL , (sd::LongType, bool, bool)
#define TTYPE_LONG_BOOL_LONG , (sd::LongType, bool, sd::LongType)
#else
#define TTYPE_LONG_BOOL_BOOL
#define TTYPE_LONG_BOOL_LONG
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_LONG_DOUBLE_DOUBLE , (sd::LongType, double, double)
#define TTYPE_LONG_DOUBLE_LONG , (sd::LongType, double, sd::LongType)
#else
#define TTYPE_LONG_DOUBLE_DOUBLE
#define TTYPE_LONG_DOUBLE_LONG
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_LONG_FLOAT_FLOAT , (sd::LongType, float, float)
#define TTYPE_LONG_FLOAT_LONG , (sd::LongType, float, sd::LongType)
#else
#define TTYPE_LONG_FLOAT_FLOAT
#define TTYPE_LONG_FLOAT_LONG
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_LONG_FLOAT16_FLOAT16 , (sd::LongType, float16, float16)
#define TTYPE_LONG_FLOAT16_LONG , (sd::LongType, float16, sd::LongType)
#else
#define TTYPE_LONG_FLOAT16_FLOAT16
#define TTYPE_LONG_FLOAT16_LONG
#endif
#if defined(HAS_INT16)
#define TTYPE_LONG_INT16_INT16 , (sd::LongType, int16_t, int16_t)
#define TTYPE_LONG_INT16_LONG , (sd::LongType, int16_t, sd::LongType)
#else
#define TTYPE_LONG_INT16_INT16
#define TTYPE_LONG_INT16_LONG
#endif
#if defined(HAS_INT32)
#define TTYPE_LONG_INT32_INT32 , (sd::LongType, int32_t, int32_t)
#define TTYPE_LONG_INT32_LONG , (sd::LongType, int32_t, sd::LongType)
#else
#define TTYPE_LONG_INT32_INT32
#define TTYPE_LONG_INT32_LONG
#endif
#if defined(HAS_INT8)
#define TTYPE_LONG_INT8_INT8 , (sd::LongType, int8_t, int8_t)
#define TTYPE_LONG_INT8_LONG , (sd::LongType, int8_t, sd::LongType)
#else
#define TTYPE_LONG_INT8_INT8
#define TTYPE_LONG_INT8_LONG
#endif
#if defined(HAS_UINT8)
#define TTYPE_LONG_UINT8_LONG , (sd::LongType, uint8_t, sd::LongType)
#define TTYPE_LONG_UINT8_UINT8 , (sd::LongType, uint8_t, uint8_t)
#else
#define TTYPE_LONG_UINT8_LONG
#define TTYPE_LONG_UINT8_UINT8
#endif
#else
#define TTYPE_LONG_LONG_LONG
#endif

#if defined(HAS_UINT8)
#define TTYPE_UINT8_UINT8_UINT8 , (uint8_t, uint8_t, uint8_t)
#if defined(HAS_BFLOAT16)
#define TTYPE_UINT8_BFLOAT16_BFLOAT16 , (uint8_t, bfloat16, bfloat16)
#define TTYPE_UINT8_BFLOAT16_UINT8 , (uint8_t, bfloat16, uint8_t)
#else
#define TTYPE_UINT8_BFLOAT16_BFLOAT16
#define TTYPE_UINT8_BFLOAT16_UINT8
#endif
#if defined(HAS_BOOL)
#define TTYPE_UINT8_BOOL_BOOL , (uint8_t, bool, bool)
#define TTYPE_UINT8_BOOL_UINT8 , (uint8_t, bool, uint8_t)
#else
#define TTYPE_UINT8_BOOL_BOOL
#define TTYPE_UINT8_BOOL_UINT8
#endif
#if defined(HAS_DOUBLE)
#define TTYPE_UINT8_DOUBLE_DOUBLE , (uint8_t, double, double)
#define TTYPE_UINT8_DOUBLE_UINT8 , (uint8_t, double, uint8_t)
#else
#define TTYPE_UINT8_DOUBLE_DOUBLE
#define TTYPE_UINT8_DOUBLE_UINT8
#endif
#if defined(HAS_FLOAT32)
#define TTYPE_UINT8_FLOAT_FLOAT , (uint8_t, float, float)
#define TTYPE_UINT8_FLOAT_UINT8 , (uint8_t, float, uint8_t)
#else
#define TTYPE_UINT8_FLOAT_FLOAT
#define TTYPE_UINT8_FLOAT_UINT8
#endif
#if defined(HAS_FLOAT16)
#define TTYPE_UINT8_FLOAT16_FLOAT16 , (uint8_t, float16, float16)
#define TTYPE_UINT8_FLOAT16_UINT8 , (uint8_t, float16, uint8_t)
#else
#define TTYPE_UINT8_FLOAT16_FLOAT16
#define TTYPE_UINT8_FLOAT16_UINT8
#endif
#if defined(HAS_INT16)
#define TTYPE_UINT8_INT16_INT16 , (uint8_t, int16_t, int16_t)
#define TTYPE_UINT8_INT16_UINT8 , (uint8_t, int16_t, uint8_t)
#else
#define TTYPE_UINT8_INT16_INT16
#define TTYPE_UINT8_INT16_UINT8
#endif
#if defined(HAS_INT32)
#define TTYPE_UINT8_INT32_INT32 , (uint8_t, int32_t, int32_t)
#define TTYPE_UINT8_INT32_UINT8 , (uint8_t, int32_t, uint8_t)
#else
#define TTYPE_UINT8_INT32_INT32
#define TTYPE_UINT8_INT32_UINT8
#endif
#if defined(HAS_INT8)
#define TTYPE_UINT8_INT8_INT8 , (uint8_t, int8_t, int8_t)
#define TTYPE_UINT8_INT8_UINT8 , (uint8_t, int8_t, uint8_t)
#else
#define TTYPE_UINT8_INT8_INT8
#define TTYPE_UINT8_INT8_UINT8
#endif
#if defined(HAS_LONG)
#define TTYPE_UINT8_LONG_LONG , (uint8_t, sd::LongType, sd::LongType)
#define TTYPE_UINT8_LONG_UINT8 , (uint8_t, sd::LongType, uint8_t)
#else
#define TTYPE_UINT8_LONG_LONG
#define TTYPE_UINT8_LONG_UINT8
#endif
#else
#define TTYPE_UINT8_UINT8_UINT8
#endif

#if defined(HAS_UNSIGNEDLONG)
#define TTYPE_ND4JULONG_ND4JULONG_ND4JULONG , (uint64_t, uint64_t, uint64_t)
#if defined(HAS_BOOL)
#define TTYPE_UINT64_BOOL_UINT64 , (uint64_t, bool, uint64_t)
#else
#define TTYPE_UINT64_BOOL_UINT64
#endif
#else
#define TTYPE_ND4JULONG_ND4JULONG_ND4JULONG
#endif

#if defined(HAS_UINT16)
#define TTYPE_UINT16_UINT16_UINT16 , (uint16_t, uint16_t, uint16_t)
#if defined(HAS_BOOL)
#define TTYPE_UINT16_BOOL_UINT16 , (uint16_t, bool, uint16_t)
#else
#define TTYPE_UINT16_BOOL_UINT16
#endif
#else
#define TTYPE_UINT16_UINT16_UINT16
#endif

#if defined(HAS_UINT32)
#define TTYPE_UINT32_UINT32_UINT32 , (uint32_t, uint32_t, uint32_t)
#if defined(HAS_BOOL)
#define TTYPE_UINT32_BOOL_UINT32 , (uint32_t, bool, uint32_t)
#else
#define TTYPE_UINT32_BOOL_UINT32
#endif
#else
#define TTYPE_UINT32_UINT32_UINT32
#endif

#define SD_PAIRWISE_TYPES_LL_0 TTYPE_FLOAT16_FLOAT16_FLOAT16 TTYPE_FLOAT16_BOOL_FLOAT16
#define SD_PAIRWISE_TYPES_LL_1 TTYPE_FLOAT_FLOAT_FLOAT TTYPE_FLOAT_BOOL_FLOAT
#define SD_PAIRWISE_TYPES_LL_2 TTYPE_DOUBLE_DOUBLE_DOUBLE TTYPE_DOUBLE_BOOL_DOUBLE
#define SD_PAIRWISE_TYPES_LL_3 TTYPE_INT8_INT8_INT8 TTYPE_INT8_BOOL_INT8
#define SD_PAIRWISE_TYPES_LL_4 TTYPE_INT16_INT16_INT16 TTYPE_INT16_BOOL_INT16
#define SD_PAIRWISE_TYPES_LL_5 TTYPE_UINT8_UINT8_UINT8 TTYPE_UINT8_BOOL_UINT8
#define SD_PAIRWISE_TYPES_LL_6 TTYPE_INT32_INT32_INT32 TTYPE_INT32_BOOL_INT32
#define SD_PAIRWISE_TYPES_LL_7 TTYPE_BOOL_BOOL_BOOL
#define SD_PAIRWISE_TYPES_LL_8 TTYPE_LONG_LONG_LONG TTYPE_LONG_BOOL_LONG
#define SD_PAIRWISE_TYPES_LL_9 TTYPE_BFLOAT16_BFLOAT16_BFLOAT16 TTYPE_BFLOAT16_BOOL_BFLOAT16
#define SD_PAIRWISE_TYPES_LL_10 TTYPE_ND4JULONG_ND4JULONG_ND4JULONG TTYPE_UINT64_BOOL_UINT64
#define SD_PAIRWISE_TYPES_LL_11 TTYPE_UINT32_UINT32_UINT32 TTYPE_UINT32_BOOL_UINT32
#define SD_PAIRWISE_TYPES_LL_12 TTYPE_UINT16_UINT16_UINT16 TTYPE_UINT16_BOOL_UINT16
#endif

// TO SUPPORT THE CURRENT CMAKE GENERATION WE WILL MANUALLY ADD TYPES_$INDEX definitions

// for pairwise we are not using GET_ELEMENT
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_0) > 1
#define SD_PAIRWISE_TYPES_0 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_0)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_1) > 1
#define SD_PAIRWISE_TYPES_1 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_1)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_2) > 1
#define SD_PAIRWISE_TYPES_2 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_2)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_3) > 1
#define SD_PAIRWISE_TYPES_3 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_3)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_4) > 1
#define SD_PAIRWISE_TYPES_4 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_4)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_5) > 1
#define SD_PAIRWISE_TYPES_5 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_5)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_6) > 1
#define SD_PAIRWISE_TYPES_6 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_6)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_7) > 1
#define SD_PAIRWISE_TYPES_7 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_7)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_8) > 1
#define SD_PAIRWISE_TYPES_8 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_8)
#endif
#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_9) > 1
#define SD_PAIRWISE_TYPES_9 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_9)
#endif

#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_10) > 1
#define SD_PAIRWISE_TYPES_10 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_10)
#endif

#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_11) > 1
#define SD_PAIRWISE_TYPES_11 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_11)
#endif

#if COUNT_NARG(SD_PAIRWISE_TYPES_LL_12) > 1
#define SD_PAIRWISE_TYPES_12 SKIP_FIRST_COMMA(SD_PAIRWISE_TYPES_LL_12)
#endif

#if !defined(SD_PAIRWISE_TYPES_0) && !defined(SD_PAIRWISE_TYPES_1) && !defined(SD_PAIRWISE_TYPES_2) &&   \
    !defined(SD_PAIRWISE_TYPES_3) && !defined(SD_PAIRWISE_TYPES_4) && !defined(SD_PAIRWISE_TYPES_5) &&   \
    !defined(SD_PAIRWISE_TYPES_6) && !defined(SD_PAIRWISE_TYPES_7) && !defined(SD_PAIRWISE_TYPES_8) &&   \
    !defined(SD_PAIRWISE_TYPES_9) && !defined(SD_PAIRWISE_TYPES_10) && !defined(SD_PAIRWISE_TYPES_11) && \
    !defined(SD_PAIRWISE_TYPES_12)

#pragma message WARN("it will use pairwise(float32,float32,float32) and 'll be defined in SD_PAIRWISE_TYPES_0")
#define SD_PAIRWISE_TYPES_0 (float, float, float)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 0
#define SD_NUMERIC_TYPES_0 GET_ELEMENT(0, SD_NUMERIC_TYPES)
#endif
#if COUNT_NARG(SD_NUMERIC_TYPES) > 1
#define SD_NUMERIC_TYPES_1 GET_ELEMENT(1, SD_NUMERIC_TYPES)
#endif
#if COUNT_NARG(SD_NUMERIC_TYPES) > 2
#define SD_NUMERIC_TYPES_2 GET_ELEMENT(2, SD_NUMERIC_TYPES)
#endif
#if COUNT_NARG(SD_NUMERIC_TYPES) > 3
#define SD_NUMERIC_TYPES_3 GET_ELEMENT(3, SD_NUMERIC_TYPES)
#endif
#if COUNT_NARG(SD_NUMERIC_TYPES) > 4
#define SD_NUMERIC_TYPES_4 GET_ELEMENT(4, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 5
#define SD_NUMERIC_TYPES_5 GET_ELEMENT(5, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 6
#define SD_NUMERIC_TYPES_6 GET_ELEMENT(6, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 7
#define SD_NUMERIC_TYPES_7 GET_ELEMENT(7, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 8
#define SD_NUMERIC_TYPES_8 GET_ELEMENT(8, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 9
#define SD_NUMERIC_TYPES_9 GET_ELEMENT(9, SD_NUMERIC_TYPES)
#endif
#if COUNT_NARG(SD_NUMERIC_TYPES) > 10
#define SD_NUMERIC_TYPES_10 GET_ELEMENT(10, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_NUMERIC_TYPES) > 11
#define SD_NUMERIC_TYPES_11 GET_ELEMENT(11, SD_NUMERIC_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 0
#define SD_COMMON_TYPES_0 GET_ELEMENT(0, SD_COMMON_TYPES)
#endif
#if COUNT_NARG(SD_COMMON_TYPES) > 1
#define SD_COMMON_TYPES_1 GET_ELEMENT(1, SD_COMMON_TYPES)
#endif
#if COUNT_NARG(SD_COMMON_TYPES) > 2
#define SD_COMMON_TYPES_2 GET_ELEMENT(2, SD_COMMON_TYPES)
#endif
#if COUNT_NARG(SD_COMMON_TYPES) > 3
#define SD_COMMON_TYPES_3 GET_ELEMENT(3, SD_COMMON_TYPES)
#endif
#if COUNT_NARG(SD_COMMON_TYPES) > 4
#define SD_COMMON_TYPES_4 GET_ELEMENT(4, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 5
#define SD_COMMON_TYPES_5 GET_ELEMENT(5, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 6
#define SD_COMMON_TYPES_6 GET_ELEMENT(6, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 7
#define SD_COMMON_TYPES_7 GET_ELEMENT(7, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 8
#define SD_COMMON_TYPES_8 GET_ELEMENT(8, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 9
#define SD_COMMON_TYPES_9 GET_ELEMENT(9, SD_COMMON_TYPES)
#endif
#if COUNT_NARG(SD_COMMON_TYPES) > 10
#define SD_COMMON_TYPES_10 GET_ELEMENT(10, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 11
#define SD_COMMON_TYPES_11 GET_ELEMENT(11, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_COMMON_TYPES) > 12
#define SD_COMMON_TYPES_12 GET_ELEMENT(12, SD_COMMON_TYPES)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES) > 0
#define SD_INTEGER_TYPES_0 GET_ELEMENT(0, SD_INTEGER_TYPES)
#endif
#if COUNT_NARG(SD_INTEGER_TYPES) > 1
#define SD_INTEGER_TYPES_1 GET_ELEMENT(1, SD_INTEGER_TYPES)
#endif
#if COUNT_NARG(SD_INTEGER_TYPES) > 2
#define SD_INTEGER_TYPES_2 GET_ELEMENT(2, SD_INTEGER_TYPES)
#endif
#if COUNT_NARG(SD_INTEGER_TYPES) > 3
#define SD_INTEGER_TYPES_3 GET_ELEMENT(3, SD_INTEGER_TYPES)
#endif
#if COUNT_NARG(SD_INTEGER_TYPES) > 4
#define SD_INTEGER_TYPES_4 GET_ELEMENT(4, SD_INTEGER_TYPES)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES) > 5
#define SD_INTEGER_TYPES_5 GET_ELEMENT(5, SD_INTEGER_TYPES)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES) > 6
#define SD_INTEGER_TYPES_6 GET_ELEMENT(6, SD_INTEGER_TYPES)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES) > 7
#define SD_INTEGER_TYPES_7 GET_ELEMENT(7, SD_INTEGER_TYPES)
#endif

#if COUNT_NARG(SD_INTEGER_TYPES) > 8
#define SD_INTEGER_TYPES_8 GET_ELEMENT(8, SD_INTEGER_TYPES)
#endif

#if COUNT_NARG(SD_FLOAT_TYPES) > 0
#define SD_FLOAT_TYPES_0 GET_ELEMENT(0, SD_FLOAT_TYPES)
#endif
#if COUNT_NARG(SD_FLOAT_TYPES) > 1
#define SD_FLOAT_TYPES_1 GET_ELEMENT(1, SD_FLOAT_TYPES)
#endif
#if COUNT_NARG(SD_FLOAT_TYPES) > 2
#define SD_FLOAT_TYPES_2 GET_ELEMENT(2, SD_FLOAT_TYPES)
#endif
#if COUNT_NARG(SD_FLOAT_TYPES) > 3
#define SD_FLOAT_TYPES_3 GET_ELEMENT(3, SD_FLOAT_TYPES)
#endif

#if COUNT_NARG(SD_INDEXING_TYPES) > 0
#define SD_INDEXING_TYPES_0 GET_ELEMENT(0, SD_INDEXING_TYPES)
#endif
#if COUNT_NARG(SD_INDEXING_TYPES) > 1
#define SD_INDEXING_TYPES_1 GET_ELEMENT(1, SD_INDEXING_TYPES)
#endif


#define INSTANTIATE_NORM(a1, b1, FUNC_NAME, ARGS) \
    template FUNC_NAME<GET_SECOND(a1), GET_SECOND(b1)>ARGS;

// Callback macro
#define CALLBACK_INSTANTIATE_NORM(a1, b1, FUNC_NAME, ARGS) \
    INSTANTIATE_NORM(a1, b1, FUNC_NAME, ARGS)
#ifndef  SD_COPY
#define SD_COPY
namespace sd {
namespace  ops {
template <typename U, typename V>
static  void safe_copy(U* dest, const V* src, size_t count) {
  if constexpr (std::is_same<U, V>::value && std::is_trivially_copyable<U>::value) {
    memcpy(dest, src, count * sizeof(U));
  } else {
    std::copy(src, src + count, dest);
  }
}

#define INSTANTIATE_COPY(a1,b1,FUNC_NAME,ARGS) template void safe_copy<GET_SECOND(a1), GET_SECOND(b1)>( GET_SECOND(a1)* dest, const GET_SECOND(b1) * src, size_t count);
ITERATE_COMBINATIONS((SD_NUMERIC_TYPES), (SD_NUMERIC_TYPES), INSTANTIATE_COPY,safe_copy,;);

template <typename T>
static  void safe_zero(T* dest, size_t count) {
  if constexpr (std::is_trivially_copyable<T>::value) {
    // For trivially copyable types, we can use memset.
    memset(dest, 0, count * sizeof(T));
  } else {
    // Otherwise, default-construct each element.
    std::fill_n(dest, count, T());
  }
}

#define INSTANTIATE_ZERO(a1) template void safe_zero<GET_SECOND(a1)>( GET_SECOND(a1)* dest, size_t count);
ITERATE_LIST((SD_NUMERIC_TYPES), INSTANTIATE_ZERO)
}

}
#endif
