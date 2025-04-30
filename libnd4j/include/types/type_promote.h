//
// Created by agibsonccc on 11/22/24.
//

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


#ifndef LIBND4J_TYPE_PROMOTE_H
#define LIBND4J_TYPE_PROMOTE_H
#include <types/types.h>
/*
 * Type Ranking System:
type_rank template and its specializations assign an integer rank to each supported type.
This ranking helps in determining the "promoted" type when combining different types.
Type Promotion Traits:
promote_type and promote_type3 templates determine the promoted type between two or three types based on their ranks.
Type Name System:
type_name template and its specializations provide a string representation for each supported type.
Helper Functions and Macros:
promote function template converts a value to the promoted type.
Macros like INSTANTIATE_PROMOTE and CALLBACK_INSTANTIATE_PROMOTE help in instantiating the promote function for different type combinations.
PROMOTE_ARGS macro handles function arguments correctly.
 */

// Type ranking system
template<typename T> struct type_rank;

// Type ranking system
template<typename T> struct type_rank;

#if defined(HAS_BOOL)
template<> struct type_rank<bool>        : std::integral_constant<int, 0> {};
#endif

#if defined(HAS_INT8)
template<> struct type_rank<int8_t>      : std::integral_constant<int, 1> {};
#endif

#if defined(HAS_UINT8)
template<> struct type_rank<uint8_t>     : std::integral_constant<int, 1> {};
#endif

#if defined(HAS_INT16)
template<> struct type_rank<int16_t>     : std::integral_constant<int, 2> {};
#endif

#if defined(HAS_UINT16)
template<> struct type_rank<uint16_t>    : std::integral_constant<int, 2> {};
#endif

#if defined(HAS_INT32)
template<> struct type_rank<int32_t>     : std::integral_constant<int, 3> {};
#endif

#if defined(HAS_UINT32)
template<> struct type_rank<uint32_t>    : std::integral_constant<int, 3> {};
#endif


template<> struct type_rank<int64_t>     : std::integral_constant<int, 4> {};
template<> struct type_rank<uint64_t>    : std::integral_constant<int, 4> {};


#if defined(HAS_FLOAT16)
template<> struct type_rank<float16>     : std::integral_constant<int, 5> {};
#endif

#if defined(HAS_BFLOAT16)
template<> struct type_rank<bfloat16>    : std::integral_constant<int, 5> {};
#endif

#if defined(HAS_FLOAT32)
template<> struct type_rank<float>       : std::integral_constant<int, 6> {};
#endif

#if defined(HAS_DOUBLE)
template<> struct type_rank<double>      : std::integral_constant<int, 7> {};
#endif


// promote_type trait
template<typename T1, typename T2>
struct promote_type {
  using type = typename std::conditional<
      (type_rank<T1>::value >= type_rank<T2>::value),
      T1,
      T2
      >::type;
};

// promote function template
template <typename Type1, typename Type2, typename ValueType>
typename promote_type<Type1, Type2>::type promote(ValueType value) {
  return static_cast<typename promote_type<Type1, Type2>::type>(value);
}

// promote_type3 trait for three types
template<typename T1, typename T2, typename T3>
struct promote_type3 {
  using type = typename promote_type<
      typename promote_type<T1, T2>::type,
      T3
      >::type;
};


// Primary template for type_name - undefined to trigger a compile-time error for unsupported types
template<typename T>
struct type_name;

#if defined(HAS_BOOL)
template<> struct type_name<bool>        { static const char* get() { return "bool"; } };
#endif

#if defined(HAS_INT8)
template<> struct type_name<int8_t>      { static const char* get() { return "int8_t"; } };
#endif

#if defined(HAS_UINT8)
template<> struct type_name<uint8_t>     { static const char* get() { return "uint8_t"; } };
#endif

#if defined(HAS_INT16)
template<> struct type_name<int16_t>     { static const char* get() { return "int16_t"; } };
#endif

#if defined(HAS_UINT16)
template<> struct type_name<uint16_t>    { static const char* get() { return "uint16_t"; } };
#endif

#if defined(HAS_INT32)
template<> struct type_name<int32_t>     { static const char* get() { return "int32_t"; } };
#endif

#if defined(HAS_UINT32)
template<> struct type_name<uint32_t>    { static const char* get() { return "uint32_t"; } };
#endif

#if defined(HAS_INT64)
template<> struct type_name<int64_t> { static const char* get() { return "int64_t"; } };

template<> struct type_name<long long int> { static const char* get() { return "long long int"; } };
#endif


#if defined(HAS_UINT64)
template<> struct type_name<uint64_t>    { static const char* get() { return "uint64_t"; } };
#endif

#if defined(HAS_FLOAT16)
template<> struct type_name<float16>     { static const char* get() { return "float16"; } };
#endif

#if defined(HAS_BFLOAT16)
template<> struct type_name<bfloat16>    { static const char* get() { return "bfloat16"; } };
#endif

#if defined(HAS_FLOAT32)
template<> struct type_name<float>       { static const char* get() { return "float"; } };
#endif

#if defined(HAS_DOUBLE)
template<> struct type_name<double>      { static const char* get() { return "double"; } };
#endif

// Helper function to get type name
template<typename T>
const char* get_type_name() {
  return type_name<T>::get();
}



// Macro to instantiate the promote function
#define INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS) \
    template sd::math::promote_type<GET_SECOND(a1), GET_SECOND(b1)>::type \
    sd::math::promote<GET_SECOND(a1), GET_SECOND(b1), GET_SECOND(a1)>(GET_SECOND(a1));

// Callback macro
#define CALLBACK_INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS) \
    INSTANTIATE_PROMOTE(a1, b1, FUNC_NAME, ARGS)

#endif  // LIBND4J_TYPE_PROMOTE_H
