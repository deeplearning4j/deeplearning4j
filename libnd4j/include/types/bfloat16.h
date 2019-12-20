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

/*

 Intel bfloat16 data type, based on https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf

 */

#ifndef __UTIL_TYPES_BFLOAT16__H__
#define __UTIL_TYPES_BFLOAT16__H__

#include <cfloat>
#include <iosfwd>
#include <iostream>

// support for half precision conversion
#ifdef __INTEL_COMPILER
#include <emmintrin.h>
#endif


#ifdef __CUDACC__
#define local_def inline __host__ __device__
#elif _MSC_VER
#define local_def inline
#elif __clang__
#define local_def inline
#elif __GNUC__
#define local_def inline
#endif

//namespace nd4j
//{
  struct bfloat16
  {
    private:
        template<typename T>
        struct isNumericType { static bool const value = std::is_same<double, T>::value || std::is_same<float, T>::value || std::is_same<int, T>::value || std::is_same<unsigned int, T>::value || std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value || std::is_same<long int, T>::value || std::is_same<long unsigned int, T>::value || std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value || std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value || std::is_same<bool, T>::value || std::is_same<float16, T>::value; };
        // struct isNumericType { static bool const value = std::is_same<double, typename std::decay<T>::type>::value || std::is_same<float, typename std::decay<T>::type>::value || std::is_same<int, typename std::decay<T>::type>::value || std::is_same<unsigned int, typename std::decay<T>::type>::value || std::is_same<long long, typename std::decay<T>::type>::value || std::is_same<unsigned long long, typename std::decay<T>::type>::value || std::is_same<long int, typename std::decay<T>::type>::value || std::is_same<long unsigned int, typename std::decay<T>::type>::value || std::is_same<int8_t, typename std::decay<T>::type>::value || std::is_same<uint8_t, typename std::decay<T>::type>::value || std::is_same<int16_t, typename std::decay<T>::type>::value || std::is_same<uint16_t, typename std::decay<T>::type>::value || std::is_same<bool, typename std::decay<T>::type>::value || std::is_same<float16, T>::value;; };

    public:
        int16_t _data;

        local_def bfloat16() {
            _data = 0;
        }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16(const T& rhs) {
            *this = rhs;
        }

        local_def operator float() const {
            int32_t temp = this->_data << 16; //((sign << 31) | (exponent << 23) | mantissa);
            return *reinterpret_cast<float*>(&temp);
        }

        local_def explicit operator bool() const {
            return this->_data == 0 ? false : true;
        }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def explicit operator T() const {
            return static_cast<T>(static_cast<float>(*this));
        }

        local_def bfloat16& operator=(const bool rhs) {
            *this = (float)rhs ? 1.f: 0.f;
            return *this;
        }

        local_def bfloat16& operator=(const float& rhs) {
            #ifdef __CUDACC__
            if(::isnan(rhs)) {
                _data = bfloat16::nan();
                return *this;
            }
            #endif
            auto x = *reinterpret_cast<int32_t*>(& const_cast<float&>(rhs));
            uint32_t lsb = (x >> 16) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            x += rounding_bias;
            this->_data = static_cast<int16_t>(x >> 16);

            return *this;
        }

        local_def bfloat16& operator=(const bfloat16& rhs) {
            _data = rhs._data;
            return *this;
        }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16& operator=(const T& rhs) {
            *this = (float)rhs;
            return *this;
        }

        local_def friend bool operator==(const bfloat16& a, const bfloat16& b) { return (a._data == b._data); }
        local_def friend bool operator!=(const bfloat16& a, const bfloat16& b) { return !(a == b); }
        local_def friend bool operator<(const bfloat16& a, const bfloat16& b) { return (float)a < (float)b; }
        local_def friend bool operator>(const bfloat16& a, const bfloat16& b) { return (float)a > (float)b; }
        local_def friend bool operator<=(const bfloat16& a, const bfloat16& b) { return (float)a <= (float)b; }
        local_def friend bool operator>=(const bfloat16& a, const bfloat16& b) { return (float)a >= (float)b; }

        local_def friend bfloat16 operator+(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a + (float)b); }
        local_def friend bfloat16 operator-(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a - (float)b); }
        local_def friend bfloat16 operator*(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a * (float)b); }
        local_def friend bfloat16 operator/(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a / (float)b); }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator+(const bfloat16& a, const T& b) { return a + static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator+(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) + b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator-(const bfloat16& a, const T& b) { return a - static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator-(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) - b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator*(const bfloat16& a, const T& b) { return a * static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator*(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) * b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator/(const bfloat16& a, const T& b) { return a / static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bfloat16 operator/(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) / b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator==(const bfloat16& a, const T& b) { return a == static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator==(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) == b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator!=(const bfloat16& a, const T& b) { return a != static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator!=(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) != b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator<(const bfloat16& a, const T& b) { return a < static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator<(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) < b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator>(const bfloat16& a, const T& b) { return a > static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator>(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) > b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator<=(const bfloat16& a, const T& b) { return a <= static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator<=(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) <= b; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator>=(const bfloat16& a, const T& b) { return a >= static_cast<bfloat16>(b); }
        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def friend bool operator>=(const T& a, const bfloat16& b) { return static_cast<bfloat16>(a) >= b; }

        local_def bfloat16& operator+=(bfloat16 rhs) { *this = (float)(*this) + (float)rhs; return *this; }

        local_def bfloat16& operator-=(bfloat16 rhs) { *this = (float)(*this) - (float)rhs; return *this; }

        local_def bfloat16& operator*=(bfloat16 rhs) { *this = (float)(*this) * (float)rhs; return *this; }

        local_def bfloat16& operator/=(bfloat16 rhs) { *this = (float)(*this) / (float)rhs; return *this; }

         template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16& operator+=(const T& rhs) { *this = *this + rhs; return *this; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16& operator-=(const T& rhs) { *this = *this - rhs; return *this; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16& operator*=(const T& rhs) { *this = *this * rhs; return *this; }

        template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
        local_def bfloat16& operator/=(const T& rhs) { *this = *this / rhs; return *this; }

        local_def bfloat16& operator++() { *this = (float)*this + (float)1.f; return *this; }

        local_def bfloat16& operator--() { *this = (float)*this - (float)1.f; return *this; }

        local_def bfloat16 operator++(int) { *this = (float)*this + (float)1.f; return *this; }

        local_def bfloat16 operator--(int) { *this = (float)*this - (float)1.f; return *this; }

        local_def bfloat16 operator-() const {
            return 0.f - (float)*this;
        }



        // local_def std::ostream& operator<<(std::ostream& os) {
        //     os << static_cast<float>(*this);
        //     return os;
        // }
        local_def static bfloat16 min() {
            bfloat16 res;
            res._data = 0xFF7F;
            return res;
        }
        local_def static bfloat16 max() {
            bfloat16 res;
            res._data = 0x7F7F;
            return res;

        }
        local_def static bfloat16 eps() {
            bfloat16 res;
            res._data = 0x3C00;
            return res;
        }

        local_def static bfloat16 inf() {
            bfloat16 res;
            res._data = 0x3C00;
            return res;
        }

        local_def static bfloat16 nan() {
            bfloat16 res;
            res._data = 0x7FC0;
            return res;
        }
};



//     local_def std::ostream& operator<<(std::ostream &os, const bfloat16 &f) {
//         os << static_cast<float>(f);
//         return os;
//     }


//   local_def bfloat16 /* constexpr */ operator+(const bfloat16& h) { return h; }

//   local_def bfloat16 operator - (const bfloat16& h) {
//     auto temp = h._data;
//     temp ^= 0x8000;
//     bfloat16 t;
//     t._data = temp;
//     return t;
// }

// WARNING: this implementation only for avoid cyclic references between float16 and bfloat16 types.
// local_def void float16::assign(const bfloat16& rhs) {
//   assign((float)rhs);
// }

//}   // namespace

#endif