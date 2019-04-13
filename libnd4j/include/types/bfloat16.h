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
  public:
    int16_t _data;
    /* constexpr */ local_def bfloat16() { _data = 0; }

    template <class T>
    local_def /*explicit*/ bfloat16(const T& rhs) {
      assign(rhs);
    }

//    local_def bfloat16(float rhs) {
//      assign(rhs);
//    }
//
//    local_def bfloat16(double rhs) {
//      assign(rhs);
//    }

    local_def operator float() const {
      int32_t temp = this->_data << 16; //((sign << 31) | (exponent << 23) | mantissa);

      return *reinterpret_cast<float*>(&temp);
    }

    local_def explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
    local_def explicit operator unsigned long long() const { return static_cast<unsigned long long>(static_cast<float>(*this)); }
    local_def explicit operator int16_t() const { return static_cast<int16_t>(static_cast<float>(*this)); }
    local_def explicit operator uint8_t() const { return static_cast<uint8_t>(static_cast<float>(*this)); }
    local_def explicit operator int8_t() const { return static_cast<int8_t>(static_cast<float>(*this)); }
    local_def explicit operator int() const { return static_cast<int>(static_cast<float>(*this)); }
    local_def explicit operator Nd4jLong() const { return static_cast<Nd4jLong>(static_cast<float>(*this)); }
    local_def explicit operator bool() const { return this->_data == 0 ? false : true; }
    local_def explicit operator float16() const { return static_cast<float16>(static_cast<float>(*this)); }

    template <class T>
    local_def bfloat16& operator=(const T& rhs) { assign(rhs); return *this; }

    local_def void assign(unsigned int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    local_def void assign(int rhs) {
      // may be a better way ?
      assign((float)rhs);
    }

    local_def void assign(double rhs) {
      assign((float)rhs);
    }

    local_def void assign(long long rhs) {
        assign((float)rhs);
    }

    local_def void assign(long int rhs) {
        assign((float)rhs);
    }

    local_def void assign(long unsigned int rhs) {
        assign((float)rhs);
    }

    local_def void assign(unsigned short rhs) {
        assign((float)rhs);
    }

    local_def void assign(float16 rhs) {
      assign((float)rhs);
    }

    local_def void assign(long long unsigned int rhs) {
        assign((float)rhs);
    }

    local_def void assign(float rhs) {
#ifdef __CUDACC__
      if(::isnan(rhs)) {
          _data = bfloat16::nan();
          return;
      }
#endif
      auto x = *reinterpret_cast<int32_t*>(&rhs);
      uint32_t lsb = (x >> 16) & 1;
      uint32_t rounding_bias = 0x7fff + lsb;
      x += rounding_bias;
      this->_data = static_cast<int16_t>(x >> 16);
    }

    local_def void assign(const bfloat16& rhs) {
      _data = rhs._data;
    }

    local_def bfloat16& operator+=(bfloat16 rhs) { assign((float)(*this) + (float)rhs); return *this; }

    local_def bfloat16& operator-=(bfloat16 rhs) { assign((float)*this - (float)rhs); return *this; }

    local_def bfloat16& operator*=(bfloat16 rhs) { assign((float)*this * (float)rhs); return *this; }

    local_def bfloat16& operator/=(bfloat16 rhs) { assign((float)*this / (float)rhs); return *this; }

    local_def bfloat16& operator+=(float rhs) { assign((float)*this + rhs); return *this; }

    local_def bfloat16& operator-=(float rhs) { assign((float)*this - rhs); return *this; }

    local_def bfloat16& operator*=(float rhs) { assign((float)*this * rhs); return *this; }

    local_def bfloat16& operator/=(float rhs) { assign((float)*this / rhs); return *this; }

    local_def bfloat16& operator++() { *this += 1.f; return *this; }

    local_def bfloat16& operator--() { *this -= 1.f; return *this; }

    local_def bfloat16 operator++(int i) { *this += i; return *this; }

    local_def bfloat16 operator--(int i) { *this -= i; return *this; }

    local_def std::ostream& operator<<(std::ostream& os) {
        os << static_cast<float>(*this);
        return os;
    }
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

    local_def bool  operator==(const bfloat16& a, const bfloat16& b) { return (a._data == b._data); }

//    template <class T>
//    local_def bool  operator==(const bfloat16& a, const T& b) { return (a == (bfloat16) b); }

    local_def bool  operator!=(const bfloat16& a, const bfloat16& b) { return !(a == b); }
//
    local_def bool  operator<(const bfloat16& a, const bfloat16& b) { return (float)a < (float)b; }

  local_def bool  operator>(const bfloat16& a, const bfloat16& b) { return (float)a > (float)b; }

    template <class T>
    local_def bool  operator>(const bfloat16& a, const T& b) { return (float)a > (float)b; }

    local_def bool  operator<=(const bfloat16& a, const bfloat16& b) { return (float)a <= (float)b; }
    template <class T>
    local_def bool  operator<=(const bfloat16& a, const T& b) { return (float)a <= (float)b; }

    local_def bool  operator>=(const bfloat16& a, const bfloat16& b) { return (float)a >= (float)b; }

    local_def bfloat16 operator+(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a + (float)b); }    
    local_def bfloat16 operator-(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a - (float)b); }
    local_def bfloat16 operator*(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a * (float)b); }
    local_def bfloat16 operator/(const bfloat16& a, const bfloat16& b) { return bfloat16((float)a / (float)b); }    
//

    local_def bfloat16 operator+(const bfloat16& a,            const double& b)             { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const float& b)              { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const float16& b)              { return static_cast<bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
    local_def bfloat16 operator+(const float16& a,             const bfloat16& b)              { return static_cast<bfloat16>(static_cast<float>(a) + static_cast<float>(b)); }
    local_def bfloat16 operator+(const bfloat16& a,            const int& b)                { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const unsigned int& b)       { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const long long& b)          { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const unsigned long long& b) { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const long int& b)           { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const bool& b)               { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const int8_t& b)             { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const uint8_t& b)            { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const int16_t& b)            { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const bfloat16& a,            const long unsigned int& b)  { return a + static_cast<bfloat16>(b); }
    local_def bfloat16 operator+(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const bool& a,               const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }
    local_def bfloat16 operator+(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) + b; }

    local_def bfloat16 operator-(const bfloat16& a,            const double& b)             { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const float& b)              { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const float16& b)            { return static_cast<bfloat16>(static_cast<float>(a) - static_cast<float>(b)); }
    local_def bfloat16 operator-(const float16& a,             const bfloat16& b)           { return static_cast<bfloat16>(static_cast<float>(a) - static_cast<float>(b)); }
    local_def bfloat16 operator-(const bfloat16& a,            const int& b)                { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const unsigned int& b)       { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const long long& b)          { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const unsigned long long& b) { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const long int& b)           { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const bool& b)               { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const int8_t& b)             { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const uint8_t& b)            { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const int16_t& b)            { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const bfloat16& a,            const long unsigned int& b)  { return a - static_cast<bfloat16>(b); }
    local_def bfloat16 operator-(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const bool& a,               const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }
    local_def bfloat16 operator-(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) - b; }

    local_def bfloat16 operator/(const bfloat16& a,            const double& b)             { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const float& b)              { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const float16& b)              { return static_cast<bfloat16>((float)a / (float)b); }
    local_def bfloat16 operator/(const float16& a,             const bfloat16& b)              { return static_cast<bfloat16>((float)a / (float)b); }
    local_def bfloat16 operator/(const bfloat16& a,            const int& b)                { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const unsigned int& b)       { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const long long& b)          { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const unsigned long long& b) { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const long int& b)           { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const bool& b)               { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const int8_t& b)             { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const uint8_t& b)            { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const int16_t& b)            { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const bfloat16& a,            const long unsigned int& b)  { return a / static_cast<bfloat16>(b); }
    local_def bfloat16 operator/(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const bool& a,               const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
    local_def bfloat16 operator/(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) / b; }
  
    local_def bfloat16 operator*(const bfloat16& a,            const double& b)             { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const float& b)              { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const float16& b)            { return static_cast<bfloat16>((float)a * (float)b); }
    local_def bfloat16 operator*(const float16& a,             const bfloat16& b)           { return static_cast<bfloat16>((float)a * (float)b); }
    local_def bfloat16 operator*(const bfloat16& a,            const int& b)                { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const unsigned int& b)       { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const long long& b)          { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const unsigned long long& b) { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const long int& b)           { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const bool& b)               { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const int8_t& b)             { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const uint8_t& b)            { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const int16_t& b)            { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const bfloat16& a,            const long unsigned int& b)  { return a * static_cast<bfloat16>(b); }
    local_def bfloat16 operator*(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const bool& a,               const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }
    local_def bfloat16 operator*(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) * b; }    

    local_def bool operator==(const bfloat16& a,            const float& b)              { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const float16& b)              { return (float)a == (float)(b); }
    local_def bool operator==(const bfloat16& a,            const double& b)             { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const int& b)                { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const unsigned int& b)       { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const long long& b)          { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const unsigned long long& b) { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const long int& b)           { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const int8_t& b)             { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const uint8_t& b)            { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const int16_t& b)            { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const bool& b)               { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bfloat16& a,            const long unsigned int& b)  { return a == static_cast<bfloat16>(b); }
    local_def bool operator==(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }
    local_def bool operator==(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) == b; }

    local_def bool operator!=(const bfloat16& a,            const float& b)              { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const float16& b)            { return (float)a != static_cast<float>(b); }
    local_def bool operator!=(const bfloat16& a,            const double& b)             { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const int& b)                { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const unsigned int& b)       { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const long long& b)          { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const unsigned long long& b) { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const long int& b)           { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const int8_t& b)             { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const uint8_t& b)            { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const int16_t& b)            { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const bool& b)               { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bfloat16& a,            const long unsigned int& b)  { return a != static_cast<bfloat16>(b); }
    local_def bool operator!=(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }
    local_def bool operator!=(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) != b; }

    local_def bool operator<(const bfloat16& a,            const float& b)              { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const float16& b)              { return (float)a < static_cast<float>(b); }
    local_def bool operator<(const bfloat16& a,            const double& b)             { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const int& b)                { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const unsigned int& b)       { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const long long& b)          { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const unsigned long long& b) { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const long int& b)           { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const int8_t& b)             { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const uint8_t& b)            { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const int16_t& b)            { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const bool& b)               { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bfloat16& a,            const long unsigned int& b)  { return a < static_cast<bfloat16>(b); }
    local_def bool operator<(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }
    local_def bool operator<(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) < b; }

    local_def bool operator>(const bfloat16& a,            const float& b)              { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const float16& b)            { return (float)a > static_cast<float>(b); }
    local_def bool operator>(const bfloat16& a,            const double& b)             { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const int& b)                { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const unsigned int& b)       { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const long long& b)          { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const unsigned long long& b) { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const long int& b)           { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const int8_t& b)             { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const uint8_t& b)            { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const int16_t& b)            { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const bool& b)               { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bfloat16& a,            const long unsigned int& b)  { return a > static_cast<bfloat16>(b); }
    local_def bool operator>(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    local_def bool operator>(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) > b; }
    
    local_def bool operator<=(const bfloat16& a,            const float& b)              { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const float16& b)            { return (float)a <= static_cast<float>(b); }
    local_def bool operator<=(const bfloat16& a,            const double& b)             { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const int& b)                { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const unsigned int& b)       { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const long long& b)          { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const unsigned long long& b) { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const long int& b)           { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const int8_t& b)             { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const uint8_t& b)            { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const int16_t& b)            { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const bool& b)               { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bfloat16& a,            const long unsigned int& b)  { return a <= static_cast<bfloat16>(b); }
    local_def bool operator<=(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }
    local_def bool operator<=(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) <= b; }

    local_def bool operator>=(const bfloat16& a,            const float& b)              { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const float16& b)            { return (float)a >= static_cast<float>(b); }
    local_def bool operator>=(const bfloat16& a,            const double& b)             { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const int& b)                { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const unsigned int& b)       { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const long long& b)          { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const unsigned long long& b) { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const long int& b)           { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const int8_t& b)             { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const uint8_t& b)            { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const int16_t& b)            { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const bool& b)               { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bfloat16& a,            const long unsigned int& b)  { return a >= static_cast<bfloat16>(b); }
    local_def bool operator>=(const bool&    a,            const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const int8_t&  a,            const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const uint8_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const int16_t& a,            const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const int& a,                const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const unsigned int& a,       const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const long long& a,          const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const unsigned long long& a, const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const long int& a,           const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const float& a,              const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const double& a,             const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
    local_def bool operator>=(const long unsigned int& a,  const bfloat16& b)            { return static_cast<bfloat16>(a) >= b; }
   

    local_def std::ostream& operator<<(std::ostream &os, const bfloat16 &f) {
        os << static_cast<float>(f);
        return os;
    }


  local_def bfloat16 /* constexpr */ operator+(const bfloat16& h) { return h; }

  local_def bfloat16 operator - (const bfloat16& h) {
    auto temp = h._data;
    temp ^= 0x8000;
    bfloat16 t;
    t._data = temp;
    return t;
}

// WARNING: this implementation only for avoid cyclic references between float16 and bfloat16 types.
local_def void float16::assign(const bfloat16& rhs) {
  assign((float)rhs);
}

//}   // namespace

#endif