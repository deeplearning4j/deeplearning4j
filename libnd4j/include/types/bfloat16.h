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

struct bfloat16i {
public:
    unsigned short x;
    inline unsigned short * getXP() {
        return &this->x;
    }

    inline unsigned short getX() const  {
        return this->x;
    }
//    void assign(bfloat16i const& another) {
//        x = another.x;
//    }
//    void assign(unsigned short internalRep) {
//       x = internalRep;
//    }
};

local_def float cpu_bfloat16i2float(bfloat16i h) {
    unsigned int temp = h.getX() << 16; //((sign << 31) | (exponent << 23) | mantissa);

    return *reinterpret_cast<float*>(&temp);
}

local_def bfloat16i cpu_float2bfloat16i_rn(float f) {
    bfloat16i ret;

    unsigned x = *reinterpret_cast<unsigned int*>(&f); // uint32_t should be used 
    *ret.getXP() = x >> 16;

    return ret;
}

//namespace nd4j
//{
  struct bfloat16
  {
  public:
    bfloat16i data;
    /* constexpr */ local_def bfloat16() { *data.getXP() = 0; }

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
      return cpu_bfloat16i2float(data);
    }

    //    local_def operator double() const { return (float)*this; }

    local_def operator bfloat16i() const { return data; }
    local_def operator float16() const { return (float16)((float)*this); }
/*
    local_def unsigned short getx() const { return (const unsigned short)data.getX(); }
    local_def bfloat16& setx(unsigned short x) { *data.getXP() = x; return *this; }
*/
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
        *data.getXP() = rhs;
    }

    local_def void assign(float16 const& rhs) {
      assign((float)rhs);
    }

    local_def void assign(long long unsigned int rhs) {
        assign((float)rhs);
    }

    local_def void assign(float rhs) {

//  #if defined(DEBUG) && defined (CPU_ONLY)
//      if (rhs > BFLOAT16_MAX || rhs < -BFLOAT16_MAX) {
//        LOG(WARNING) << "Overflow: " << rhs;
//      } else if (rhs != 0.F && rhs < HLF_MIN && rhs > -HLF_MIN) {
//        LOG(WARNING) << "Underflow: " << rhs;
//      }
//  #endif
      data = cpu_float2bfloat16i_rn(rhs);
    }

    local_def void assign(const bfloat16i& rhs) {
        *data.getXP() = rhs.getX();
    }

#ifdef __CUDACC__
    local_def void assign(const bfloat16& rhs) {
      //data = rhs;
      data.assign(rhs);
    }
#endif

    local_def void assign(const bfloat16& rhs) {
      data = rhs.data;
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

    // Utility contants
    static const bfloat16 zero;
    static const bfloat16 one;
    static const bfloat16 minus_one;
  };

    local_def bool  operator==(const bfloat16& a, const bfloat16& b) { return ((bfloat16i) a.data).getX() == ((bfloat16i)b.data).getX(); }

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


  //template <class T>
  //local_def bfloat16 operator+(const bfloat16& a, const T& b) { return bfloat16((float)a + (float)b); }

  //template <class T>
  //local_def bfloat16 operator+(const T& a, const bfloat16& b) { return bfloat16((float)a + (float)b); }


  //template <class T>
  //local_def bfloat16 operator-(const bfloat16& a, const T& b) { return bfloat16((float)a - (float)b); }


//  template <class T>
//  local_def int operator&(const T& a, const bfloat16& b) { return a & (float)b; }

  //template <class T>
  //local_def bfloat16 operator*(const bfloat16& a, const T& b) { return bfloat16((float)a * (float)b); }

  //template <class T>
  //local_def bfloat16 operator*(const T& a, const bfloat16& b) { return bfloat16((float)a * (float)b); }


  // this operator is special case, for division by larger types, like int, long long etc
  //template <class T>
  //local_def bfloat16 operator/(const bfloat16& a, const T& b) { return bfloat16((float)a / (float)b); }


  local_def bfloat16 /* constexpr */ operator+(const bfloat16& h) { return h; }

  local_def bfloat16 operator - (const bfloat16& h) {
    const bfloat16i * tmp = &h.data;
    unsigned short temp = tmp->getX();
    temp ^= 0x8000;
    return bfloat16(temp);
}

// WARNING: this implementation only for avoid cyclic references between float16 and bfloat16 types.
local_def void float16::assign(const bfloat16& rhs) {
  assign((float)rhs);
}

#ifdef __CUDACC__
  local_def int isnan(const bfloat16& h)  { return ishnan_(((bfloat16i)h.data).getX()); }

  local_def int isinf(const bfloat16& h) { return ishinf_(((bfloat16i)h.data).getX()); }
#endif

///  std::ostream& operator << (std::ostream& s, const bfloat16&);


//}   // namespace caffe

#endif