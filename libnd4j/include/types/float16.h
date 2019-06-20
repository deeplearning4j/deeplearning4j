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

#ifndef LIBND4J_FLOAT16_H
#define LIBND4J_FLOAT16_H

#include <cfloat>
#include <iosfwd>
#include <iostream>
#include <pointercast.h>
#if defined(__INTEL_COMPILER) || defined(__F16C__)
    #include <immintrin.h>
#endif


struct bfloat16;

#ifdef __CUDACC__
#include <cuda_fp16.h>

#ifndef CUDA_8
// CUDA_9 and above

struct ihalf : public __half {
    public:
        __host__ __device__ ihalf() : half() {
            //
        }

        inline __host__ __device__ unsigned short * getXP() {
           return &this->__x;
        }

        inline __host__ __device__ unsigned short getX() const  {
            return this->__x;
        }

        inline __host__ __device__ void assign(const half f) {
            this->__x = ((__half_raw *) &f)->x;
        }
};

#else
struct ihalf : public __half {
    public:
        __host__ __device__ ihalf() : half() {
            //
        }

        inline __host__ __device__ unsigned short * getXP() {
            return &this->x;
        }

        inline __host__ __device__ unsigned short getX() const {
            return this->x;
        }

        inline __host__ __device__ void assign(const half f) {
            this->x = ((__half *) &f)->x;
        }
};
#endif // CUDA_8

#else
struct __half {
public:
    unsigned short x;
    inline unsigned short * getXP() {
        return &this->x;
    }

    inline unsigned short getX() const  {
        return this->x;
    }
};

typedef __half half;
typedef __half ihalf;


#endif // CUDA

#ifdef __CUDACC__
#define local_def inline __host__ __device__
#elif _MSC_VER
#define local_def inline
#elif __clang__
#define local_def inline
#elif __GNUC__
#define local_def inline
#endif


static local_def int ishnan_(unsigned short h) {
     return (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) != 0;
}

static local_def int ishinf_(unsigned short h) {
    return (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) == 0;
}

static local_def int ishequ_(unsigned short x, unsigned short y) {
    return ishnan_(x) == 0 && ishnan_(y) == 0 && x == y;
}

static local_def unsigned short hneg(unsigned short h) {
    h ^= 0x8000U;
    return h;
}


#if defined(__INTEL_COMPILER) || defined(__F16C__)
//_Pragma("omp declare simd") inline
local_def  float cpu_ihalf2float(ihalf h) {
    return _cvtsh_ss(h.getX());
}
#else
local_def float cpu_ihalf2float(ihalf h) {
    unsigned sign = ((h.getX() >> 15) & 1);
    unsigned exponent = ((h.getX() >> 10) & 0x1f);
    unsigned mantissa = ((h.getX() & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));
}
#endif

#if defined(__INTEL_COMPILER) || defined(__F16C__)
//_Pragma("omp declare simd") inline
local_def ihalf cpu_float2ihalf_rn(float f) {
    ihalf ret;
    ret.x = _cvtss_sh(f, 0);
    return ret;
}

#else
local_def ihalf cpu_float2ihalf_rn(float f)
{
    ihalf ret;

    unsigned x = *((int*)(void*)(&f));
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        *ret.getXP() = 0x7fffU;
        return ret;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        *ret.getXP() = sign | 0x7c00U;
        return ret;
    }
    if (u < 0x33000001) {
        *ret.getXP() = (sign | 0x0000);
        return ret;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    *ret.getXP() = (sign | (exponent << 10) | mantissa);

    return ret;
}
#endif

  struct float16
  {
  public:
    ihalf data;
    local_def float16() { *data.getXP() = 0; }

    template <class T>
    local_def float16(const T& rhs) {
      assign(rhs);
    }

    local_def operator float() const {
#ifdef __CUDA_ARCH__
      return __half2float(data);
#else
      return cpu_ihalf2float(data);
#endif
    }

    local_def explicit operator double() const {
        return static_cast<double>(static_cast<float>(*this));
    }

    local_def explicit operator Nd4jLong() const {
        return static_cast<Nd4jLong>(static_cast<float>(*this));
    }

    local_def explicit operator int() const {
        return static_cast<int>(static_cast<float>(*this));
    }

    local_def explicit operator bool() const {
        return static_cast<float>(*this) > 0.0f;
    }

    local_def explicit operator int16_t() const {
        return static_cast<int16_t>(static_cast<float>(*this));
    }

    local_def explicit operator uint16_t() const {
        return static_cast<uint16_t>(static_cast<float>(*this));
    }

    local_def explicit operator uint8_t() const {
        return static_cast<uint8_t>(static_cast<float>(*this));
    }

    local_def explicit operator int8_t() const {
        return static_cast<int8_t>(static_cast<float>(*this));
    }

    local_def operator half() const { return data; }

    template <class T>
    local_def float16& operator=(const T& rhs) { assign(rhs); return *this; }

    local_def void assign(unsigned int rhs) {
      assign((float)rhs);
    }

    local_def void assign(int rhs) {
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

    local_def void assign(const bool rhs) {
        assign(rhs ? 1.0f : 0.0f);
    }

    local_def void assign(long unsigned int rhs) {
        assign((float)rhs);
    }

    local_def void assign(unsigned short rhs) {
        *data.getXP() = rhs;
    }

    local_def void assign(long long unsigned int rhs) {
        assign((float)rhs);
    }

    local_def void assign(float rhs) {
#ifdef __CUDA_ARCH__
      auto t = __float2half_rn(rhs);
      auto b = *(data.getXP());

#ifdef CUDA_8
      *(data.getXP()) = t;
#else
      data.assign(t);
#endif

#else
      data = cpu_float2ihalf_rn(rhs);
#endif
    }

    local_def void assign(const ihalf& rhs) {
        *data.getXP() = ((ihalf) rhs).getX();
    }

    local_def void assign(const bfloat16& rhs);

#ifdef __CUDACC__
    local_def void assign(const half& rhs) {
      data.assign(rhs);
    }
#endif

    local_def void assign(const float16& rhs) {
      data = rhs.data;
    }

    local_def float16& operator+=(float16 rhs) { assign((float)*this + rhs); return *this; }

    local_def float16& operator-=(float16 rhs) { assign((float)*this - rhs); return *this; }

    local_def float16& operator*=(float16 rhs) { assign((float)*this * rhs); return *this; }

    local_def float16& operator/=(float16 rhs) { assign((float)*this / rhs); return *this; }

    local_def float16& operator+=(float rhs) { assign((float)*this + rhs); return *this; }

    local_def float16& operator-=(float rhs) { assign((float)*this - rhs); return *this; }

    local_def float16& operator*=(float rhs) { assign((float)*this * rhs); return *this; }

    local_def float16& operator/=(float rhs) { assign((float)*this / rhs); return *this; }

    local_def float16& operator++() { assign(*this + 1.f); return *this; }

    local_def float16& operator--() { assign(*this - 1.f); return *this; }

    local_def float16 operator++(int i) { assign(*this + (float)i); return *this; }

    local_def float16 operator--(int i) { assign(*this - (float)i); return *this; }

    local_def std::ostream& operator<<(std::ostream& os) {
        os << static_cast<float>(*this);
        return os;
    }
  };


#ifdef NATIVE_HALFS
    local_def bool  operator==(const float16& a, const float16& b) { return __hequ(a.data, b.data); }
#else
    local_def bool  operator==(const float16& a, const float16& b) { return ishequ_(((ihalf) a.data).getX(), ((ihalf)b.data).getX()); }
#endif

#ifdef NATIVE_HALFS
    local_def bool  operator!=(const float16& a, const float16& b) { return !(__hequ(a.data, b.data)); }
#else
    local_def bool  operator!=(const float16& a, const float16& b) { return !(a == b); }
#endif

#ifdef NATIVE_HALFS
    local_def bool  operator<(const float16& a, const float16& b) { return __hlt(a.data, b.data); }
#else
    local_def bool  operator<(const float16& a, const float16& b) { return (float)a < (float)b; }
#endif

#ifdef NATIVE_HALFS
  local_def bool  operator>(const float16& a, const float16& b) { return __hgt(a.data, b.data); }
#else
  local_def bool  operator>(const float16& a, const float16& b) { return (float)a > (float)b; }
#endif

    template <class T>
    local_def bool  operator>(const float16& a, const T& b) { return (float)a > (float)b; }

#ifdef NATIVE_HALFS
    local_def bool  operator<=(const float16& a, const float16& b) { return __hle(a.data, b.data); }
#else
    local_def bool  operator<=(const float16& a, const float16& b) { return (float)a <= (float)b; }
#endif
    template <class T>
    local_def bool  operator<=(const float16& a, const T& b) { return (float)a <= (float)b; }

#ifdef NATIVE_HALFS
    local_def bool  operator>=(const float16& a, const float16& b) { return __hge(a.data, b.data); }
#else
    local_def bool  operator>=(const float16& a, const float16& b) { return (float)a >= (float)b; }
#endif

#ifdef NATIVE_HALFS
    local_def float16 operator+(const float16& a, const float16& b) { return __hadd(a.data, b.data); }

    local_def float16 operator-(const float16& a, const float16& b) { return __hsub(a.data, b.data); }

    local_def float16 operator*(const float16& a, const float16& b) { return __hmul(a.data, b.data); }

    local_def float16 operator/(const float16& a, const float16& b) { 
        #ifdef CUDA_8
            return hdiv(a.data, b.data); 
        #else
            return __hdiv(a.data, b.data); 
        #endif
    }
#else
    local_def float16 operator+(const float16& a, const float16& b) { return float16((float)a + (float)b); }    
    local_def float16 operator-(const float16& a, const float16& b) { return float16((float)a - (float)b); }
    local_def float16 operator*(const float16& a, const float16& b) { return float16((float)a * (float)b); }
    local_def float16 operator/(const float16& a, const float16& b) { return float16((float)a / (float)b); }    
#endif

    local_def float16 operator+(const float16& a,            const double& b)             { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const float& b)              { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const int& b)                { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const unsigned int& b)       { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const long long& b)          { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const unsigned long long& b) { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const long int& b)           { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const bool& b)               { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const int8_t& b)             { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const uint8_t& b)            { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const int16_t& b)            { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const uint16_t& b)            { return a + static_cast<float16>(b); }
    local_def float16 operator+(const float16& a,            const long unsigned int& b)  { return a + static_cast<float16>(b); }
    local_def float16 operator+(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const uint16_t& a,            const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const bool& a,               const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const int& a,                const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const long long& a,          const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const long int& a,           const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const float& a,              const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const double& a,             const float16& b)            { return static_cast<float16>(a) + b; }
    local_def float16 operator+(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) + b; }

    local_def float16 operator-(const float16& a,            const double& b)             { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const float& b)              { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const int& b)                { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const unsigned int& b)       { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const long long& b)          { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const unsigned long long& b) { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const long int& b)           { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const bool& b)               { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const int8_t& b)             { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const uint8_t& b)            { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const int16_t& b)            { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const uint16_t& b)            { return a - static_cast<float16>(b); }
    local_def float16 operator-(const float16& a,            const long unsigned int& b)  { return a - static_cast<float16>(b); }
    local_def float16 operator-(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const bool& a,               const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const int& a,                const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const long long& a,          const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const long int& a,           const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const float& a,              const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const double& a,             const float16& b)            { return static_cast<float16>(a) - b; }
    local_def float16 operator-(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) - b; }

    local_def float16 operator/(const float16& a,            const double& b)             { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const float& b)              { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const int& b)                { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const unsigned int& b)       { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const long long& b)          { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const unsigned long long& b) { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const long int& b)           { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const bool& b)               { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const int8_t& b)             { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const uint8_t& b)            { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const int16_t& b)            { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const uint16_t& b)            { return a / static_cast<float16>(b); }
    local_def float16 operator/(const float16& a,            const long unsigned int& b)  { return a / static_cast<float16>(b); }
    local_def float16 operator/(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const bool& a,               const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const int& a,                const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const long long& a,          const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const long int& a,           const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const float& a,              const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const double& a,             const float16& b)            { return static_cast<float16>(a) / b; }
    local_def float16 operator/(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) / b; }
  
    local_def float16 operator*(const float16& a,            const double& b)             { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const float& b)              { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const int& b)                { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const unsigned int& b)       { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const long long& b)          { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const unsigned long long& b) { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const long int& b)           { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const bool& b)               { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const int8_t& b)             { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const uint8_t& b)            { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const int16_t& b)            { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const uint16_t& b)            { return a * static_cast<float16>(b); }
    local_def float16 operator*(const float16& a,            const long unsigned int& b)  { return a * static_cast<float16>(b); }
    local_def float16 operator*(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const bool& a,               const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const int& a,                const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const long long& a,          const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const long int& a,           const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const float& a,              const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const double& a,             const float16& b)            { return static_cast<float16>(a) * b; }
    local_def float16 operator*(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) * b; }    

    local_def bool operator==(const float16& a,            const float& b)              { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const double& b)             { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const int& b)                { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const unsigned int& b)       { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const long long& b)          { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const unsigned long long& b) { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const long int& b)           { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const int8_t& b)             { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const uint8_t& b)            { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const int16_t& b)            { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const uint16_t& b)            { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const bool& b)               { return a == static_cast<float16>(b); }
    local_def bool operator==(const float16& a,            const long unsigned int& b)  { return a == static_cast<float16>(b); }
    local_def bool operator==(const bool&    a,            const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const int& a,                const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const long long& a,          const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const long int& a,           const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const float& a,              const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const double& a,             const float16& b)            { return static_cast<float16>(a) == b; }
    local_def bool operator==(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) == b; }

    local_def bool operator!=(const float16& a,            const float& b)              { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const double& b)             { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const int& b)                { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const unsigned int& b)       { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const long long& b)          { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const unsigned long long& b) { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const long int& b)           { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const int8_t& b)             { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const uint8_t& b)            { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const int16_t& b)            { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const uint16_t& b)            { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const bool& b)               { return a != static_cast<float16>(b); }
    local_def bool operator!=(const float16& a,            const long unsigned int& b)  { return a != static_cast<float16>(b); }
    local_def bool operator!=(const bool&    a,            const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const int& a,                const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const long long& a,          const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const long int& a,           const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const float& a,              const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const double& a,             const float16& b)            { return static_cast<float16>(a) != b; }
    local_def bool operator!=(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) != b; }

    local_def bool operator<(const float16& a,            const float& b)              { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const double& b)             { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const int& b)                { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const unsigned int& b)       { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const long long& b)          { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const unsigned long long& b) { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const long int& b)           { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const int8_t& b)             { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const uint8_t& b)            { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const int16_t& b)            { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const uint16_t& b)            { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const bool& b)               { return a < static_cast<float16>(b); }
    local_def bool operator<(const float16& a,            const long unsigned int& b)  { return a < static_cast<float16>(b); }
    local_def bool operator<(const bool&    a,            const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const int& a,                const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const long long& a,          const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const long int& a,           const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const float& a,              const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const double& a,             const float16& b)            { return static_cast<float16>(a) < b; }
    local_def bool operator<(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) < b; }

    local_def bool operator>(const float16& a,            const float& b)              { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const double& b)             { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const int& b)                { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const unsigned int& b)       { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const long long& b)          { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const unsigned long long& b) { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const long int& b)           { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const int8_t& b)             { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const uint8_t& b)            { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const int16_t& b)            { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const uint16_t& b)            { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const bool& b)               { return a > static_cast<float16>(b); }
    local_def bool operator>(const float16& a,            const long unsigned int& b)  { return a > static_cast<float16>(b); }
    local_def bool operator>(const bool&    a,            const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const int& a,                const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const long long& a,          const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const long int& a,           const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const float& a,              const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const double& a,             const float16& b)            { return static_cast<float16>(a) > b; }
    local_def bool operator>(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) > b; }
    
    local_def bool operator<=(const float16& a,            const float& b)              { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const double& b)             { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const int& b)                { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const unsigned int& b)       { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const long long& b)          { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const unsigned long long& b) { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const long int& b)           { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const int8_t& b)             { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const uint8_t& b)            { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const int16_t& b)            { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const uint16_t& b)            { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const bool& b)               { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const float16& a,            const long unsigned int& b)  { return a <= static_cast<float16>(b); }
    local_def bool operator<=(const bool&    a,            const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const int& a,                const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const long long& a,          const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const long int& a,           const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const float& a,              const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const double& a,             const float16& b)            { return static_cast<float16>(a) <= b; }
    local_def bool operator<=(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) <= b; }

    local_def bool operator>=(const float16& a,            const float& b)              { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const double& b)             { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const int& b)                { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const unsigned int& b)       { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const long long& b)          { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const unsigned long long& b) { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const long int& b)           { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const int8_t& b)             { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const uint8_t& b)            { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const int16_t& b)            { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const uint16_t& b)            { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const bool& b)               { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const float16& a,            const long unsigned int& b)  { return a >= static_cast<float16>(b); }
    local_def bool operator>=(const bool&    a,            const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const int8_t&  a,            const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const uint8_t& a,            const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const int16_t& a,            const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const uint16_t& a,           const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const int& a,                const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const unsigned int& a,       const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const long long& a,          const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const unsigned long long& a, const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const long int& a,           const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const float& a,              const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const double& a,             const float16& b)            { return static_cast<float16>(a) >= b; }
    local_def bool operator>=(const long unsigned int& a,  const float16& b)            { return static_cast<float16>(a) >= b; }
   

    local_def std::ostream& operator<<(std::ostream &os, const float16 &f) {
        os << static_cast<float>(f);
        return os;
    }

  local_def float16 operator+(const float16& h) { return h; }

  local_def float16 operator - (const float16& h) {
    const ihalf * tmp = &h.data;
    return float16(hneg(tmp->getX()));
}

#ifdef __CUDACC__
  local_def int isnan(const float16& h)  { return ishnan_(((ihalf)h.data).getX()); }

  local_def int isinf(const float16& h) { return ishinf_(((ihalf)h.data).getX()); }
#endif

  std::ostream& operator << (std::ostream& s, const float16&);

#endif
