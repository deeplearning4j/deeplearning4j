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

#ifndef LIBND4J_FLOAT16_H
#define LIBND4J_FLOAT16_H
#include <system/common.h>

#include <cfloat>
#include <iosfwd>
#include <iostream>
#include <type_traits>

#if defined(__INTEL_COMPILER) || defined(SD_F16C)
#include <immintrin.h>
#endif

struct bfloat16;

#if defined(__CUDACC__)
#include <cuda_fp16.h>

#if CUDA_VERSION_MAJOR != 8
// CUDA_9 and above

struct ihalf : public __half {
public:
 SD_HOST_DEVICE ihalf() : __half() {
   //
 }

 SD_INLINE SD_HOST_DEVICE unsigned short* getXP() { return &this->__x; }

 SD_INLINE SD_HOST_DEVICE unsigned short getX() const { return this->__x; }

 SD_INLINE SD_HOST_DEVICE void assign(const __half f) { this->__x = ((__half_raw*)&f)->x; }
};

#else
struct ihalf : public __half {
public:
 SD_HOST_DEVICE ihalf() : __half() {
   //
 }

 SD_INLINE SD_HOST_DEVICE unsigned short* getXP() { return &this->x; }

 SD_INLINE SD_HOST_DEVICE unsigned short getX() const { return this->x; }

 SD_INLINE SD_HOST_DEVICE void assign(const __half f) { this->x = ((__half*)&f)->x; }
};
#endif  // CUDA_8

#else
struct alignas(2) __half {
public:
 unsigned short x;
 inline unsigned short* getXP() { return &this->x; }

 inline unsigned short getX() const { return this->x; }
};

typedef __half half;
typedef __half ihalf;

#endif  // CUDA

static SD_INLINE SD_HOST_DEVICE int ishnan_(unsigned short h) { return (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) != 0; }

static SD_INLINE SD_HOST_DEVICE int ishinf_(unsigned short h) { return (h & 0x7c00U) == 0x7c00U && (h & 0x03ffU) == 0; }

static SD_INLINE SD_HOST_DEVICE int ishequ_(unsigned short x, unsigned short y) {
 return ishnan_(x) == 0 && ishnan_(y) == 0 && x == y;
}

static SD_INLINE SD_HOST_DEVICE unsigned short hneg(unsigned short h) {
 h ^= 0x8000U;
 return h;
}

#if defined(__INTEL_COMPILER) || defined(SD_F16C)
//_Pragma("omp declare simd") inline
SD_INLINE SD_HOST_DEVICE float cpu_ihalf2float(ihalf h) { return _cvtsh_ss(h.getX()); }
#else
SD_INLINE SD_HOST_DEVICE float cpu_ihalf2float(ihalf h) {
 unsigned sign = ((h.getX() >> 15) & 1);
 unsigned exponent = ((h.getX() >> 10) & 0x1f);
 unsigned mantissa = ((h.getX() & 0x3ff) << 13);

 if (exponent == 0x1f) { /* NaN or Inf */
   mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
   exponent = 0xff;
 } else if (!exponent) { /* Denorm or Zero */
   if (mantissa) {
     unsigned int msb;
     exponent = 0x71;
     do {
       msb = (mantissa & 0x400000);
       mantissa <<= 1; /* normalize */
       --exponent;
     } while (!msb);
     mantissa &= 0x7fffff; /* 1.mantissa is implicit */
   }
 } else {
   exponent += 0x70;
 }

 union {
   int i;
   float f;
 } u;
 u.i = ((sign << 31) | (exponent << 23) | mantissa);
 return u.f;
}
#endif

#if defined(__INTEL_COMPILER) || defined(SD_F16C)
//_Pragma("omp declare simd") inline
SD_INLINE SD_HOST_DEVICE ihalf cpu_float2ihalf_rn(float f) {
 ihalf ret;
 ret.x = _cvtss_sh(f, 0);
 return ret;
}

#else
SD_INLINE SD_HOST_DEVICE ihalf cpu_float2ihalf_rn(float f) {
 ihalf ret;

 union {
   float f;
   int i;
 } u;
 u.f = f;
 unsigned x = u.i;
 unsigned u_val = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
 unsigned sign, exponent, mantissa;

 // Get rid of +NaN/-NaN case first.
 if (u_val > 0x7f800000) {
   *ret.getXP() = 0x7fffU;
   return ret;
 }

 sign = ((x >> 16) & 0x8000);

 // Get rid of +Inf/-Inf, +0/-0.
 if (u_val > 0x477fefff) {
   *ret.getXP() = sign | 0x7c00U;
   return ret;
 }
 if (u_val < 0x33000001) {
   *ret.getXP() = (sign | 0x0000);
   return ret;
 }

 exponent = ((u_val >> 23) & 0xff);
 mantissa = (u_val & 0x7fffff);

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

struct alignas(2) float16 {
private:
 template <typename T>
 struct isNumericType {
   static bool const value = std::is_same<double, T>::value || std::is_same<float, T>::value ||
                             std::is_same<int, T>::value || std::is_same<unsigned int, T>::value ||
                             std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value ||
                             std::is_same<long int, T>::value || std::is_same<long unsigned int, T>::value ||
                             std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value ||
                             std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value ||
                             std::is_same<bool, T>::value;
 };

public:
 constexpr float16(const float16&) = default;

 ihalf data;
 SD_INLINE SD_HOST_DEVICE float16() { *data.getXP() = 0; }

 template <typename T,
           typename = typename std::enable_if<isNumericType<T>::value || std::is_same<bfloat16, T>::value>::type>
 SD_INLINE SD_HOST_DEVICE explicit float16(const T& rhs) {
   *this = rhs;
 }

 SD_INLINE SD_HOST_DEVICE float16(const half& rhs) {
#ifdef __CUDACC__
   data.assign(rhs);
#endif
 }

 SD_INLINE SD_HOST_DEVICE operator float() const {
#if defined(__CUDA_ARCH__) 
   return __half2float(data);
#else
   return cpu_ihalf2float(data);
#endif
 }

 SD_INLINE SD_HOST_DEVICE float16 operator|(int rhs) const {
   float16 result;
   *result.data.getXP() = static_cast<unsigned short>(this->data.getX() | static_cast<unsigned short>(rhs));
   return result;
 }

 // Bitwise OR with float16
 SD_INLINE SD_HOST_DEVICE float16 operator|(const float16& rhs) const {
   float16 result;
   *result.data.getXP() = static_cast<unsigned short>(this->data.getX() | rhs.data.getX());
   return result;
 }

 // Friend function for int | float16
 SD_INLINE SD_HOST_DEVICE friend float16 operator|(int lhs, const float16& rhs) {
   float16 result;
   *result.data.getXP() = static_cast<unsigned short>(static_cast<unsigned short>(lhs) | rhs.data.getX());
   return result;
 }

 SD_INLINE SD_HOST_DEVICE explicit operator bool() const { return static_cast<float>(*this) != 0.0f; }

 SD_INLINE SD_HOST_DEVICE explicit operator half() const { return data; }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE explicit operator T() const {
   return static_cast<T>(static_cast<float>(*this));
 }

 SD_INLINE SD_HOST_DEVICE float16& operator=(const float& rhs) {
#if defined(__CUDA_ARCH__) 
   auto t = __float2half_rn(rhs);
   auto b = *(data.getXP());

#if CUDA_VERSION_MAJOR == 8
   *(data.getXP()) = t;
#else
   data.assign(t);
#endif

#else
   data = cpu_float2ihalf_rn(rhs);
#endif

   return *this;
 }

 // Correct operator overload for bitwise AND
 SD_INLINE SD_HOST_DEVICE float16 operator&(const float16& rhs) const {
   float16 result;
   *result.data.getXP() = static_cast<unsigned short>(this->data.getX() & rhs.data.getX());
   return result;
 }

 // Additional overload for int if needed
 SD_INLINE SD_HOST_DEVICE float16 operator&(int rhs) const {
   float16 result;
   *result.data.getXP() = static_cast<unsigned short>(this->data.getX() & static_cast<unsigned short>(rhs));
   return result;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator=(const unsigned short rhs) {
   *data.getXP() = rhs;
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator=(const bool rhs) {
   *this = rhs ? 1.0f : 0.0f;
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator=(const ihalf& rhs) {
   *data.getXP() = ((ihalf)rhs).getX();
   return *this;
 }

#if defined(__CUDACC__) 
 SD_INLINE SD_HOST_DEVICE float16& operator=(const half& rhs) {
   data.assign(rhs);
   return *this;
 }
#endif

 SD_INLINE SD_HOST_DEVICE float16& operator=(const float16& rhs) {
   data = rhs.data;
   return *this;
 }

 template <typename T,
           typename = typename std::enable_if<isNumericType<T>::value || std::is_same<bfloat16, T>::value>::type>
 SD_INLINE SD_HOST_DEVICE float16& operator=(const T& rhs) {
   *this = static_cast<float>(rhs);
   return *this;
 }

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator==(const float16& a, const float16& b) { return __hequ(a.data, b.data); }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator==(const float16& a, const float16& b) {
   return ishequ_(((ihalf)a.data).getX(), ((ihalf)b.data).getX());
 }
#endif

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator!=(const float16& a, const float16& b) {
   return !(__hequ(a.data, b.data));
 }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator!=(const float16& a, const float16& b) { return !(a == b); }
#endif

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator<(const float16& a, const float16& b) { return __hlt(a.data, b.data); }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator<(const float16& a, const float16& b) { return static_cast<float>(a) < static_cast<float>(b); }
#endif

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator>(const float16& a, const float16& b) { return __hgt(a.data, b.data); }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator>(const float16& a, const float16& b) { return static_cast<float>(a) > static_cast<float>(b); }
#endif

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator<=(const float16& a, const float16& b) { return __hle(a.data, b.data); }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator<=(const float16& a, const float16& b) { return static_cast<float>(a) <= static_cast<float>(b); }
#endif

#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend bool operator>=(const float16& a, const float16& b) { return __hge(a.data, b.data); }
#else
 SD_INLINE SD_HOST_DEVICE friend bool operator>=(const float16& a, const float16& b) { return static_cast<float>(a) >= static_cast<float>(b); }
#endif

// Arithmetic operators - optimized for native CUDA half support when available
#ifdef SD_NATIVE_HALFS
 SD_INLINE SD_HOST_DEVICE friend float16 operator+(const float16& a, const float16& b) {
   return __hadd(a.data, b.data);
 }

 SD_INLINE SD_HOST_DEVICE friend float16 operator-(const float16& a, const float16& b) {
   return __hsub(a.data, b.data);
 }

 SD_INLINE SD_HOST_DEVICE friend float16 operator*(const float16& a, const float16& b) {
   return __hmul(a.data, b.data);
 }

 SD_INLINE SD_HOST_DEVICE friend float16 operator/(const float16& a, const float16& b) {
#if CUDA_VERSION_MAJOR == 8
   return hdiv(a.data, b.data);
#else
   return __hdiv(a.data, b.data);
#endif
 }
#else
 SD_INLINE SD_HOST_DEVICE friend float16 operator+(const float16& a, const float16& b) {
   return float16(static_cast<float>(a) + static_cast<float>(b));
 }
 
 SD_INLINE SD_HOST_DEVICE friend float16 operator-(const float16& a, const float16& b) {
   return float16(static_cast<float>(a) - static_cast<float>(b));
 }
 
 SD_INLINE SD_HOST_DEVICE friend float16 operator*(const float16& a, const float16& b) {
   return float16(static_cast<float>(a) * static_cast<float>(b));
 }
 
 SD_INLINE SD_HOST_DEVICE friend float16 operator/(const float16& a, const float16& b) {
   return float16(static_cast<float>(a) / static_cast<float>(b));
 }
#endif

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator+(const float16& a, const T& b) {
   return float16(static_cast<float>(a) + static_cast<float>(b));
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator+(const T& a, const float16& b) {
   return float16(static_cast<float>(a) + static_cast<float>(b));
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator-(const float16& a, const T& b) {
   return float16(static_cast<float>(a) - static_cast<float>(b));
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator-(const T& a, const float16& b) {
   return float16(static_cast<float>(a) - static_cast<float>(b));
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator*(const float16& a, const T& b) {
   return float16(static_cast<float>(a) * static_cast<float>(b));
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator*(const T& a, const float16& b) {
   return float16(static_cast<float>(a) * static_cast<float>(b));
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator/(const float16& a, const T& b) {
   return float16(static_cast<float>(a) / static_cast<float>(b));
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend float16 operator/(const T& a, const float16& b) {
   return float16(static_cast<float>(a) / static_cast<float>(b));
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator==(const float16& a, const T& b) {
   return static_cast<float>(a) == static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator==(const T& a, const float16& b) {
   return static_cast<float>(a) == static_cast<float>(b);
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator!=(const float16& a, const T& b) {
   return static_cast<float>(a) != static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator!=(const T& a, const float16& b) {
   return static_cast<float>(a) != static_cast<float>(b);
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator<(const float16& a, const T& b) {
   return static_cast<float>(a) < static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator<(const T& a, const float16& b) {
   return static_cast<float>(a) < static_cast<float>(b);
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator>(const float16& a, const T& b) {
   return static_cast<float>(a) > static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator>(const T& a, const float16& b) {
   return static_cast<float>(a) > static_cast<float>(b);
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator<=(const float16& a, const T& b) {
   return static_cast<float>(a) <= static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator<=(const T& a, const float16& b) {
   return static_cast<float>(a) <= static_cast<float>(b);
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator>=(const float16& a, const T& b) {
   return static_cast<float>(a) >= static_cast<float>(b);
 }
 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE friend bool operator>=(const T& a, const float16& b) {
   return static_cast<float>(a) >= static_cast<float>(b);
 }

 // Compound assignment operators
 SD_INLINE SD_HOST_DEVICE float16& operator+=(const float16& rhs) {
   *this = static_cast<float>(*this) + static_cast<float>(rhs);
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator-=(const float16& rhs) {
   *this = static_cast<float>(*this) - static_cast<float>(rhs);
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator*=(const float16& rhs) {
   *this = static_cast<float>(*this) * static_cast<float>(rhs);
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator/=(const float16& rhs) {
   *this = static_cast<float>(*this) / static_cast<float>(rhs);
   return *this;
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE float16& operator+=(const T& rhs) {
   *this = static_cast<float>(*this) + static_cast<float>(rhs);
   return *this;
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE float16& operator-=(const T& rhs) {
   *this = static_cast<float>(*this) - static_cast<float>(rhs);
   return *this;
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE float16& operator*=(const T& rhs) {
   *this = static_cast<float>(*this) * static_cast<float>(rhs);
   return *this;
 }

 template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
 SD_INLINE SD_HOST_DEVICE float16& operator/=(const T& rhs) {
   *this = static_cast<float>(*this) / static_cast<float>(rhs);
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator++() {
   *this = static_cast<float>(*this) + 1.0f;
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16& operator--() {
   *this = static_cast<float>(*this) - 1.0f;
   return *this;
 }

 SD_INLINE SD_HOST_DEVICE float16 operator++(int) {
   float16 tmp = *this;
   *this = static_cast<float>(*this) + 1.0f;
   return tmp;
 }

 SD_INLINE SD_HOST_DEVICE float16 operator--(int) {
   float16 tmp = *this;
   *this = static_cast<float>(*this) - 1.0f;
   return tmp;
 }

 SD_INLINE SD_HOST_DEVICE float16 operator-() const { 
   float16 result;
   *result.data.getXP() = data.getX() ^ 0x8000U; // Flip sign bit
   return result;
 }

 // Helper methods
 SD_INLINE SD_HOST_DEVICE static float as_float(const float16& f16) {
   return static_cast<float>(f16);
 }

 SD_INLINE SD_HOST_DEVICE static float16 from_float(float f) {
   return float16(f);
 }
};

// Ensure proper alignment for SIMD operations
static_assert(sizeof(float16) == 2, "float16 must be 2 bytes");
static_assert(alignof(float16) >= 2, "float16 must be at least 2-byte aligned");

// Template specializations to ensure SIMD compatibility
namespace std {
  template<>
  struct is_arithmetic<float16> : std::true_type {};
  
  template<>
  struct is_floating_point<float16> : std::true_type {};
}

#ifdef __CUDACC__
SD_INLINE SD_HOST_DEVICE int isnan(const float16& h) { return ishnan_(((ihalf)h.data).getX()); }

SD_INLINE SD_HOST_DEVICE int isinf(const float16& h) { return ishinf_(((ihalf)h.data).getX()); }
#endif

#endif