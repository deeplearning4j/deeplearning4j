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

/*

Intel bfloat16 data type, based on
https://software.intel.com/sites/default/files/managed/40/8b/bf16-hardware-numerics-definition-white-paper.pdf

*/

#ifndef __UTIL_TYPES_BFLOAT16__H__
#define __UTIL_TYPES_BFLOAT16__H__

#include <system/common.h>
#include <types/float16.h>

#include <cfloat>
#include <iosfwd>
#include <iostream>
#include <type_traits>
#include <limits>
#include <cmath>

#include "bfloat16.h"
#include "float16.h"

// Type trait for SIMD safety
template<typename T>
struct is_simd_native {
  static constexpr bool value = std::is_same<T, float>::value ||
                                std::is_same<T, double>::value ||
                                std::is_integral<T>::value;
};

template<typename T>
struct is_bfloat16_type {
  static constexpr bool value = false;
};

template <typename T>
struct isNumericType {
  static bool constexpr value = std::is_same<double, T>::value || std::is_same<float, T>::value ||
                                std::is_same<int, T>::value || std::is_same<unsigned int, T>::value ||
                                std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value ||
                                std::is_same<long int, T>::value || std::is_same<long unsigned int, T>::value ||
                                std::is_same<int8_t, T>::value || std::is_same<uint8_t, T>::value ||
                                std::is_same<int16_t, T>::value || std::is_same<uint16_t, T>::value ||
                                std::is_same<bool, T>::value || std::is_same<float16, T>::value;
};

// Forward declaration
struct bfloat16;

// Specialization for bfloat16
template<>
struct is_bfloat16_type<bfloat16> {
  static constexpr bool value = true;
};

struct alignas(2) bfloat16 {

 public:
  constexpr bfloat16(const bfloat16&) = default;
  constexpr bfloat16& operator=(const bfloat16&) = default;

  uint16_t _data;

  // Default constructor - can be constexpr since we use simple uint16_t
  SD_INLINE SD_HOST_DEVICE constexpr bfloat16() : _data(0) {}

  // Constexpr constructor from raw bits
  SD_INLINE SD_HOST_DEVICE constexpr explicit bfloat16(uint16_t raw_bits) : _data(raw_bits) {}

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE explicit bfloat16(const T& rhs) : _data(0) {
    *this = rhs;
  }

  // SIMD-safe float conversion
  SD_INLINE SD_HOST_DEVICE operator float() const {
    union {
      uint32_t i;
      float f;
    } u;
    u.i = static_cast<uint32_t>(_data) << 16;
    return u.f;
  }

  SD_INLINE SD_HOST_DEVICE explicit operator bool() const { return this->_data != 0; }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE explicit operator T() const {
    return static_cast<T>(static_cast<float>(*this));
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator=(const bool rhs) {
    *this = rhs ? 1.0f : 0.0f;
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator=(const float& rhs) {
    // Handle special cases first
#if defined(__CUDA_ARCH__)
    if (isnan(rhs)) {
      _data = 0x7FC0; // NaN pattern
      return *this;
    }
#else
    if (std::isnan(rhs)) {
      _data = 0x7FC0; // NaN pattern
      return *this;
    }
#endif

    union {
      float f;
      uint32_t i;
    } u;
    u.f = rhs;

    // Round to nearest even with proper bias
    uint32_t lsb = (u.i >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    u.i += rounding_bias;
    this->_data = static_cast<uint16_t>(u.i >> 16);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE bfloat16& operator=(const T& rhs) {
    *this = static_cast<float>(rhs);
    return *this;
  }

  // Comparison operators - SIMD compatible
  // Handle both positive and negative zero correctly by converting to float for comparison
  SD_INLINE SD_HOST_DEVICE friend bool operator==(const bfloat16& a, const bfloat16& b) {
    // Handle NaN cases first - NaN is never equal to anything, including itself
    if ((a._data & 0x7F80) == 0x7F80 && (a._data & 0x007F) != 0) return false; // a is NaN
    if ((b._data & 0x7F80) == 0x7F80 && (b._data & 0x007F) != 0) return false; // b is NaN

    // For all other values, including positive/negative zero, use float comparison
    return static_cast<float>(a) == static_cast<float>(b);
  }

  SD_INLINE SD_HOST_DEVICE friend bool operator!=(const bfloat16& a, const bfloat16& b) {
    return !(a == b);
  }

  SD_INLINE SD_HOST_DEVICE friend bool operator<(const bfloat16& a, const bfloat16& b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }

  SD_INLINE SD_HOST_DEVICE friend bool operator>(const bfloat16& a, const bfloat16& b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }

  SD_INLINE SD_HOST_DEVICE friend bool operator<=(const bfloat16& a, const bfloat16& b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }

  SD_INLINE SD_HOST_DEVICE friend bool operator>=(const bfloat16& a, const bfloat16& b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }

  // Arithmetic operators - using float intermediates for compatibility
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
  }

  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) - static_cast<float>(b));
  }

  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) * static_cast<float>(b));
  }

  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) / static_cast<float>(b));
  }

  // Mixed type operators
  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator+(const bfloat16& a, const T& b) {
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator+(const T& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator-(const bfloat16& a, const T& b) {
    return bfloat16(static_cast<float>(a) - static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator-(const T& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) - static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator*(const bfloat16& a, const T& b) {
    return bfloat16(static_cast<float>(a) * static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator*(const T& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) * static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator/(const bfloat16& a, const T& b) {
    return bfloat16(static_cast<float>(a) / static_cast<float>(b));
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bfloat16 operator/(const T& a, const bfloat16& b) {
    return bfloat16(static_cast<float>(a) / static_cast<float>(b));
  }

  // Mixed type comparison operators
  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator==(const bfloat16& a, const T& b) {
    return static_cast<float>(a) == static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator==(const T& a, const bfloat16& b) {
    return static_cast<float>(a) == static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator!=(const bfloat16& a, const T& b) {
    return static_cast<float>(a) != static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator!=(const T& a, const bfloat16& b) {
    return static_cast<float>(a) != static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator<(const bfloat16& a, const T& b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator<(const T& a, const bfloat16& b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator>(const bfloat16& a, const T& b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator>(const T& a, const bfloat16& b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator<=(const bfloat16& a, const T& b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator<=(const T& a, const bfloat16& b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator>=(const bfloat16& a, const T& b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE friend bool operator>=(const T& a, const bfloat16& b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }

  // Compound assignment operators
  SD_INLINE SD_HOST_DEVICE bfloat16& operator+=(const bfloat16& rhs) {
    *this = static_cast<float>(*this) + static_cast<float>(rhs);
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator-=(const bfloat16& rhs) {
    *this = static_cast<float>(*this) - static_cast<float>(rhs);
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator*=(const bfloat16& rhs) {
    *this = static_cast<float>(*this) * static_cast<float>(rhs);
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator/=(const bfloat16& rhs) {
    *this = static_cast<float>(*this) / static_cast<float>(rhs);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE bfloat16& operator+=(const T& rhs) {
    *this = static_cast<float>(*this) + static_cast<float>(rhs);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE bfloat16& operator-=(const T& rhs) {
    *this = static_cast<float>(*this) - static_cast<float>(rhs);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE bfloat16& operator*=(const T& rhs) {
    *this = static_cast<float>(*this) * static_cast<float>(rhs);
    return *this;
  }

  template <typename T, typename = typename std::enable_if<isNumericType<T>::value>::type>
  SD_INLINE SD_HOST_DEVICE bfloat16& operator/=(const T& rhs) {
    *this = static_cast<float>(*this) / static_cast<float>(rhs);
    return *this;
  }

  // Increment/decrement operators
  SD_INLINE SD_HOST_DEVICE bfloat16& operator++() {
    *this = static_cast<float>(*this) + 1.0f;
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16& operator--() {
    *this = static_cast<float>(*this) - 1.0f;
    return *this;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16 operator++(int) {
    bfloat16 tmp = *this;
    *this = static_cast<float>(*this) + 1.0f;
    return tmp;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16 operator--(int) {
    bfloat16 tmp = *this;
    *this = static_cast<float>(*this) - 1.0f;
    return tmp;
  }

  SD_INLINE SD_HOST_DEVICE bfloat16 operator-() const {
    bfloat16 result;
    result._data = _data ^ 0x8000; // Flip sign bit
    return result;
  }

  // Static utility methods - constexpr compatible
  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 min() {
    return bfloat16(static_cast<uint16_t>(0x0080)); // Smallest positive normal
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 lowest() {
    return bfloat16(static_cast<uint16_t>(0xFF7F)); // Most negative finite
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 max() {
    return bfloat16(static_cast<uint16_t>(0x7F7F)); // Largest positive finite
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 epsilon() {
    return bfloat16(static_cast<uint16_t>(0x3C00)); // Machine epsilon
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 round_error() {
    return bfloat16(static_cast<uint16_t>(0x3F00)); // 0.5f
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 infinity() {
    return bfloat16(static_cast<uint16_t>(0x7F80)); // Positive infinity
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 quiet_NaN() {
    return bfloat16(static_cast<uint16_t>(0x7FC0)); // Quiet NaN
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 signaling_NaN() {
    return bfloat16(static_cast<uint16_t>(0x7FA0)); // Signaling NaN
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 denorm_min() {
    return bfloat16(static_cast<uint16_t>(0x0001)); // Smallest positive subnormal
  }

  // Legacy aliases for compatibility
  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 eps() {
    return epsilon();
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 inf() {
    return infinity();
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 nan() {
    return quiet_NaN();
  }

  SD_INLINE SD_HOST_DEVICE static constexpr bfloat16 min_positive() {
    return denorm_min();
  }

  // Helper methods for template operations
  SD_INLINE SD_HOST_DEVICE static float as_float(const bfloat16& bf16) {
    return static_cast<float>(bf16);
  }

  SD_INLINE SD_HOST_DEVICE static bfloat16 from_float(float f) {
    return bfloat16(f);
  }
};

// Ensure proper alignment for SIMD operations
static_assert(sizeof(bfloat16) == 2, "bfloat16 must be 2 bytes");
static_assert(alignof(bfloat16) >= 2, "bfloat16 must be at least 2-byte aligned");

// Template specializations to ensure SIMD compatibility
namespace std {
template<>
struct is_arithmetic<bfloat16> : std::true_type {};

template<>
struct is_floating_point<bfloat16> : std::true_type {};

// std::numeric_limits specialization
template<>
struct numeric_limits<bfloat16> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_to_nearest;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;     // 7 mantissa bits + 1 implicit
  static constexpr int digits10 = 2;   // Decimal digits of precision
  static constexpr int max_digits10 = 4; // Maximum decimal digits needed for round-trip
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;  // Minimum binary exponent
  static constexpr int min_exponent10 = -37; // Minimum decimal exponent
  static constexpr int max_exponent = 128;   // Maximum binary exponent
  static constexpr int max_exponent10 = 38;  // Maximum decimal exponent

  static constexpr bfloat16 min() noexcept { return bfloat16::min(); }
  static constexpr bfloat16 lowest() noexcept { return bfloat16::lowest(); }
  static constexpr bfloat16 max() noexcept { return bfloat16::max(); }
  static constexpr bfloat16 epsilon() noexcept { return bfloat16::epsilon(); }
  static constexpr bfloat16 round_error() noexcept { return bfloat16::round_error(); }
  static constexpr bfloat16 infinity() noexcept { return bfloat16::infinity(); }
  static constexpr bfloat16 quiet_NaN() noexcept { return bfloat16::quiet_NaN(); }
  static constexpr bfloat16 signaling_NaN() noexcept { return bfloat16::signaling_NaN(); }
  static constexpr bfloat16 denorm_min() noexcept { return bfloat16::denorm_min(); }
};

// Utility functions for CUDA compatibility
#ifdef __CUDACC__
SD_INLINE SD_HOST_DEVICE bool isnan(const bfloat16& bf) {
  return (bf._data & 0x7F80) == 0x7F80 && (bf._data & 0x007F) != 0;
}

SD_INLINE SD_HOST_DEVICE bool isinf(const bfloat16& bf) {
  return (bf._data & 0x7F80) == 0x7F80 && (bf._data & 0x007F) == 0;
}

SD_INLINE SD_HOST_DEVICE bool isfinite(const bfloat16& bf) {
  return (bf._data & 0x7F80) != 0x7F80;
}
#endif

} // namespace std

#endif