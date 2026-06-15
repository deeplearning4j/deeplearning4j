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
// FP8 types: E4M3FN and E5M2
//
// E4M3FN: 1 sign, 4 exponent, 3 mantissa bits. Bias=7. Range ±448. No Inf (NaN = 0x7F/0xFF).
// E5M2:   1 sign, 5 exponent, 2 mantissa bits. Bias=15. Range ±57344. Has Inf and NaN.
//

#ifndef LIBND4J_FLOAT8_H
#define LIBND4J_FLOAT8_H

#include <system/op_boilerplate.h>
#include <cmath>
#include <cstring>

// CUTLASS interop: on CUDA builds with CUTLASS, include cutlass float8 types
#if defined(HAVE_CUTLASS) && HAVE_CUTLASS && defined(__CUDACC__)
#include <cute/numeric/numeric_types.hpp>
#endif

namespace sd {

// ============================================================================
// E4M3FN format: 1-sign, 4-exp (bias=7), 3-mantissa, no Inf, NaN=0x7F/0xFF
// ============================================================================

typedef struct {
  unsigned char x;
} __quarter_e4m3;

typedef __quarter_e4m3 quarter_e4m3;

// Forward declarations
quarter_e4m3 SD_INLINE SD_HOST_DEVICE cpu_float2e4m3_rn(float f);
float SD_INLINE SD_HOST_DEVICE cpu_e4m3_2float(quarter_e4m3 b);

// ---- E4M3FN -> float ----
float cpu_e4m3_2float(quarter_e4m3 b) {
  unsigned char val = b.x;
  unsigned sign = (val >> 7) & 1;
  unsigned exponent = (val >> 3) & 0xF;   // 4-bit exponent
  unsigned mantissa = val & 0x7;           // 3-bit mantissa

  // NaN: exponent=0xF, mantissa=0x7 (E4M3FN has no Inf)
  if (exponent == 0xF && mantissa == 0x7) {
    unsigned result = (sign << 31) | 0x7FC00000;  // quiet NaN
    float ret;
    memcpy(&ret, &result, sizeof(float));
    return ret;
  }

  float fval;
  if (exponent == 0) {
    if (mantissa == 0) {
      // Zero
      fval = 0.0f;
    } else {
      // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa) = (-1)^sign * 2^(-6) * (mantissa/8)
      fval = static_cast<float>(mantissa) / 8.0f * (1.0f / 64.0f);  // 2^(-6) = 1/64
    }
  } else {
    // Normal: value = (-1)^sign * 2^(exponent-bias) * (1 + mantissa/8)
    fval = (1.0f + static_cast<float>(mantissa) / 8.0f) * powf(2.0f, static_cast<float>(exponent) - 7.0f);
  }

  return sign ? -fval : fval;
}

// ---- float -> E4M3FN (round to nearest even) ----
quarter_e4m3 cpu_float2e4m3_rn(float f) {
  quarter_e4m3 ret;

  unsigned x;
  memcpy(&x, &f, sizeof(float));
  unsigned sign = (x >> 31) & 1;
  unsigned f_exp = (x >> 23) & 0xFF;
  unsigned f_mant = x & 0x7FFFFF;

  // Handle special cases
  if (f_exp == 0xFF) {
    // Inf or NaN -> E4M3 NaN
    ret.x = (sign << 7) | 0x7F;
    return ret;
  }

  // Compute the float value magnitude for clamping
  float abs_f = sign ? -f : f;

  // E4M3FN max normal = 448.0
  if (abs_f > 448.0f) {
    // Clamp to max representable value (exponent=0xE=14, mantissa=0x7=7)
    // value = 2^(14-7) * (1 + 7/8) = 128 * 1.875 = 240... wait
    // Actually max = 2^8 * (1 + 6/8) = 256 * 1.75 = 448
    // exponent=15 is NaN row, so max exponent=14 -> 2^(14-7)=2^7=128
    // max mantissa without NaN at exp=15: at exp=14, mantissa=7 -> 128*(1+7/8)=128*1.875=240
    // Hmm, let me recalculate. E4M3FN: exp=14, mant=7 is NOT NaN (only exp=15,mant=7 is NaN)
    // Wait, E4M3FN has no infinity. NaN = 0x7F (exp=15, mant=7).
    // Max = exp=15, mant=6 -> 2^(15-7) * (1+6/8) = 256*1.75 = 448. Yes.
    ret.x = (sign << 7) | 0x7E;  // exp=15, mant=6 = 448
    return ret;
  }

  if (abs_f == 0.0f) {
    ret.x = (sign << 7);
    return ret;
  }

  // E4M3FN: bias=7, mantissa bits=3
  // Convert via direct computation
  int e4m3_exp;
  float mant_f;

  // Extract exponent: abs_f = 2^e * (1.xxx)
  int unbiased_exp = static_cast<int>(f_exp) - 127;  // float unbiased exponent

  // E4M3 biased exponent range: 1..15 (0 = subnormal, 15 with mant<7 = normal, 15 with mant=7 = NaN)
  // E4M3 unbiased range: -6..8
  if (unbiased_exp < -9) {
    // Too small, rounds to zero
    ret.x = (sign << 7);
    return ret;
  }

  if (unbiased_exp < -6) {
    // Subnormal in E4M3
    e4m3_exp = 0;
    // Subnormal: value = 2^(-6) * (mantissa/8)
    // So mantissa = abs_f / 2^(-6) * 8
    mant_f = abs_f * 64.0f * 8.0f;  // abs_f * 2^6 * 8
  } else {
    e4m3_exp = unbiased_exp + 7;  // bias=7
    if (e4m3_exp > 15) e4m3_exp = 15;
    // Normal: mantissa = (abs_f / 2^unbiased_exp - 1) * 8
    float power = powf(2.0f, static_cast<float>(unbiased_exp));
    mant_f = (abs_f / power - 1.0f) * 8.0f;
  }

  // Round to nearest even
  int mant_int = static_cast<int>(mant_f);
  float frac = mant_f - static_cast<float>(mant_int);
  if (frac > 0.5f || (frac == 0.5f && (mant_int & 1))) {
    mant_int++;
  }

  // Handle mantissa overflow
  if (e4m3_exp == 0) {
    // Subnormal
    if (mant_int >= 8) {
      mant_int = 0;
      e4m3_exp = 1;
    }
  } else {
    // Normal
    if (mant_int >= 8) {
      mant_int = 0;
      e4m3_exp++;
    }
  }

  // Clamp to avoid NaN encoding (exp=15, mant=7)
  if (e4m3_exp == 15 && mant_int >= 7) {
    mant_int = 6;  // max value = 448
  }
  if (e4m3_exp > 15) {
    e4m3_exp = 15;
    mant_int = 6;
  }

  ret.x = (sign << 7) | (e4m3_exp << 3) | (mant_int & 0x7);
  return ret;
}

// ============================================================================
// E5M2 format: 1-sign, 5-exp (bias=15), 2-mantissa, has Inf and NaN
// (IEEE 754 binary8 variant — same as BF16/FP16 pattern but 8-bit)
// ============================================================================

typedef struct {
  unsigned char x;
} __quarter_e5m2;

typedef __quarter_e5m2 quarter_e5m2;

quarter_e5m2 SD_INLINE SD_HOST_DEVICE cpu_float2e5m2_rn(float f);
float SD_INLINE SD_HOST_DEVICE cpu_e5m2_2float(quarter_e5m2 b);

// ---- E5M2 -> float ----
float cpu_e5m2_2float(quarter_e5m2 b) {
  unsigned char val = b.x;
  unsigned sign = (val >> 7) & 1;
  unsigned exponent = (val >> 2) & 0x1F;  // 5-bit exponent
  unsigned mantissa = val & 0x3;           // 2-bit mantissa

  if (exponent == 0x1F) {
    if (mantissa != 0) {
      // NaN
      unsigned result = (sign << 31) | 0x7FC00000;
      float ret;
      memcpy(&ret, &result, sizeof(float));
      return ret;
    } else {
      // Infinity
      unsigned result = (sign << 31) | 0x7F800000;
      float ret;
      memcpy(&ret, &result, sizeof(float));
      return ret;
    }
  }

  float fval;
  if (exponent == 0) {
    if (mantissa == 0) {
      fval = 0.0f;
    } else {
      // Subnormal: value = (-1)^sign * 2^(1-bias) * (0.mantissa) = (-1)^sign * 2^(-14) * (mantissa/4)
      fval = static_cast<float>(mantissa) / 4.0f * (1.0f / 16384.0f);  // 2^(-14)
    }
  } else {
    // Normal: value = (-1)^sign * 2^(exponent-15) * (1 + mantissa/4)
    fval = (1.0f + static_cast<float>(mantissa) / 4.0f) * powf(2.0f, static_cast<float>(exponent) - 15.0f);
  }

  return sign ? -fval : fval;
}

// ---- float -> E5M2 (round to nearest even) ----
quarter_e5m2 cpu_float2e5m2_rn(float f) {
  quarter_e5m2 ret;

  unsigned x;
  memcpy(&x, &f, sizeof(float));
  unsigned sign = (x >> 31) & 1;
  unsigned f_exp = (x >> 23) & 0xFF;
  unsigned f_mant = x & 0x7FFFFF;

  if (f_exp == 0xFF) {
    if (f_mant != 0) {
      // NaN
      ret.x = (sign << 7) | 0x7F;  // exp=31, mant=3
    } else {
      // Inf
      ret.x = (sign << 7) | 0x7C;  // exp=31, mant=0
    }
    return ret;
  }

  float abs_f = sign ? -f : f;

  // E5M2 max = 2^(30-15) * (1 + 3/4) = 2^15 * 1.75 = 57344
  if (abs_f > 57344.0f) {
    // Overflow -> Inf
    ret.x = (sign << 7) | 0x7C;
    return ret;
  }

  if (abs_f == 0.0f) {
    ret.x = (sign << 7);
    return ret;
  }

  int unbiased_exp = static_cast<int>(f_exp) - 127;
  int e5m2_exp;
  float mant_f;

  if (unbiased_exp < -16) {
    ret.x = (sign << 7);
    return ret;
  }

  if (unbiased_exp < -14) {
    e5m2_exp = 0;
    mant_f = abs_f * 16384.0f * 4.0f;  // abs_f * 2^14 * 4
  } else {
    e5m2_exp = unbiased_exp + 15;
    if (e5m2_exp > 30) e5m2_exp = 30;
    float power = powf(2.0f, static_cast<float>(unbiased_exp));
    mant_f = (abs_f / power - 1.0f) * 4.0f;
  }

  int mant_int = static_cast<int>(mant_f);
  float frac = mant_f - static_cast<float>(mant_int);
  if (frac > 0.5f || (frac == 0.5f && (mant_int & 1))) {
    mant_int++;
  }

  if (e5m2_exp == 0) {
    if (mant_int >= 4) {
      mant_int = 0;
      e5m2_exp = 1;
    }
  } else {
    if (mant_int >= 4) {
      mant_int = 0;
      e5m2_exp++;
    }
  }

  if (e5m2_exp >= 31) {
    // Overflow to Inf
    ret.x = (sign << 7) | 0x7C;
    return ret;
  }

  ret.x = (sign << 7) | (e5m2_exp << 2) | (mant_int & 0x3);
  return ret;
}

// ============================================================================
// float8_e4m3 struct (primary FP8 type)
// ============================================================================

struct float8_e4m3 {
  constexpr float8_e4m3(const float8_e4m3&) = default;

  quarter_e4m3 data;

  SD_INLINE SD_HOST_DEVICE float8_e4m3();

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e4m3(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e4m3& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);
  SD_INLINE SD_HOST_DEVICE void assign(float rhs);

#if defined(HAVE_CUTLASS) && HAVE_CUTLASS && defined(__CUDACC__)
  SD_INLINE SD_HOST_DEVICE operator cutlass::float_e4m3_t() const {
    cutlass::float_e4m3_t result;
    result.storage = data.x;
    return result;
  }

  SD_INLINE SD_HOST_DEVICE float8_e4m3(const cutlass::float_e4m3_t& rhs) {
    data.x = rhs.storage;
  }
#endif
};

float8_e4m3::float8_e4m3() { data = cpu_float2e4m3_rn(0.0f); }

template <class T>
float8_e4m3::float8_e4m3(const T& rhs) {
  assign(static_cast<float>(rhs));
}

template <class T>
float8_e4m3& float8_e4m3::operator=(const T& rhs) {
  assign(static_cast<float>(rhs));
  return *this;
}

float8_e4m3::operator float() const { return cpu_e4m3_2float(data); }

void float8_e4m3::assign(double rhs) { assign(static_cast<float>(rhs)); }

void float8_e4m3::assign(float rhs) { data = cpu_float2e4m3_rn(rhs); }

// ============================================================================
// float8_e5m2 struct (gradient/accumulation FP8 type)
// ============================================================================

struct float8_e5m2 {
  constexpr float8_e5m2(const float8_e5m2&) = default;

  quarter_e5m2 data;

  SD_INLINE SD_HOST_DEVICE float8_e5m2();

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e5m2(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e5m2& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);
  SD_INLINE SD_HOST_DEVICE void assign(float rhs);

#if defined(HAVE_CUTLASS) && HAVE_CUTLASS && defined(__CUDACC__)
  SD_INLINE SD_HOST_DEVICE operator cutlass::float_e5m2_t() const {
    cutlass::float_e5m2_t result;
    result.storage = data.x;
    return result;
  }

  SD_INLINE SD_HOST_DEVICE float8_e5m2(const cutlass::float_e5m2_t& rhs) {
    data.x = rhs.storage;
  }
#endif
};

float8_e5m2::float8_e5m2() { data = cpu_float2e5m2_rn(0.0f); }

template <class T>
float8_e5m2::float8_e5m2(const T& rhs) {
  assign(static_cast<float>(rhs));
}

template <class T>
float8_e5m2& float8_e5m2::operator=(const T& rhs) {
  assign(static_cast<float>(rhs));
  return *this;
}

float8_e5m2::operator float() const { return cpu_e5m2_2float(data); }

void float8_e5m2::assign(double rhs) { assign(static_cast<float>(rhs)); }

void float8_e5m2::assign(float rhs) { data = cpu_float2e5m2_rn(rhs); }

// ============================================================================
// Backward compatibility: float8 = float8_e4m3 (the primary inference type)
// ============================================================================

using float8 = float8_e4m3;

// Legacy aliases
using quarter = quarter_e4m3;
using __quarter = __quarter_e4m3;
SD_INLINE SD_HOST_DEVICE quarter cpu_float2quarter_rn(float f) { return cpu_float2e4m3_rn(f); }
SD_INLINE SD_HOST_DEVICE float cpu_quarter2float(quarter b) { return cpu_e4m3_2float(b); }

}  // namespace sd

#endif  // LIBND4J_FLOAT8_H
