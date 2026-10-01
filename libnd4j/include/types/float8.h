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
// NaN is canonical, as in float16 and bfloat16: every FP32 NaN converts to the
// positive NaN 0x7F, and every FP8 NaN converts to the FP32 NaN 0x7FFFFFFF.
// These are the encodings the sm_89+ hardware conversions produce (they drop
// the NaN sign), so on device the types convert in hardware (SD_NATIVE_FP8)
// with results identical to the software conversions below.
//

#ifndef LIBND4J_FLOAT8_H
#define LIBND4J_FLOAT8_H

#include <system/op_boilerplate.h>
#include <types/float16.h>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_fp8.h>
#endif

// CUTLASS interop: on CUDA builds with CUTLASS, include cutlass float8 types
#if defined(HAVE_CUTLASS) && HAVE_CUTLASS && defined(__CUDACC__)
#include <cute/numeric/numeric_types.hpp>
#endif

namespace sd {

// The FP32 value of every FP8 NaN encoding: the canonical NaN that float16
// decodes NaN to, and that the hardware FP8 -> FP16 -> FP32 conversion yields.
SD_INLINE SD_HOST_DEVICE float cpu_fp8_nan_2float() {
  const unsigned bits = 0x7FFFFFFF;
  float ret;
  memcpy(&ret, &bits, sizeof(float));
  return ret;
}

// ============================================================================
// E4M3FN format: 1-sign, 4-exp (bias=7), 3-mantissa, no Inf, NaN=0x7F/0xFF
// ============================================================================

typedef struct {
  unsigned char x;
} __quarter_e4m3;

typedef __quarter_e4m3 quarter_e4m3;

// Forward declarations
quarter_e4m3 SD_INLINE SD_HOST_DEVICE cpu_float2e4m3_rn(float f);
quarter_e4m3 SD_INLINE SD_HOST_DEVICE cpu_float2e4m3_rn_satfinite(float f);
float SD_INLINE SD_HOST_DEVICE cpu_e4m3_2float(quarter_e4m3 b);

// ---- E4M3FN -> float ----
float cpu_e4m3_2float(quarter_e4m3 b) {
  unsigned char val = b.x;
  unsigned sign = (val >> 7) & 1;
  unsigned exponent = (val >> 3) & 0xF;   // 4-bit exponent
  unsigned mantissa = val & 0x7;           // 3-bit mantissa

  // NaN: exponent=0xF, mantissa=0x7 (E4M3FN has no Inf)
  if (exponent == 0xF && mantissa == 0x7) return cpu_fp8_nan_2float();

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
    // Normal: value = (-1)^sign * 2^(exponent-bias) * (1 + mantissa/8). Every E4M3
    // normal is exactly a float normal, so build its IEEE bits directly (rebias
    // the exponent, left-align the mantissa) instead of evaluating powf.
    unsigned bits = ((exponent - 7 + 127) << 23) | (mantissa << 20);
    memcpy(&fval, &bits, sizeof(float));
  }

  return sign ? -fval : fval;
}

// ---- float -> E4M3FN (round to nearest even) ----
// Finite values beyond the largest finite value saturate to it. E4M3FN has no
// infinity: infinities convert to NaN of the same sign (as NVIDIA's conversion
// without saturation does), and every NaN to the canonical NaN.
quarter_e4m3 cpu_float2e4m3_rn(float f) {
  quarter_e4m3 ret;

  unsigned x;
  memcpy(&x, &f, sizeof(float));
  unsigned sign = (x >> 31) & 1;
  unsigned f_exp = (x >> 23) & 0xFF;
  unsigned f_mant = x & 0x7FFFFF;

  // Handle special cases
  if (f_exp == 0xFF) {
    ret.x = f_mant != 0 ? 0x7F : ((sign << 7) | 0x7F);
    return ret;
  }

  // Compute the float value magnitude for clamping
  float abs_f = sign ? -f : f;

  // E4M3FN max = 2^(15-7) * (1 + 6/8) = 448 (exponent 15, mantissa 6; exponent
  // 15 with mantissa 7 is NaN)
  if (abs_f > 448.0f) {
    ret.x = (sign << 7) | 0x7E;
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
  if (unbiased_exp < -10) {
    // Below half the smallest subnormal (2^-10), rounds to zero.
    // Exponent -10 must reach ties-to-even rounding below.
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
    // Normal: mantissa = (abs_f / 2^unbiased_exp - 1) * 8, the FP32 mantissa
    // field in units of the 3-bit mantissa (exact: f_mant < 2^24, scaled by a
    // power of two).
    mant_f = static_cast<float>(f_mant) * (1.0f / (1 << 20));
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

// ---- float -> E4M3FN, saturating (cvt.rn.satfinite) ----
// cpu_float2e4m3_rn, except that infinities saturate to ±448 too.
quarter_e4m3 cpu_float2e4m3_rn_satfinite(float f) {
  unsigned x;
  memcpy(&x, &f, sizeof(float));
  if ((x & 0x7FFFFFFF) == 0x7F800000) {
    quarter_e4m3 ret;
    ret.x = ((x >> 24) & 0x80) | 0x7E;
    return ret;
  }
  return cpu_float2e4m3_rn(f);
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
quarter_e5m2 SD_INLINE SD_HOST_DEVICE cpu_float2e5m2_rn_satfinite(float f);
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
      return cpu_fp8_nan_2float();
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
    // Normal: value = (-1)^sign * 2^(exponent-15) * (1 + mantissa/4). Every E5M2
    // normal is exactly a float normal, so build its IEEE bits directly (rebias
    // the exponent, left-align the mantissa) instead of evaluating powf.
    unsigned bits = ((exponent - 15 + 127) << 23) | (mantissa << 21);
    memcpy(&fval, &bits, sizeof(float));
  }

  return sign ? -fval : fval;
}

// ---- float -> E5M2 (round to nearest even) ----
// Overflow rounds to infinity; every NaN converts to the canonical NaN.
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
      ret.x = 0x7F;  // exp=31, mant=3
    } else {
      // Inf
      ret.x = (sign << 7) | 0x7C;  // exp=31, mant=0
    }
    return ret;
  }

  float abs_f = sign ? -f : f;

  // E5M2 max = 2^(30-15) * (1 + 3/4) = 2^15 * 1.75 = 57344. Rounding to nearest
  // even overflows from the midpoint between it and 2^16: 61440 ties to the even
  // encoding, which is infinity.
  if (abs_f >= 61440.0f) {
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

  if (unbiased_exp < -17) {
    // Below half the smallest subnormal (2^-17), rounds to zero.
    // Exponent -17 must reach ties-to-even rounding below.
    ret.x = (sign << 7);
    return ret;
  }

  if (unbiased_exp < -14) {
    e5m2_exp = 0;
    mant_f = abs_f * 16384.0f * 4.0f;  // abs_f * 2^14 * 4
  } else {
    e5m2_exp = unbiased_exp + 15;
    if (e5m2_exp > 30) e5m2_exp = 30;
    // The FP32 mantissa field in units of the 2-bit mantissa (exact).
    mant_f = static_cast<float>(f_mant) * (1.0f / (1 << 21));
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

// ---- float -> E5M2, saturating (cvt.rn.satfinite) ----
// cpu_float2e5m2_rn, except that overflow and infinities saturate to ±57344.
quarter_e5m2 cpu_float2e5m2_rn_satfinite(float f) {
  quarter_e5m2 ret = cpu_float2e5m2_rn(f);
  if ((ret.x & 0x7F) == 0x7C) ret.x = (ret.x & 0x80) | 0x7B;
  return ret;
}

// ============================================================================
// float8_e4m3 struct (primary FP8 type)
// ============================================================================

struct float8_e4m3 {
  constexpr float8_e4m3(const float8_e4m3&) = default;

  quarter_e4m3 data;

  SD_INLINE SD_HOST_DEVICE constexpr float8_e4m3() : data{0} {}

  // Wraps an encoding: no numeric conversion.
  SD_INLINE SD_HOST_DEVICE constexpr explicit float8_e4m3(quarter_e4m3 bits) : data(bits) {}

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e4m3(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e4m3& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);
  SD_INLINE SD_HOST_DEVICE void assign(float rhs);

  // Round to nearest even that also saturates infinities (cvt.rn.satfinite):
  // the conversion of the value clamped to [lowest(), max()], NaN kept.
  SD_INLINE SD_HOST_DEVICE static float8_e4m3 from_float_satfinite(float value);

  // from_float_satfinite of in[0] and in[1]: one packed conversion in hardware.
  // The pair functions move their two codes as one 16-bit word, so codes held
  // in registers stay packed instead of splitting into bytes.
  SD_INLINE SD_HOST_DEVICE static void from_float2_satfinite(const float* in, float8_e4m3* out);

  // in[0] and in[1] to FP32: one packed conversion in hardware.
  SD_INLINE SD_HOST_DEVICE static void to_float2(const float8_e4m3* in, float* out);

  // in[0] and in[1] to FP16, exactly: one packed conversion in hardware.
  SD_INLINE SD_HOST_DEVICE static void to_half2(const float8_e4m3* in, float16* out);

  // std::numeric_limits semantics. E4M3FN has no infinity and no signaling NaN.
  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 min() {
    return float8_e4m3(quarter_e4m3{0x08});  // 2^-6
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 lowest() {
    return float8_e4m3(quarter_e4m3{0xFE});  // -448
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 max() {
    return float8_e4m3(quarter_e4m3{0x7E});  // 448
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 epsilon() {
    return float8_e4m3(quarter_e4m3{0x20});  // 2^-3
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 round_error() {
    return float8_e4m3(quarter_e4m3{0x30});  // 0.5
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 quiet_NaN() {
    return float8_e4m3(quarter_e4m3{0x7F});
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 denorm_min() {
    return float8_e4m3(quarter_e4m3{0x01});  // 2^-9
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e4m3 min_positive() {
    return denorm_min();
  }

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

static_assert(sizeof(float8_e4m3) == 1, "float8_e4m3 must be 1 byte");

template <class T>
float8_e4m3::float8_e4m3(const T& rhs) {
  assign(static_cast<float>(rhs));
}

template <class T>
float8_e4m3& float8_e4m3::operator=(const T& rhs) {
  assign(static_cast<float>(rhs));
  return *this;
}

float8_e4m3::operator float() const {
#if defined(SD_NATIVE_FP8)
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(data.x, __NV_E4M3)));
#else
  return cpu_e4m3_2float(data);
#endif
}

void float8_e4m3::assign(double rhs) { assign(static_cast<float>(rhs)); }

void float8_e4m3::assign(float rhs) {
#if defined(SD_NATIVE_FP8)
  // The hardware conversion saturates, and E4M3FN converts infinities to NaN.
  const unsigned bits = __float_as_uint(rhs);
  if ((bits & 0x7FFFFFFF) == 0x7F800000)
    data.x = ((bits >> 24) & 0x80) | 0x7F;
  else
    data.x = __nv_cvt_float_to_fp8(rhs, __NV_SATFINITE, __NV_E4M3);
#else
  data = cpu_float2e4m3_rn(rhs);
#endif
}

float8_e4m3 float8_e4m3::from_float_satfinite(float value) {
#if defined(SD_NATIVE_FP8)
  return float8_e4m3(quarter_e4m3{__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)});
#else
  return float8_e4m3(cpu_float2e4m3_rn_satfinite(value));
#endif
}

// The packed pair holds element 0 in its low byte (low half for FP16 pairs):
// memory order on the little-endian device.
void float8_e4m3::from_float2_satfinite(const float* in, float8_e4m3* out) {
#if defined(SD_NATIVE_FP8)
  const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(make_float2(in[0], in[1]), __NV_SATFINITE, __NV_E4M3);
  memcpy(out, &pair, sizeof(pair));
#else
  out[0] = from_float_satfinite(in[0]);
  out[1] = from_float_satfinite(in[1]);
#endif
}

void float8_e4m3::to_float2(const float8_e4m3* in, float* out) {
#if defined(SD_NATIVE_FP8)
  __nv_fp8x2_storage_t pair;
  memcpy(&pair, in, sizeof(pair));
  const float2 values = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
  out[0] = values.x;
  out[1] = values.y;
#else
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
#endif
}

void float8_e4m3::to_half2(const float8_e4m3* in, float16* out) {
#if defined(SD_NATIVE_FP8)
  __nv_fp8x2_storage_t pair;
  memcpy(&pair, in, sizeof(pair));
  const __half2_raw values = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  memcpy(out, &values, sizeof(values));
#else
  out[0] = static_cast<float16>(static_cast<float>(in[0]));
  out[1] = static_cast<float16>(static_cast<float>(in[1]));
#endif
}

// ============================================================================
// float8_e5m2 struct (gradient/accumulation FP8 type)
// ============================================================================

struct float8_e5m2 {
  constexpr float8_e5m2(const float8_e5m2&) = default;

  quarter_e5m2 data;

  SD_INLINE SD_HOST_DEVICE constexpr float8_e5m2() : data{0} {}

  // Wraps an encoding: no numeric conversion.
  SD_INLINE SD_HOST_DEVICE constexpr explicit float8_e5m2(quarter_e5m2 bits) : data(bits) {}

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e5m2(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE float8_e5m2& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);
  SD_INLINE SD_HOST_DEVICE void assign(float rhs);

  // Round to nearest even that saturates instead of overflowing to infinity
  // (cvt.rn.satfinite): the conversion of the value clamped to [lowest(), max()].
  SD_INLINE SD_HOST_DEVICE static float8_e5m2 from_float_satfinite(float value);

  // from_float_satfinite of in[0] and in[1]: one packed conversion in hardware.
  SD_INLINE SD_HOST_DEVICE static void from_float2_satfinite(const float* in, float8_e5m2* out);

  // in[0] and in[1] to FP32: one packed conversion in hardware.
  SD_INLINE SD_HOST_DEVICE static void to_float2(const float8_e5m2* in, float* out);

  // in[0] and in[1] to FP16, exactly: one packed conversion in hardware.
  SD_INLINE SD_HOST_DEVICE static void to_half2(const float8_e5m2* in, float16* out);

  // std::numeric_limits semantics.
  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 min() {
    return float8_e5m2(quarter_e5m2{0x04});  // 2^-14
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 lowest() {
    return float8_e5m2(quarter_e5m2{0xFB});  // -57344
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 max() {
    return float8_e5m2(quarter_e5m2{0x7B});  // 57344
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 epsilon() {
    return float8_e5m2(quarter_e5m2{0x34});  // 2^-2
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 round_error() {
    return float8_e5m2(quarter_e5m2{0x38});  // 0.5
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 infinity() {
    return float8_e5m2(quarter_e5m2{0x7C});
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 quiet_NaN() {
    return float8_e5m2(quarter_e5m2{0x7F});
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 signaling_NaN() {
    return float8_e5m2(quarter_e5m2{0x7D});
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 denorm_min() {
    return float8_e5m2(quarter_e5m2{0x01});  // 2^-16
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float8_e5m2 min_positive() {
    return denorm_min();
  }

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

static_assert(sizeof(float8_e5m2) == 1, "float8_e5m2 must be 1 byte");

template <class T>
float8_e5m2::float8_e5m2(const T& rhs) {
  assign(static_cast<float>(rhs));
}

template <class T>
float8_e5m2& float8_e5m2::operator=(const T& rhs) {
  assign(static_cast<float>(rhs));
  return *this;
}

float8_e5m2::operator float() const {
#if defined(SD_NATIVE_FP8)
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(data.x, __NV_E5M2)));
#else
  return cpu_e5m2_2float(data);
#endif
}

void float8_e5m2::assign(double rhs) { assign(static_cast<float>(rhs)); }

void float8_e5m2::assign(float rhs) {
#if defined(SD_NATIVE_FP8)
  // The hardware conversion saturates; rounding to nearest even overflows to
  // infinity from 61440 (bits 0x47700000), infinities included.
  const unsigned bits = __float_as_uint(rhs);
  const unsigned magnitude = bits & 0x7FFFFFFF;
  if (magnitude >= 0x47700000 && magnitude <= 0x7F800000)
    data.x = ((bits >> 24) & 0x80) | 0x7C;
  else
    data.x = __nv_cvt_float_to_fp8(rhs, __NV_SATFINITE, __NV_E5M2);
#else
  data = cpu_float2e5m2_rn(rhs);
#endif
}

float8_e5m2 float8_e5m2::from_float_satfinite(float value) {
#if defined(SD_NATIVE_FP8)
  return float8_e5m2(quarter_e5m2{__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2)});
#else
  return float8_e5m2(cpu_float2e5m2_rn_satfinite(value));
#endif
}

// Pairs in memory order, as float8_e4m3's.
void float8_e5m2::from_float2_satfinite(const float* in, float8_e5m2* out) {
#if defined(SD_NATIVE_FP8)
  const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(make_float2(in[0], in[1]), __NV_SATFINITE, __NV_E5M2);
  memcpy(out, &pair, sizeof(pair));
#else
  out[0] = from_float_satfinite(in[0]);
  out[1] = from_float_satfinite(in[1]);
#endif
}

void float8_e5m2::to_float2(const float8_e5m2* in, float* out) {
#if defined(SD_NATIVE_FP8)
  __nv_fp8x2_storage_t pair;
  memcpy(&pair, in, sizeof(pair));
  const float2 values = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E5M2)));
  out[0] = values.x;
  out[1] = values.y;
#else
  out[0] = static_cast<float>(in[0]);
  out[1] = static_cast<float>(in[1]);
#endif
}

void float8_e5m2::to_half2(const float8_e5m2* in, float16* out) {
#if defined(SD_NATIVE_FP8)
  __nv_fp8x2_storage_t pair;
  memcpy(&pair, in, sizeof(pair));
  const __half2_raw values = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E5M2);
  memcpy(out, &values, sizeof(values));
#else
  out[0] = static_cast<float16>(static_cast<float>(in[0]));
  out[1] = static_cast<float16>(static_cast<float>(in[1]));
#endif
}

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

// Limits only: the FP8 types are storage formats, deliberately not arithmetic
// or floating-point types (arithmetic happens in a wider accumulation type).
namespace std {
template <>
struct numeric_limits<sd::float8_e4m3> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr float_denorm_style has_denorm = denorm_present;
  static constexpr bool has_denorm_loss = false;
  static constexpr float_round_style round_style = round_to_nearest;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 3;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -5;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 9;
  static constexpr int max_exponent10 = 2;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  SD_HOST_DEVICE static constexpr sd::float8_e4m3 min() noexcept { return sd::float8_e4m3::min(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 lowest() noexcept { return sd::float8_e4m3::lowest(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 max() noexcept { return sd::float8_e4m3::max(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 epsilon() noexcept { return sd::float8_e4m3::epsilon(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 round_error() noexcept { return sd::float8_e4m3::round_error(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 infinity() noexcept { return sd::float8_e4m3(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 quiet_NaN() noexcept { return sd::float8_e4m3::quiet_NaN(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 signaling_NaN() noexcept { return sd::float8_e4m3(); }
  SD_HOST_DEVICE static constexpr sd::float8_e4m3 denorm_min() noexcept { return sd::float8_e4m3::denorm_min(); }
};

template <>
struct numeric_limits<sd::float8_e5m2> {
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
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;

  SD_HOST_DEVICE static constexpr sd::float8_e5m2 min() noexcept { return sd::float8_e5m2::min(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 lowest() noexcept { return sd::float8_e5m2::lowest(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 max() noexcept { return sd::float8_e5m2::max(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 epsilon() noexcept { return sd::float8_e5m2::epsilon(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 round_error() noexcept { return sd::float8_e5m2::round_error(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 infinity() noexcept { return sd::float8_e5m2::infinity(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 quiet_NaN() noexcept { return sd::float8_e5m2::quiet_NaN(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 signaling_NaN() noexcept { return sd::float8_e5m2::signaling_NaN(); }
  SD_HOST_DEVICE static constexpr sd::float8_e5m2 denorm_min() noexcept { return sd::float8_e5m2::denorm_min(); }
};
}  // namespace std

#endif  // LIBND4J_FLOAT8_H
