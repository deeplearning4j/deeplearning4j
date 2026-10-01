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
// FP4 type: E2M1 (OCP MX FP4, NVFP4 elements)
//
// E2M1: 1 sign, 2 exponent, 1 mantissa bits. Bias=1. Values ±{0, 0.5, 1, 1.5,
// 2, 3, 4, 6}; no infinity and no NaN. Codes are packed two per byte, the even
// element in the low nibble.
//
// A storage type: libnd4j decodes E2M1 weights (ModelOpt NVFP4, ADR 0122) and
// never produces them, so the type converts to float only. Every E2M1 value is
// an FP32 value, so the conversion is exact.
//

#ifndef LIBND4J_FLOAT4_H
#define LIBND4J_FLOAT4_H

#include <system/common.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace sd {

struct float4_e2m1 {
  static constexpr int kExponentBits = 2;
  static constexpr int kMantissaBits = 1;
  static constexpr int kExponentBias = 1;
  static constexpr int kBits = 1 + kExponentBits + kMantissaBits;
  static constexpr int kCodesPerByte = 8 / kBits;
  static constexpr uint32_t kCodeMask = (1u << kBits) - 1;

  uint8_t code;  // the low kBits bits

  SD_INLINE SD_HOST_DEVICE constexpr float4_e2m1() : code(0) {}

  // Wraps an encoding (the low kBits bits of bits): no numeric conversion.
  SD_INLINE SD_HOST_DEVICE constexpr explicit float4_e2m1(uint8_t bits)
      : code(static_cast<uint8_t>(bits & kCodeMask)) {}

  // Codes held by one packed storage word.
  template <typename Storage>
  SD_INLINE SD_HOST_DEVICE static constexpr int codesPer() {
    static_assert(std::is_unsigned<Storage>::value, "packed E2M1 storage is an unsigned integer");
    return static_cast<int>(sizeof(Storage)) * 8 / kBits;
  }

  // The index-th code of a packed storage word, code 0 in the low bits.
  template <typename Storage>
  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 unpack(Storage packed, int index) {
    static_assert(std::is_unsigned<Storage>::value, "packed E2M1 storage is an unsigned integer");
    return float4_e2m1(static_cast<uint8_t>(packed >> (kBits * index)));
  }

  // Branch-free, as it runs once per weight in every NVFP4 kernel. A normal
  // code's FP32 pattern is its magnitude bits at the top of the FP32 mantissa
  // plus the exponent rebias; the one subnormal (exponent 0, mantissa 1) is
  // 2^(1 - bias - mantissa bits); code 0 is +0 and the sign bit makes -0 of
  // code 8.
  SD_INLINE SD_HOST_DEVICE operator float() const {
    static_assert(kMantissaBits == 1, "E2M1 has exactly one subnormal magnitude");
    constexpr uint32_t kFloatMantissaBits = 23;
    constexpr uint32_t kFloatBias = 127;
    constexpr int kSignBit = kExponentBits + kMantissaBits;
    constexpr uint32_t kMagnitudeMask = (1u << kSignBit) - 1;
    constexpr uint32_t kMinNormal = 1u << kMantissaBits;
    constexpr uint32_t kRebias = (kFloatBias - kExponentBias) << kFloatMantissaBits;
    constexpr uint32_t kSubnormal = (kFloatBias + 1 - kExponentBias - kMantissaBits) << kFloatMantissaBits;
    const uint32_t magnitude = code & kMagnitudeMask;
    const uint32_t magnitudeBits = magnitude >= kMinNormal ? (magnitude << (kFloatMantissaBits - kMantissaBits)) + kRebias
                                                           : (magnitude != 0 ? kSubnormal : 0u);
    const uint32_t bits = magnitudeBits | ((code & (1u << kSignBit)) << (31 - kSignBit));
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
  }

  // std::numeric_limits semantics. E2M1 has no infinity and no NaN.
  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 min() {
    return float4_e2m1(static_cast<uint8_t>(0x02));  // 1
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 lowest() {
    return float4_e2m1(static_cast<uint8_t>(0x0F));  // -6
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 max() {
    return float4_e2m1(static_cast<uint8_t>(0x07));  // 6
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 epsilon() {
    return float4_e2m1(static_cast<uint8_t>(0x01));  // 0.5
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 round_error() {
    return float4_e2m1(static_cast<uint8_t>(0x01));  // 0.5
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 denorm_min() {
    return float4_e2m1(static_cast<uint8_t>(0x01));  // 0.5
  }

  SD_INLINE SD_HOST_DEVICE static constexpr float4_e2m1 min_positive() {
    return denorm_min();
  }
};

static_assert(sizeof(float4_e2m1) == 1, "float4_e2m1 must be 1 byte");

}  // namespace sd

#endif  // LIBND4J_FLOAT4_H
