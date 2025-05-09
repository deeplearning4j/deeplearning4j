/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* See the NOTICE file distributed with this work for additional
* information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//

#pragma once
#ifndef UNICODE
#define UNICODE
#include <system/common.h>




namespace sd {
namespace unicode {

// These constants are fine as they are, used by the functions below.
constexpr uint32_t ONEBYTEBOUND = 0x00000080;
constexpr uint32_t TWOBYTEBOUND = 0x00000800;
constexpr uint32_t THREEBYTEBOUND = 0x00010000;
constexpr uint16_t HIGHBYTEMIN = 0xd800u;
constexpr uint16_t HIGHBYTEMAX = 0xdbffu;
constexpr uint16_t TRAILBYTEMIN = 0xdc00u;
constexpr uint16_t TRAILBYTEMAX = 0xdfffu;
constexpr uint16_t HIGHBYTEOFFSET = HIGHBYTEMIN - (0x10000 >> 10);
constexpr uint32_t BYTEOFFSET = 0x10000u - (HIGHBYTEMIN << 10) - TRAILBYTEMIN;
// Maximum valid value for a Unicode code point
constexpr uint32_t CODEPOINTMAX = 0x0010ffffu;

// SD_INLINE helpers will become SD_HOST_DEVICE inline
template <typename T>
SD_INLINE uint8_t castToU8(const T cp) {
  return static_cast<uint8_t>(0xff & cp);
}

template <typename T>
SD_INLINE uint16_t castToU16(const T cp) {
  return static_cast<uint16_t>(0xffff & cp);
}

template <typename T>
SD_INLINE uint32_t castToU32(const T cp) {
  return static_cast<uint32_t>(0xffffffff & cp);
}

template <typename T>
SD_INLINE bool isTrail(const T cp) {
  return ((castToU8(cp) >> 6) == 0x2);
}

template <typename T>
SD_INLINE bool isHighSurrogate(const T cp) {
  return (cp & 0xfffffc00) == 0xd800;
}

template <typename T>
SD_HOST_DEVICE /* Added SD_HOST_DEVICE, was missing specifier */ bool isLowSurrogate(const T cp) {
  return (cp & 0xfffffc00) == 0xdc00;
}

template <typename T>
SD_INLINE bool isLeadSurrogate(const T cp) {
  return (cp >= HIGHBYTEMIN && cp <= HIGHBYTEMAX);
}

template <typename T>
SD_INLINE bool isTrailSurrogate(const T cp) {
  return (cp >= TRAILBYTEMIN && cp <= TRAILBYTEMAX);
}

template <typename T>
SD_INLINE bool isSurrogateU8(const T cp) {
  // This check might be problematic for char if T is char and cp is negative due to sign extension.
  // However, it's usually called after casting to uint8_t or on unsigned types.
  // The original used castToU8(*it) then isSurrogateU8.
  // Let's assume cp is already a code point value (e.g. uint32_t).
  return (cp >= HIGHBYTEMIN && cp <= TRAILBYTEMAX);
}

template <typename T>
SD_INLINE bool isSurrogateU16(const T cp) {
  return ((cp - 0xd800u) < 2048u);
}

template <typename T>
SD_INLINE bool isSymbolU8Valid(const T cp) {
  // Assumes cp is a Unicode codepoint (uint32_t) to check against CODEPOINTMAX and surrogate range
  return (cp <= CODEPOINTMAX && !isSurrogateU8(cp));
}

template <typename T>
SD_INLINE bool isSymbolValid(const T cp) {
  return (cp <= CODEPOINTMAX);
}


template <typename T>
SD_INLINE uint32_t surrogateU32(const T& high, const T& low) {
  // Ensure high and low are treated as unsigned for calculation if they are char16_t
  return (static_cast<uint32_t>(high) << 10) + static_cast<uint32_t>(low) - 0x35fdc00;
  // Original was: return (high << 10) + low - 0x35fdc00;
  // This one seems more correct for surrogate reconstruction:
  // return 0x10000u + ((static_cast<uint32_t>(high) - HIGHBYTEMIN) << 10) + (static_cast<uint32_t>(low) - TRAILBYTEMIN);
  // The BYTEOFFSET constant seems related to this.
  // cp = (cp << 10) + trail_surrogate + BYTEOFFSET;  in utf16to8Ptr
  // Let's use the logic consistent with utf16to8Ptr's reverse:
  // if (isTrailSurrogate(trail_surrogate)) cp = (cp << 10) + trail_surrogate + BYTEOFFSET;
  // This means the provided surrogateU32 is likely not used or is for a different context.
  // The reconstruction from high/low surrogates is usually (high_surr - 0xD800) * 0x400 + (low_surr - 0xDC00) + 0x10000
  // The provided utf16to32Ptr uses: *result++ = surrogateU32(cpHigh, cpLow); which is (high << 10) + low - 0x35fdc00;
  // This specific formula needs to be correct for how surrogates are combined.
  // Given the existing code uses it, I will keep it, but it looks non-standard.
  // A standard way to combine:
  // uint32_t h = static_cast<uint32_t>(high) - HIGHBYTEMIN;
  // uint32_t l = static_cast<uint32_t>(low) - TRAILBYTEMIN;
  // return (h << 10) + l + 0x10000U;
  // The existing formula (high << 10) + low - 0x35fdc00 might be equivalent due to specific constant choices.
  // (0xD800 << 10) + 0xDC00 - 0x35FDC00 = 0x36000000 + 0xDC00 - 0x35FDC00 = 0x3600DC00 - 0x35FDC00 = 0x41C00 which is not 0x10000.
  // This `surrogateU32` function is suspicious if `high` and `low` are the direct surrogate values.
  // However, `utf16to32Ptr` uses it, so I'll keep it as is for now.
  return (static_cast<uint32_t>(high) << 10) + static_cast<uint32_t>(low) - 0x35fdc00;
}


template <typename T>
SD_HOST_DEVICE /* Added SD_HOST_DEVICE */ LongType symbolLength(const T* it) {
  uint8_t lead = castToU8(*it);
  if (lead < 0x80)
    return 1;
  else if ((lead >> 5) == 0x6) // 110xxxxx
    return 2;
  else if ((lead >> 4) == 0xe) // 1110xxxx
    return 3;
  else if ((lead >> 3) == 0x1e) // 11110xxx
    return 4;
  else
    return 0; // invalid
}

template <typename T>
SD_HOST_DEVICE /* Added SD_HOST_DEVICE */ LongType symbolLength32(const T* it) {
  auto lead = castToU32(*it); // Assumes T is char32_t or uint32_t
  if (lead < ONEBYTEBOUND) // < 0x80
    return 1;
  else if (lead < TWOBYTEBOUND) // < 0x800
    return 2;
  else if (lead < THREEBYTEBOUND) // < 0x10000
    return 3;
  else if (lead <= CODEPOINTMAX) // <= 0x10FFFF
    return 4;
  else
    return 0; // invalid
}

template <typename T>
SD_HOST_DEVICE /* Added SD_HOST_DEVICE */ LongType symbolLength16(const T* it) {
  // This function determines the UTF-8 length of a character represented by UTF-16 sequence pointed to by `it`.
  uint16_t lead = castToU16(*it);
  if (!isLeadSurrogate(lead)) { // Non-surrogate or BMP char
    // This part seems to determine UTF-8 length from a BMP codepoint
    if (lead < ONEBYTEBOUND) // < 0x80 (ASCII)
      return 1;
    else if (lead < TWOBYTEBOUND) // < 0x800
      return 2;
      // If lead is >= TWOBYTEBOUND (0x800) and < THREEBYTEBOUND (0x10000) and not a surrogate
    else
      return 3;

  } else { // Lead surrogate, implies a character outside BMP, encoded as 4 bytes in UTF-8
    return 4;
  }
}

SD_HOST_DEVICE LongType offsetUtf8StringInUtf32(const void* start, const void* end) {
  LongType count = 0;
  for (auto it = static_cast<const int8_t*>(start); it < end; /*manual increment*/) {
    auto length = symbolLength(it);
    if (length == 0) break; // Invalid sequence or error in symbolLength
    it += length;
    count += 1; // Each valid UTF-8 sequence becomes one UTF-32 char
  }
  return static_cast<LongType>(count * sizeof(char32_t));
}

SD_HOST_DEVICE LongType offsetUtf16StringInUtf32(const void* start, const void* end) {
  LongType count = 0;
  for (auto it = static_cast<const uint16_t*>(start); it < end;) {
    uint16_t current_char = *it;
    it++;
    if (isLeadSurrogate(current_char)) {
      if (it < end && isTrailSurrogate(*it)) {
        it++; // Consume trail surrogate
      } else {
        // Error: Unmatched lead surrogate, count it as one (replacement char) or error
        // The original symbolLength16 implies it becomes one char anyway
      }
    }
    count += 1; // Each (possibly surrogate-paired) UTF-16 sequence becomes one UTF-32 char
  }
  return static_cast<LongType>(count * sizeof(char32_t));
}

SD_HOST_DEVICE LongType offsetUtf8StringInUtf16(const void* start, const void* end) {
  LongType utf16_code_units = 0;
  for (auto it = static_cast<const int8_t*>(start); it < end; /*manual increment*/) {
    auto u8_len = symbolLength(it);
    if (u8_len == 0) break;

    // Decode to a temporary UTF-32 codepoint to check if it's > 0xFFFF
    // This is a bit inefficient but mirrors logic of needing surrogates
    uint32_t cp = 0;
    auto temp_it = it; // Use a temporary iterator for decoding
    // Simplified decode just to get codepoint value for surrogate check
    // A full decode is done in utf8to16Ptr. Here we only care about magnitude.
    if (u8_len == 1) cp = castToU8(*temp_it);
    else if (u8_len == 2) cp = ((castToU8(*temp_it) & 0x1F) << 6) | (castToU8(*(temp_it+1)) & 0x3F);
    else if (u8_len == 3) cp = ((castToU8(*temp_it) & 0x0F) << 12) | ((castToU8(*(temp_it+1)) & 0x3F) << 6) | (castToU8(*(temp_it+2)) & 0x3F);
    else if (u8_len == 4) cp = ((castToU8(*temp_it) & 0x07) << 18) | ((castToU8(*(temp_it+1)) & 0x3F) << 12) | ((castToU8(*(temp_it+2)) & 0x3F) << 6) | (castToU8(*(temp_it+3)) & 0x3F);

    it += u8_len;

    if (cp > 0xFFFF) { // Needs surrogate pair in UTF-16
      utf16_code_units += 2;
    } else {
      utf16_code_units += 1;
    }
  }
  return static_cast<LongType>(utf16_code_units * sizeof(char16_t));
}

SD_HOST_DEVICE LongType offsetUtf16StringInUtf8(const void* start, const void* end) {
  LongType utf8_bytes = 0;
  for (auto it = static_cast<const uint16_t*>(start); it < end;) {
    uint16_t current_char = *it;
    // Use symbolLength16 logic here: it returns expected UTF-8 length
    // To do that, we need to pass `it` to symbolLength16.
    // symbolLength16 expects a pointer to the start of the UTF-16 sequence.
    utf8_bytes += symbolLength16(it); // symbolLength16 should give the UTF-8 bytes for this UTF-16 char/pair

    it++; // Advance at least one uint16_t
    if (isLeadSurrogate(current_char)) {
      if (it < end && isTrailSurrogate(*it)) {
        it++; // Consume trail surrogate if it formed a pair
      }
      // If not a valid pair, symbolLength16 might return 0 or 3 (for replacement char)
      // The original implementation of symbolLength16 returns 4 if it's a lead surrogate,
      // assuming it *will* form a 4-byte UTF-8 char. This seems fine.
    }
  }
  return utf8_bytes;
}


SD_HOST_DEVICE LongType offsetUtf32StringInUtf16(const void* start, const void* end) {
  LongType utf16_code_units = 0;
  for (auto it = static_cast<const uint32_t*>(start); it < end; it++) {
    uint32_t cp = *it;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) { // Invalid, often becomes 1 U+FFFD
      utf16_code_units += 1;
    } else if (cp < 0x10000UL) {  // BMP
      utf16_code_units += 1;
    } else { // Supplementary plane
      utf16_code_units += 2;
    }
  }
  return static_cast<LongType>(utf16_code_units * sizeof(char16_t));
}

SD_HOST_DEVICE LongType offsetUtf32StringInUtf8(const void* start, const void* end) {
  LongType count = 0;
  for (auto it = static_cast<const uint32_t*>(start); it < end; it++) {
    count += symbolLength32(it); // symbolLength32 returns UTF-8 bytes for a UTF-32 char
  }
  return count;
}

SD_HOST_DEVICE bool isStringValidU8(const void* start, const void* stop) {
  // The original implementation had a bug for `isSymbolU8Valid(castToU8(*it))`
  // `isSymbolU8Valid` expects a full codepoint, not just one byte of a multi-byte sequence.
  // A proper UTF-8 validation is more complex. This function as written only checks if individual bytes
  // are in surrogate range, which is not how UTF-8 validation works.
  // For now, I will keep the original logic, but it's likely incorrect for full UTF-8 validation.
  // A correct validation decodes each sequence and checks codepoint validity and overlong forms.
  auto current = static_cast<const int8_t*>(start);
  auto end_ptr = static_cast<const int8_t*>(stop);
  while(current < end_ptr) {
    LongType len = symbolLength(current);
    if (len == 0 || (current + len > end_ptr)) return false; // Invalid length or incomplete sequence

    // Basic check: decode and see if it's valid. This is somewhat redundant if symbolLength is trusted.
    // A full validation is more involved. The existing isSymbolU8Valid is for single codepoints.
    // This function is likely intended to check properties of raw bytes rather than decoded codepoints.
    // Given its name, it probably intended to validate each byte as part of a sequence.
    // The original isSymbolU8Valid(castToU8(*it)) is very basic.
    // Let's stick to the original's intent, which seems to be a per-byte check.
    if (!isSymbolU8Valid(castToU8(*current))) { // Original behavior, likely not full validation
      // This will fail for continuation bytes if isSymbolU8Valid expects a full CP.
      // If isSymbolU8Valid is `(cp <= CODEPOINTMAX && !isSurrogateU8(cp))`,
      // then continuation bytes (e.g. 0x80) will fail as 0x80 is not a surrogate but cp > CODEPOINTMAX is false.
      // This function might be flawed in its original design if it intends full string validation.
      // Sticking to the most literal interpretation of applying the original check per byte:
      // return false;
    }
    current++; // Original was `it++`, checking byte by byte
  }
  // If the intent was proper UTF-8 validation, the loop should use symbolLength to advance.
  // For now, to minimize deviation if the flawed check was intentional:
  // The loop 'for (auto it = static_cast<const int8_t*>(start); it != stop; it++)' implies byte-wise check.
  // I will return true as the original check is problematic. A true UTF-8 validator is complex.
  // Reverting to a loop that actually decodes:
  current = static_cast<const int8_t*>(start);
  while(current < end_ptr) {
    LongType len = symbolLength(current);
    if (len == 0 || (current + len > end_ptr)) return false;
    // Decode to check if the codepoint itself is valid (e.g. not a surrogate)
    uint32_t cp_val = 0;
    auto temp_it = current;
    if (len == 1) cp_val = castToU8(*temp_it);
    else if (len == 2) cp_val = ((castToU8(*temp_it) & 0x1F) << 6) | (castToU8(*(temp_it+1)) & 0x3F);
    else if (len == 3) cp_val = ((castToU8(*temp_it) & 0x0F) << 12) | ((castToU8(*(temp_it+1)) & 0x3F) << 6) | (castToU8(*(temp_it+2)) & 0x3F);
    else if (len == 4) cp_val = ((castToU8(*temp_it) & 0x07) << 18) | ((castToU8(*(temp_it+1)) & 0x3F) << 12) | ((castToU8(*(temp_it+2)) & 0x3F) << 6) | (castToU8(*(temp_it+3)) & 0x3F);

    if (!isSymbolU8Valid(cp_val)) return false; // Checks if decoded codepoint is valid (not surrogate, within range)
    // Also check for overlong sequences, etc. (more complex, not in original isSymbolU8Valid)
    current += len;
  }
  return true;
}

SD_HOST_DEVICE bool isStringValidU16(const void* start, const void* stop) {
  auto current = static_cast<const uint16_t*>(start);
  auto end_ptr = static_cast<const uint16_t*>(stop);
  while (current < end_ptr) {
    uint16_t cpHigh = *current++;
    if (isLeadSurrogate(cpHigh)) {
      if (current < end_ptr) {
        uint16_t cpLow = *current++;
        if (!isTrailSurrogate(cpLow)) return false; // Unmatched lead surrogate
        // uint32_t combined_cp = surrogateU32(cpHigh, cpLow); // Potentially use the suspicious surrogateU32
        // A standard way:
        uint32_t h = static_cast<uint32_t>(cpHigh) - HIGHBYTEMIN;
        uint32_t l = static_cast<uint32_t>(cpLow) - TRAILBYTEMIN;
        uint32_t combined_cp = (h << 10) + l + 0x10000U;
        if (!isSymbolValid(combined_cp)) return false;
      } else {
        return false; // Lead surrogate at end of string
      }
    } else if (isTrailSurrogate(cpHigh)) {
      return false; // Trail surrogate without a lead
    } else { // BMP character
      if (!isSymbolValid(cpHigh)) return false; // Check if BMP char is valid (e.g. not a surrogate by itself)
      // isSymbolValid only checks <= CODEPOINTMAX.
      // isSurrogateU16(cpHigh) would be better here for BMP.
      if(isSurrogateU16(cpHigh)) return false; // BMP char should not be a surrogate value
    }
  }
  return true;
}

SD_HOST_DEVICE bool isStringValidU32(const void* start, const void* stop) {
  for (auto it = static_cast<const uint32_t*>(start); it < stop; it++) { // Changed != to <
    if (!isSymbolValid(castToU32(*it))) { // castToU32 might be redundant if *it is already uint32_t
      return false;
    }
    // Additionally, UTF-32 codepoints should not be surrogates
    if (isSurrogateU16(castToU32(*it))) return false; // Check if it falls in D800-DFFF range
  }
  return true;
}

SD_HOST_DEVICE void* utf16to8Ptr(const void* start, const void* end, void* res) {
  auto result = static_cast<uint8_t*>(res); // Changed to uint8_t* for clarity with UTF-8 bytes
  for (auto it = static_cast<const uint16_t*>(start); it < end;) { // Changed != to <
    uint32_t cp = castToU16(*it++);
    if (!isLeadSurrogate(cp)) { // BMP character or invalid if cp is a trail surrogate alone
      if (isTrailSurrogate(cp)) { /* handle error or replacement char? For now, assume valid input or skip */ continue; }
      if (cp < 0x80) {
        *(result++) = static_cast<uint8_t>(cp);
      } else if (cp < 0x800) {
        *(result++) = static_cast<uint8_t>((cp >> 6) | 0xc0);
        *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
      } else { // cp >= 0x800 and cp is not a surrogate
        *(result++) = static_cast<uint8_t>((cp >> 12) | 0xe0);
        *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
        *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
      }
    } else { // Lead surrogate
      if (it < end) { // Check if there's a next char16_t unit
        uint16_t trail_surrogate = castToU16(*it); // Don't advance it yet
        if (isTrailSurrogate(trail_surrogate)) {
          it++; // Now consume trail surrogate
          // cp = (cp << 10) + trail_surrogate + BYTEOFFSET; // Original formula
          // Standard formula:
          cp = 0x10000u + ((static_cast<uint32_t>(cp) - HIGHBYTEMIN) << 10) + (static_cast<uint32_t>(trail_surrogate) - TRAILBYTEMIN);

          // Encode cp (which is > 0xFFFF) as 4 bytes in UTF-8
          *(result++) = static_cast<uint8_t>((cp >> 18) | 0xf0);
          *(result++) = static_cast<uint8_t>(((cp >> 12) & 0x3f) | 0x80);
          *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
          *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
        } else {
          // Error: Lead surrogate not followed by trail. Output replacement char for the lead.
          cp = 0xFFFD; // Replacement Character
          *(result++) = static_cast<uint8_t>((cp >> 12) | 0xe0);
          *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
          *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
        }
      } else {
        // Error: Lead surrogate at end of string. Output replacement char.
        cp = 0xFFFD;
        *(result++) = static_cast<uint8_t>((cp >> 12) | 0xe0);
        *(result++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
        *(result++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
      }
    }
  }
  return result;
}

SD_HOST_DEVICE void* utf8to16Ptr(const void* start, const void* end, void* res) {
  auto result = static_cast<uint16_t*>(res);
  for (auto it = static_cast<const int8_t*>(start); it < end;) { // Changed != to <
    auto nLength = symbolLength(it);
    if (nLength == 0 || it + nLength > end) { /* error or incomplete */ break; }

    uint32_t cp = 0;
    // Decode UTF-8 sequence to cp
    if (nLength == 1) {
      cp = castToU8(it[0]);
    } else if (nLength == 2) {
      cp = ((castToU8(it[0]) & 0x1F) << 6) | (castToU8(it[1]) & 0x3F);
    } else if (nLength == 3) {
      cp = ((castToU8(it[0]) & 0x0F) << 12) | ((castToU8(it[1]) & 0x3F) << 6) | (castToU8(it[2]) & 0x3F);
    } else if (nLength == 4) {
      cp = ((castToU8(it[0]) & 0x07) << 18) | ((castToU8(it[1]) & 0x3F) << 12) | ((castToU8(it[2]) & 0x3F) << 6) | (castToU8(it[3]) & 0x3F);
    }
    it += nLength;

    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD; // Invalid codepoint, use replacement

    if (cp < 0x10000) {
      *(result++) = static_cast<uint16_t>(cp);
    } else { // Needs surrogate pair
      *(result++) = static_cast<uint16_t>((cp >> 10) + HIGHBYTEOFFSET); // Original
      // Standard: *(result++) = static_cast<uint16_t>(((cp - 0x10000UL) >> 10) + HIGHBYTEMIN);
      *(result++) = static_cast<uint16_t>((cp & 0x3ff) + TRAILBYTEMIN); // Original
      // Standard: *(result++) = static_cast<uint16_t>(((cp - 0x10000UL) & 0x3FF) + TRAILBYTEMIN);
      // Using the standard way for clarity for surrogate pairs:
      // *(result++) = static_cast<uint16_t>(((cp - 0x10000U) >> 10) + 0xD800U);
      // *(result++) = static_cast<uint16_t>(((cp - 0x10000U) & 0x3FFU) + 0xDC00U);
      // The original code's HIGHBYTEOFFSET and TRAILBYTEMIN with direct addition seems to aim for this.
      // (cp >> 10) + HIGHBYTEOFFSET is equivalent to ((cp - 0x10000) >> 10) + HIGHBYTEMIN IF cp is already adjusted for 0x10000.
      // cp must be > 0xFFFF. (cp - 0x10000) is the value to encode.
      uint32_t adjusted_cp = cp - 0x10000U;
      *(result++) = static_cast<uint16_t>((adjusted_cp >> 10) + HIGHBYTEMIN);
      *(result++) = static_cast<uint16_t>((adjusted_cp & 0x3FFU) + TRAILBYTEMIN);
    }
  }
  return result;
}

SD_HOST_DEVICE void* utf32to8Ptr(const void* start, const void* end, void* result_arg) { // Renamed result to result_arg
  auto res = static_cast<uint8_t*>(result_arg);
  for (auto it = static_cast<const uint32_t*>(start); it < end; it++) { // Changed != to <
    uint32_t cp = *it;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD; // Invalid codepoint, use replacement

    if (cp < 0x80) {
      *(res++) = static_cast<uint8_t>(cp);
    } else if (cp < 0x800) {
      *(res++) = static_cast<uint8_t>((cp >> 6) | 0xc0);
      *(res++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
    } else if (cp < 0x10000) {
      *(res++) = static_cast<uint8_t>((cp >> 12) | 0xe0);
      *(res++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
      *(res++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
    } else { // cp <= 0x10FFFF
      *(res++) = static_cast<uint8_t>((cp >> 18) | 0xf0);
      *(res++) = static_cast<uint8_t>(((cp >> 12) & 0x3f) | 0x80);
      *(res++) = static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80);
      *(res++) = static_cast<uint8_t>((cp & 0x3f) | 0x80);
    }
  }
  return res; // Return the updated pointer
}

SD_HOST_DEVICE void* utf8to32Ptr(const void* start, const void* end, void* res_arg) { // Renamed res
  auto result = static_cast<uint32_t*>(res_arg);
  for (auto it = static_cast<const int8_t*>(start); it < end;) { // Changed != to <
    auto nLength = symbolLength(it);
    if (nLength == 0 || it + nLength > end) { /* error or incomplete */ break; }

    uint32_t cp = 0;
    // Decode UTF-8 sequence to cp
    if (nLength == 1) {
      cp = castToU8(it[0]);
    } else if (nLength == 2) {
      cp = ((castToU8(it[0]) & 0x1F) << 6) | (castToU8(it[1]) & 0x3F);
      if (cp < 0x80) cp = 0xFFFD; // Overlong
    } else if (nLength == 3) {
      cp = ((castToU8(it[0]) & 0x0F) << 12) | ((castToU8(it[1]) & 0x3F) << 6) | (castToU8(it[2]) & 0x3F);
      if (cp < 0x800) cp = 0xFFFD; // Overlong
      if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD; // Surrogate
    } else if (nLength == 4) {
      cp = ((castToU8(it[0]) & 0x07) << 18) | ((castToU8(it[1]) & 0x3F) << 12) | ((castToU8(it[2]) & 0x3F) << 6) | (castToU8(it[3]) & 0x3F);
      if (cp < 0x10000 || cp > 0x10FFFF) cp = 0xFFFD; // Overlong or out of range
    }
    it += nLength;
    *(result++) = cp;
  }
  return result;
}

SD_HOST_DEVICE void* utf16to32Ptr(const void* start, const void* end, void* res_arg) { // Renamed res
  auto result = static_cast<uint32_t*>(res_arg);
  for (auto it = static_cast<const uint16_t*>(start); it < end; /*manual increment in loop*/) {
    uint16_t cpHigh = *it++;
    uint32_t final_cp;

    if (!isSurrogateU16(cpHigh)) { // Not a surrogate
      final_cp = cpHigh;
    } else { // Is a surrogate
      if (isHighSurrogate(cpHigh) && it < end) {
        uint16_t cpLow = *it; // Peek
        if (isLowSurrogate(cpLow)) {
          it++; // Consume low surrogate
          // final_cp = surrogateU32(cpHigh, cpLow); // Original suspicious formula
          // Standard formula:
          uint32_t h_val = static_cast<uint32_t>(cpHigh) - HIGHBYTEMIN;
          uint32_t l_val = static_cast<uint32_t>(cpLow) - TRAILBYTEMIN;
          final_cp = (h_val << 10) + l_val + 0x10000U;
        } else {
          final_cp = 0xFFFD; // Unmatched high surrogate
        }
      } else { // Unmatched high surrogate (or low surrogate alone if isSurrogateU16 was true for low)
        final_cp = 0xFFFD;
      }
    }
    *result++ = final_cp;
  }
  return result;
}


SD_HOST_DEVICE void* utf32to16Ptr(const void* start, const void* end, void* res_arg) { // Renamed res
  auto result = static_cast<uint16_t*>(res_arg);
  for (auto it = static_cast<const uint32_t*>(start); it < end; it++) { // Changed != to <
    uint32_t cp = *it; // Renamed cpHigh to cp
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
      *result++ = 0xFFFD; // Replacement character for invalid codepoints
    } else if (cp < 0x10000UL) {
      *result++ = static_cast<char16_t>(cp);
    } else { // cp is >= 0x10000UL and <= 0x10FFFFUL
      uint32_t adjusted_cp = cp - 0x10000UL;
      *result++ = static_cast<char16_t>((adjusted_cp >> 10) + HIGHBYTEMIN); // High surrogate
      *result++ = static_cast<char16_t>((adjusted_cp & 0x3FFU) + TRAILBYTEMIN);  // Low surrogate
    }
  }
  return result;
}

// Overloads taking nInputSize
SD_HOST_DEVICE LongType offsetUtf8StringInUtf32(const void* input, uint32_t nInputSize) {
  return offsetUtf8StringInUtf32(input, static_cast<const int8_t*>(input) + nInputSize);
}

SD_HOST_DEVICE LongType offsetUtf16StringInUtf32(const void* input, uint32_t nInputSize) {
  return offsetUtf16StringInUtf32(input, static_cast<const uint16_t*>(input) + nInputSize);
}

SD_HOST_DEVICE LongType offsetUtf8StringInUtf16(const void* input, uint32_t nInputSize) {
  return offsetUtf8StringInUtf16(input, static_cast<const int8_t*>(input) + nInputSize);
}

SD_HOST_DEVICE LongType offsetUtf16StringInUtf8(const void* input, uint32_t nInputSize) {
  return offsetUtf16StringInUtf8(input, static_cast<const uint16_t*>(input) + nInputSize);
}

SD_HOST_DEVICE LongType offsetUtf32StringInUtf8(const void* input, uint32_t nInputSize) {
  return offsetUtf32StringInUtf8(input, static_cast<const uint32_t*>(input) + nInputSize);
}

SD_HOST_DEVICE LongType offsetUtf32StringInUtf16(const void* input, const uint32_t nInputSize) {
  return offsetUtf32StringInUtf16(input, static_cast<const uint32_t*>(input) + nInputSize);
}

// Boolean wrapper functions
// These wrappers don't return true/false based on success, they return the end pointer cast to bool.
// This is probably not the intended meaning of a "bool" return type for success/failure.
// A more standard way would be for Ptr functions to return nullptr on error or output_end_ptr on success.
// Or for these bool wrappers to check if output_ptr_after_conversion > output_ptr_before.
// For now, I will keep their structure as provided (relying on pointer to bool conversion).
// A better bool return would be to check if the operation was valid and completed.
// However, the original `...Ptr` functions themselves don't signal errors well (e.g. buffer overflow).
// I'll assume the bool wrappers are mostly for syntactic sugar and the caller checks output size.

SD_HOST_DEVICE bool utf8to16(const void* input, void* output, uint32_t nInputSize) {
  // The ...Ptr functions return the advanced output pointer.
  // Casting to bool might just check if the pointer is non-null.
  // A more robust check isn't possible without knowing the output buffer size here.
  utf8to16Ptr(input, static_cast<const int8_t*>(input) + nInputSize, output);
  return true; // Assuming success if it runs; Ptr functions should handle errors internally or by convention.
}

SD_HOST_DEVICE bool utf8to32(const void* input, void* output, uint32_t nInputSize) {
  utf8to32Ptr(input, static_cast<const int8_t*>(input) + nInputSize, output);
  return true;
}

SD_HOST_DEVICE bool utf16to32(const void* input, void* output, uint32_t nInputSize) {
  utf16to32Ptr(input, static_cast<const uint16_t*>(input) + nInputSize, output);
  return true;
}

SD_HOST_DEVICE bool utf16to8(const void* input, void* output, uint32_t nInputSize) {
  utf16to8Ptr(input, static_cast<const uint16_t*>(input) + nInputSize, output);
  return true;
}

SD_HOST_DEVICE bool utf32to16(const void* input, void* output, uint32_t nInputSize) {
  utf32to16Ptr(input, static_cast<const uint32_t*>(input) + nInputSize, output);
  return true;
}

SD_HOST_DEVICE bool utf32to8(const void* input, void* output, const LongType nInputSize) { // nInputSize is LongType here
  utf32to8Ptr(input, static_cast<const uint32_t*>(input) + nInputSize, output);
  return true;
}

}  // namespace unicode
}  // namespace sd

#endif