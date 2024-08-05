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
// Created by raver119 on 10.11.2017.
//

#ifndef LIBND4J_BITWISEUTILS_H
#define LIBND4J_BITWISEUTILS_H
#include <array/ByteOrder.h>
#include <system/op_boilerplate.h>

#include <climits>
#include <vector>

namespace sd {
class SD_LIB_EXPORT BitwiseUtils {
 public:
  /**
   * This method returns first non-zero bit index
   * @param holder
   * @return
   */
  static int valueBit(int holder);

  /**
   *  This method returns vector representation of bits.
   *
   *  PLEASE NOTE: Result is ALWAYS left-to-right
   */
  static std::vector<LongType> valueBits(int holder);

  /**
   *  This method returns TRUE if it's called on Big-Endian system, and false otherwise
   */
  static bool isBE();

  /**
   * This method returns enum
   * @return
   */
  static ByteOrder asByteOrder();

  /**
   * This method swaps bytes: LE vs BE
   * @tparam T
   * @param v
   * @return
   */
  template <typename T>
  static SD_INLINE T swap_bytes(T v) {
    static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");

    union S {
      T v;
      unsigned char u8[sizeof(T)];
      S(T val) { v = val; }
    };

    S source(v);
    S dest(v);

    for (size_t k = 0; k < sizeof(T); k++) dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.v;
  }

  /**
   * This method flips bits in given value
   *
   * @tparam T
   * @param v
   * @return
   */
  static int SD_INLINE flip_bits(int v) { return ~v; }

  static int8_t SD_INLINE flip_bits(int8_t v) { return ~v; }

  static int16_t SD_INLINE flip_bits(int16_t v) { return ~v; }

  static uint8_t SD_INLINE flip_bits(uint8_t v) { return ~v; }

  static uint16_t SD_INLINE flip_bits(uint16_t v) { return ~v; }

  static uint32_t SD_INLINE flip_bits(uint32_t v) { return ~v; }

  static uint64_t SD_INLINE flip_bits(uint64_t v) { return ~v; }

  static LongType SD_INLINE flip_bits(LongType v) { return ~v; }
};
}  // namespace sd

#endif  // LIBND4J_BITWISEUTILS_H
