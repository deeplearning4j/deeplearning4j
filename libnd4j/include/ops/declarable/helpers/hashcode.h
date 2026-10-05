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
// @author raver119@gmail.com
//

#ifndef DEV_TESTS_HASHCODE_H
#define DEV_TESTS_HASHCODE_H
#include "helpers.h"

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
SD_INLINE SD_HOST_DEVICE LongType longBytes(T value);

template <>
SD_INLINE SD_HOST_DEVICE LongType longBytes(float value) {
  int intie = *(int *)&value;
  return static_cast<LongType>(intie);
}

template <>
SD_INLINE SD_HOST_DEVICE LongType longBytes(double value) {
  LongType longie = *(LongType *)&value;
  return longie;
}

template <>
SD_INLINE SD_HOST_DEVICE LongType longBytes(float16 value) {
  return longBytes<float>((float)value);
}

template <>
SD_INLINE SD_HOST_DEVICE LongType longBytes(LongType value) {
  return value;
}

template <>
SD_INLINE SD_HOST_DEVICE LongType longBytes(bfloat16 value) {
  return longBytes<float>((float)value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE LongType longBytes(T value) {
  return longBytes<LongType>((LongType)value);
}

// One step of the hash of a block, r = 31 * r + v from r = 1. The hash of an array is a tree of these: the elements in
// C order are hashed in blocks of 32, then the hashes of a level in blocks of 32, until one hash is left. The
// arithmetic wraps around in 64 bits (a signed overflow would be undefined).
SD_INLINE SD_HOST_DEVICE LongType hashCodeStep(const LongType r, const LongType v) {
  return static_cast<LongType>(31ULL * static_cast<unsigned long long>(r) + static_cast<unsigned long long>(v));
}

SD_LIB_HIDDEN void hashCode(LaunchContext *context, NDArray &array, NDArray &result);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_HASHCODE_H
