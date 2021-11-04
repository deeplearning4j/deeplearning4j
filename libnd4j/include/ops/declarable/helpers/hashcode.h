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
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(T value);

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(float value) {
  int intie = *(int *)&value;
  return static_cast<sd::LongType>(intie);
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(double value) {
  sd::LongType longie = *(sd::LongType *)&value;
  return longie;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(float16 value) {
  return longBytes<float>((float)value);
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(sd::LongType value) {
  return value;
}

template <>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(bfloat16 value) {
  return longBytes<float>((float)value);
}

template <typename T>
SD_INLINE SD_HOST_DEVICE sd::LongType longBytes(T value) {
  return longBytes<sd::LongType>((sd::LongType)value);
}

SD_LIB_HIDDEN void hashCode(LaunchContext *context, NDArray &array, NDArray &result);
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_HASHCODE_H
