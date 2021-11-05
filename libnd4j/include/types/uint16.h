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

#ifndef LIBND4J_UINT16_H
#define LIBND4J_UINT16_H
#include <stdint.h>
#include <system/op_boilerplate.h>

namespace sd {

uint16_t SD_INLINE SD_HOST_DEVICE cpu_float2uint16(float data);
float SD_INLINE SD_HOST_DEVICE cpu_uint162float(uint16_t data);

struct uint16 {
  uint16_t data;

  SD_INLINE SD_HOST_DEVICE uint16();
  SD_INLINE SD_HOST_DEVICE ~uint16();

  template <class T>
  SD_INLINE SD_HOST_DEVICE uint16(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE uint16& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);

  SD_INLINE SD_HOST_DEVICE void assign(float rhs);
};

//////////////////// IMPLEMENTATIONS

float SD_HOST_DEVICE cpu_uint162float(uint16_t data) { return static_cast<float>(data); }

uint16_t SD_HOST_DEVICE cpu_float2uint16(float data) {
  auto t = static_cast<int>(data);
  if (t > 65536) t = 65536;
  if (t < 0) t = 0;

  return static_cast<uint16_t>(t);
}

SD_HOST_DEVICE uint16::uint16() { data = cpu_float2uint16(0.0f); }

SD_HOST_DEVICE uint16::~uint16() {
  //
}

template <class T>
SD_HOST_DEVICE uint16::uint16(const T& rhs) {
  assign(rhs);
}

template <class T>
SD_HOST_DEVICE uint16& uint16::operator=(const T& rhs) {
  assign(rhs);
  return *this;
}

SD_HOST_DEVICE uint16::operator float() const { return cpu_uint162float(data); }

SD_HOST_DEVICE void uint16::assign(float rhs) { data = cpu_float2uint16(rhs); }

SD_HOST_DEVICE void uint16::assign(double rhs) { assign((float)rhs); }
}  // namespace sd

#endif  // LIBND4J_UINT16_H
