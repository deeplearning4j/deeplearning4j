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

#ifndef LIBND4J_INT8_H
#define LIBND4J_INT8_H
#include <stdint.h>
#include <system/op_boilerplate.h>

namespace sd {

float SD_INLINE SD_HOST_DEVICE cpu_int82float(int8_t data);
int8_t SD_INLINE SD_HOST_DEVICE cpu_float2int8(float data);

struct int8 {
  int8_t data;

  SD_INLINE SD_HOST_DEVICE int8();
  SD_INLINE SD_HOST_DEVICE ~int8() = default;

  template <class T>
  SD_INLINE SD_HOST_DEVICE int8(const T& rhs);

  template <class T>
  SD_INLINE SD_HOST_DEVICE int8& operator=(const T& rhs);

  SD_INLINE SD_HOST_DEVICE operator float() const;

  SD_INLINE SD_HOST_DEVICE void assign(double rhs);

  SD_INLINE SD_HOST_DEVICE void assign(float rhs);
};

float cpu_int82float(int8_t data) { return (float)((int)data); }

int8_t cpu_float2int8(float data) {
  int t = (int)data;
  if (t > 127) t = 127;
  if (t < -128) t = -128;

  return (int8_t)t;
}

int8::int8() { data = cpu_float2int8(0.0f); }

template <class T>
int8::int8(const T& rhs) {
  assign(rhs);
}

template <class T>
int8& int8::operator=(const T& rhs) {
  assign(rhs);
  return *this;
}

int8::operator float() const { return cpu_int82float(data); }

void int8::assign(double rhs) { assign((float)rhs); }

void int8::assign(float rhs) { data = cpu_float2int8(rhs); }
}  // namespace sd

#endif  // LIBND4J_INT8_H
