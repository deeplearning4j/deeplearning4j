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
// Created by GS aka shugeo <sgazeos@gmail.com> on 3/12/19.
//

#ifndef LIBND4J__DEBUG_INFO_HELPER__H
#define LIBND4J__DEBUG_INFO_HELPER__H

#include <helpers/StringUtils.h>
#include <math/templatemath.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>

#include <string>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#endif

namespace sd {
struct DebugInfo {
  double _minValue;
  double _maxValue;
  double _meanValue;
  double _stdDevValue;
  sd::LongType _zeroCount;
  sd::LongType _positiveCount;
  sd::LongType _negativeCount;
  sd::LongType _infCount;
  sd::LongType _nanCount;
};

SD_INLINE bool operator==(DebugInfo const& first, DebugInfo const& second) {
  return sd::math::sd_abs(first._minValue - second._minValue) < 0.000001 &&
         sd::math::sd_abs(first._maxValue - second._maxValue) < 0.000001 &&
         sd::math::sd_abs(first._meanValue - second._meanValue) < 0.000001 &&
         sd::math::sd_abs(first._stdDevValue - second._stdDevValue) < 0.000001 &&
         first._zeroCount == second._zeroCount && first._positiveCount == second._positiveCount &&
         first._negativeCount == second._negativeCount && first._infCount == second._infCount &&
         first._nanCount == second._nanCount;
}

}  // namespace sd

#endif  // LIBND4J_DEBUGHELPER_H
