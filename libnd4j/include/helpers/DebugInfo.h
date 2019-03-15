/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <pointercast.h>
#include <op_boilerplate.h>
#include <Environment.h>
#include <StringUtils.h>
#include <string>
#include <dll.h>
#include <templatemath.h>

#ifdef __CUDACC__

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#endif

namespace nd4j {
    struct ND4J_EXPORT DebugInfo {
       double _minValue;
       double _maxValue;
       double _meanValue;
       double _stdDevValue;
       Nd4jLong _zeroCount;
       Nd4jLong _positiveCount;
       Nd4jLong _negativeCount;
       Nd4jLong _infCount;
       Nd4jLong _nanCount;
    };

    FORCEINLINE bool operator==(DebugInfo const& first, DebugInfo const& second) {
        return nd4j::math::nd4j_abs(first._minValue - second._minValue) < 0.000001 &&
        nd4j::math::nd4j_abs(first._maxValue  -   second._maxValue) < 0.000001  &&
        nd4j::math::nd4j_abs(first._meanValue -  second._meanValue) < 0.000001  &&
        nd4j::math::nd4j_abs(first._stdDevValue - second._stdDevValue) < 0.000001  &&
        first._zeroCount   ==   second._zeroCount &&
        first._positiveCount == second._positiveCount &&
        first._negativeCount == second._negativeCount &&
        first._infCount ==      second._infCount &&
        first._nanCount ==      second._nanCount;

    }

}


#endif //LIBND4J_DEBUGHELPER_H
