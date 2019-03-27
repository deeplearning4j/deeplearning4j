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
// @author raver119@gmail.com
//

#include "ReductionLoops.hpp"
#include <pointercast.h>
#include <types/types.h>

using namespace simdOps;

namespace nd4j {

    template<typename X, typename Z>
    template <typename OpType>
    void ReductionFloatLoops<X, Z>::innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, int* dimsToExclude, int dimsLen, Z* extraParams) {
        ReductionLoops<X,Z,Z>::template loopTadXZ<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, dimsToExclude, dimsLen, extraParams);
    }

    template<typename X, typename Y>
    void ReductionFloatLoops<X, Y>::wrapper(const int opNum, X *vx, Nd4jLong *xShapeInfo, Y *vz,
                                                  Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo,
                                                  Nd4jLong *tadOffsets, int *dimsToExclude,
                                                  int dimsLen, Y *vextraParams) {
        const auto x = reinterpret_cast<X *>(vx);
        auto z = reinterpret_cast<Y *>(vz);
        auto extraParams = reinterpret_cast<Y *>(vextraParams);

        DISPATCH_BY_OPNUM_TT(innerloopTadXZ, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, dimsToExclude, dimsLen, extraParams), REDUCE_FLOAT_OPS);
    }
    template class ReductionFloatLoops<float16, float>;
    template class ReductionFloatLoops<bfloat16, float>;
    template class ReductionFloatLoops<float, float>;
    template class ReductionFloatLoops<double, float>;
    template class ReductionFloatLoops<bool, float>;
    template class ReductionFloatLoops<int8_t, float>;
    template class ReductionFloatLoops<uint8_t, float>;
    template class ReductionFloatLoops<int16_t, float>;
    template class ReductionFloatLoops<int, float>;
    template class ReductionFloatLoops<Nd4jLong , float>;

    template class ReductionFloatLoops<float16, double>;
    template class ReductionFloatLoops<bfloat16, double>;
    template class ReductionFloatLoops<float, double>;
    template class ReductionFloatLoops<double, double>;
    template class ReductionFloatLoops<bool, double>;
    template class ReductionFloatLoops<int8_t, double>;
    template class ReductionFloatLoops<uint8_t, double>;
    template class ReductionFloatLoops<int16_t, double>;
    template class ReductionFloatLoops<int, double>;
    template class ReductionFloatLoops<Nd4jLong , double>;

    template class ReductionFloatLoops<float16, bfloat16>;
    template class ReductionFloatLoops<bfloat16, bfloat16>;
    template class ReductionFloatLoops<float, bfloat16>;
    template class ReductionFloatLoops<double, bfloat16>;
    template class ReductionFloatLoops<bool, bfloat16>;
    template class ReductionFloatLoops<int8_t, bfloat16>;
    template class ReductionFloatLoops<uint8_t, bfloat16>;
    template class ReductionFloatLoops<int16_t, bfloat16>;
    template class ReductionFloatLoops<int, bfloat16>;
    template class ReductionFloatLoops<Nd4jLong , bfloat16>;

    template class ReductionFloatLoops<float16, float16>;
    template class ReductionFloatLoops<bfloat16, float16>;
    template class ReductionFloatLoops<float, float16>;
    template class ReductionFloatLoops<double, float16>;
    template class ReductionFloatLoops<bool, float16>;
    template class ReductionFloatLoops<int8_t, float16>;
    template class ReductionFloatLoops<uint8_t, float16>;
    template class ReductionFloatLoops<int16_t, float16>;
    template class ReductionFloatLoops<int, float16>;
    template class ReductionFloatLoops<Nd4jLong , float16>;


}


