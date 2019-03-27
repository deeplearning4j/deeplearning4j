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

using namespace simdOps;

namespace nd4j {

    template<typename X, typename Z>
    template <typename OpType>
    void ReductionBoolLoops<X, Z>::innerloopTadXZ(const X* x, const Nd4jLong* xShapeInfo, Z* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, X* extraParams) {
        ReductionLoops<X,Z,Z>::template loopTadXZ<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, dimsToExclude, dimsLen, extraParams);
    }

    template<typename X, typename Y>
    void ReductionBoolLoops<X, Y>::wrapper(const int opNum, const X *vx, const Nd4jLong *xShapeInfo, Y *vz,
                                            const Nd4jLong *zShapeInfo, const Nd4jLong *tadShapeInfo,
                                            const Nd4jLong *tadOffsets, const int *dimsToExclude,
                                            const int dimsLen, X *vextraParams) {
        const auto x = reinterpret_cast<const X *>(vx);
        auto z = reinterpret_cast<Y *>(vz);
        auto extraParams = reinterpret_cast<X *>(vextraParams);

        DISPATCH_BY_OPNUM_TT(innerloopTadXZ, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, dimsToExclude, dimsLen, extraParams), REDUCE_BOOL_OPS);
    }

    BUILD_DOUBLE_TEMPLATE(template class ReductionFloatLoops, , LIBND4J_TYPES, BOOL_TYPES);
}

