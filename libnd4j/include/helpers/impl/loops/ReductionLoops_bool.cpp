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
    void ReductionBoolLoops<X, Z>::innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams) {
        ReductionLoops<X,Z,X>::template loopTadXZ<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams);
    }

    template<typename X, typename Y>
    void ReductionBoolLoops<X, Y>::wrapper(const int opNum, X *x, Nd4jLong *xShapeInfo, Y *z,
                                            Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo,
                                            Nd4jLong *tadOffsets,
                                            X *extraParams) {
        DISPATCH_BY_OPNUM_TT(innerloopTadXZ, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams), REDUCE_BOOL_OPS);
    }

    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReductionBoolLoops, , LIBND4J_TYPES, BOOL_TYPES);
}

