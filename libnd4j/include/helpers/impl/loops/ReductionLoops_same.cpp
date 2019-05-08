/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

    template<typename X>
    template <typename OpType>
    void ReductionSameLoops<X>::innerloopReduce(X* x, Nd4jLong* xShapeInfo, X* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams) {
        ReductionLoops<X,X,X>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams);
    }

    template<typename X>
    void ReductionSameLoops<X>::wrapper(const int opNum, X *vx, Nd4jLong *xShapeInfo, X *vz,
                                           Nd4jLong *zShapeInfo, Nd4jLong *tadShapeInfo,
                                           Nd4jLong *tadOffsets,
                                           X *vextraParams) {
        auto x = reinterpret_cast<X *>(vx);
        auto z = reinterpret_cast<X *>(vz);
        auto extraParams = reinterpret_cast<X *>(vextraParams);

        DISPATCH_BY_OPNUM_T(innerloopReduce, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams), REDUCE_SAME_OPS);
    }

    BUILD_SINGLE_TEMPLATE(template class ReductionSameLoops, , LIBND4J_TYPES);
}
