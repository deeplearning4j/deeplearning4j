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

namespace sd {

    template<typename X>
    template <typename OpType>
    void ReductionSameLoops<X>::innerloopReduce(const X* x, const Nd4jLong* xShapeInfo, X* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, X* extraParams, int64_t start, int64_t stop) {
#ifndef INLINE_LOOPS
        ReductionLoops<X,X,X>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams, start, stop);
#endif
    }

    template<typename X>
    void ReductionSameLoops<X>::wrapper(const int opNum, const X *vx, const Nd4jLong *xShapeInfo, X *vz,
                                        const Nd4jLong *zShapeInfo, const Nd4jLong *tadShapeInfo,
                                        const Nd4jLong *tadOffsets,
                                           X *vextraParams, int64_t start, int64_t stop) {
#ifndef INLINE_LOOPS
        auto x = reinterpret_cast<X *>(vx);
        auto z = reinterpret_cast<X *>(vz);
        auto extraParams = reinterpret_cast<X *>(vextraParams);

        DISPATCH_BY_OPNUM_T(innerloopReduce, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams, start, stop), REDUCE_SAME_OPS);
#endif
    }

    BUILD_SINGLE_TEMPLATE(template class ReductionSameLoops, , LIBND4J_TYPES);
}
