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
#include <system/pointercast.h>
#include <types/types.h>

using namespace simdOps;

namespace sd {

    template<typename X, typename Z>
    template <typename OpType>
    void ReductionFloatLoops<X, Z>::innerloopReduce(const X * x, const Nd4jLong* xShapeInfo, Z* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, Z* extraParams, int64_t start, int64_t stop) {
#ifndef INLINE_LOOPS
        ReductionLoops<X,Z,Z>::template loopReduce<OpType>(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams, start, stop);
#endif
    }

    template<typename X, typename Y>
    void ReductionFloatLoops<X, Y>::wrapper(const int opNum,
                                            const X *x, const Nd4jLong *xShapeInfo,
                                            Y *z, const Nd4jLong *zShapeInfo,
                                            const Nd4jLong *tadShapeInfo, const Nd4jLong *tadOffsets,
                                            Y *extraParams,
                                            int64_t start, int64_t stop) {
#ifndef INLINE_LOOPS
        DISPATCH_BY_OPNUM_TT(innerloopReduce, PARAMS(x, xShapeInfo, z, zShapeInfo, tadShapeInfo, tadOffsets, extraParams, start, stop), REDUCE_FLOAT_OPS);
#endif
    }

    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ReductionFloatLoops, , LIBND4J_TYPES, FLOAT_TYPES_1);
}


