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

#include <Loops.h>
#include <pointercast.h>
#include <types/types.h>

using namespace simdOps;

namespace nd4j {

    template<typename X, typename Z>
    template <typename OpType>
    void Reduction3Loops<X,Z>::innerloopReduce3(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, int* dims, int dimsLen, Z* extraParams, int64_t start, int64_t stop) {
         Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dims, dimsLen, extraParams, start, stop);
    }

    template<typename X, typename Z>
    template <typename OpType>
    void Reduction3Loops<X,Z>::innerloopReduce3All(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets, Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets, Z* extraParams, int64_t start, int64_t stop) {
         Reduction3Loops<X,Z>::template loopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xTadOffsets, yTadShapeInfo, yTadOffsets, extraParams, start, stop);
    }

    template<typename X, typename Y>
    void Reduction3Loops<X, Y>::wrapper(const int opNum, X *x, Nd4jLong *xShapeInfo, X *y, Nd4jLong *yShapeInfo, Y *z, Nd4jLong *zShapeInfo, int* dims, int dimsLen, Y *extraParams, int64_t start, int64_t stop) {
        DISPATCH_BY_OPNUM_TT(innerloopReduce3, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dims, dimsLen, extraParams, start, stop), REDUCE3_OPS);
    }

    template<typename X, typename Y>
    void Reduction3Loops<X, Y>::wrapperAll(const int opNum, X *x, Nd4jLong *xShapeInfo, X *y, Nd4jLong *yShapeInfo, Y *z, Nd4jLong *zShapeInfo, Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets, Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets, Y* extraParams, int64_t start, int64_t stop) {
        DISPATCH_BY_OPNUM_TT(innerloopReduce3All, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,  xTadShapeInfo, xTadOffsets, yTadShapeInfo, yTadOffsets, extraParams, start, stop), REDUCE3_OPS);
    }

    BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT Reduction3Loops, , LIBND4J_TYPES, FLOAT_TYPES_3);
}
