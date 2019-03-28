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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//

#ifndef LIBND4J_LOOPS_H
#define LIBND4J_LOOPS_H

#include <functional>
#include <pointercast.h>
#include <shape.h>
#include <OmpLaunchHelper.h>
#include <DataTypeUtils.h>
#include <ops.h>
#include <indexreduce.h>
#include <openmp_pragmas.h>

namespace nd4j {

    class Loops {
    private:
        enum LoopKind {SMALLARR2DX, EWS1, EWSNONZERO, RANK1, RANK2, RANK3, RANK4, RANK5, X_EWSNONZERO, Z_EWSNONZERO, COMMON};

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopTadXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo);
    public:
        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Y, typename Z> 
        static void loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                            const Y* y, const Nd4jLong* yShapeInfo,
                                  Z* z, const Nd4jLong* zShapeInfo,
                                  Z* extraParams,
                            std::function<Z(X,Y,Z*)> op);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Z, typename E, typename OpType>
        static void loopTadXZ(const X* x, const Nd4jLong* xShapeInfo,
                                    Z* z, const Nd4jLong* zShapeInfo,
                              const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                              const int* dimsToExclude,
                              const int dimsLen,
                              E* extraParams);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X,typename OpType>
        static void loopIndexTadXZ(const X* x, const Nd4jLong* xShapeInfo,  Nd4jLong* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, X* extraParams);
    };
}


#endif //LIBND4J_LOOPS_H
