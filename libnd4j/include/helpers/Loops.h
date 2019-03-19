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
        enum LoopKind {EWS1, EWSNONZERO, RANK1, RANK2, RANK3, RANK4, RANK5, X_EWSNONZERO, Z_EWSNONZERO, COMMON};

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        static LoopKind deduceKindOfLoopTadXZ(const Nd4jLong* tadShapeInfo, const Nd4jLong* zShapeInfo);

        template<typename X, typename Z>
        static void _loopTadXZ(const void* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, void* z, const Nd4jLong* zShapeInfo, void* extraParams);

        template<typename X>
        static void _loopIndexTadXZ(const void* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, Nd4jLong* z, const Nd4jLong* zShapeInfo, void* extraParams);

    public:
        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Y, typename Z> 
        static void loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                            const Y* y, const Nd4jLong* yShapeInfo,
                                  Z* z, const Nd4jLong* zShapeInfo,
                                  Z* extraParams,
                            std::function<Z(X,Y,Z*)> op);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Z, typename E>
        static void loopTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                                Z* z, const Nd4jLong* zShapeInfo,
                                                E* extraParams,
                                                std::function<X(const X*)>      startVal, 
                                                std::function<Z(Z,Z,E*)>        update,
                                                std::function<Z(X,E*)>          op,
                                                std::function<Z(Z,Nd4jLong,E*)> postPr);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename E>
        static void loopIndexTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                              Nd4jLong* z, const Nd4jLong* zShapeInfo,
                                              E* extraParams,
                                              std::function<functions::indexreduce::IndexValue<X>(X*)> startVal, 
                                              std::function<functions::indexreduce::IndexValue<X>(functions::indexreduce::IndexValue<X>&, functions::indexreduce::IndexValue<X>, E*)> update);
    };
}


#endif //LIBND4J_LOOPS_H
