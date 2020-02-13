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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_TRUEBROADCASTHELPER_H
#define LIBND4J_TRUEBROADCASTHELPER_H

#include <NDArray.h>

namespace nd4j    {
namespace helpers {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
class TrueBroadcastHelper {

        #ifdef __CUDACC__
            template <typename OpType>
            static __host__ void execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo);
        #else
            template <typename OpType>
            static void exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
        #endif

    public:
        static void exec(const nd4j::broadcast::Ops opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
};

template <typename X, typename Y>
class TrueBroadcastBoolHelper {

        #ifdef __CUDACC__
            template <typename OpType>
            static __host__ void execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo);
        #else
            template <typename OpType>
            static void exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
        #endif

    public:

        static void exec(const nd4j::broadcast::BoolOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
};

////////////////////////////////////////////////////////////////////////
template <typename X>
class TrueBroadcastIntHelper {

        #ifdef __CUDACC__
            template <typename OpType>
            static __host__ void execLauncher(dim3 launchDims, cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, const void *vy, const Nd4jLong *yShapeInfo, void *vz, const Nd4jLong *zShapeInfo);
        #else
            template <typename OpType>
            static void exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
        #endif

    public:

        static void exec(const nd4j::broadcast::IntOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr);
};


}
}



#endif //LIBND4J_BIDIAGONALUP_H
