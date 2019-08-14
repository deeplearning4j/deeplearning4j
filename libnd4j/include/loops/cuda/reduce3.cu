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


#include <op_boilerplate.h>
#include <loops/reduce3.h>
#include <loops/legacy_ops.h>
#include <types/types.h>
#include <specials_cuda.h>

namespace functions {
    namespace reduce3 {
        template <typename X, typename Y>
        template<typename OpType>
        void Reduce3<X,Y>::execScalar(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo) {

        }


        template <typename X, typename Y>
        void Reduce3<X,Y>::execScalar(const int opNum, void *x, Nd4jLong *xShapeInfo, void *extraParamsVals, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo) {

        }


        template <typename X, typename Y>
        template<typename OpType>
        void Reduce3<X,Y>::exec(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength) {

        }


        template <typename X, typename Y>
        template<typename OpType>
        void Reduce3<X,Y>::exec(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

        }


        template <typename X, typename Y>
        template<typename OpType>
        void Reduce3<X,Y>::execAll(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength,  Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

        }


        template <typename X, typename Y>
        void Reduce3<X,Y>::exec(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength) {

        }


        template <typename X, typename Y>
        void Reduce3<X,Y>::exec(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

        }


        template <typename X, typename Y>
        void Reduce3<X,Y>::execAll(const int opNum, void *vx, Nd4jLong *xShapeInfo, void *extraParamsVals, void *vy, Nd4jLong *yShapeInfo, void *vz, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

        }

    }
}