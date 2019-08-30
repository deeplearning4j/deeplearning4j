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

#include "../pairwise_transform.h"

namespace functions {
    namespace pairwise_transforms {
        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *y,
                Nd4jLong *yShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *extraParams) {

        }

        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *x,
                Nd4jLong xStride,
                void *y,
                Nd4jLong yStride,
                void *z,
                Nd4jLong resultStride,
                void *extraParams,
                Nd4jLong len) {

        }


        template <typename X, typename Y, typename Z>
        template<typename OpType>
        void PairWiseTransform<X, Y, Z>:: exec(
                void *vx,
                Nd4jLong* xShapeInfo,
                void *vy,
                Nd4jLong* yShapeInfo,
                void *vresult,
                Nd4jLong* zShapeInfo,
                void *vextraParams) {

        }

        template <typename X, typename Y, typename Z>
        template<typename OpType>
        void PairWiseTransform<X, Y, Z>::exec(void *vx,
                         Nd4jLong xStride,
                         void *vy,
                         Nd4jLong yStride,
                         void *vresult,
                         Nd4jLong resultStride,
                         void *vextraParams,
                         const Nd4jLong len) {

        }
    }
}