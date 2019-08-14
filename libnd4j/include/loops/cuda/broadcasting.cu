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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>
#include <types/types.h>
#include <Environment.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <StringUtils.h>
#include <specials_cuda.h>

namespace functions {
    namespace broadcast {
        template <typename X, typename Y, typename Z>
        void Broadcast<X, Y, Z>::execInverse(int opNum,
                                void *x,
                                Nd4jLong *xShapeInfo,
                                void *y,
                                Nd4jLong *yShapeInfo,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffset,
                                Nd4jLong *tadShapeInfoZ,
                                Nd4jLong *tadOffsetZ) {
            //
        }

        template <typename X, typename Y, typename Z>
        void Broadcast<X, Y, Z>::exec(int opNum,
                         void *x,
                         Nd4jLong *xShapeInfo,
                         void *y,
                         Nd4jLong *yShapeInfo,
                         void *result,
                         Nd4jLong *resultShapeInfo,
                         int *dimension,
                         int dimensionLength,
                         Nd4jLong *tadShapeInfo,
                         Nd4jLong *tadOffset,
                         Nd4jLong *tadShapeInfoZ,
                         Nd4jLong *tadOffsetZ) {

        }

        /**
         * CPU execution
         * @param x the input
         * @param xShapeInfo the x shape information
         * @param y the y data
         * @param yShapeInfo the y shape information
         * @param result the result
         * @param resultShapeInfo the result shape information
         * @param dimension the dimension to broadcast along long
         * @param dimensionLength the length of the dimension buffer
         */
        template <typename X, typename Y, typename Z>
        template<typename OpType>
        void Broadcast<X, Y, Z>::exec(void *x,
                         Nd4jLong *xShapeInfo,
                         void *y,
                         Nd4jLong *yShapeInfo,
                         void *result,
                         Nd4jLong *resultShapeInfo,
                         int *dimension,
                         int dimensionLength,
                         Nd4jLong *tadShapeInfo,
                         Nd4jLong *tadOffset,
                         Nd4jLong *tadShapeInfoZ,
                         Nd4jLong *tadOffsetZ) {
            //
        }


        template <typename X, typename Y, typename Z>
        template<typename OpType>
        void Broadcast<X, Y, Z>::execInverse(void *x,
                                Nd4jLong *xShapeInfo,
                                void *y,
                                Nd4jLong *yShapeInfo,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffset,
                                Nd4jLong *tadShapeInfoZ,
                                Nd4jLong *tadOffsetZ) {

        }
    }
}