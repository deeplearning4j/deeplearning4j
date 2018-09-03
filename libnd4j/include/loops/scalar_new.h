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
// Created by raver on 9/2/2018.
//

#ifndef LIBND4J_SCALAR_NEW_H
#define LIBND4J_SCALAR_NEW_H

#include <pointercast.h>

namespace functions {
    namespace scalar {
        template <typename X, typename Y>
        class NewScalarTransform {
        public:
            static void transform(const int opNum, X *x, Nd4jLong *xShapeInfo, X *result, Nd4jLong *resultShapeInfo, Y scalar, X *extraParams);

            template<typename OpType>
            static void transform(X *x, Nd4jLong *xShapeInfo, X *result, Nd4jLong *resultShapeInfo, Y scalar, X *extraParams);

            template<typename OpType>
            static void transform(X *x, Nd4jLong xStride, X *result, Nd4jLong resultStride, Y scalar, X *extraParams, const Nd4jLong n);
        };
    }
}

#endif //DEV_TESTS_SCALAR_NEW_H
