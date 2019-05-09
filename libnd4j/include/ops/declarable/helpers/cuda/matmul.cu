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
// Created by raver119 on 20.12.17.
//

#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename X, typename Y, typename Z>
            void __matmul(NDArray *vA, NDArray *vB, NDArray *vC, int transA, int transB, double alpha, double beta) {

            }


            void _matmul(nd4j::LaunchContext * context, NDArray *vA, NDArray *vB, NDArray *vC, int transA, int transB, double alpha, double beta) {
                BUILD_TRIPLE_SELECTOR(vA->dataType(), vB->dataType(), vC->dataType(), __matmul, (vA, vB, vC, transA, transB, alpha, beta), LIBND4J_TYPES, LIBND4J_TYPES, LIBND4J_TYPES);
            }

            BUILD_TRIPLE_TEMPLATE(template void __matmul, (NDArray *A, NDArray *B, NDArray *C, int transA, int transB, double alpha, double beta), LIBND4J_TYPES, LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}
