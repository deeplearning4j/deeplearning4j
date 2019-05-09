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
#include <types/float16.h>
#include <ops/declarable/helpers/batched_gemm.h>
#include <helpers/BlasHelper.h>


namespace nd4j {
    namespace ops {
        namespace helpers {
        

            template <typename T>
            void __bgemm(std::vector<NDArray*>& vA, std::vector<NDArray*>& vB, std::vector<NDArray*>& vC, NDArray* alphas, NDArray* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC) {

            };


            void _bgemm(nd4j::LaunchContext * context, std::vector<NDArray*>& vA, std::vector<NDArray*>& vB, std::vector<NDArray*>& vC, NDArray* alphas, NDArray* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC) {
                auto xType = vA.at(0)->dataType();

                BUILD_SINGLE_SELECTOR(xType, __bgemm, (vA, vB, vC, alphas, betas, transA, transB, M, N, K, ldA, ldB, ldC), FLOAT_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void __bgemm, (std::vector<NDArray*>& vA, std::vector<NDArray*>& vB, std::vector<NDArray*>& vC, NDArray* alphas, NDArray* betas, int transA, int transB, int M, int N, int K, int ldA, int ldB, int ldC), FLOAT_TYPES);
        }
    }
}