/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#ifndef LIBND4J_HELPERS_MATMUL_H
#define LIBND4J_HELPERS_MATMUL_H

#include <array/NDArray.h>

namespace sd {
    namespace ops {
        namespace helpers {

            void _matmul(sd::LaunchContext * context, NDArray *A, NDArray *B, NDArray *C, int transA, int transB, double alpha = 1., double beta = 0.);
        }
    }
}

#endif //LIBND4J_MATMUL_H
