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
//  @created by George A. Shulinok <sgazeos@gmail.com> 4/18/2019
//

#ifndef LIBND4J_HEADERS_BARNES_HUT_TSNE_H
#define LIBND4J_HEADERS_BARNES_HUT_TSNE_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation used as helper with BarnesHutTsne class 
         * to compute edge forces using barnes hut
         *
         * Expected input:
         * 0: 1D row-vector (or with shape (1, m))
         * 1: 1D integer vector with slice nums
         * 2: 1D float-point values vector with same shape as above
         * 3: 2D float-point matrix with data to search
         * 
         * Int args:
         * 0: N - number of slices
         * 
         * Output:
         * 0: 2D matrix with the same shape and type as the 3th argument
         */
        #if NOT_EXCLUDED(OP_barnes_edge_forces)
        DECLARE_CUSTOM_OP(barnes_edge_forces, 4, 1, false, 0, 1);
        #endif

        /**
         * This operation used as helper with BarnesHutTsne class
         * to Symmetrize the value matrix
         *
         * Expected input:
         * 0: 1D int row-vector
         * 1: 1D int col-vector
         * 2: 1D float vector with values
         *
         * Output:
         * 0: 1D int result row-vector
         * 1: 1D int result col-vector
         * 2: a float-point tensor with shape 1xN, with values from the last input vector
         */
        #if NOT_EXCLUDED(OP_barnes_symmetrized)
        DECLARE_CUSTOM_OP(barnes_symmetrized, 3, 3, false, 0, -1);
        #endif

        /**
         * This operation used as helper with BranesHutTsne class 
         * to compute x = x + 2 * yGrads / abs(yGrads) != yIncs / abs(yIncs)
         *
         * Expected input:
         * 0: input tensor
         * 1: input gradient
         * 2: gradient step tensor
         *
         * Output:
         * 0: result of expression above
         */
        #if NOT_EXCLUDED(OP_barnes_gains)
        DECLARE_OP(barnes_gains, 3, 1, true);
        #endif

        /**
         * This operation used as helper with Cell class
         * to check vals in given set 
         *
         * Expected input:
         * 0: 1D float row-vector (corners)
         * 1: 1D float col-vector (widths)
         * 2: 1D float vector (point)
         * 
         * Output:
         * 0: bool val
         */
        #if NOT_EXCLUDED(OP_cell_contains)
        DECLARE_CUSTOM_OP(cell_contains, 3, 1, false, 0, 1);
        #endif

    }
}

#endif // LIBND4J_HEADERS_BARNES_HUT_TSNE_H
