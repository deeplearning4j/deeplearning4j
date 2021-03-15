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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#ifndef LIBND4J_FULLPIVLU_H
#define LIBND4J_FULLPIVLU_H

#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

// class solves equation A*x = b for x, by procedure of LU decomposition of input matrix A with complete pivoting
// LU decomposition of a matrix is:
// A = P^-1 * L * U * Q^-1
// L is unit-lower-triangular,
// U is upper-triangular,
// and P and Q are permutation matrices for rows and columns correspondingly

template <typename T>
class FullPivLU {

    public:

        // A{M,K} * x{K,N} = b{M,N}
        static void solve(const NDArray& A, const NDArray& b, NDArray& x);
};


}
}
}


#endif //LIBND4J_FULLPIVLU_H
