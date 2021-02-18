/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_H
#define LIBND4J_MMULHELPER_H

#include "array/NDArray.h"

namespace sd {
    class ND4J_EXPORT MmulHelper {

    private:

        // multiptication N-dimensions tensor on other N-dimensions one
        static sd::NDArray* mmulNxN(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C, const double alpha = 1.0, const double beta = 0.0, const char outOrder = 'f');

        // dot product of vectors (X * Y) = Z[0]
        static sd::NDArray* dot(const sd::NDArray* X, const sd::NDArray* Y, sd::NDArray* Z, const double alpha = 1.0, const double beta = 0.0);

        // multiptication Matrix to Matrix
        static sd::NDArray* mmulMxM(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C, double alpha = 1.0, double beta = 0.0, const char outOrder = 'f');

        // multiptication Matrix to vector
        static sd::NDArray* mmulMxV(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C, double alpha = 1.0, double beta = 0.0, const char outOrder = 'f');

    public:

        static sd::NDArray* mmul(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C = nullptr, const double alpha = 1.0, const double beta = 0.0, const char outOrder = 'f');

        static sd::NDArray* tensorDot(const sd::NDArray* A, const sd::NDArray* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB = {});

        static sd::NDArray* tensorDot(const sd::NDArray* A, const sd::NDArray* B, const std::vector<int>& axesA, const std::vector<int>& axesB);

        static void tensorDot(const sd::NDArray* a, const sd::NDArray* b, sd::NDArray* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC = {});


#ifndef __JAVACPP_HACK__
        /**
        *  modif - (can be empty) vector containing a subsequence of permutation/reshaping arrays (in any order), user must take care of correctness of such arrays by himself
        */
        static void tensorDot(const sd::NDArray* a, const sd::NDArray* b, sd::NDArray* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC);
        static sd::NDArray* tensorDot(const sd::NDArray* a, const sd::NDArray* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB);
#endif

        static void matmul(const sd::NDArray* x, const sd::NDArray* y, sd::NDArray* z, const bool transX, const bool transY, double alpha = 1.0, double beta = 0.0);
    };
}


#endif //LIBND4J_MMULHELPER_H