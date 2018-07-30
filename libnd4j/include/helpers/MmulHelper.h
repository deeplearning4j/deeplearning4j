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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_H
#define LIBND4J_MMULHELPER_H

#include "NDArray.h"

namespace nd4j {
    template<typename T>
    class MmulHelper {

    private:
        // helpers for helper 
        // multiptication N-dimensions tensor on other N-dimensions one
        static nd4j::NDArray<T>* mmulNxN(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);
        // multiptication Matrix to vector
        static nd4j::NDArray<T>* mmulMxV(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);
        // multiptication Matrix to Matrix
        static nd4j::NDArray<T>* mmulMxM(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, T alpha, T beta);

    public:

        static nd4j::NDArray<T>* mmul(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C = nullptr, T alpha = 1.0f, T beta = 0.0f);

        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB = {});

        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::vector<int>& axesA, const std::vector<int>& axesB);

        static void tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC = {});

#ifndef __JAVACPP_HACK__
        /**
        *  modif - (can be empty) vector containing a subsequence of permutation/reshaping arrays (in any order), user must take care of correctness of such arrays by himself 
        */
        static void tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC);
        static nd4j::NDArray<T>* tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB);
#endif

        static NDArray<T>* simpleMMul(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c , const T alpha, const T beta);

        static void matmul(const nd4j::NDArray<T>* x, const nd4j::NDArray<T>* y, nd4j::NDArray<T>* z, const bool transX, const bool transY);
    };
}


#endif //LIBND4J_MMULHELPER_H
