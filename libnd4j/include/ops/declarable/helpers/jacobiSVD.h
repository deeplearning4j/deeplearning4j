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
// Created by Yurii Shyrma on 11.01.2018
//

#ifndef LIBND4J_JACOBISVD_H
#define LIBND4J_JACOBISVD_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hhSequence.h>
#include "NDArray.h"

namespace nd4j {
namespace ops {
namespace helpers {


template<typename T>
class JacobiSVD {

    public:                

        NDArray<T> _m;
        NDArray<T> _s;          // vector with singular values
        NDArray<T> _u;
        NDArray<T> _v;
    
        int _diagSize;
        int _rows;
        int _cols;

        // bool _transp;
        bool _calcU;
        bool _calcV;
        bool _fullUV;

        JacobiSVD(const NDArray<T>& matrix, const bool calcU, const bool calcV, const bool fullUV);

        bool isBlock2x2NotDiag(NDArray<T>& block, int p, int q, T& maxElem);

        static bool createJacobiRotation(const T& x, const T& y, const T& z, NDArray<T>& rotation);
        
        static void svd2x2(const NDArray<T>& block, int p, int q, NDArray<T>& left, NDArray<T>& right);

        static void mulRotationOnLeft(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation);

        static void mulRotationOnRight(const int i, const int j, NDArray<T>& block, const NDArray<T>& rotation);

        void evalData(const NDArray<T>& matrix);
};



}
}
}


#endif //LIBND4J_JACOBISVD_H
