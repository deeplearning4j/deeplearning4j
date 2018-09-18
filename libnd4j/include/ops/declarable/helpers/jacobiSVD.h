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

template <typename T>
class JacobiSVD {

    public:                

        NDArray _m;
        NDArray _s;          // vector with singular values
        NDArray _u;
        NDArray _v;
    
        int _diagSize;
        int _rows;
        int _cols;

        // bool _transp;
        bool _calcU;
        bool _calcV;
        bool _fullUV;

        JacobiSVD(const NDArray& matrix, const bool calcU, const bool calcV, const bool fullUV);

        bool isBlock2x2NotDiag(NDArray& block, int p, int q, T& maxElem);

        static bool createJacobiRotation(const T& x, const T& y, const T& z, NDArray& rotation);
        
        static void svd2x2(const NDArray& block, int p, int q, NDArray& left, NDArray& right);

        static void mulRotationOnLeft(const int i, const int j, NDArray& block, const NDArray& rotation);

        static void mulRotationOnRight(const int i, const int j, NDArray& block, const NDArray& rotation);

        void evalData(const NDArray& matrix);
};



}
}
}


#endif //LIBND4J_JACOBISVD_H
