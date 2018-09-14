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
// Created by Yurii Shyrma on 03.01.2018
//

#ifndef LIBND4J_SVD_H
#define LIBND4J_SVD_H

#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hhSequence.h>
#include "NDArray.h"

namespace nd4j    {
namespace ops     {
namespace helpers {


class SVD {

    public:
    
    int _switchSize = 10;

    NDArray _m;
    NDArray _s;
    NDArray _u;
    NDArray _v;
    
    int _diagSize;

    bool _transp;
    bool _calcU;
    bool _calcV;
    bool _fullUV;

    /**
    *  constructor
    */
    SVD(const NDArray& matrix, const int switchSize, const bool calcV, const bool calcU, const bool fullUV);

    SVD(const NDArray& matrix, const int switchSize, const bool calcV, const bool calcU, const bool fullUV, const char t);

    void deflation1(int col1, int shift, int ind, int size);
    
    void deflation2(int col1U , int col1M, int row1W, int col1W, int ind1, int ind2, int size);
    
    void deflation(int col1, int col2, int ind, int row1W, int col1W, int shift);    

    // FIXME: proper T support required here
    double secularEq(const double diff, const NDArray& col0, const NDArray& diag, const NDArray &permut, const NDArray& diagShifted, const double shift);

    void calcSingVals(const NDArray& col0, const NDArray& diag, const NDArray& permut, NDArray& singVals, NDArray& shifts, NDArray& mus);

    void perturb(const NDArray& col0, const NDArray& diag, const NDArray& permut, const NDArray& singVals,  const NDArray& shifts, const NDArray& mus, NDArray& zhat);

    void calcSingVecs(const NDArray& zhat, const NDArray& diag, const NDArray& perm, const NDArray& singVals, const NDArray& shifts, const NDArray& mus, NDArray& U, NDArray& V);

    void calcBlockSVD(int firstCol, int size, NDArray& U, NDArray& singVals, NDArray& V);

    void DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift);

    void exchangeUV(const HHsequence& hhU, const HHsequence& hhV, const NDArray U, const NDArray V);

    void evalData(const NDArray& matrix);

    FORCEINLINE NDArray& getS();
    FORCEINLINE NDArray& getU();
    FORCEINLINE NDArray& getV();

};


//////////////////////////////////////////////////////////////////////////
FORCEINLINE NDArray& SVD::getS() {
  return _s;
}

//////////////////////////////////////////////////////////////////////////
FORCEINLINE NDArray& SVD::getU() {
  return _u;
}

//////////////////////////////////////////////////////////////////////////
FORCEINLINE NDArray& SVD::getV() {
  return _v;
}



//////////////////////////////////////////////////////////////////////////
// svd operation, this function is not method of SVD class, it is standalone function
void svd(const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum);



}
}
}

#endif //LIBND4J_SVD_H
