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

#ifndef LIBND4J_MMULHELPER_CPP
#define LIBND4J_MMULHELPER_CPP

#include "../MmulHelper.h"
#include <helpers/ShapeUtils.h>
#include <helpers/BlasHelper.h>
#include <array/NDArrayFactory.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////
sd::NDArray* sd::MmulHelper::tensorDot(const sd::NDArray* A, const sd::NDArray* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB) {
    std::vector<int> aA(axesA);
    std::vector<int> aB(axesB);
    return tensorDot(A, B, aA, aB);
}

//////////////////////////////////////////////////////////////////////////
sd::NDArray* sd::MmulHelper::tensorDot(const sd::NDArray* a, const sd::NDArray* b, const std::vector<int>& axes_0, const std::vector<int>& axes_1) {

    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;

    auto outShape = ShapeUtils::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);

    // check whether permutation is necessary
    const NDArray* aP = permutAt.empty() ? a : new NDArray(a->permute(permutAt));
    const NDArray* bP = permutBt.empty() ? b : new NDArray(b->permute(permutBt));

    // check whether reshape is necessary
    const NDArray* aPR = aP->isSameShape(shapeAt) ? aP : new NDArray(aP->reshape(aP->ordering(), shapeAt));
    const NDArray* bPR = bP->isSameShape(shapeAt) ? bP : new NDArray(bP->reshape(bP->ordering(), shapeBt));

    NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);

    c->reshapei(outShape);

    if(aP != aPR)
        delete aPR;
    if(bP != bPR)
        delete bPR;
    if(a != aP)
        delete aP;
    if(b != bP)
        delete bP;

    return c;
}

//////////////////////////////////////////////////////////////////////////
void sd::MmulHelper::tensorDot(const sd::NDArray* a, const sd::NDArray* b, sd::NDArray* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC) {

    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;
    ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);

    // check whether permutation is required
    NDArray* cP = permutForC.empty() ? c : new NDArray(c->permute(permutForC));

    // check whether permutation is necessary
    const NDArray* aP = permutAt.empty() ? a : new NDArray(a->permute(permutAt));
    const NDArray* bP = permutBt.empty() ? b : new NDArray(b->permute(permutBt));

    // check whether reshape is necessary
    const NDArray* aPR = aP->isSameShape(shapeAt) ? aP : new NDArray(aP->reshape(aP->ordering(), shapeAt));
    const NDArray* bPR = bP->isSameShape(shapeAt) ? bP : new NDArray(bP->reshape(bP->ordering(), shapeBt));

    std::vector<Nd4jLong> requiredCshape = {aPR->sizeAt(0), bPR->sizeAt(1)};

    NDArray* cPR = cP->isSameShape(requiredCshape) ? cP : new NDArray(cP->reshape(cP->ordering(), requiredCshape, false));

    mmul(aPR, bPR, cPR, 1.0, 0.0);

    if(cPR->buffer() != cP->buffer() || cPR->specialBuffer() != cP->specialBuffer() )   // this means both permute and reshape have been performed on c, cP always points on c->buffer()
        cP->assign(cPR);

   if(aP != aPR)
        delete aPR;
    if(bP != bPR)
        delete bPR;
    if(a != aP)
        delete aP;
    if(b != bP)
        delete bP;

    if(cP != cPR)
        delete cPR;
    if(c != cP)
        delete cP;
}


#ifndef __JAVACPP_HACK__
//////////////////////////////////////////////////////////////////////////
void sd::MmulHelper::tensorDot(const NDArray* a, const NDArray* b, NDArray* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC) {

    NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
    std::string whatToDoWithA, whatToDoWithB, whatToDoWithC;         // "" - nothing; "p" - permutation; "r" - reshaping; "pr" - permutation+reshaping; "rp" - reshaping/permutation, and so on; if another string is produced - throw exception

    for(const auto& arr : modifA)
        whatToDoWithA = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithA + "p" : whatToDoWithA + "r";        // when 0 is present in arr then it is permutation array, otherwise - it is reshaping array
    for(const auto& arr : modifB)
        whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithB + "p" : whatToDoWithB + "r";
    for(const auto& arr : modifC)
        whatToDoWithC = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithC + "p" : whatToDoWithC + "r";

    // first step for a array
    if(!whatToDoWithA.empty())
        aPR = (whatToDoWithA[0] == 'p') ? new NDArray(a->permute(modifA[0])) : new NDArray(a->reshape(a->ordering(), modifA[0]));
    // first step for b array
    if(!whatToDoWithB.empty())
        bPR = (whatToDoWithB[0] == 'p') ? new NDArray(b->permute(modifB[0])) : new NDArray(b->reshape(b->ordering(), modifB[0]));
    // rest steps for a array
    for(int i = 1; i < whatToDoWithA.size(); ++i)
        if(whatToDoWithA[i] == 'p') aPR->permutei(modifA[i]); else aPR->reshapei(modifA[i]);
    // rest steps for b array
    for(int i = 1; i < whatToDoWithB.size(); ++i)
        if(whatToDoWithB[i] == 'p') bPR->permutei(modifB[i]); else bPR->reshapei(modifB[i]);

    // now work with c array
    std::vector<NDArray*> cArrs = {c};
    if(!whatToDoWithC.empty()) {
        cArrs = std::vector<NDArray*>(whatToDoWithC.size()+1, c);
        for(int i = 0; i < cArrs.size()-1; ++i)
            cArrs[i+1] = (whatToDoWithC[i] == 'p') ? new NDArray(cArrs[i]->permute(modifC[i])) : new NDArray(cArrs[i]->reshape(c->ordering(), modifC[i], false));  // since we ignore first element in cArrs (that is cArrs[0]) then it is always equal to c
    }

    mmul(aPR, bPR, cArrs[cArrs.size()-1], 1.0, 0.0);

    // check whether new buffer allocation was happened for c array
    if(!whatToDoWithC.empty()) {
        for(int i = cArrs.size()-1; i > 0; --i) {
            if(cArrs[i]->buffer() != cArrs[i-1]->buffer() || cArrs[i]->specialBuffer() != cArrs[i-1]->specialBuffer())
                cArrs[i-1]->assign(cArrs[i]);
            delete cArrs[i];
        }
    }

    if(aPR != a)
        delete aPR;
    if(bPR != b)
        delete bPR;
}

//////////////////////////////////////////////////////////////////////////
NDArray* sd::MmulHelper::tensorDot(const sd::NDArray* a, const sd::NDArray* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB) {

    NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
    std::string whatToDoWithA, whatToDoWithB;         // "" - nothing; "p" - permutation only; "r" - reshaping only; "pr" - permutation+reshaping; "rp" - reshaping/permutation; another string - throw exception

    for(const auto& arr : modifA)
        whatToDoWithA = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithA + "p" : whatToDoWithA + "r";        // when 0 is present in arr then it is permutation array, otherwise - it is reshaping array
    for(const auto& arr : modifB)
        whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithB + "p" : whatToDoWithB + "r";

    // first step for a array
    if(!whatToDoWithA.empty())
        aPR = (whatToDoWithA[0] == 'p') ? new NDArray(a->permute(modifA[0])) : new NDArray(a->reshape(a->ordering(), modifA[0]));
    // first step for b array
    if(!whatToDoWithB.empty())
        bPR = (whatToDoWithB[0] == 'p') ? new NDArray(b->permute(modifB[0])) : new NDArray(b->reshape(b->ordering(), modifB[0]));
    // rest steps for a array
    for(int i = 1; i < whatToDoWithA.size(); ++i)
        if(whatToDoWithA[i] == 'p') aPR->permutei(modifA[i]); else aPR->reshapei(modifA[i]);
    // rest steps for b array
    for(int i = 1; i < whatToDoWithB.size(); ++i)
        if(whatToDoWithB[i] == 'p') bPR->permutei(modifB[i]); else bPR->reshapei(modifB[i]);

    NDArray* result = mmul(aPR, bPR, nullptr, 1.0, 0.0);

    if(aPR != a)
        delete aPR;
    if(bPR != b)
        delete bPR;
    return result;
}
#endif


//////////////////////////////////////////////////////////////////////////
sd::NDArray* MmulHelper::mmul(const sd::NDArray* A, const sd::NDArray* B, sd::NDArray* C , const double alpha, const double beta, const char outOrder) {

    int lenDim;
    const int aRank = A->rankOf();
    const int bRank = B->rankOf();
    const bool isAVector = shape::isCommonVector(A->shapeInfo(), lenDim);
    const bool isBVector = shape::isCommonVector(B->shapeInfo(), lenDim);

    // dot product of 2 vectors
    if(isAVector && isBVector && (aRank != 2 || aRank == 2 && (A->isSameShape(B) || bRank == 1 && A->sizeAt(1) == 1)))  // (1x1x1 * 1x1) or (1x4 * 1*4) or (4x1 * 4x1) or (4x1 * 4)
        return dot(A, B, C, alpha, beta);

    // matrix x matrix
    if(aRank == 2 && bRank == 2)
        return mmulMxM(A, B, C, alpha, beta, outOrder);

    // matrix x vector
    if(aRank == 2 && isBVector)
        return mmulMxV(A, B, C, alpha, beta, outOrder);

    // vector x matrix, A{M} x B{M,N} = C{N} -> reduce to matrix x matrix A2{1,M} x B{M,N} = C2{1,N}, since there is no corresponding blas operation sgevm
    if(isAVector && bRank == 2) {
        NDArray* A2 = new NDArray(A->reshape(A->ordering(), {1, A->lengthOf()}));               // A{M} -> A2{1,M}
        NDArray* C2 = C ? new NDArray(C->reshape(C->ordering(), {1, C->lengthOf()}, false)) : nullptr; // C{N} -> C2{1,N}
        auto result = mmulMxM(A2, B, C2, alpha, beta, outOrder);                                // result{1,N}
        delete A2;
        delete C2;

        if(!C) {
            result->reshapei({result->lengthOf()});                                             // result{1,N} -> result{N}
            return result;
        }
        return C;
    }

    // batched matrix multiplication
    return mmulNxN(A, B, C, alpha, beta, outOrder);
}


//////////////////////////////////////////////////////////////////////////
    void MmulHelper::matmul(const sd::NDArray* x, const sd::NDArray* y, sd::NDArray* z, const bool transX, const bool transY, double alpha, double beta) {
        int xRank = x->rankOf();
        int yRank = y->rankOf();

        auto outShape = ShapeUtils::evalShapeForMatmul(x->shapeInfo(), y->shapeInfo(), transX, transY);
        if(!z->isSameShape(outShape)) {
            nd4j_printf("NDArrayFactory::matmul static method: input shape of output array is wrong, actual is %s and expected is %s ! \n", ShapeUtils::shapeAsString(z).c_str(), ShapeUtils::shapeAsString(outShape).c_str());
            throw std::invalid_argument("");
        }

        if (z->isEmpty())
            return;

        NDArray* xT(const_cast<NDArray*>(x)), *yT(const_cast<NDArray*>(y)), *zT(z);

        if((transX && xRank > 1) || (transY && yRank > 1)) {
            const int rank = xRank >= yRank ? xRank : yRank;
            std::vector<int> permut(rank);
            for (int i = 0; i < rank-2; ++i)
                permut[i] = i;
            permut[rank-2] = rank - 1;
            permut[rank-1] = rank - 2;

            if(transX)
                xT = new NDArray(x->permute(permut));

            if(transY)
                yT = new NDArray(y->permute(permut));
        }

        if(xRank <= 2 && yRank <= 2) {  // dot (1Dx1D), vector-matrix (1Dx2D), matrix-vector (2Dx1D), matrix-matrix (2Dx2D) product cases

            if(xRank == 1 && yRank == 2) {   // reduce vector-matrix to matrix-matrix case
                xT = new NDArray(x->reshape(x->ordering(), {1, x->lengthOf()})); // please note x is not transposed in this case (since xRank=1)
                zT = new NDArray(z->reshape(z->ordering(), {1, z->lengthOf()}));
            }

            mmul(xT, yT, zT, alpha, beta);
        }
        else {  // rest cases -  batched mmul

            const int batchRank = xRank - 2;
            std::vector<int> dimsToExclude(batchRank);
            for(int i = 0; i < batchRank; ++i)
                dimsToExclude[i] = i;

            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(xT->shapeInfo(), dimsToExclude);

//PRAGMA_OMP_PARALLEL_FOR
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto xSubArr = (*xT)(i, dimsToExclude);
                auto ySubArr = (*yT)(i, dimsToExclude);
                auto zSubArr = (*zT)(i, dimsToExclude);
                mmul(&xSubArr, &ySubArr, &zSubArr, alpha, beta);
            }
        }

        if(xT != x)
            delete xT;
        if(yT != y)
            delete yT;
        if(zT != z)
            delete zT;
    }

//BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const char aOrder, const int M, const int N, const double alpha, const void* A, const int lda, const void* B, const int incx, const double beta, void* C, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}

#endif