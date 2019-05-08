/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
#include <NDArrayFactory.h>

namespace nd4j { 


//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
template <typename T1, typename T2, typename T3>
static void usualGemm(const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* B = reinterpret_cast<T2*>(const_cast<void*>(vB));
    T3* C = reinterpret_cast<T3*>(vC);
    T3 alphaZ(alpha), betaZ(beta);
    
    const bool flagC = cOrder == 'f';
    const bool flagA = (flagC && transA) || (!flagC && !transA);
    const bool flagB = (flagC && transB) || (!flagC && !transB);   

    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
    for(uint row = 0; row < M; ++row) {
       for(uint col = 0; col < N; ++col) {
            
            T3* c = flagC ? (C + row + col * ldc) : (C + row * ldc + col);
            T3 val = 0;

           PRAGMA_OMP_SIMD
            for(uint i = 0; i < K; ++i) {
                T3 a = flagA ? *(A + row * lda + i) : *(A + row + i * lda);
                T3 b = flagB ? *(B + col + i * ldb) : *(B + col * ldb + i);             
                val += alphaZ * a * b;
            }
            
            if(betaZ)
                *c = val + betaZ * *c;
            else
                *c = val;
       }
    }
}

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M
template <typename T1, typename T2, typename T3>
static void usualGemv(const char aOrder, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* X = reinterpret_cast<T2*>(const_cast<void*>(vX));
    T3* Y = reinterpret_cast<T3*>(vY);
    T3 alphaZ(alpha), betaZ(beta);
    
    const bool flagA = aOrder == 'f';

    PRAGMA_OMP_PARALLEL_FOR_IF(M > Environment::getInstance()->tadThreshold())
    for(int row = 0; row < M; ++row) {
                        
        T3* y = Y + row * incy;
        T3 val = 0;

        PRAGMA_OMP_SIMD
        for(int i = 0; i < N; ++i) {
            T3 a = flagA ? *(A + row + i * lda) : *(A + row * lda + i);
            T3 x = *(X + i * incx);
            val += alphaZ * a * x;
        }
        
        if(betaZ)
            *y = val + betaZ * *y;
        else
            *y = val;
    }
}

//////////////////////////////////////////////////////////////////////////////
// (X*Y) = Z[0]
template <typename T1, typename T2, typename T3>
static void usualDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);
    T3 alphaZ(alpha), betaZ(beta);

    T3 sum = 0;
    PRAGMA_OMP_PARALLEL_FOR_SIMD_REDUCTION(sumT:sum)
    for(unsigned int i = 0; i < length; ++i)
        sum = sum + X[i * incx] * Y[i * incy];        
    
    *Z = alphaZ * sum + betaZ * *Z;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {    

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");    

    const auto M     = A->sizeAt(0);
    const auto K     = A->sizeAt(1);
    const auto N     = B->sizeAt(1);
    const auto bRows = B->sizeAt(0);
    
    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if(bRows != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of columns !");

    if(C == nullptr)
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()));

    NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));    

    const auto cOrder = C->ordering();

    if(A->ews() != 1)
        pA = pA->dup(cOrder);
    if(B->ews() != 1)
        pB = pB->dup(cOrder);
    if(C->ews() != 1)
        pC = pC->dup(cOrder);

    const auto aOrder = pA->ordering();
    const auto bOrder = pB->ordering();    

    const bool transA = aOrder != cOrder;
    const bool transB = bOrder != cOrder;
    
    const CBLAS_ORDER blasOrder  = cOrder == 'f' ? CblasColMajor : CblasRowMajor;    
    const CBLAS_TRANSPOSE transAblas = transA ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE transBblas = transB ? CblasTrans : CblasNoTrans;

    const int lda = aOrder == 'f' ? M : K;
    const int ldb = bOrder == 'f' ? K : N;
    const int ldc = cOrder == 'f' ? M : N;    

    const auto aType = pA->dataType();
    const auto bType = pB->dataType();
    const auto cType = pC->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
    const bool hasGemm = BlasHelper::getInstance()->hasGEMM(aType);    
    
    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (ABC && hasGemm && aType == DataType::FLOAT32) {
        nd4j_debug("MMUL: Using provided BLAS impl\n","");
        BlasHelper::getInstance()->sgemm()(blasOrder, transAblas, transBblas, M, N, K, (float) alpha, reinterpret_cast<float *>(pA->getBuffer()), lda, reinterpret_cast<float *>(pB->getBuffer()), ldb, (float) beta, reinterpret_cast<float *>(pC->getBuffer()), ldc);
    }
    else if (ABC && hasGemm && aType == DataType::DOUBLE) {
        nd4j_debug("MMUL: Using provided BLAS impl\n","");
        BlasHelper::getInstance()->dgemm()(blasOrder, transAblas, transBblas, M, N, K, (double) alpha, reinterpret_cast<double *>(pA->getBuffer()), lda, reinterpret_cast<double *>(pB->getBuffer()), ldb, (double) beta, reinterpret_cast<double *>(pC->getBuffer()), ldc);
    }
    else {
        nd4j_debug("MMUL: Using fallback BLAS impl\n","");
        BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (cOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }    

    if(pC != C) {
        C->assign(pC);
        delete pC;
    }
    if(pA != A)
        delete pA;
    if(pB != B)
        delete pB;

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, nd4j::NDArray* Y, const double alpha, const double beta, const char outOrder) {

    int xLenDim, yLenDim(0);

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxV: rank of A array is not equal 2 !");
    if(!shape::isCommonVector(X->getShapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: X array must be vector !");    

    const auto M = A->sizeAt(0);    
    const auto N = A->sizeAt(1);
    
    if(Y != nullptr && !shape::isCommonVector(Y->getShapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: Y array must be vector !");
    if(X->lengthOf() != N)
        throw std::runtime_error("MmulHelper::mmulMxV: X vector has wrong length !");
    if(Y != nullptr && Y->lengthOf() != M)
        throw std::runtime_error("MmulHelper::mmulMxV: Y array has wrong length !");    

    if(Y == nullptr)        
        Y = new NDArray(outOrder, {M}, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()));
    
    NDArray *pA(const_cast<NDArray*>(A));

    if(A->ews() != 1)
        pA = pA->dup();
    
    CBLAS_ORDER blasOrder;
    int lda;
    if (pA->ordering() == 'f')  {blasOrder = CblasColMajor; lda = M; }
    else                        {blasOrder = CblasRowMajor; lda = N; }
         
    const int incx = X->stridesOf()[xLenDim];
    const int incy = Y->stridesOf()[yLenDim];

    const auto aType = pA->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);
    const bool hasGemv = BlasHelper::getInstance()->hasGEMV(aType);
    
    // choose appropriate gemm api depending on data types    
    if(AXY && hasGemv && aType == DataType::DOUBLE) {
        BlasHelper::getInstance()->dgemv()(blasOrder, CblasNoTrans, M, N, alpha,       (double*)pA->getBuffer(), lda, (double*)X->getBuffer(), incx, beta,        (double*)Y->getBuffer(), incy);
    }
    else if(AXY && hasGemv && aType == DataType::FLOAT32) {                
        BlasHelper::getInstance()->sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->getBuffer(), lda, (float*)X->getBuffer(),  incx, (float)beta, (float*)Y->getBuffer(),  incy);
    }
    else {
        BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (pA->ordering(), M, N, alpha, pA->getBuffer(), lda, X->getBuffer(), incx, beta, Y->getBuffer(), incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);        
    }

    if(pA != A)
        delete pA;
    
    return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, nd4j::NDArray* Z, const double alpha, const double beta) {

    int xLenDim(0), yLenDim(0);

    if(!shape::isCommonVector(X->getShapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::dot: X array must be vector !");
    if(!shape::isCommonVector(Y->getShapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::dot: Y array must be vector !");
    if(Z != nullptr && !Z->isScalar())
        throw std::runtime_error("MmulHelper::dot: Z array must be scalar !");

    const auto length = X->lengthOf();

    if(Y->lengthOf() != length)
        throw std::runtime_error("MmulHelper::dot: lengths of input vectors are different !");

    if(Z == nullptr)        
        Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()));
    
    const Nd4jLong incx = X->stridesOf()[xLenDim];
    const Nd4jLong incy = Y->stridesOf()[yLenDim];

    const auto xType = X->dataType();
    const auto yType = Y->dataType();
    const auto zType = Z->dataType();
    
    BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (length, alpha, X->getBuffer(), incx, Y->getBuffer(), incy, beta, Z->getBuffer()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);        

    return Z;
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
    }

    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))
            throw std::runtime_error("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else {
        C = new NDArray(outOrder, cExpectedShape, B->dataType());
    }


    // multiplication
    const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(C->rankOf(), {-2, -1});
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(C->getShapeInfo(), dimsToExclude);
    std::vector<Nd4jLong> idxRanges(2 * C->rankOf());

        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, C->getShapeInfo(), dimsToExclude, idxRanges.data());
            NDArray cSubArr = (*C)(idxRanges);

            if(aRank > bRank) {
                NDArray aSubArr = (*A)(idxRanges);
                mmulMxM(&aSubArr, B, &cSubArr, 1., 0., outOrder);
            }
            else if(bRank > aRank) {
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(A, &bSubArr, &cSubArr, 1., 0, outOrder);
            }
            else {
                NDArray aSubArr = (*A)(idxRanges);
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(&aSubArr, &bSubArr, &cSubArr, 1., 0., outOrder);
            }
        }

    return C;
}

//////////////////////////////////////////////////////////////////////////
nd4j::NDArray* MmulHelper::mmul(const nd4j::NDArray* A, const nd4j::NDArray* B, nd4j::NDArray* C , const double alpha, const double beta, const char outOrder) {

    int lenDim;
    const int aRank = A->rankOf();
    const int bRank = B->rankOf();
    const bool isAVector = shape::isCommonVector(A->getShapeInfo(), lenDim);
    const bool isBVector = shape::isCommonVector(B->getShapeInfo(), lenDim);

    // dot product of 2 vectors
    if(isAVector && isBVector && (aRank != 2 || aRank == 2 && (A->isSameShape(B) || bRank == 1 && A->sizeAt(1) == 1)))  // (1x1x1 * 1x1) or (1x4 * 1*4) or (4x1 * 4x1) or (4x1 * 4)
        return dot(A, B, C, alpha, beta);

    // matrix x matrix
    if(aRank == 2 && bRank == 2)
        return mmulMxM(A, B, C, alpha, beta, outOrder);

    // matrix x vector
    if(aRank == 2 && isBVector)
        return mmulMxV(A, B, C, alpha, beta, outOrder);

    // batched matrix multiplication
    return mmulNxN(A, B, C, alpha, beta, outOrder);
}

//////////////////////////////////////////////////////////////////////////
nd4j::NDArray* nd4j::MmulHelper::tensorDot(const nd4j::NDArray* A, const nd4j::NDArray* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB) {
    std::vector<int> aA(axesA);
    std::vector<int> aB(axesB);
    return tensorDot(A, B, aA, aB);
}

//////////////////////////////////////////////////////////////////////////
nd4j::NDArray* nd4j::MmulHelper::tensorDot(const nd4j::NDArray* a, const nd4j::NDArray* b, const std::vector<int>& axes_0, const std::vector<int>& axes_1) {

    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;        

    auto outShape = ShapeUtils::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);
    
    NDArray* aPR = a->permute(permutAt);        
    NDArray* bPR = b->permute(permutBt);
    
    // check whether reshape is necessary
    if(!aPR->isSameShape(shapeAt))
        aPR->reshapei(shapeAt);
    if(!bPR->isSameShape(shapeBt)) 
        bPR->reshapei(shapeBt);  
    
    NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);

    c->reshapei('c', outShape);
        
    delete aPR;        
    delete bPR;

    return c;
}


















// //////////////////////////////////////////////////////////////////////////
// void nd4j::MmulHelper::tensorDot(const nd4j::NDArray* a, const nd4j::NDArray* b, nd4j::NDArray* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC, char aOrder, char bOrder, char cOrder) {
    
//     std::vector<int> permutAt, permutBt;
//     std::vector<Nd4jLong> shapeAt, shapeBt;
//     auto outShape = ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);
//     NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b)), *cP(c), *cPR(c);
//     // check whether permutation is required
//     if(!permutForC.empty())
//         cP = c->permute(permutForC);            
    
//     aPR = a->permute(permutAt);        
//     bPR = b->permute(permutBt);
        
//     // check whether reshape is necessary        
//     if(!aPR->isSameShape(shapeAt)) {
//         if(aPR == a)
//             aPR = a->reshape(a->ordering(), shapeAt);
//         else 
//             aPR->reshapei(a->ordering(), shapeAt);
//     }
//     if(!bPR->isSameShape(shapeBt)) {
//         if(bPR == b) {     
//             bPR = b->reshape(b->ordering(), shapeBt);                    
//         }
//         else {
//             bPR->reshapei(b->ordering(), shapeBt);                

//         }
//     }
//     if(!cP->isSameShape({aPR->sizeAt(0), bPR->sizeAt(1)}))
//         cPR = cP->reshape(c->ordering(), {aPR->sizeAt(0), bPR->sizeAt(1)});
            
//     mmul(aPR, bPR, cPR, 1.0, 0.0);
//     if(cPR->getBuffer() != cP->getBuffer())                     // this means both permute and reshape have been performed on c, cP always points on c->getBuffer()
//         cP->assign(cPR);                        
    
//     if(cPR != c)
//         delete cPR;
//     if(aPR != a)
//         delete aPR;        
//     if(bPR != b)
//         delete bPR;
//     if(cP != c)
//         delete cP;
// }

////////////////////////////////////////////////////////////////////////
void nd4j::MmulHelper::tensorDot(const nd4j::NDArray* a, const nd4j::NDArray* b, nd4j::NDArray* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC) {

    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;
    ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);
    
    NDArray *cP(c), *cPR(c);
    
    // check whether permutation is required    
    if(!permutForC.empty())
        cP = c->permute(permutForC);        

    auto aPR = a->permute(permutAt);
    auto bPR = b->permute(permutBt);    

    // check whether reshape is necessary
    if(!aPR->isSameShape(shapeAt))    
            aPR->reshapei(shapeAt);    
    if(!bPR->isSameShape(shapeBt))
            bPR->reshapei(shapeBt);

    if(!cP->isSameShape({aPR->sizeAt(0), bPR->sizeAt(1)}))
        cPR = cP->reshape(cP->ordering(), {aPR->sizeAt(0), bPR->sizeAt(1)});
            
    mmul(aPR, bPR, cPR, 1.0, 0.0);
    if(cPR->getBuffer() != cP->getBuffer())          // this means both permute and reshape have been performed on c, cP always points on c->getBuffer()
        cP->assign(cPR);
    
    if(cPR != c)
        delete cPR;
    if(cP != c)
        delete cP;
    delete aPR;    
    delete bPR;    
}


#ifndef __JAVACPP_HACK__
//////////////////////////////////////////////////////////////////////////
void nd4j::MmulHelper::tensorDot(const NDArray* a, const NDArray* b, NDArray* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC) {
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
        aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0]) : a->reshape(a->ordering(), modifA[0]);
    // first step for b array
    if(!whatToDoWithB.empty())
        bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0]) : b->reshape(b->ordering(), modifB[0]);
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
            cArrs[i+1] = (whatToDoWithC[i] == 'p') ? cArrs[i]->permute(modifC[i]) : cArrs[i]->reshape(c->ordering(), modifC[i]);  // since we ignore first element in cArrs (that is cArrs[0]) then it is always equal to c
    }
    
    mmul(aPR, bPR, cArrs[cArrs.size()-1], 1.0, 0.0);

    // check whether new buffer allocation was happened for c array
    if(!whatToDoWithC.empty()) {
        for(int i = cArrs.size()-1; i > 0; --i) {
            if(cArrs[i]->getBuffer() != cArrs[i-1]->getBuffer())
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
NDArray* nd4j::MmulHelper::tensorDot(const nd4j::NDArray* a, const nd4j::NDArray* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB) {
    NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
    std::string whatToDoWithA, whatToDoWithB;         // "" - nothing; "p" - permutation only; "r" - reshaping only; "pr" - permutation+reshaping; "rp" - reshaping/permutation; another string - throw exception
    for(const auto& arr : modifA) 
        whatToDoWithA = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithA + "p" : whatToDoWithA + "r";        // when 0 is present in arr then it is permutation array, otherwise - it is reshaping array            
    for(const auto& arr : modifB) 
        whatToDoWithB = (std::find(arr.begin(), arr.end(), 0) != arr.end()) ? whatToDoWithB + "p" : whatToDoWithB + "r";    
    // first step for a array
    if(!whatToDoWithA.empty())
        aPR = (whatToDoWithA[0] == 'p') ? a->permute(modifA[0]) : a->reshape(a->ordering(), modifA[0]);
    // first step for b array
    if(!whatToDoWithB.empty())
        bPR = (whatToDoWithB[0] == 'p') ? b->permute(modifB[0]) : b->reshape(b->ordering(), modifB[0]);
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
    void MmulHelper::matmul(const nd4j::NDArray* x, const nd4j::NDArray* y, nd4j::NDArray* z, const bool transX, const bool transY) {
        int xRank = x->rankOf();
        int yRank = y->rankOf();

        auto outShape = ShapeUtils::evalShapeForMatmul(x->getShapeInfo(), y->getShapeInfo(), transX, transY);
        if(!z->isSameShape(outShape)) {
            nd4j_printf("NDArrayFactory::matmul static method: input shape of output array is wrong, actual is %s and expected is %s ! \n", ShapeUtils::shapeAsString(z).c_str(), ShapeUtils::shapeAsString(outShape).c_str());
            throw std::invalid_argument("");
        }
        
        NDArray* xT(const_cast<NDArray*>(x)), *yT(const_cast<NDArray*>(y)), *zT(z);
    
        if((transX && xRank > 1) || (transY && yRank > 1)) {
            const int rank = xRank >= yRank ? xRank : yRank;
            std::vector<int> permut(rank);
            for (int i = 0; i < rank-2; ++i)
                permut[i] = i;
            permut[rank-2] = rank - 1;
            permut[rank-1] = rank - 2;
        
            if(transX)
                xT = x->permute(permut);

            if(transY)
                yT = y->permute(permut);
        }

        if(xRank <= 2 && yRank <= 2) {  // dot (1Dx1D), vector-matrix (1Dx2D), matrix-vector (2Dx1D), matrix-matrix (2Dx2D) product cases

            if(xRank == 1 && yRank == 2) {   // reduce vector-matrix to matrix-matrix case
                xT = x->reshape(x->ordering(), {1, x->lengthOf()}); // please note x is not transposed in this case (since xRank=1)
                zT = z->reshape(z->ordering(), {1, z->lengthOf()});
            }
        
            mmul(xT, yT, zT, 1., 0.);
        }
        else {  // rest cases -  batched mmul
        
            const int batchRank = xRank - 2;
            std::vector<int> dimsToExclude(batchRank);
            for(int i = 0; i < batchRank; ++i)
                dimsToExclude[i] = i;

            const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(xT->getShapeInfo(), dimsToExclude);

            //PRAGMA_OMP_PARALLEL_FOR
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                auto xSubArr = (*xT)(i, dimsToExclude);
                auto ySubArr = (*yT)(i, dimsToExclude);
                auto zSubArr = (*zT)(i, dimsToExclude);
                mmul(&xSubArr, &ySubArr, &zSubArr, 1., 0.);
            }
        }

        if(xT != x)
            delete xT;
        if(yT != y)
            delete yT;
        if(zT != z)
            delete zT;
    }

BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const char aOrder, const int M, const int N, const double alpha, const void* A, const int lda, const void* B, const int incx, const double beta, void* C, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}


#endif
