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
#include <NDArrayFactory.h>

namespace nd4j { 

    
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
    NDArray* aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b));
    aPR = a->permute(permutAt);        
    bPR = b->permute(permutBt);
    
    // check whether reshape is necessary
    if(!aPR->isSameShape(shapeAt)) {
        if(aPR == a)
            aPR = a->reshape('c', shapeAt);
        else 
            aPR->reshapei('c', shapeAt);
    }
    if(!bPR->isSameShape(shapeBt)) {
        if(bPR == b)
            bPR = b->reshape('c', shapeBt);
        else 
            bPR->reshapei('c', shapeBt);                
    }
    NDArray* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);
    c->reshapei('c', outShape);
    
    if(aPR != a)
        delete aPR;        
    if(bPR != b)
        delete bPR;
    return c;
}


//////////////////////////////////////////////////////////////////////////
void nd4j::MmulHelper::tensorDot(const nd4j::NDArray* a, const nd4j::NDArray* b, nd4j::NDArray* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC) {
    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;
    auto outShape = ShapeUtils::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);
    NDArray *aPR(const_cast<NDArray*>(a)), *bPR(const_cast<NDArray*>(b)), *cP(c), *cPR(c);
    // check whether permutation is required
    if(!permutForC.empty())
        cP = c->permute(permutForC);            
    
    aPR = a->permute(permutAt);        
    bPR = b->permute(permutBt);    
    // check whether reshape is necessary        
    if(!aPR->isSameShape(shapeAt)) {
        if(aPR == a)
            aPR = a->reshape('c', shapeAt);
        else 
            aPR->reshapei('c', shapeAt);
    }
    if(!bPR->isSameShape(shapeBt)) {
        if(bPR == b)
            bPR = b->reshape('c', shapeBt);
        else 
            bPR->reshapei('c', shapeBt);                
    }
    if(!cP->isSameShape({aPR->sizeAt(0), bPR->sizeAt(1)}))
        cPR = cP->reshape('c', {aPR->sizeAt(0), bPR->sizeAt(1)});
            
    mmul(aPR, bPR, cPR, 1.0, 0.0);
    if(cPR->getBuffer() != cP->getBuffer())                     // this means both permute and reshape have been performed on c, cP always points on c->getBuffer()
        cP->assign(cPR);                        
    
    if(cPR != c)
        delete cPR;
    if(aPR != a)
        delete aPR;        
    if(bPR != b)
        delete bPR;
    if(cP != c)
        delete cP;
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
// nd4j::NDArray* MmulHelper::mmulNxN(nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C , double alpha, double beta) {
//        nd4j::NDArray* result = C;
//         // matmul
//         if (A->rankOf() != B->rankOf()) {
//             // FIXME (r119): this is temporary fix for @shyrma, proper impl required here
//             int pRows = A->sizeAt(-2);
//             int pCols = B->sizeAt(-1);
//             if (A->sizeAt(-1) != B->sizeAt(-2)) {
//                 nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
//                             A->sizeAt(-1), B->sizeAt(-2))
//                 throw std::runtime_error("Numbers of rows/columns should match");
//             }
//             std::vector<Nd4jLong> newShape;
//             if (A->rankOf() > B->rankOf())
//                 for (int e = 0; e < A->rankOf() - 2; e++)
//                     newShape.emplace_back(A->sizeAt(e));
//             else
//                 for (int e = 0; e < B->rankOf() - 2; e++)
//                     newShape.emplace_back(B->sizeAt(e));
//             newShape.push_back(pRows);
//             newShape.push_back(pCols);
//             if (result == nullptr)
//                 result = new NDArray('c', newShape);
//             else if (!result->isSameShape(newShape)) {
//                 nd4j_printf("Bad result shape for MatMul\n", "");
//                 throw std::runtime_error("Bad result shape");
//             }
//             if (A->rankOf() > B->rankOf()) {
//                 auto aL = A->allTensorsAlongDimension({A->rankOf() - 2, A->rankOf() - 1});
//                 auto cL = result->allTensorsAlongDimension({result->rankOf() - 2, result->rankOf() - 1});
//                 nd4j_debug("NumTads: %i\n", aL->size());
//                 for (int e = 0; e < aL->size(); e++) {
//                     auto c_ = mmul(aL->at(e), B, cL->at(e));
//                     if (c_ != cL->at(e)) {
//                         cL->at(e)->assign(c_);
//                         delete c_;
//                     }
//                 }
//                 delete aL;
//                 delete cL;
//             } else {
//                 auto bL = B->allTensorsAlongDimension({B->rankOf() - 2, B->rankOf() - 1});
//                 auto cL = result->allTensorsAlongDimension({result->rankOf() - 2, result->rankOf() - 1});
//                 nd4j_debug("NumTads: %i\n", bL->size());
//                 for (int e = 0; e < bL->size(); e++) {
//                     auto c_ = mmul(A, bL->at(e), cL->at(e));
//                     if (cL->at(e) != c_) {
//                         cL->at(e)->assign(c_);
//                         delete c_;
//                     }
//                 }
//                 delete bL;
//                 delete cL;
//             }
//         } else {
//             //int dims = A->rankOf();
//             std::vector<Nd4jLong> newShape;
//             for (int e = 0; e < A->rankOf() - 2; e++)
//                 if (A->sizeAt(e) != B->sizeAt(e)) {
//                     nd4j_printf("Dimension [%i] differs for A and B: %i vs %i", e, A->sizeAt(e), B->sizeAt(e));
//                     throw std::runtime_error("Outer dimensions for A & B should be equal");
//                 } else {
//                     newShape.push_back(A->sizeAt(e));
//                 }
//             int pRows = A->sizeAt(-2);
//             int pCols = B->sizeAt(-1);
//             if (A->sizeAt(-1) != B->sizeAt(-2)) {
//                 nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
//                             A->sizeAt(-1), B->sizeAt(-2))
//                 throw std::runtime_error("Numbers of rows/columns should match");
//             }
//             newShape.push_back(pRows);
//             newShape.push_back(pCols);
//             //Nd4jLong prod = shape::prodLong(newShape.data(), newShape.size());
//             if (result == nullptr)
//                 result = new NDArray('c', newShape);
//             else if (!result->isSameShape(newShape)) {
//                 nd4j_printf("Bad result shape for MatMul\n", "");
//                 throw std::runtime_error("Bad result shape");
//             }
//             auto aL = A->allTensorsAlongDimension({A->rankOf() - 2, A->rankOf() - 1});
//             auto bL = B->allTensorsAlongDimension({B->rankOf() - 2, B->rankOf() - 1});
//             auto cL = result->allTensorsAlongDimension({result->rankOf() - 2, result->rankOf() - 1});
//             int aL_size = aL->size();
//             int bL_size = bL->size();
//             int cL_size = cL->size();
//             nd4j_debug("NumTads: %i\n", aL->size());
//             for (int e = 0; e < aL->size(); e++) {
//                 auto aLt = aL->at(e);
//                 auto bLt = bL->at(e);
//                 auto cLt = cL->at(e);

//                 auto c_ = mmul(aLt, bLt, cLt);
//                 if (c_ != cLt) {
//                     cLt->assign(c_);
//                     delete c_;
//                 }
//             }
//             delete aL;
//             delete bL;
//             delete cL;
//         }
//     return result;
// }

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(NDArray* A, NDArray* B, NDArray* C, double alpha, double beta) {
    
    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)         
        throw std::runtime_error("Rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)        
        throw std::runtime_error("Rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper<T>::mmulNxN op: shapes of A and B arrays are not suitable for matrix multiplication !");
    }
    
    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper<T>::mmulNxN op: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array    
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);    

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))    
            throw std::runtime_error("MmulHelper<T>::mmulNxN op: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else 
        C = new NDArray('c', cExpectedShape, B->dataType());
        
    // multiplication
    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(C->rankOf(), {-2, -1});
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(C->getShapeInfo(), dimsToExclude);
    std::vector<Nd4jLong> idxRanges(2 * C->rankOf());

#pragma omp parallel for schedule(guided) firstprivate(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, C->getShapeInfo(), dimsToExclude, idxRanges.data());
            NDArray cSubArr = (*C)(idxRanges);
            NDArray* c = nullptr;

            if(aRank > bRank) {
                NDArray aSubArr = (*A)(idxRanges);
                BUILD_TRIPLE_SELECTOR(A->dataType(), B->dataType(), C->dataType(), c = mmulMxM, (&aSubArr, B, &cSubArr, 1., 0.), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
            }
            else if(bRank > aRank) {
                NDArray bSubArr = (*B)(idxRanges);                
                BUILD_TRIPLE_SELECTOR(A->dataType(), B->dataType(), C->dataType(), c = mmulMxM, (A, &bSubArr, &cSubArr, 1., 0.), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
            }
            else {                
                NDArray aSubArr = (*A)(idxRanges);
                NDArray bSubArr = (*B)(idxRanges);                
                BUILD_TRIPLE_SELECTOR(A->dataType(), B->dataType(), C->dataType(), c = mmulMxM, (&aSubArr, &bSubArr, &cSubArr, 1., 0.), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
            }                   

            if (c != &cSubArr) { cSubArr.assign(c); delete c; }
        }

    return C;
}
//////////////////////////////////////////////////////////////////////////////
// static
template<typename X, typename Y, typename Z>
nd4j::NDArray* MmulHelper::mmulMxM(nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C , double alpha, double beta) {
    nd4j::NDArray* result = C;
    bool needAllocA = false;
    bool needAllocB = false;
    if (A->isView()) {
        needAllocA = true;
    }
    if (B->isView()) {
        needAllocB = true;
    }
    if (result == nullptr) {
        nd4j_verbose("mmulMxM: Creating new array: [%i x %i]\n", A->rows(), B->columns());
        result = NDArrayFactory::create_<Z>('f', {A->rows(), B->columns()}, nullptr);
    }
        
    auto aShape = A->shapeOf();
    auto bShape = B->shapeOf();
    auto cShape = result->shapeOf();
    char rOrder;
    int M, N, K, lda, ldb, ldc;
    CBLAS_TRANSPOSE transA = CblasNoTrans, 
                    transB = CblasNoTrans;
    M = cShape[0]; // c.rows
    N = cShape[1]; // c.columns
    K = aShape[1]; // a.columns
    rOrder = 'f'; //aOrder;
    nd4j::NDArray* pA = nullptr;
    nd4j::NDArray* pB = nullptr;
    nd4j::NDArray* pC = nullptr;;
    nd4j::NDArray* tA;
    nd4j::NDArray* tB;
    nd4j::NDArray* tC = result;
    
    if (needAllocA) {
        tA = new nd4j::NDArray(A->getBuffer(), A->getShapeInfo(), A->getWorkspace());
        nd4j_verbose("Matrix A was recreated from view.\n", "");
    }
    else 
        tA = A; 
    if (needAllocB) {
        tB = new nd4j::NDArray(B->getBuffer(), B->getShapeInfo(), B->getWorkspace());
        nd4j_verbose("Matrix B was recreated from view.\n", "");
    }
    else 
        tB = B; 
    char aOrder = tA->ordering();
    char bOrder = tB->ordering();
    char cOrder = tC->ordering();
    if (cOrder != rOrder) {
        pC = tC->dup('f');
    } else {
        pC = tC;
    }

// the lines in gemm.cpp for reference
//        bool transAFlag = TransA == CblasTrans;
//        bool transBFlag = TransB == CblasTrans;
    if (tB->ews() == -1) {
        pB = tB->dup('f');
        transB = CblasNoTrans;
    }
    else 
        pB = tB; //->dup('f');
    if (tA->ews() == -1) {
        pA = tA->dup('c');
        transA = CblasNoTrans;
    }
    else 
        pA = tA; //->dup('c');
    
    lda = pA->ordering() == 'f' ? pA->rows() : pA->columns();
    ldb = pB->ordering() == 'f' ? pB->rows() : pB->columns();
    ldc = pC->rows();
    transA = (pA->ordering() == 'c'? CblasTrans:CblasNoTrans);
    transB = (pB->ordering() == 'c' ? CblasTrans:CblasNoTrans);

    auto xType = A->dataType();
    auto yType = B->dataType();
    auto zType = result->dataType();

    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (xType == yType && yType == zType && BlasHelper::getInstance()->template hasGEMM<X>()) {
        nd4j_debug("Using provided GEMM pointer\n","");
        if (xType == FLOAT32)
            BlasHelper::getInstance()->sgemm()(CblasColMajor, transA, transB, M, N, K, (float) alpha, reinterpret_cast<float *>(pA->getBuffer()), lda, reinterpret_cast<float *>(pB->getBuffer()), ldb, (float) beta, reinterpret_cast<float *>(pC->getBuffer()), ldc);
        else if (xType == DOUBLE)
            BlasHelper::getInstance()->dgemm()(CblasColMajor, transA, transB, M, N, K, (double) alpha, reinterpret_cast<double *>(pA->getBuffer()), lda, reinterpret_cast<double *>(pB->getBuffer()), ldb, (double) beta, reinterpret_cast<double *>(pC->getBuffer()), ldc);
        else {
            BUILD_TRIPLE_SELECTOR(xType, yType, zType, nd4j::blas::GEMM, ::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        }
    } else {
        nd4j_debug("mmulMxM: Using fallback GEMM impl\n","");
       
        nd4j::blas::GEMM<X, Y, Z>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
    }
    if (tC != pC) {
        tC->assign(pC);
    }
    if (tA != pA)
        delete pA;
    if (tB != pB)
        delete pB;
    if (tC != pC)
        delete pC;
    if (tA != A)
        delete tA;
    if (tB != B)
        delete tB;
    return result;
}

////////////////////////////////////////////////////////////////////////////
// static
template <typename X, typename Y, typename Z>
nd4j::NDArray* MmulHelper::mmulMxV(nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C , double alpha, double beta) {
    
    nd4j::NDArray* result = C;
        // gemv
        if (A->columns() != B->lengthOf())
            throw std::runtime_error("A columns != B length");
        if (result == nullptr)
            result = NDArrayFactory::create_<Z>('f', {A->rows(), 1});

        auto xType = A->dataType();
        auto yType = B->dataType();
        auto zType = result->dataType();

        // TODO: strides!!!
        if (xType == yType && xType == zType && BlasHelper::getInstance()->hasGEMV<X>()) {
            nd4j_debug("Using provided GEMV pointer\n","");
            auto layout = A->ordering() == 'f' ? CblasColMajor : CblasRowMajor;
            if (std::is_same<X, float>::value)
                BlasHelper::getInstance()->sgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (float) alpha, reinterpret_cast<float *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<float *>(B->getBuffer()), 1, (float) beta, reinterpret_cast<float *>(result->getBuffer()), 1);
            else if (std::is_same<X, double>::value)
                BlasHelper::getInstance()->dgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (double) alpha, reinterpret_cast<double *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<double *>(B->getBuffer()), 1, (double) beta, reinterpret_cast<double *>(result->getBuffer()), 1);
            else
                nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        } else {
            nd4j_debug("Using fallback GEMV impl\n","");
            nd4j::blas::GEMV<X, Y, Z>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        }
    return result;
}



//////////////////////////////////////////////////////////////////////////
nd4j::NDArray* MmulHelper::mmul(nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C , double alpha, double beta) {
    nd4j::NDArray* result = C;
    auto xType = A->dataType();
    auto yType = B->dataType();
    auto zType = C != nullptr ? C->dataType() : yType;

    if (A->rankOf() > 2 || B->rankOf() > 2) {
        return mmulNxN(A, B, C, alpha, beta);
    } else if ((A->isMatrix() && B->isRowVector()) || (A->isMatrix() && B->isColumnVector())) {
        BUILD_TRIPLE_SELECTOR(xType, yType, zType, return mmulMxV, (A, B, C, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    } else if ((A->isRowVector() && B->isRowVector()) || (A->isColumnVector() && B->isColumnVector())) {
        // dot
        if (A->lengthOf() != B->lengthOf())
            throw std::runtime_error("A length != B length");
        if (result == nullptr)
            result = NDArrayFactory::create_('c', {1, 1},B->dataType());
        //result->p(0, nd4j::math::nd4j_dot(A->getBuffer(), B->getBuffer(), A->lengthOf()));
        BUILD_TRIPLE_SELECTOR(xType, yType, result->dataType(), _dot, (A->buffer(), B->buffer(), result->buffer(), A->lengthOf()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        return result;
    } else { //if ((A->isMatrix() && B->isMatrix()) || (A->isVector() && B->isMatrix()) || (A->isColumnVector() && B->isRowVector())) {
        // gemm
        // int[] shape = {rows(), other.columns()};
        BUILD_TRIPLE_SELECTOR(xType, yType, zType, return mmulMxM, (A, B, C, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::simpleMMul(const NDArray* a, const NDArray* b, NDArray* c, const double alpha, const double beta) {
    
    if(a->rankOf() != 2 || b->rankOf() != 2)
        throw std::runtime_error("NDArrayFactory::simpleMMul static function: some of input arrays has rank not equal to 2 !");

    if(a->shapeOf()[1] != b->shapeOf()[0])
        throw std::runtime_error("NDArrayFactory::simpleMMul static function: the number of A columns is not equal to number of B rows !");

    NDArray* dot = c;
    if(c == nullptr) 
        c = NDArrayFactory::create_('f', {a->shapeOf()[0], b->shapeOf()[1]}, b->dataType(), a->getWorkspace());
    else {
        if( c->shapeOf()[0] != a->shapeOf()[0] || c->shapeOf()[1] != b->shapeOf()[1])
            throw std::runtime_error("NDArrayFactory::simpleMMul static function: wrong shape of C array !");
        if(beta != 0. ) {
            dot = NDArrayFactory::create_(c->ordering(), {a->shapeOf()[0], b->shapeOf()[1]}, c->dataType(), a->getWorkspace());
            if( beta != 1.)
                c->applyScalar(scalar::Multiply, beta, c, nullptr);
        }        
    }
    int M = a->shapeOf()[0];
    int N = b->shapeOf()[1];
    int K = a->shapeOf()[1];

    // FIXME: double?
    for(int row = 0; row < M; ++row)
        for(int col = 0; col < N; ++col)
            for(int j = 0; j < K; ++j)
                    dot->p(row,col, a->e<double>(row,j) * b->e<double>(j,col));

    if(alpha != 1.)
        dot->applyScalar(scalar::Multiply, alpha, dot, nullptr);

    if(beta != 0.) {
        c->applyPairwiseTransform(pairwise::Add, dot, nullptr);
        delete dot;
    }
    
    return c;
}

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

#pragma omp parallel for schedule(guided)
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

    template <typename X, typename Y, typename Z>
    void MmulHelper::_dot(void* vA, void* vB, void* vC, Nd4jLong length) {
        auto A = reinterpret_cast<X *>(vA);
        auto B = reinterpret_cast<Y *>(vB);
        auto C = reinterpret_cast<Z *>(vC);

        C[0] = nd4j::math::nd4j_dot<X, Y, Z>(A, B, length);
    }

    BUILD_TRIPLE_TEMPLATE(template nd4j::NDArray* MmulHelper::mmulMxM, (nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C, double alpha, double beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    BUILD_TRIPLE_TEMPLATE(template nd4j::NDArray* MmulHelper::mmulMxV, (nd4j::NDArray* A, nd4j::NDArray* B, nd4j::NDArray* C, double alpha, double beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    BUILD_TRIPLE_TEMPLATE(template void MmulHelper::_dot, (void* vA, void* vB, void* vC, Nd4jLong length), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
}


#endif
