//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.06.2018
//

#ifndef LIBND4J_MMULHELPER_CPP
#define LIBND4J_MMULHELPER_CPP

#include "../MmulHelper.h"
#include "../NDArrayFactory.h"
#include <helpers/ShapeUtils.h>
#include <helpers/BlasHelper.h>

namespace nd4j { 

    
//////////////////////////////////////////////////////////////////////////
template<typename T>
nd4j::NDArray<T>* nd4j::MmulHelper<T>::tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::initializer_list<int>& axesA, const std::initializer_list<int>& axesB) {
    std::vector<int> aA(axesA);
    std::vector<int> aB(axesB);
    return tensorDot(A, B, aA, aB);
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
nd4j::NDArray<T>* nd4j::MmulHelper<T>::tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, const std::vector<int>& axes_0, const std::vector<int>& axes_1) {
    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;        
    auto outShape = ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);
    NDArray<T>* aPR(const_cast<NDArray<T>*>(a)), *bPR(const_cast<NDArray<T>*>(b));
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
    NDArray<T>* c = mmul(aPR, bPR, nullptr, 1.0, 0.0);
    c->reshapei('c', outShape);
    
    if(aPR != a)
        delete aPR;        
    if(bPR != b)
        delete bPR;
    return c;
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void nd4j::MmulHelper<T>::tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, const std::vector<int>& axes_a, const std::vector<int>& axes_b, const std::vector<int>& permutForC) {
    std::vector<int> permutAt, permutBt;
    std::vector<Nd4jLong> shapeAt, shapeBt;
    auto outShape = ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_a, axes_b, permutAt, permutBt, shapeAt, shapeBt);
    NDArray<T> *aPR(const_cast<NDArray<T>*>(a)), *bPR(const_cast<NDArray<T>*>(b)), *cP(c), *cPR(c);
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
template<typename T>
void nd4j::MmulHelper<T>::tensorDot(const NDArray<T>* a, const NDArray<T>* b, NDArray<T>* c, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB, const std::vector<std::vector<Nd4jLong>>& modifC) {
    NDArray<T> *aPR(const_cast<NDArray<T>*>(a)), *bPR(const_cast<NDArray<T>*>(b));
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
    std::vector<NDArray<T>*> cArrs = {c}; 
    if(!whatToDoWithC.empty()) {
        cArrs = std::vector<NDArray<T>*>(whatToDoWithC.size()+1, c);
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
template<typename T>
NDArray<T>* nd4j::MmulHelper<T>::tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, const std::vector<std::vector<Nd4jLong>>& modifA, const std::vector<std::vector<Nd4jLong>>& modifB) {
    NDArray<T> *aPR(const_cast<NDArray<T>*>(a)), *bPR(const_cast<NDArray<T>*>(b));
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
            
    NDArray<T>* result = mmul(aPR, bPR, nullptr, 1.0, 0.0);
    
    if(aPR != a)
        delete aPR;
    if(bPR != b)
        delete bPR;
    return result;
}
#endif

//////////////////////////////////////////////////////////////////////////
template<typename T>
nd4j::NDArray<T>* MmulHelper<T>::mmulNxN(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , 
    T alpha, T beta) {
       nd4j::NDArray<T>* result = C;
        // matmul
        if (A->rankOf() != B->rankOf()) {
            // FIXME (r119): this is temporary fix for @shyrma, proper impl required here
            int pRows = A->sizeAt(-2);
            int pCols = B->sizeAt(-1);
            if (A->sizeAt(-1) != B->sizeAt(-2)) {
                nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
                            A->sizeAt(-1), B->sizeAt(-2))
                throw std::runtime_error("Numbers of rows/columns should match");
            }
            std::vector<Nd4jLong> newShape;
            if (A->rankOf() > B->rankOf())
                for (int e = 0; e < A->rankOf() - 2; e++)
                    newShape.emplace_back(A->sizeAt(e));
            else
                for (int e = 0; e < B->rankOf() - 2; e++)
                    newShape.emplace_back(B->sizeAt(e));
            newShape.push_back(pRows);
            newShape.push_back(pCols);
            if (result == nullptr)
                result = new NDArray<T>('c', newShape);
            else if (!result->isSameShape(newShape)) {
                nd4j_printf("Bad result shape for MatMul\n", "");
                throw std::runtime_error("Bad result shape");
            }
            if (A->rankOf() > B->rankOf()) {
                auto aL = NDArrayFactory<T>::allTensorsAlongDimension(A, {A->rankOf() - 2, A->rankOf() - 1});
                auto cL = NDArrayFactory<T>::allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});
                nd4j_debug("NumTads: %i\n", aL->size());
                for (int e = 0; e < aL->size(); e++) {
                    auto c_ = mmul(aL->at(e), B, cL->at(e));
                    if (c_ != cL->at(e)) {
                        cL->at(e)->assign(c_);
                        delete c_;
                    }
                }
                delete aL;
                delete cL;
            } else {
                auto bL = NDArrayFactory<T>::allTensorsAlongDimension(B, {B->rankOf() - 2, B->rankOf() - 1});
                auto cL = NDArrayFactory<T>::allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});
                nd4j_debug("NumTads: %i\n", bL->size());
                for (int e = 0; e < bL->size(); e++) {
                    auto c_ = mmul(A, bL->at(e), cL->at(e));
                    if (cL->at(e) != c_) {
                        cL->at(e)->assign(c_);
                        delete c_;
                    }
                }
                delete bL;
                delete cL;
            }
        } else {
            //int dims = A->rankOf();
            std::vector<Nd4jLong> newShape;
            for (int e = 0; e < A->rankOf() - 2; e++)
                if (A->sizeAt(e) != B->sizeAt(e)) {
                    nd4j_printf("Dimension [%i] differs for A and B: %i vs %i", e, A->sizeAt(e), B->sizeAt(e));
                    throw std::runtime_error("Outer dimensions for A & B should be equal");
                } else {
                    newShape.push_back(A->sizeAt(e));
                }
            int pRows = A->sizeAt(-2);
            int pCols = B->sizeAt(-1);
            if (A->sizeAt(-1) != B->sizeAt(-2)) {
                nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
                            A->sizeAt(-1), B->sizeAt(-2))
                throw std::runtime_error("Numbers of rows/columns should match");
            }
            newShape.push_back(pRows);
            newShape.push_back(pCols);
            //Nd4jLong prod = shape::prodLong(newShape.data(), newShape.size());
            if (result == nullptr)
                result = new NDArray<T>('c', newShape);
            else if (!result->isSameShape(newShape)) {
                nd4j_printf("Bad result shape for MatMul\n", "");
                throw std::runtime_error("Bad result shape");
            }
            auto aL = NDArrayFactory<T>::allTensorsAlongDimension(A, {A->rankOf() - 2, A->rankOf() - 1});
            auto bL = NDArrayFactory<T>::allTensorsAlongDimension(B, {B->rankOf() - 2, B->rankOf() - 1});
            auto cL = NDArrayFactory<T>::allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});
            int aL_size = aL->size();
            int bL_size = bL->size();
            int cL_size = cL->size();
            nd4j_debug("NumTads: %i\n", aL->size());
            for (int e = 0; e < aL->size(); e++) {
                auto aLt = aL->at(e);
                auto bLt = bL->at(e);
                auto cLt = cL->at(e);
                
                auto c_ = mmul(aLt, bLt, cLt);
                if (c_ != cLt) {
                    cLt->assign(c_);
                    delete c_;
                }
            }
            delete aL;
            delete bL;
            delete cL;
        }
    return result;
}

//////////////////////////////////////////////////////////////////////////////
// static
template<typename T>
nd4j::NDArray<T>* MmulHelper<T>::mmulMxM(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , 
    T alpha, T beta) {
    nd4j::NDArray<T>* result = C;
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
        result = new NDArray<T>('f', {A->rows(), B->columns()});
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
    nd4j::NDArray<T>* pA = nullptr;
    nd4j::NDArray<T>* pB = nullptr;
    nd4j::NDArray<T>* pC = nullptr;;
    nd4j::NDArray<T>* tA;
    nd4j::NDArray<T>* tB;
    nd4j::NDArray<T>* tC = result; 
    
    if (needAllocA) {
        tA = new nd4j::NDArray<T>(A->getBuffer(), A->getShapeInfo(), A->getWorkspace());
        nd4j_verbose("Matrix A was recreated from view.\n", "");
    }
    else 
        tA = A; 
    if (needAllocB) {
        tB = new nd4j::NDArray<T>(B->getBuffer(), B->getShapeInfo(), B->getWorkspace());
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
    // we'll use platform-specific gemm here eventually. maybe tomorrow.
    // TODO: put proper _gemm here
    if (BlasHelper::getInstance()->template hasGEMM<T>()) {
        nd4j_debug("Using provided GEMM pointer\n","");
        if (sizeof(T) == 4)
            BlasHelper::getInstance()->sgemm()(CblasColMajor, transA, transB, M, N, K, (float) alpha, reinterpret_cast<float *>(pA->getBuffer()), lda, reinterpret_cast<float *>(pB->getBuffer()), ldb, (float) beta, reinterpret_cast<float *>(pC->getBuffer()), ldc);
        else if (sizeof(T) == 8)
            BlasHelper::getInstance()->dgemm()(CblasColMajor, transA, transB, M, N, K, (double) alpha, reinterpret_cast<double *>(pA->getBuffer()), lda, reinterpret_cast<double *>(pB->getBuffer()), ldb, (double) beta, reinterpret_cast<double *>(pC->getBuffer()), ldc);
        else
            nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
    } else {
        nd4j_debug("mmulMxM: Using fallback GEMM impl\n","");
       
        nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
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
template<typename T>
nd4j::NDArray<T>* MmulHelper<T>::mmulMxV(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , 
    T alpha, T beta) {
    
    nd4j::NDArray<T>* result = C;
        // gemv
        if (A->columns() != B->lengthOf())
            throw std::runtime_error("A columns != B length");
        if (result == nullptr)
            result = new NDArray<T>('f', {A->rows(), 1});
        // TODO: strides!!!
        if (BlasHelper::getInstance()->hasGEMV<T>()) {
            nd4j_debug("Using provided GEMV pointer\n","");
            auto layout = A->ordering() == 'f' ? CblasColMajor : CblasRowMajor;
            if (sizeof(T) == 4)
                BlasHelper::getInstance()->sgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (float) alpha, reinterpret_cast<float *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<float *>(B->getBuffer()), 1, (float) beta, reinterpret_cast<float *>(result->getBuffer()), 1);
            else if (sizeof(T) == 8)
                BlasHelper::getInstance()->dgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (double) alpha, reinterpret_cast<double *>(A->getBuffer()), layout == CblasColMajor ? A->rows() : A->columns(), reinterpret_cast<double *>(B->getBuffer()), 1, (double) beta, reinterpret_cast<double *>(result->getBuffer()), 1);
            else
                nd4j::blas::GEMV<T>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        } else {
            nd4j_debug("Using fallback GEMV impl\n","");
            nd4j::blas::GEMV<T>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->lengthOf(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
        }
    return result;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
nd4j::NDArray<T>* MmulHelper<T>::mmul(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , 
    T alpha, T beta) {
    nd4j::NDArray<T>* result = C;
    if (A->rankOf() > 2 || B->rankOf() > 2) {
        return mmulNxN(A, B, C, alpha, beta);
    } else if ((A->isMatrix() && B->isRowVector()) || (A->isMatrix() && B->isColumnVector())) {
        return mmulMxV(A, B, C, alpha, beta);
    } else if ((A->isRowVector() && B->isRowVector()) || (A->isColumnVector() && B->isColumnVector())) {
        // dot
        if (A->lengthOf() != B->lengthOf())
            throw std::runtime_error("A length != B length");
        if (result == nullptr)
            result = new NDArray<T>('c', {1, 1});
        result->putScalar(0, nd4j::math::nd4j_dot(A->getBuffer(), B->getBuffer(), A->lengthOf()));
        return result;
    } else { //if ((A->isMatrix() && B->isMatrix()) || (A->isVector() && B->isMatrix()) || (A->isColumnVector() && B->isRowVector())) {
        // gemm
        // int[] shape = {rows(), other.columns()};
        return mmulMxM(A, B, C, alpha, beta);
    }
    return result;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* MmulHelper<T>::simpleMMul(const NDArray<T>* a, const NDArray<T>* b, NDArray<T>* c, const T alpha, const T beta) {
    
    if(a->rankOf() != 2 || b->rankOf() != 2)
        throw std::runtime_error("NDArrayFactory::simpleMMul static function: some of input arrays has rank not equal to 2 !");

    if(a->shapeOf()[1] != b->shapeOf()[0])
        throw std::runtime_error("NDArrayFactory::simpleMMul static function: the number of A columns is not equal to number of B rows !");

    NDArray<T>* dot = c;
    if(c == nullptr) 
        c = new NDArray<T>('f', {a->shapeOf()[0], b->shapeOf()[1]}, a->getWorkspace());        
    else {
        if( c->shapeOf()[0] != a->shapeOf()[0] || c->shapeOf()[1] != b->shapeOf()[1])
            throw std::runtime_error("NDArrayFactory::simpleMMul static function: wrong shape of C array !");
        if(beta != (T)0. ) {
            dot = new NDArray<T>(c->ordering(), {a->shapeOf()[0], b->shapeOf()[1]},  a->getWorkspace());
            if( beta != (T)1.)
                c->template applyScalar<simdOps::Multiply<T>>(beta);            
        }        
    }
    int M = a->shapeOf()[0];
    int N = b->shapeOf()[1];
    int K = a->shapeOf()[1];
    for(int row = 0; row < M; ++row)
        for(int col = 0; col < N; ++col)
            for(int j = 0; j < K; ++j)
                    (*dot)(row,col) += (*a)(row,j)*(*b)(j,col);

    if(alpha != (T)1.)
        dot->template applyScalar<simdOps::Multiply<T>>(alpha);

    if(beta != (T)0.) {
        c->template applyPairwiseTransform<simdOps::Add<T>>(dot, nullptr);
        delete dot;
    }
    
    return c;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void MmulHelper<T>::matmul(const nd4j::NDArray<T>* x, const nd4j::NDArray<T>* y, nd4j::NDArray<T>* z, const bool transX, const bool transY) {
   
    int xRank = x->rankOf();
    int yRank = y->rankOf();

    std::vector<Nd4jLong> outShape = ShapeUtils<T>::evalShapeForMatmul(x->getShapeInfo(), y->getShapeInfo(), transX, transY);
    if(!z->isSameShape(outShape)) {
        nd4j_printf("NDArrayFactory::matmul static method: input shape of output array is wrong, actual is %s and expected is %s ! \n", ShapeUtils<T>::shapeAsString(z).c_str(), ShapeUtils<T>::shapeAsString(outShape).c_str());
        throw std::invalid_argument("");       
    }
        
    NDArray<T>* xT(const_cast<NDArray<T>*>(x)), *yT(const_cast<NDArray<T>*>(y)), *zT(z);
    
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
        
        mmul(xT, yT, zT, (T)1., (T)0.);        
    }
    else {  // rest cases -  batched mmul
        
        const int batchRank = xRank - 2;    
        std::vector<int> dimsToExclude(batchRank);
        for(int i = 0; i < batchRank; ++i)
            dimsToExclude[i] = i;

        const Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(xT->getShapeInfo(), dimsToExclude);
        Nd4jLong idxRanges[2 * xRank];

#pragma omp parallel for schedule(guided) private(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils<T>::evalIdxRangesForSubArr(i, xT->getShapeInfo(), dimsToExclude, idxRanges);
            NDArray<T> xSubArr = (*xT)(idxRanges, false);
            NDArray<T> ySubArr = (*yT)(idxRanges, false);
            NDArray<T> zSubArr = (*zT)(idxRanges, false);
            mmul(&xSubArr, &ySubArr, &zSubArr, (T)1., (T)0.);
        }
    }

    if(xT != x)
        delete xT;
    if(yT != y)
        delete yT;
    if(zT != z)
        delete zT;
}



template class ND4J_EXPORT MmulHelper<float>;
template class ND4J_EXPORT MmulHelper<float16>;
template class ND4J_EXPORT MmulHelper<double>;
}


#endif
