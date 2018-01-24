//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_NDARRAYFACTORY_CPP
#define LIBND4J_NDARRAYFACTORY_CPP

#include "../NDArrayFactory.h"
#include "../NDArray.h"
#include <memory/Workspace.h>
#include <ops/gemm.h>
#include <types/float16.h>
#include <helpers/ShapeUtils.h>
#include <helpers/BlasHelper.h>

namespace nd4j {

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allExamples(NDArray<T>* ndArray) {
        std::vector<int> dimensions;
        for (int e = 1; e < ndArray->rankOf(); e++)
            dimensions.push_back(e);

        return allTensorsAlongDimension(ndArray, dimensions);
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions) {
        auto result = new ResultSet<T>();

        if (indices.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(ndArray->getShapeInfo(), copy.data(), copy.size());
        Nd4jIndex numTads = ndArray->lengthOf() / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(ndArray->getShapeInfo(), copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        int* shapeInfo = new int[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (auto idx: indices) {
            if (idx >= numTads) {
                nd4j_printf("Index %i is higher then number of TADs: %i\n", idx, numTads);
                throw "Bad index";
            }


            T* buffer = ndArray->getBuffer() + tad->tadOffsets[idx];
            auto array = new NDArray<T>(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allTensorsAlongDimension(const NDArray<T>* ndArray, const std::initializer_list<int> dimensions) {
        std::vector<int> vec(dimensions);
        return allTensorsAlongDimension(ndArray, vec);
    }

    template<typename T>
    ResultSet<T>* NDArrayFactory<T>::allTensorsAlongDimension(const NDArray<T>* ndArray, const std::vector<int> &dimensions) {
        auto result = new ResultSet<T>();

        if(dimensions.size() == 0)
            return result;

        std::vector<int> copy(dimensions);

        // we need to sort dimensions (?)
        if (dimensions.size() > 1)
            std::sort (copy.begin(), copy.end());

        Nd4jIndex tadLength = shape::tadLength(ndArray->getShapeInfo(), copy.data(), copy.size());
        Nd4jIndex numTads = ndArray->lengthOf() / tadLength;

        std::unique_ptr<shape::TAD> tad(new shape::TAD(ndArray->getShapeInfo(), copy.data(), copy.size()));
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        int* shapeInfo = new int[shape::shapeInfoLength(tad->tadOnlyShapeInfo[0])];
        std::memcpy(shapeInfo, tad->tadOnlyShapeInfo, shape::shapeInfoByteLength(tad->tadOnlyShapeInfo));

        for (int idx = 0; idx < numTads; idx++ ) {
            T* buffer = const_cast<NDArray<T>*>(ndArray)->getBuffer() + tad->tadOffsets[idx];
            auto array = new NDArray<T>(buffer, shapeInfo);
            result->push_back(array);
        }

        // if we have no indices - just delete shapeInfo
        if (result->size() > 0)
            result->at(0)->triggerAllocationFlag(false, true);
        else
            delete[] shapeInfo;

        return result;
    }

    
    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    nd4j::NDArray<T>* nd4j::NDArrayFactory<T>::tensorDot(const nd4j::NDArray<T>* A, const nd4j::NDArray<T>* B, const std::initializer_list<int> axesA, const std::initializer_list<int> axesB) {
        std::vector<int> aA(axesA);
        std::vector<int> aB(axesB);
        return tensorDot(A, B, aA, aB);
    }

    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    nd4j::NDArray<T>* nd4j::NDArrayFactory<T>::tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, std::vector<int>& axes_0, std::vector<int>& axes_1) {

        std::vector<int> permutAt, permutBt, shapeAt, shapeBt;
        std::vector<int> outShape = ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);

        NDArray<T>* aT = a->permute(permutAt);
        NDArray<T>* bT = b->permute(permutBt);
        aT->reshapei('c', shapeAt);
        bT->reshapei('c', shapeBt);        

        NDArray<T>* c = nd4j::NDArrayFactory<T>::mmulHelper(aT, bT, nullptr, 1.0, 0.0);
        c->reshapei('c', outShape);

        if (aT != a)
            delete aT;

        if (bT != b)
            delete bT;

        return c;
    }


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    void nd4j::NDArrayFactory<T>::tensorDot(const nd4j::NDArray<T>* a, const nd4j::NDArray<T>* b, nd4j::NDArray<T>* c, std::vector<int>& axes_0, std::vector<int>& axes_1) {

        if(c->rankOf() != 2 || c->shapeOf()[0] != a->shapeOf()[0] || c->shapeOf()[1] != b->shapeOf()[1])
            throw "NDArrayFactory::simpleMMul static function: wrong shape of C array !";

        std::vector<int> permutAt, permutBt, shapeAt, shapeBt;
        std::vector<int> outShape = ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);

        NDArray<T>* aT = a->permute(permutAt);
        NDArray<T>* bT = b->permute(permutBt);
        aT->reshapei('c', shapeAt);
        bT->reshapei('c', shapeBt);        

        nd4j::NDArrayFactory<T>::mmulHelper(aT, bT, c, 1.0, 0.0);
        c->reshapei('c', outShape);

        if (aT != a)
            delete aT;

        if (bT != b)
            delete bT;
        
    }


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    nd4j::NDArray<T>* NDArrayFactory<T>::mmulHelper(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , T alpha, T beta) {
        nd4j::NDArray<T>* result = C;

        if (A->rankOf() > 2 || B->rankOf() > 2) {
            // matmul
            if (A->rankOf() != B->rankOf()) {
                // FIXME (r119): this is temporary fix for @shyrma, proper impl required here
                int pRows = A->sizeAt(-2);
                int pCols = B->sizeAt(-1);

                if (A->sizeAt(-1) != B->sizeAt(-2)) {
                    nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
                                A->sizeAt(-1), B->sizeAt(-2))
                    throw "Numbers of rows/columns should match";
                }

                std::vector<int> newShape;
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
                    throw "Bad result shape";
                }


                if (A->rankOf() > B->rankOf()) {
                    auto aL = allTensorsAlongDimension(A, {A->rankOf() - 2, A->rankOf() - 1});
                    auto cL = allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});

                    nd4j_debug("NumTads: %i\n", aL->size());
                    for (int e = 0; e < aL->size(); e++) {
                        auto c_ = mmulHelper(aL->at(e), B);

                        cL->at(e)->assign(c_);
                        delete c_;
                    }

                    delete aL;
                    delete cL;
                } else {
                    auto bL = allTensorsAlongDimension(B, {B->rankOf() - 2, B->rankOf() - 1});
                    auto cL = allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});

                    nd4j_debug("NumTads: %i\n", bL->size());
                    for (int e = 0; e < bL->size(); e++) {
                        auto c_ = mmulHelper(A, bL->at(e));

                        cL->at(e)->assign(c_);
                        delete c_;
                    }

                    delete bL;
                    delete cL;
                }

            } else {
                //int dims = A->rankOf();

                std::vector<int> newShape;
                for (int e = 0; e < A->rankOf() - 2; e++)
                    if (A->sizeAt(e) != B->sizeAt(e)) {
                        nd4j_printf("Dimension [%i] differs for A and B: %i vs %i", e, A->sizeAt(e), B->sizeAt(e));
                        throw "Outer dimensions for A & B should be equal";
                    } else {
                        newShape.push_back(A->sizeAt(e));
                    }

                int pRows = A->sizeAt(-2);
                int pCols = B->sizeAt(-1);

                if (A->sizeAt(-1) != B->sizeAt(-2)) {
                    nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead",
                                A->sizeAt(-1), B->sizeAt(-2))
                    throw "Numbers of rows/columns should match";
                }

                newShape.push_back(pRows);
                newShape.push_back(pCols);

                //Nd4jIndex prod = shape::prodLong(newShape.data(), newShape.size());

                if (result == nullptr)
                    result = new NDArray<T>('c', newShape);
                else if (!result->isSameShape(newShape)) {
                    nd4j_printf("Bad result shape for MatMul\n", "");
                    throw "Bad result shape";
                }

                auto aL = allTensorsAlongDimension(A, {A->rankOf() - 2, A->rankOf() - 1});
                auto bL = allTensorsAlongDimension(B, {B->rankOf() - 2, B->rankOf() - 1});
                auto cL = allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});

                int aL_size = aL->size();
                int bL_size = bL->size();
                int cL_size = cL->size();

                nd4j_debug("NumTads: %i\n", aL->size());
                for (int e = 0; e < aL->size(); e++) {
                    auto aLt = aL->at(e);
                    auto bLt = bL->at(e);
                    auto cLt = cL->at(e);

                    auto c_ = mmulHelper(aLt, bLt);

                    cLt->assign(c_);

                    delete c_;
                }

                delete aL;
                delete bL;
                delete cL;
            }
        } else if ((A->isMatrix() && B->isRowVector()) || (A->isMatrix() && B->isColumnVector())) {
            // gemv
            if (A->columns() != B->rows())
                throw "A columns != B length";

            if (result == nullptr)
                result = new NDArray<T>(A->rows(), 1, 'f');

            // TODO: strides!!!
            if (BlasHelper::getInstance()->hasGEMV<T>()) {
                nd4j_debug("Using provided GEMV pointer\n","");

                auto layout = A->ordering() == 'f' ? CblasColMajor : CblasRowMajor;

                if (sizeof(T) == 4)
                    BlasHelper::getInstance()->sgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (float) alpha, (float *) A->getBuffer(), layout == CblasColMajor ? A->rows() : A->columns(), (float *) B->getBuffer(), 1, (float) beta, (float *) result->getBuffer(), 1);
                else if (sizeof(T) == 8)
                    BlasHelper::getInstance()->dgemv()(layout, CblasNoTrans, A->rows(), A->columns(), (double) alpha, (double *) A->getBuffer(), layout == CblasColMajor ? A->rows() : A->columns(), (double *) B->getBuffer(), 1, (double) beta, (double *) result->getBuffer(), 1);
                else
                    nd4j::blas::GEMV<T>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->rows(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
            } else {
                nd4j_debug("Using fallback GEMV impl\n","");

                nd4j::blas::GEMV<T>::op(A->ordering() == 'f' ? CblasTrans : 0, A->rows(), A->columns(), alpha, A->getBuffer(), B->rows(), B->getBuffer(), 1, beta, result->getBuffer(), 1);
            }
        } else if ((A->isRowVector() && B->isRowVector()) || (A->isColumnVector() && B->isColumnVector())) {
            // dot
            if (A->lengthOf() != B->lengthOf())
                throw "A length != B length";

            if (result == nullptr)
                result = new NDArray<T>(1,1, 'c');

            result->putScalar(0, nd4j::math::nd4j_dot(A->getBuffer(), B->getBuffer(), A->lengthOf()));
        } else { //if ((A->isMatrix() && B->isMatrix()) || (A->isVector() && B->isMatrix()) || (A->isColumnVector() && B->isRowVector())) {
            // gemm
            // int[] shape = {rows(), other.columns()};
            
            if (result == nullptr) {
                nd4j_verbose("Creating new array: [%i x %i]\n", A->rows(), B->columns());
                result = new NDArray<T>(A->rows(), B->columns(), 'f');
            }


            char aOrder = A->ordering();
            char bOrder = B->ordering();
            char cOrder = result->ordering();

            int *aShape = A->shapeOf();
            int *bShape = B->shapeOf();
            int *cShape = result->shapeOf();

            char rOrder;

            int M, N, K, lda, ldb, ldc;
            char transA, transB;

            nd4j::NDArray<T>* pA = nullptr;
            nd4j::NDArray<T>* pB = nullptr;
            nd4j::NDArray<T>* pC = nullptr;;

            //_C = new NDArray<T>(C, cShapeInfo);

            auto tA = new nd4j::NDArray<T>(A->getBuffer(), A->getShapeInfo());
            auto tB = new nd4j::NDArray<T>(B->getBuffer(), B->getShapeInfo());
            auto tC = new nd4j::NDArray<T>(result->getBuffer(), result->getShapeInfo());

            if (cOrder != 'f') {
                pC = tC->dup('f');
            } else {
                pC = tC;
            }

            if (aOrder == bOrder) {
                //printf("Going dRoute here\n");

                if (aOrder == 'c') {
                    // we might need to transpose matrices,
                    // todo: we need dup(c/f) helper here
                    pA = tA->dup('f');
                    pB = tB->dup('f');
                } else {
                    pA = tA;
                    pB = tB;
                }

                rOrder = 'f';

                M = cShape[0];
                N = cShape[1];
                K = aShape[1];

                lda = aShape[0];
                ldb = bShape[0];
                ldc = cShape[0];

                transA = 'N';
                transB = 'N';
            } else {
                //printf("Going tRoute here\n");
                if (aOrder == 'c') {
                    // dup(F) A here
                    pA = tA->dup('f');
                    pB = tB;
                } else {
                    // dup(F) B here
                    pA = tA;
                    pB = tB->dup('f');
                }

                // pC = tC->dup('f');

                M = cShape[0];
                N = cShape[1];
                K = aShape[1];

                rOrder = aOrder;

                lda = aShape[0];
                ldb = bShape[0];
                ldc = cShape[0];

                transA = 'N';
                transB = 'N';
            }

            // we'll use platform-specific gemm here eventually. maybe tomorrow.
            // TODO: put proper _gemm here
            if (BlasHelper::getInstance()->template hasGEMM<T>()) {
                nd4j_debug("Using provided GEMM pointer\n","");
                if (sizeof(T) == 4)
                    BlasHelper::getInstance()->sgemm()(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, (float) alpha, (float *) pA->getBuffer(), lda, (float *) pB->getBuffer(), ldb, (float) beta, (float *) pC->getBuffer(), ldc);
                else if (sizeof(T) == 8)
                    BlasHelper::getInstance()->dgemm()(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, (double) alpha, (double *) pA->getBuffer(), lda, (double *) pB->getBuffer(), ldb, (double) beta, (double *) pC->getBuffer(), ldc);
                else
                    nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
            } else {
                nd4j_debug("Using fallback GEMM impl\n","");

                nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);
            }

            if (cOrder != 'f') {
                tC->assign(pC);
            }

            if (tA != pA)
                delete pA;

            if (tB != pB)
                delete pB;

            if (tC != pC)
                delete pC;


            delete tA;
            delete tB;
            delete tC;
        }

        return result;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory<T>::tile(NDArray<T> *original, std::vector<int> &dimensions) {
        return nullptr;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory<T>::repeat(NDArray<T> *original, std::vector<int> &repeats) {
        return nullptr;
    }

    template<typename T>
    NDArray<T>* NDArrayFactory<T>::linspace(T from, T to, Nd4jIndex numElements) {
        auto result = new NDArray<T>(numElements, 'c');

        for (Nd4jIndex e = 0; e < numElements; e++) {
            T step = (T) e / ((T) numElements - (T) 1.0f);
            result->getBuffer()[e] = (from * ((T) 1.0f - step) + step * to);
        }

        return result;
    }

    template<typename T>
    void NDArrayFactory<T>::linspace(T from, NDArray<T>& arr, T step) {
        
        int size = arr.lengthOf();
        for (Nd4jIndex i = 0; i < size; ++i)
            arr(i) = from + (step * i);
    }

    template<typename T>
    NDArray<T>* NDArrayFactory<T>::createUninitialized(NDArray<T>* other) {
        auto workspace = other->getWorkspace();

        int* newShape;
        ALLOCATE(newShape, workspace, shape::shapeInfoLength(other->getShapeInfo()), int);
        memcpy(newShape, other->getShapeInfo(), shape::shapeInfoByteLength(other->getShapeInfo()));

        T* buffer;
        ALLOCATE(buffer, workspace, other->lengthOf(), T);
        auto result = new NDArray<T>(buffer, newShape, workspace);
        result->triggerAllocationFlag(true, true);

        return result;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::scalar(T value) {
        auto res = new NDArray<T>('c', {1, 1});
        res->putScalar(0, value);

        return res;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::valueOf(std::initializer_list<int> shape, T value, char order) {
        auto result = new NDArray<T>(order, shape);
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::valueOf(std::vector<int>& shape, T value, char order) {
        auto result = new NDArray<T>(order, shape);
        result->assign(value);
        return result;
    }

    template <typename T>
    NDArray<T>* NDArrayFactory<T>::concat(const std::vector<NDArray<T> *>& vectors, int axis, NDArray<T>* target) {
        NDArray<T>* result = target;

        if (vectors.size() == 1) {
            if (result == nullptr)
                result = vectors.at(0)->dup();
            else
                result->assign(vectors.at(0));
        } else {
            Nd4jPointer* buffers = new Nd4jPointer[vectors.size()];
            Nd4jPointer* shapes = new Nd4jPointer[vectors.size()];

            NDArray<T> *first = vectors.at(0);

            if (axis < 0)
                axis += first->rankOf();

            buffers[0] = (Nd4jPointer) first->buffer();
            shapes[0] = (Nd4jPointer) first->shapeInfo();

            std::vector<int> shape((unsigned int)first->rankOf());
            for (int e = 0; e < first->rankOf(); e++)
                shape[e] = first->sizeAt(e);

            for (int e = 1; e < (int) vectors.size(); e++) {
                NDArray<T>* array = vectors.at(e);

                buffers[e] = (Nd4jPointer) array->buffer();
                shapes[e] = (Nd4jPointer) array->shapeInfo();

                shape[axis] += array->sizeAt(axis);
            }

            if (result == nullptr)
                result = new NDArray<T>(first->ordering(), shape);


            nd4j::SpecialMethods<T>::concatCpuGeneric(axis, vectors.size(), buffers, shapes, result->buffer(), result->shapeInfo());

            delete[] buffers;
            delete[] shapes;
        }

        return result;
    }


//////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray<T>* NDArrayFactory<T>::simpleMMul(const NDArray<T>* a, const NDArray<T>* b, NDArray<T>* c, const T alpha, const T beta) {
    
    if(a->rankOf() != 2 || b->rankOf() != 2)
        throw "NDArrayFactory::simpleMMul static function: some of input arrays has rank not equal to 2 !";

    if(a->shapeOf()[1] != b->shapeOf()[0])
        throw "NDArrayFactory::simpleMMul static function: the number of A columns is not equal to number of B rows !";

    NDArray<T>* dot = c;
    if(c == nullptr) 
        c = new NDArray<T>(a->shapeOf()[0], b->shapeOf()[1], 'f', a->getWorkspace());        
    else {
        if( c->shapeOf()[0] != a->shapeOf()[0] || c->shapeOf()[1] != b->shapeOf()[1])
            throw "NDArrayFactory::simpleMMul static function: wrong shape of C array !";
        if(beta != (T)0. ) {
            dot = new NDArray<T>(a->shapeOf()[0], b->shapeOf()[1], c->ordering(), a->getWorkspace());
            if( beta != (T)1.)
                c->template applyScalar<simdOps::Multiply<T>>(beta);            
        }        
    }

    for(int row = 0; row < a->shapeOf()[0]; ++row)
        for(int col = 0; col < b->shapeOf()[1]; ++col)
            for(int j = 0; j < a->shapeOf()[1]; ++j)
                for(int i = 0; i < b->shapeOf()[0]; ++i)
                    (*dot)(row,col) += (*a)(row,j)*(*b)(i,col);

    if(alpha != (T)1.)
        dot->template applyScalar<simdOps::Multiply<T>>(alpha);

    if(beta != (T)0.) {
        c->template applyPairwiseTransform<simdOps::Add<T>>(dot, nullptr);
        delete dot;
    }
    
    return c;
}


    template class ND4J_EXPORT NDArrayFactory<float>;
    template class ND4J_EXPORT NDArrayFactory<float16>;
    template class ND4J_EXPORT NDArrayFactory<double>;
}


#endif