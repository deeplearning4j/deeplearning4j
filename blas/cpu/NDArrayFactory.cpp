//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_NDARRAYFACTORY_CPP
#define LIBND4J_NDARRAYFACTORY_CPP

#include "../NDArrayFactory.h"
#include "../NDArray.h"
#include <memory/Workspace.h>
#include <ops/gemm.h>
#include "NDArray.cpp"

namespace nd4j {

    template<typename T>
    ArrayList<T>* NDArrayFactory::allExamples(NDArray<T>* ndArray) {
        std::vector<int> dimensions;
        for (int e = 1; e < ndArray->rankOf(); e++)
            dimensions.push_back(e);

        return allTensorsAlongDimension(ndArray, dimensions);
    }

    template<typename T>
    ArrayList<T>* NDArrayFactory::multipleTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &indices, std::vector<int> &dimensions) {
        auto result = new ArrayList<T>();

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
    ArrayList<T>* NDArrayFactory::allTensorsAlongDimension(NDArray<T>* ndArray, std::initializer_list<int> dimensions) {
        std::vector<int> vec(dimensions);
        return allTensorsAlongDimension<T>(ndArray, vec);
    }

    template<typename T>
    ArrayList<T>* NDArrayFactory::allTensorsAlongDimension(NDArray<T>* ndArray, std::vector<int> &dimensions) {
        auto result = new ArrayList<T>();

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
    nd4j::NDArray<T>* nd4j::NDArrayFactory::tensorDot(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C, std::initializer_list<int> axesA, std::initializer_list<int> axesB) {
        std::vector<int> aA(axesA);
        std::vector<int> aB(axesB);
        return tensorDot(A, B, C, aA, aB);
    }

    template<typename T>
    nd4j::NDArray<T>* nd4j::NDArrayFactory::tensorDot(nd4j::NDArray<T>* a, nd4j::NDArray<T>* b, nd4j::NDArray<T>* C, std::vector<int>& axes_0, std::vector<int>& axes_1) {

        int axe0_size = (int) axes_0.size();
        int axe1_size = (int) axes_1.size();
        // validating axes
        int validationLength = nd4j::math::nd4j_min<int>(axe0_size, axe1_size);
        for (int i = 0; i < validationLength; i++) {
            if (a->sizeAt(axes_0[i]) != b->sizeAt(axes_1[i]))
                throw "Size of the given axes at each dimension must be the same size.";
            if (axes_0[i] < 0)
                axes_0[i] += a->rankOf();
            if (axes_1[i] < 0)
                axes_1[i] += b->rankOf();
        }


        std::vector<int> list_A, list_B;
        for (int i = 0; i < a->rankOf(); i++)
            if (std::find(axes_0.begin(), axes_0.end(), i) == axes_0.end())
                list_A.push_back(i);

        for (int i = 0; i < b->rankOf(); i++)
            if (std::find(axes_1.begin(), axes_1.end(), i) == axes_1.end())
                list_B.push_back(i);


        std::vector<int> newAxesA(list_A);
        std::vector<int> newAxesB;
        for (auto v: axes_0)
            newAxesA.push_back(v);

        for (auto v: axes_1)
            newAxesB.push_back(v);

        for (auto v: list_B)
            newAxesB.push_back(v);

        int n2 = 1;
        int aLength = nd4j::math::nd4j_min<int>(a->rankOf(), axes_0.size());
        for (int i = 0; i < aLength; i++)
            n2 *= a->sizeAt(axes_0[i]);

        std::vector<int> newShapeA({-1, n2});
        std::vector<int> oldShapeA;
        if (list_A.size() == 0) {
            oldShapeA.push_back(1);
        } else {
            for (auto v: list_A)
                oldShapeA.push_back(v);

            for (int i = 0; i < (int) oldShapeA.size(); i++)
                oldShapeA[i] = a->sizeAt(oldShapeA[i]);
        }

        int n3 = 1;
        int bNax = nd4j::math::nd4j_min<int>(b->rankOf(), axes_1.size());
        for (int i = 0; i < bNax; i++)
            n3 *= b->sizeAt(axes_1[i]);

        std::vector<int> newShapeB({n3, -1});
        std::vector<int> oldShapeB;
        if (list_B.size() == 0) {
            oldShapeB.push_back(1);
        } else {
            for (auto v: list_B)
                oldShapeB.push_back(v);
            for (int i = 0; i < (int) oldShapeB.size(); i++)
                oldShapeB[i] = b->sizeAt(oldShapeB[i]);
        }

        //d4j::Logger::printv("newAxesA: ", newAxesA);
        //nd4j::Logger::printv("newAxesB: ", newAxesB);
        auto aT = a->permute(newAxesA);
        auto bT = b->permute(newAxesB);

        //aT->printShapeInfo("at pshape");
        //bT->printShapeInfo("bt pshape");

        //aT->permutei(newAxesA);
        aT->reshapei('c', newShapeA);
        //aT->printShapeInfo("at rshape");

        //bT->permutei(newAxesB);
        bT->reshapei('c', newShapeB);
        //bT->printShapeInfo("bt rshape");

        auto c = nd4j::NDArrayFactory::mmulHelper<T>(aT, bT, nullptr, 1.0, 0.0);

        std::vector<int> aPlusB(oldShapeA);
        for (auto v: oldShapeB)
            aPlusB.push_back(v);

        c->reshapei('c', aPlusB);

        if (aT != a)
            delete aT;

        if (bT != b)
            delete bT;

        return c;
    }


    //////////////////////////////////////////////////////////////////////////
    template<typename T>
    nd4j::NDArray<T>* NDArrayFactory::mmulHelper(nd4j::NDArray<T>* A, nd4j::NDArray<T>* B, nd4j::NDArray<T>* C , T alpha, T beta) {
        nd4j::NDArray<T>* result = C;

        if (A->rankOf() > 2 || B->rankOf() > 2) {
            // matmul
            if (A->rankOf() != B->rankOf()) {
                nd4j_printf("Ranks of A and B should match, but got %i/%i instead\n",A->rankOf(), B->rankOf());
                throw "Ranks of A and B should match";
            }

            int dims = A->rankOf();

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
                nd4j_printf("Number of A \"columns\" should match number of B \"rows\", but got %i/%i instead", A->sizeAt(-1), B->sizeAt(-2))
                throw "Numbers of rows/columns should match";
            }

            newShape.push_back(pRows);
            newShape.push_back(pCols);

            Nd4jIndex prod = shape::prodLong(newShape.data(), newShape.size());

            if (result == nullptr)
                result = new NDArray<T>('c', newShape);
            else
                if (!result->isSameShape(newShape)) {
                    nd4j_printf("Bad result shape for MatMul\n","");
                    throw "Bad result shape";
                }

            auto aL = allTensorsAlongDimension(A, {A->rankOf() - 2, A->rankOf() - 1});
            auto bL = allTensorsAlongDimension(B, {B->rankOf() - 2, B->rankOf() - 1});
            auto cL = allTensorsAlongDimension(result, {result->rankOf() - 2, result->rankOf() - 1});

            nd4j_debug("NumTads: %i\n", aL->size());
            for (int e = 0; e < aL->size(); e++) {
                auto c_ = mmulHelper(aL->at(e), bL->at(e));

                cL->at(e)->assign(c_);
                delete c_;
            }

            delete aL;
            delete bL;
            delete cL;
        } else if (A->isVector() && B->isVector()) {
            // dot
            if (A->lengthOf() != B->lengthOf())
                throw "A length != B length";

            if (result == nullptr)
                result = new NDArray<T>(1,1, 'c');

            result->putScalar(0, nd4j::math::nd4j_dot(A->getBuffer(), B->getBuffer(), A->lengthOf()));
        } if (A->isMatrix() && B->isVector()) {
            // gemv
            if (A->columns() != B->rows())
                throw "A columns != B length";

            if (result == nullptr)
                result = new NDArray<T>(A->rows(), 1, 'f');

            // TODO: strides!!!
            nd4j::blas::GEMV<T>::op(A->ordering() == 'f' ? CblasTrans : 0,  A->rows(), A->columns(), alpha, A->getBuffer(), B->rows(), B->getBuffer(), 1, beta, C->getBuffer(), 1);
        } else if ((A->isMatrix() && B->isMatrix()) || (A->isVector() && B->isMatrix())) {
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
            nd4j::blas::GEMM<T>::op(rOrder, transA, transB, M, N, K, alpha, pA->getBuffer(), lda, pB->getBuffer(), ldb, beta, pC->getBuffer(), ldc);

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
    NDArray<T>* NDArrayFactory::tile(NDArray<T> *original, std::vector<int> &dimensions) {
        return nullptr;
    }


    template<typename T>
    NDArray<T>* NDArrayFactory::repeat(NDArray<T> *original, std::vector<int> &repeats) {
        return nullptr;
    }

    template<typename T>
    NDArray<T>* NDArrayFactory::linspace(T from, T to, Nd4jIndex numElements) {
        auto result = new NDArray<T>(numElements, 'c');

        for (Nd4jIndex e = 0; e < numElements; e++) {
            T step = (T) e / ((T) numElements - (T) 1.0f);
            result->getBuffer()[e] = (from * ((T) 1.0f - step) + step * to);
        }

        return result;
    }

    template<typename T>
    void NDArrayFactory::linspace(T from, NDArray<T>& arr) {
        
        int size = arr.lengthOf();
        for (Nd4jIndex i = 0; i < size; ++i)
            arr(i) = from++;
    }

    template<typename T>
    NDArray<T>* NDArrayFactory::createUninitialized(NDArray<T>* other) {
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
}

#endif