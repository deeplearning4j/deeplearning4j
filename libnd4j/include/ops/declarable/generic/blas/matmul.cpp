//
// Created by raver119 on 07.10.2017.
// Modified by GS <sgazeos@gmail.com> 01.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_matmul)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matmul, 2, 1, false, -2, -2) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() <= 2 && y->rankOf() <= 2 && z->rankOf() <= 2, 0, "MatMul: Input and Output NDArrays should have rank less or equal to 2");

            int iSize = (int) block.getIArguments()->size();
            int transA = 0;
            int transB = 0;

            if (iSize > 0)
                transA = INT_ARG(0);

            if (iSize > 1)
                transB = INT_ARG(1);

            T alpha = (T) 1.0f;
            T beta = (T) 0.0f;
            if (block.getTArguments()->size() > 0)
                alpha = block.getTArguments()->at(0);

            if (block.getTArguments()->size() > 1)
                beta = block.getTArguments()->at(1);


            if (transA == 0)
                transA = 111;

            if (transB == 0)
                transB = 111;

            if (transA == 1)
                transA = 112;

            if (transB == 1)
                transB = 112;

            REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0, "BatchedGemm: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
            if (x->rankOf() == 1 && y->isMatrix()) {
                NDArray<T> *_x = x->reshape(x->ordering(), {1, (int) x->lengthOf()});
                NDArray<T> *_y = transB == 111 ? y : y->transpose();
                //NDArray<T> *_z = z->reshape(z->ordering(), {1, (int) z->lengthOf()});
        
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                delete _x;
                //delete _z;

                if (transB == 112)
                    delete _y;
            } else if (x->isMatrix() && y->isVector()) {
                NDArray<T> *_x = transA == 111 ? x : x->transpose();
                NDArray<T> *_y = transB == 111 ? y : y->transpose();
                // gemv
                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transA == 112)
                    delete _x;

                if (transB == 112)
                    delete _y;
            } else if (x->isVector() && y->isMatrix() && iSize > 0) {
                // gemm
                NDArray<T> *_x = transA == 111 ? x : x->transpose();
                NDArray<T> *_y = transB == 111 ? y : y->transpose();

                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transA == 112)
                    delete _x;

                if (transB == 112)
                    delete _y;
            } else if (x->isVector() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if ((x->isMatrix() && y->isMatrix() || (x->isColumnVector() || (x->isRowVector() && transA == 112)) && (y->isRowVector() || (y->isColumnVector() && transB == 112))) && iSize > 0) {
                // gemm
                NDArray<T> *_x = transA == 111 ? x : x->transpose();
                NDArray<T> *_y = transB == 111 ? y : y->transpose();

                REQUIRE_TRUE(_x->rankOf() == 2 && _y->rankOf() == 2, 0, "MatMul: both operands should have rank 2");
                REQUIRE_TRUE(_x->columns() == _y->rows(), 0, "MatMul: number of A.colums() should be equal to number of B.rows()");

                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transA == 112)
                    delete _x;

                if (transB == 112)
                    delete _y;
            } else if ((x->isMatrix() && y->isMatrix()) || (x->isColumnVector() && y->isRowVector())) {
                // gemm

                REQUIRE_TRUE(x->rankOf() == 2 && y->rankOf() == 2, 0, "MatMul: both operands should have rank 2");
                REQUIRE_TRUE(x->columns() == y->rows(), 0, "MatMul: number of A.colums() should be equal to number of B.rows()");

                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if (x->isVector() && y->isVector()) {
                // dot
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if (x->isVector() && y->isScalar()) {
                // elementwise mul

                x->template applyScalar<simdOps::Multiply<T>>(y->getScalar(0), z, nullptr);
             } else if (x->isScalar() && y->isVector()) {
                // elementwise mul, reverse op

                y->template applyScalar<simdOps::Multiply<T>>(x->getScalar(0), z, nullptr);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mMul, matmul);
        DECLARE_SYN(mmul, matmul);
        DECLARE_SYN(gemm, matmul);
        DECLARE_SYN(gemv, matmul);
        DECLARE_SYN(dot, matmul);

        DECLARE_SHAPE_FN(matmul) {

            int iSize = (int) block.getIArguments()->size();
            int transA = 0;
            int transB = 0;

            if (iSize > 0)
                transA = INT_ARG(0);

            if (iSize > 1)
                transB = INT_ARG(1);

            if (transA == 0)
                transA = 111;

            if (transB == 0)
                transB = 111;

            if (transA == 1)
                transA = 112;

            if (transB == 1)
                transB = 112;

            auto outputShape = ShapeUtils<T>::matrixProductShape(inputShape->at(0), inputShape->at(1), transA == 112, transB == 112, block.getWorkspace());

            return SHAPELIST(outputShape);
        }
    }
}

#endif