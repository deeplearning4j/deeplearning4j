//
//
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matmul, 2, 1, false, -2, -2) {
            // FIXME: we might want to have gemv/dot fallback here
            REQUIRE_OK(this->validateInput2D(block));


            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            //x->printShapeInfo("x shape");
            //y->printShapeInfo("y shape");

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

            if (x->isMatrix() && y->isVector()) {
                // gemv
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);

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

                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transA == 112)
                    delete _x;

                if (transB == 112)
                    delete _y;
            } else if ((x->isMatrix() && y->isMatrix()) || (x->isColumnVector() && y->isRowVector())) {
                // gemm
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
            int *inA = inputShape->at(0);
            int *inB = inputShape->at(1);
            int *shape;
            ALLOCATE(shape, block.getWorkspace(), 2, int);

            int *tmpA, *tmpB;
            COPY_SHAPE(inA, tmpA);
            COPY_SHAPE(inB, tmpB);


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

            if (transA == 112)
                shape::transposeInplace(tmpA);

            if (transB == 112)
                shape::transposeInplace(tmpB);

            if (shape::isScalar(tmpA) && shape::isScalar(tmpB)) {
                // just scalar vs scalar
                shape[0] = 1;
                shape[1] = 1;
            }  else if (shape::isMatrix(tmpA) && shape::isVector(tmpB)) {
                // gemv case
                shape[0] = inA[1];
                shape[1] = inB[2];
            } else if ((shape::isMatrix(tmpA) && shape::isMatrix(tmpB)) || (shape::isVector(tmpA) && shape::isMatrix(tmpB)) || (shape::isColumnVector(tmpA) && shape::isVector(tmpB))) {
                // gemm case
                shape[0] = tmpA[1];
                shape[1] = tmpB[2];
            } else if ((shape::isVector(tmpA) && shape::isScalar(tmpB)) || (shape::isScalar(tmpA) && shape::isVector(tmpB))) {
                // element-wise
                shape[0] = 1;
                shape[1] = (int) nd4j::math::nd4j_max<Nd4jIndex>(shape::length(tmpA), shape::length(tmpB));
            } else if (shape::isRowVector(tmpA) && shape::isRowVector(tmpB)) {
                // dot case
                shape[0] = 1;
                shape[1] = 1;
            }

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBufferFortran(2, shape, newShape);

            RELEASE(shape, block.getWorkspace());

            RELEASE(tmpA, block.getWorkspace());
            RELEASE(tmpB, block.getWorkspace());
            return new ShapeList(newShape);
        }
    }
}