//
// @author raver119@gmail.com, created on 07.10.2017.
// @author GS <sgazeos@gmail.com>, modified
// @author Yurii Shyrma (iuriish@yahoo.com), fully rewritten
//

#include <op_boilerplate.h> 
#if NOT_EXCLUDED(OP_matmul)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(matmul, 2, 1, false, 0, -2) {
    
    NDArray<T> *x = INPUT_VARIABLE(0);
    NDArray<T> *y = INPUT_VARIABLE(1);
    NDArray<T> *z = OUTPUT_VARIABLE(0);    

    const int iSize  = (int) block.getIArguments()->size();   
          int transX = iSize > 0 ? INT_ARG(0) : 0;
          int transY = iSize > 1 ? INT_ARG(1) : 0;
    const int transZ = iSize > 2 ? INT_ARG(2) : 0;

    const int xRank = x->rankOf();
    const int yRank = y->rankOf();
    const int zRank = z->rankOf();

    if(transZ) {
        x = INPUT_VARIABLE(1);
        y = INPUT_VARIABLE(0);
        bool temp = transX;
        transX = !transY;
        transY = !temp;
    }

    const int xLastDim       = transX ? -2 : -1; 
    const int yLastDim       = transY ? -2 : -1;
    const int xLastButOneDim = transX ? -1 : -2; 
    const int yLastButOneDim = transY ? -1 : -2;

    // ******* input validation ******* //
    REQUIRE_TRUE(xRank > 0 && yRank > 0, 0, "MATMUL OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank = %i, y rank = %i !", xRank, yRank);
    
    if(xRank == 1 && yRank == 1) {  // dot case, output is scalar (or vector with length = 1)
        REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0, "MATMUL OP: since input arrays are vectors they must have the same length, but got x length = %i, y length = %i !", x->lengthOf(), y->lengthOf()); 
    }
    else if(xRank == 1 && yRank == 2) {  // vector x matrix, i.e. [4] x [4,5] = [5], output is vector
        REQUIRE_TRUE(x->lengthOf() == y->sizeAt(yLastButOneDim), 0, "MATMUL OP: input arrays have inconsistent shapes for vector-matrix product: x %s, y %s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(y).c_str());
    }
    else if(xRank == 2 && yRank == 1) {   // matrix x vector , i.e. [4,5] x [5] = [4], output is vector
        REQUIRE_TRUE(x->sizeAt(xLastDim) == y->lengthOf(), 0, "MATMUL OP: input arrays have inconsistent shapes for matrix-vector product: x %s, y %s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(y).c_str());
    }
    else {               
        REQUIRE_TRUE(xRank == yRank && yRank == zRank, 0, "MATMUL OP: input and output arrays must have the same rank, but got instead: x rank = %i, y rank = %i, z rank = %i !", xRank, yRank, zRank);
        REQUIRE_TRUE(x->sizeAt(xLastDim) == y->sizeAt(yLastButOneDim) && x->sizeAt(xLastButOneDim) == z->sizeAt(-2) && y->sizeAt(yLastDim) == z->sizeAt(-1), 0, "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(y).c_str(), ShapeUtils<T>::shapeAsString(z).c_str());
    
        if(xRank > 2)   // outer dims must be the same
            for(int i = 0; i < xRank-2; ++i)
                REQUIRE_TRUE(x->sizeAt(i) == y->sizeAt(i) && y->sizeAt(i) == z->sizeAt(i), 0, "MATMUL OP: input/output arrays have inconsistent shapes for matrix product: x %s, y %s, z %s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(y).c_str(), ShapeUtils<T>::shapeAsString(z).c_str());
    }
    // ******* end of input validation ******* //
    
    NDArrayFactory<T>::matmul(x, y, z, transX, transY);

    return Status::OK();
}
DECLARE_SYN(mMul, matmul);
DECLARE_SYN(mmul, matmul);
DECLARE_SYN(gemm, matmul);
DECLARE_SYN(gemv, matmul);
DECLARE_SYN(dot, matmul);


DECLARE_SHAPE_FN(matmul) {
    
    Nd4jLong* xShapeInfo = inputShape->at(0);
    Nd4jLong* yShapeInfo = inputShape->at(1);    

    const int iSize  = (int) block.getIArguments()->size();   
          int transX = iSize > 0 ? INT_ARG(0) : 0;
          int transY = iSize > 1 ? INT_ARG(1) : 0;
    const int transZ = iSize > 2 ? INT_ARG(2) : 0;

    REQUIRE_TRUE(xShapeInfo[0] > 0 && yShapeInfo[0] > 0, 0, "MATMUL OP: input arrays must have rank bigger than 0 (should not be scalars), but got instead: x rank = %i, y rank = %i !", xShapeInfo[0], yShapeInfo[0]);

    if(transZ) {
        xShapeInfo = inputShape->at(1);
        yShapeInfo = inputShape->at(0);
        bool temp = transX;
        transX = !transY;
        transY = !temp;
    }

    std::vector<Nd4jLong> zShapeOnly = ShapeUtils<T>::evalShapeForMatmul(xShapeInfo, yShapeInfo, transX, transY);

    return SHAPELIST( ShapeUtils<T>::createShapeInfo('f', zShapeOnly, block.getWorkspace()) );    
}

}
}

#endif


   /*CUSTOM_OP_IMPL(matmul, 2, 1, false, -2, -2) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() <= 2 && y->rankOf() <= 2 && z->rankOf() <= 2, 0, "MatMul: Input and Output NDArrays should have rank less or equal to 2");

            int iSize = (int) block.getIArguments()->size();
            int transX = 0;
            int transY = 0;

            if (iSize > 0)
                transX = INT_ARG(0);

            if (iSize > 1)
                transY = INT_ARG(1);

            T alpha = (T) 1.0f;
            T beta = (T) 0.0f;
            if (block.getTArguments()->size() > 0)
                alpha = block.getTArguments()->at(0);

            if (block.getTArguments()->size() > 1)
                beta = block.getTArguments()->at(1);


            if (transX == 0)
                transX = 111;

            if (transY == 0)
                transY = 111;

            if (transX == 1)
                transX = 112;

            if (transY == 1)
                transY = 112;

            REQUIRE_TRUE((transX == 111 || transX == 112) && (transY == 111 || transY == 112), 0, "BatchedGemm: valid values for transX and transY are: 0/1 or 111/112, for NoTrans/Trans respectively")
            if (x->rankOf() == 1 && y->isMatrix()) {
                NDArray<T> *_x = x->reshape(x->ordering(), {1, (int) x->lengthOf()});
                NDArray<T> *_y = transY == 111 ? y : y->transpose();
                //NDArray<T> *_z = z->reshape(z->ordering(), {1, (int) z->lengthOf()});
        
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                delete _x;
                //delete _z;

                if (transY == 112)
                    delete _y;
            } else if (x->isMatrix() && y->isVector()) {
                NDArray<T> *_x = transX == 111 ? x : x->transpose();
                NDArray<T> *_y = transY == 111 ? y : y->transpose();
                // gemv
                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transX == 112)
                    delete _x;

                if (transY == 112)
                    delete _y;
            } else if (x->isVector() && y->isMatrix() && iSize > 0) {
                // gemm
                NDArray<T> *_x = transX == 111 ? x : x->transpose();
                NDArray<T> *_y = transY == 111 ? y : y->transpose();

                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transX == 112)
                    delete _x;

                if (transY == 112)
                    delete _y;
            } else if (x->isVector() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if ((x->isMatrix() && y->isMatrix() || (x->isColumnVector() || (x->isRowVector() && transX == 112)) && (y->isRowVector() || (y->isColumnVector() && transY == 112))) && iSize > 0) {
                // gemm
                NDArray<T> *_x = transX == 111 ? x : x->transpose();
                NDArray<T> *_y = transY == 111 ? y : y->transpose();

                REQUIRE_TRUE(_x->rankOf() == 2 && _y->rankOf() == 2, 0, "MatMul: both operands should have rank 2");
                REQUIRE_TRUE(_x->columns() == _y->rows(), 0, "MatMul: number of A.colums() should be equal to number of B.rows()");

                nd4j::NDArrayFactory<T>::mmulHelper(_x, _y, z, alpha, beta);

                if (transX == 112)
                    delete _x;

                if (transY == 112)
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
            int transX = 0;
            int transY = 0;

            if (iSize > 0)
                transX = INT_ARG(0);

            if (iSize > 1)
                transY = INT_ARG(1);

            if (transX == 0)
                transX = 111;

            if (transY == 0)
                transY = 111;

            if (transX == 1)
                transX = 112;

            if (transY == 1)
                transY = 112;

            auto outputShape = ShapeUtils<T>::matrixProductShape(inputShape->at(0), inputShape->at(1), transX == 112, transY == 112, block.getWorkspace());

            return SHAPELIST(outputShape);
        }*/