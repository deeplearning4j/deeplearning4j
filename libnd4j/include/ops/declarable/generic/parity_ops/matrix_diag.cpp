//
// Created to use with batched tensor by GS <sgazeos@gmail.com> 3/21/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matrix_diag.h>


namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(matrix_diag, 1, 1, false, 0, 0) {
            NDArray<T>* input  = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(!input->isScalar(), 0, "CUSTOM_OP matrix_diag: input array must be at list a vector, but scalar was given!");
            return helpers::matrixDiag(input, output);
        }

        DECLARE_SHAPE_FN(matrix_diag) {

            Nd4jLong* outShapeInfo = nullptr;
            auto in = inputShape->at(0);
            int inRank = shape::rank(in);

            int outRank = inRank + 1;
            auto lastDimension = shape::sizeAt(in, -1);

            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
            outShapeInfo[0] = outRank;
            for(int i = 0; i < inRank; ++i)
                outShapeInfo[i + 1] = shape::sizeAt(in, i);
            outShapeInfo[outRank] = lastDimension;

            shape::updateStrides(outShapeInfo, shape::order(in));

            return SHAPELIST(outShapeInfo);
        }
}
}

