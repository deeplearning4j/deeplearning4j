//
// Created by Yurii Shyrma on 07.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(matrix_set_diag, 2, 1, false, 0, 0) {
            NDArray<T>* input    = INPUT_VARIABLE(0);
            NDArray<T>* diagonal = INPUT_VARIABLE(1);

            NDArray<T>* output   = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(diagonal->rankOf() == input->rankOf()-1, 0, "matrix_set_diag: rank of diagonal array must be smaller by one compared to rank of input array !");

            for(int i = 0;  i < diagonal->rankOf() - 1; ++i)        
                REQUIRE_TRUE(diagonal->sizeAt(i) == input->sizeAt(i), 0, "matrix_set_diag: the shapes of diagonal and input arrays must be equal till last diagonal dimension but one !");

            REQUIRE_TRUE(diagonal->sizeAt(-1) == (int)nd4j::math::nd4j_min<Nd4jIndex>(input->sizeAt(-1), input->sizeAt(-2)), 
                0, "matrix_set_diag: the shape of diagonal at last dimension must be equal to min(input_last_shape, input_last_but_one_shape) !");

            helpers::matrixSetDiag(input, diagonal, output);
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MatrixSetDiag, matrix_set_diag);
    }
}