//
// Created by GS <sgazeos@gmail.com> at 2/27/2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_matrix_inverse)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lup.h>
namespace nd4j {
    namespace ops {
        OP_IMPL(matrix_inverse, 1, 1, true) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() >=2, 0, "matrix_inverse: The rank of input array should not less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE(input->sizeAt(-1) == input->sizeAt(-2), 0, "matrix_inverse: The last two dimmensions should be equal, but %i and %i are given", input->sizeAt(-1), input->sizeAt(-2));

            return helpers::inverse(input, output);
        }
    }
}

#endif