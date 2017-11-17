//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        OP_IMPL(biasadd, 2, 1, true) {
            //REQUIRE_OK(this->validateInput2D(block));

            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *bias = INPUT_VARIABLE(1);

            REQUIRE_TRUE(bias->isRowVector(), 0, "Bias array should be a vector");

            NDArray<T> *z = OUTPUT_VARIABLE(0);

            if (input->isMatrix())
                input->addRowVector(bias, z);
            else {
                std::vector<int> shape({-1, (int) bias->lengthOf()});
                //nd4j_debug("Reshaping to: [%i, %i]\n", -1, (int) bias->lengthOf());
                auto tArr = input->reshape(input->ordering(), shape);
                auto zArr = z->reshape(z->ordering(), shape);
                tArr->addRowVector(bias, zArr);

                delete tArr;
                delete zArr;
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(bias_add, biasadd);
    }
}