//
// Created by raver119 on 24.11.17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /////////////////////////////////////////
        OP_IMPL(assign, 2, 1, false) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            if (y->isScalar()) {

                z->assign(y->getScalar(0));
            } else {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                REQUIRE_OK(this->validateInputDimensionsMatch(block));

                z->assign(y);
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);
        DECLARE_SYN(copy, assign);

    }
}