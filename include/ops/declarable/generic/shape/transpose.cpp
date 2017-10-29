//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        OP_IMPL(transpose, 1, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);

            if(block.isInplace()) {
                x->transposei();
                STORE_RESULT(*x);
            }
            else {
                NDArray<T>* ret = x->transpose();
                STORE_RESULT(*ret);
            }
            return ND4J_STATUS_OK;
        }
    }
}