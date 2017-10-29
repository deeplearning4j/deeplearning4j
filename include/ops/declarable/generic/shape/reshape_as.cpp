//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        OP_IMPL(reshapeas, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);

            NDArray<T>* z = this->getZ(block);
            std::vector<int> shapeNew(y->shapeOf(), y->shapeOf() + y->rankOf());
            char order = y->ordering();

            if (x->reshapei(order, shapeNew)) {
                *z = *x;
                STORE_RESULT(*z);
                return ND4J_STATUS_OK;
            }

            return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SYN(shape, reshapeas);
    }
}