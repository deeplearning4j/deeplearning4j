//
// Created by raver119 on 29/10/17.
//


#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // here iArgs is int vector of ordered set of dimensions to be permuted
        CONFIGURABLE_OP_IMPL(permute, 1, 1, true, 0, -1) {
            std::vector<int>* argumets = block.getIArguments();
            NDArray<T> *x = INPUT_VARIABLE(0);

            if(block.isInplace()) {		// in-place
                x->permutei(*argumets);
                STORE_RESULT(*x);
            }
            else {						// not-in-place
                NDArray<T>* ret = x->permute(*argumets);
                STORE_RESULT(*ret);
            }
            return ND4J_STATUS_OK;
        }
    }
}

