//
// Created by raver119 on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(argmax, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            // FIXME: optional vector
            input->template applyIndexReduce<simdOps::IndexMax<T>>(output, *block.getIArguments());

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
    }
}
