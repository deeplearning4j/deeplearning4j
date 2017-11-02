//
// Created by raver119 on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(argmin, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            input->template applyIndexReduce<simdOps::IndexMin<T>>(output, *block.getIArguments());

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
    }
}
