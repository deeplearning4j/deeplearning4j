//
// Created by raver119 on 12.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_zeros_as)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(zeros_as, 1, 1, false) {
            auto input = INPUT_VARIABLE(0);

            auto out = OUTPUT_VARIABLE(0);

            STORE_RESULT(*out);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(zeroslike, zeros_as);
        DECLARE_SYN(zeros_like, zeros_as);
    }
}

#endif