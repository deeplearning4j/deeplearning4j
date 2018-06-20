//
// Created by raver119 on 01.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_ones_as)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(ones_as, 1, 1, false) {
            auto output = OUTPUT_VARIABLE(0);

            *output = static_cast<T>(1.f);

            return ND4J_STATUS_OK;
        }
    }
}

#endif