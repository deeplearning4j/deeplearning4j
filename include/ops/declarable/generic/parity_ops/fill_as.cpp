//
// Created by raver119 on 01.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_fill_as)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(fill_as, 1, 1, true, 0, 0) {
            auto output = OUTPUT_VARIABLE(0);
            T scalr = 0;

            if (block.width() > 1) {
                auto s = INPUT_VARIABLE(1);
                scalr = s->getScalar(0);
            } else if (block.getTArguments()->size() > 0) {
                scalr = T_ARG(0);
            };

            output->assign(scalr);

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(filllike, fill_as);
        DECLARE_SYN(fill_like, fill_as);
    }
}

#endif