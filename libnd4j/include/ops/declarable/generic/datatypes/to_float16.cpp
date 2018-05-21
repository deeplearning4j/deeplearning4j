//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_to_float16)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(to_float16, 1, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            STORE_RESULT(output);

            return ND4J_STATUS_OK;
        }
    }
}

#endif