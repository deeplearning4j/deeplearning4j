//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>

#if NOT_EXCLUDED(test_output_reshape)
#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(test_output_reshape, 1, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            output->reshapei({-1});

            return Status::OK();
        }
    }
}

#endif
