//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_testreduction)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(testreduction, 1, 1, false, 0, -1) {
            auto z = OUTPUT_VARIABLE(0);

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
    }
}

#endif
