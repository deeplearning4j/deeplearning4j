//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_noop)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }
    }
}

#endif
