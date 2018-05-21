//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_Log1p)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(Log1p, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            x->template applyTransform<simdOps::Log1p<T>>(z, nullptr);

            STORE_RESULT(z);
            
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(log1p, Log1p);
    }
}

#endif