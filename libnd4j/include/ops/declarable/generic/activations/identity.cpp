//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_identity)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(identity, 1, 1, true) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            // just for lulz
            first->template applyTransform<simdOps::Identity<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(linear, identity);


        OP_IMPL(identity_bp, 2, 1, true) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            z->assign(epsilon);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(LinearGrad, identity_bp);
    }
}

#endif