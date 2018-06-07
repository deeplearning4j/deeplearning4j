//
// Created by raver on 6/6/2018.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_neq_scalar)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        BROADCASTABLE_OP_IMPL(boolean_or, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            auto tZ = BroadcastHelper<T>::template broadcast_apply<simdOps::Or<T>>(x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z)
                throw std::runtime_error("boolean_and: result was overwritten");

            return Status::OK();
        }
    }
}

#endif