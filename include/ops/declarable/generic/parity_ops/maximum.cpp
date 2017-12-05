//
// Created by raver119 on 12.10.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(maximum, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y),0, "Maximum: operands should have same shape");

            x->template applyPairwiseTransform<simdOps::Max<T>>(y, z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
    }
}