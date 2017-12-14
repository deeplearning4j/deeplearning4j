//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(less, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y), 0, "Both inputs should have equal shape");

            auto lambda = LAMBDA_TT(_x, _y) {
                return _x < _y ? (T) 1.0f : (T) 0.0f;
            };

            x->applyPairwiseLambda(y, lambda, z);

            STORE_RESULT(z);

            return ND4J_STATUS_OK;
        }
    }
}