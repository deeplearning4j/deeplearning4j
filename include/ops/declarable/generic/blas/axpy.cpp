//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_axpy)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(axpy, 2, 1, false, -2, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y),0, "Axpy: both arguments should have the same shape")

            T a = (T) 1.0f;

            if (block.width() > 2) {
                auto alpha = INPUT_VARIABLE(2);
                REQUIRE_TRUE(alpha->isScalar(), 0, "Axpy: alpha argument should be scalar or TArg"); 
            } else if (block.getTArguments()->size() > 0) {
                a = T_ARG(0);
            }

            auto lambda = LAMBDA_TT(_y, _x, a) {
                return a * _x + _y;
            };

            y->applyPairwiseLambda(x, lambda, z);

            return ND4J_STATUS_OK;
        }
    }
}

#endif