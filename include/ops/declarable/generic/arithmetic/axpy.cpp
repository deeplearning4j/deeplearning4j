//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(axpy, 2, 1, false, -2, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            T a = (T) 1.0f;

            auto lambda = LAMBDA_TT(_y, _x, a) {
                return a * _x + _y;
            };

            y->applyPairwiseLambda(x, lambda, z);

            return ND4J_STATUS_OK;
        }
    }
}