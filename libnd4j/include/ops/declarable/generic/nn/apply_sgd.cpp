//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_apply_sgd)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(apply_sgd, 2, 1, true, -2, 0) {
            auto parameters = INPUT_VARIABLE(0);
            auto gradients = INPUT_VARIABLE(1);

            T lr = (T) 0.0f;

            REQUIRE_TRUE(parameters->isSameShape(gradients), 0, "ApplySGD: parameters and gradients should have the same shape, but got parameters = %s and gradients = %s !", ShapeUtils<T>::shapeAsString(parameters).c_str(), ShapeUtils<T>::shapeAsString(gradients).c_str());

            if (block.width() == 3) {
                auto tarr = INPUT_VARIABLE(2);
                lr = tarr->getScalar(0);
            } else if (block.getTArguments()->size() == 1) {
                lr = T_ARG(0);
            } else {
                REQUIRE_TRUE(false, 0, "ApplyGradients op should have LR announced either es T argument or additional NDArray!");
            }

            auto Z = OUTPUT_VARIABLE(0);

            auto lambda = LAMBDA_TT(_x, _y, lr) {
                return _x - (_y * lr);
            };

            parameters->applyPairwiseLambda(gradients, lambda, Z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ApplyGradientDescent, apply_sgd);
    }
}

#endif