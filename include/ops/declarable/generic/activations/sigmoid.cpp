//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sigmoid)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(sigmoid, 1, 1, true, 0, 0) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Sigmoid<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        CONFIGURABLE_OP_IMPL(sigmoid_bp, 2, 1, true, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsilon = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto lambda = LAMBDA_TT(_x, _e) {
                T s = nd4j::math::nd4j_sigmoid<T>(_x);
                return  _e * (s * ((T) 1.0f - s));
            };

            input->applyPairwiseLambda(epsilon, lambda, z);  

            return ND4J_STATUS_OK;
        }
    }
}

#endif