//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_tanh)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
        namespace ops {
        CONFIGURABLE_OP_IMPL(tanh, 1, 1, true, 0, 0) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Tanh<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        CONFIGURABLE_OP_IMPL(tanh_bp, 2, 1, true, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsilon = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto lambda = LAMBDA_TT(_x, _e) {
                T t = nd4j::math::nd4j_tanh<T>(_x);
                return _e * ((T) 1.0f - (t * t));
            };

            input->applyPairwiseLambda(epsilon, lambda, z);  

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TanhGrad, tanh_bp);
    }
}

#endif