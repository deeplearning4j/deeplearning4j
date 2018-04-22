//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lrelu)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(lrelu, 1, 1, true, 1, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            input->template applyTransform<simdOps::LeakyRELU<T>>(output, block.getTArguments()->data());
            STORE_RESULT(output);
            
            return ND4J_STATUS_OK;
        }
        
        CONFIGURABLE_OP_IMPL(lrelu_bp, 2, 1, true, 0, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsilon = INPUT_VARIABLE(1);

            auto z = OUTPUT_VARIABLE(0);

            auto lambda = LAMBDA_TT(_x, _e) {
                return _x >= (T) 0.0f ? _e : (T) 0.0f;
            };

            input->applyPairwiseLambda(epsilon, lambda, z);  

            return ND4J_STATUS_OK;
        }
    }
}

#endif