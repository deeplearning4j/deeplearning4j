//
// Created by raver119 on 23.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_toggle_bits)

#include <ops/declarable/CustomOperations.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(toggle_bits, -1, -1, true) {

            for (int i = 0; i < block.width(); i++) {
                auto x = INPUT_VARIABLE(i);
                auto z = OUTPUT_VARIABLE(i);

             //   auto lambda = LAMBDA_T(_x) {
             //       return BitwiseUtils::flip_bits<T>(_x);
             //   };
                
             //   x->applyLambda(lambda, z);
                return ND4J_STATUS_OK;
            }
        }
    }
}

#endif