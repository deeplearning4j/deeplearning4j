//
// Created by raver119 on 23.11.17.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation is possible only on integer datatypes
         * @tparam T
         */
        OP_IMPL(toggle_bits, -1, -1, true) {

            for (int i = 0; i < block.width(); i++) {
                auto x = INPUT_VARIABLE(i);
                auto z = OUTPUT_VARIABLE(i);

             //   auto lambda = LAMBDA_T(_x) {
             //       return BitwiseUtils::flip_bits<T>(_x);
             //   };

             //   x->applyLambda(lambda, z);
            }
        }
    }
}