//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        ///////////////////////
        /**
         * uniform distribution
         * takes 1 ndarray
         *
         * T argumens map:
         * TArgs[0] - min for rng
         * TArgs[1] - max for rng
         */
        CONFIGURABLE_OP_IMPL(randomuniform, 1, 1, true, 2, 0) {
            // uniform distribution
            auto rng = block.getRNG();

            if (rng == nullptr)
                return ND4J_STATUS_BAD_RNG;

            if (block.getTArguments()->size() != 2)
                return ND4J_STATUS_BAD_ARGUMENTS;

            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = x;
            if (!block.isInplace())
                z = new NDArray<T>(x);

            functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
    }
}