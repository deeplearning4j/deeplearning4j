//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(floormod, 2, 1, true) {
            auto first = INPUT_VARIABLE(0);
            auto second = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            first->template applyPairwiseTransform<simdOps::FloorMod<T>>(second, z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
    }
}