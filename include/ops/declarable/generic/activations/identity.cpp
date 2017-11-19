//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        OP_IMPL(identity, 1, 1, true) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            // just for lulz
            first->template applyTransform<simdOps::Identity<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(linear, identity);
    }
}