//
//  @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(realdiv, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            x->template applyPairwiseTransform<simdOps::Divide<T>>(y, z, nullptr);

            STORE_RESULT(z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RealDiv, realdiv);
    }
}