//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(not_equals, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y), 0, "Both inputs should have equal shape");

#pragma omp parallel for simd
            for (int e = 0; e < z->lengthOf(); e++) {
                T v = x->getIndexedScalar(e) != y->getIndexedScalar(e) ? (T) 1.0f : (T) 0.0f;
                z->putIndexedScalar(e, v);
            }

            STORE_RESULT(z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(not_equal, not_equals);
    }
}