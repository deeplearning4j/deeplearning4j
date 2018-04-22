//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_testop2i2o)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // test op, non-divergent
        OP_IMPL(testop2i2o, 2, 2, true) {
            //nd4j_printf("CPU op used!\n","");
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto xO = OUTPUT_VARIABLE(0);
            auto yO = OUTPUT_VARIABLE(1);

            x->template applyScalar<simdOps::Add<T>>(1.0, xO, nullptr);
            y->template applyScalar<simdOps::Add<T>>(2.0, yO, nullptr);

            STORE_2_RESULTS(*xO, *yO);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);
    }
}

#endif
