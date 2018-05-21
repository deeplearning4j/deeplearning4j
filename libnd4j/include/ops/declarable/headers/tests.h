//
//  @author raver119@gmail.com
//
#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if NOT_EXCLUDED(OP_test_output_reshape)
        DECLARE_OP(test_output_reshape, 1, 1, true);
        #endif

        #if NOT_EXCLUDED(OP_test_scalar)
        DECLARE_CUSTOM_OP(test_scalar, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_testreduction)
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);
        #endif

        #if NOT_EXCLUDED(OP_noop)
        DECLARE_OP(noop, -1, -1, true);
        #endif

        #if NOT_EXCLUDED(OP_testop2i2o)
        DECLARE_OP(testop2i2o, 2, 2, true);
        #endif

        #if NOT_EXCLUDED(OP_testcustom)
        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        #endif
    }
}