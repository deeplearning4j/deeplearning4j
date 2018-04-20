//
//  @author raver119@gmail.com
//
#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_OP(test_output_reshape, 1, 1, true);
        DECLARE_CUSTOM_OP(test_scalar, 1, 1, false, 0, 0);
    }
}