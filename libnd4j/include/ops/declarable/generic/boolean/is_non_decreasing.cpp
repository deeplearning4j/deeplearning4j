//
//  @author @cpuheater
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_is_non_decreasing)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/compare_elem.h>

namespace nd4j {
    namespace ops {
        BOOLEAN_OP_IMPL(is_non_decreasing, 1, true) {

            auto input = INPUT_VARIABLE(0);

            bool isNonDecreasing = true;

            nd4j::ops::helpers::compare_elem(input, false, isNonDecreasing);

            if (isNonDecreasing)
                return ND4J_STATUS_TRUE;
            else
                return ND4J_STATUS_FALSE;
        }
    }
}

#endif