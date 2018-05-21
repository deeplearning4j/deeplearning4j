//
//  @author @cpuheater
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_is_numeric_tensor)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/compare_elem.h>

namespace nd4j {
    namespace ops {
        BOOLEAN_OP_IMPL(is_numeric_tensor, 1, true) {

            auto input = INPUT_VARIABLE(0);

            return ND4J_STATUS_TRUE;
        }
    }
}

#endif