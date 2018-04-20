//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation toggles individual bits of each element in array
         * 
         * PLEASE NOTE: This operation is possible only on integer datatypes
         * 
         * @tparam T
         */
        DECLARE_OP(toggle_bits, -1, -1, true);
    }
}