//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        // recurrent ops
        DECLARE_CUSTOM_OP(sru,         5, 2, false, 0, 0);


        DECLARE_CUSTOM_OP(sru_logic,   5, 2, false, 0, 0);


        DECLARE_CUSTOM_OP(sru_bi,      5, 2, true,  0, 0);


        DECLARE_CUSTOM_OP(sru_bp,      8, 4, true,  0, 0);


        DECLARE_CUSTOM_OP(sru_bp_logic,8, 4, true,  0, 0);


        DECLARE_CUSTOM_OP(sru_bi_bp,   8, 4, true,  0, 0);


        DECLARE_CUSTOM_OP(lstmCell, 8, 2, false, 3, 2);


        DECLARE_CUSTOM_OP(sruCell, 4, 2, false, 0, 0);


        DECLARE_CUSTOM_OP(gruCell, 5, 1, false, 0, 0);
    }
}