//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0);
        DECLARE_CONFIGURABLE_OP(clipbynorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(clipbyavgnorm, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(cumsum, 1, 1, true, 0, -2);
        DECLARE_CONFIGURABLE_OP(cumprod, 1, 1, true, 0, -2);
        DECLARE_CUSTOM_OP(tile, 1, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(repeat, 1, 1, true, 0, -1); 
        DECLARE_CONFIGURABLE_OP(invert_permutation, 1, 1, false, 0, 0);  

        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(concat_bp, -1, -1, false, 0, 1);

        DECLARE_OP(mergemax, -1, 1, false);
        DECLARE_OP(mergemaxindex, -1, 1, false);
        DECLARE_OP(mergeadd, -1, 1, false);
        DECLARE_OP(mergeavg, -1, 1, false);   

        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1); 

        DECLARE_OP(Floor, 1, 1, true);

        DECLARE_OP(Log1p, 2, 1, true);

        DECLARE_CONFIGURABLE_OP(reverse, 1, 1, true, 0, -2);

        DECLARE_CUSTOM_OP(gather, 2, 1, false, 0, 1);

        DECLARE_CUSTOM_OP(pad, 2, 1, false, 0, 1);
    }
}