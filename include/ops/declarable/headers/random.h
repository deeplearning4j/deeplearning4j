//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
        DECLARE_CUSTOM_OP(set_seed, -2, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(get_seed, -2, 1, false, 0, 0);
       
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0);        
    }
}