//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
        DECLARE_CUSTOM_OP(set_seed, -2, 1, false, 0, -2);
        DECLARE_CUSTOM_OP(get_seed, -2, 1, false, 0, 0);
       
        DECLARE_CUSTOM_OP(randomuniform, 1, 1, true, 2, 0);
        DECLARE_CUSTOM_OP(random_normal, 1, 1, true, 2, 0);
        DECLARE_CUSTOM_OP(random_bernoulli, 1, 1, true, 0, 1);
        DECLARE_CUSTOM_OP(random_exponential, 1, 1, true, 1, 0);
        DECLARE_CUSTOM_OP(random_crop, 2, 1, false, 0, 0);
    }
}