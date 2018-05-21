//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_broadcastgradientargs)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        /**
         * PLEASE NOTE: This op is disabled atm, and reserved for future releases.
         */
         OP_IMPL(broadcastgradientargs, 2, 2, true) {
            
            nd4j_printf("BroadcastGradientArgs: Not implemented yet\n", "");

            return ND4J_STATUS_KERNEL_FAILURE;
        }
        DECLARE_SYN(BroadcastGradientArgs, broadcastgradientargs);
    }
}

#endif