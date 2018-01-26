//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
        namespace ops {
        OP_IMPL(polisq, 2, 1, true) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
//            if (x.shape() == y.shape()) {
            
            auto mainProc = LAMBDA_TT(_x, _y) { 
                return (_x * _x + _y); 
            };

            x->applyPairwiseLambda(y, mainProc);
            return ND4J_STATUS_OK;
//            }
//            else {
//                return ND4J_STATUS_FAILURE;
//            }
        }

    }
}
