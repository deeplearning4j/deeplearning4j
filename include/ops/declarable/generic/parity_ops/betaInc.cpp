//
// Created by Yurii Shyrma on 12.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/betaInc.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(betainc, 3, 1, false, 0, 0) {
	        NDArray<T>* a = INPUT_VARIABLE(0);
            NDArray<T>* b = INPUT_VARIABLE(1);
            NDArray<T>* x = INPUT_VARIABLE(2);

	        NDArray<T>* output   = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(a->isSameShape(b) && a->isSameShape(x), 0, "CONFIGURABLE_OP betainc: all three input arrays must have the same shapes !");

            int arrLen = a->lengthOf();

            for(int i = 0; i < arrLen; ++i ) {
            
                REQUIRE_TRUE((*a)(i) > (T)0. && (*b)(i) > (T)0.,   0, "CONFIGURABLE_OP betainc: arrays a and b must contain elements > 0 !");            
                REQUIRE_TRUE((T)0. <= (*x)(i) && (*x)(i) <= (T)1., 0, "CONFIGURABLE_OP betainc: elements of x array must be within [0, 1] range!");
                
            }

            *output = helpers::betaInc<T>(*a, *b, *x);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(BetaInc, betainc);
        DECLARE_SYN(betaInc, betainc);
    }
}

