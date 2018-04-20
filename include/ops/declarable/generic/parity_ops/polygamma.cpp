//
// Created by Yurii Shyrma on 13.12.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_polygamma)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/polyGamma.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(polygamma, 2, 1, false, 0, 0) {

            NDArray<T>* n = INPUT_VARIABLE(0);
            NDArray<T>* x = INPUT_VARIABLE(1);

            NDArray<T>* output   = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(n->isSameShape(x), 0, "CONFIGURABLE_OP polygamma: two input arrays must have the same shapes !");

            int arrLen = n->lengthOf();

            for(int i = 0; i < arrLen; ++i ) {

                // TODO case for n == 0 (digamma) should be of OK
                REQUIRE_TRUE((*n)(i) > (T)0., 0, "CONFIGURABLE_OP polygamma: all elements of n array must be > 0 !");

                REQUIRE_TRUE((*x)(i) > (T)0., 0, "CONFIGURABLE_OP polygamma: all elements of x array must be > 0 !");
            }

            *output = helpers::polyGamma<T>(*n, *x);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(polyGamma, polygamma);
        DECLARE_SYN(PolyGamma, polygamma);
    }
}

#endif