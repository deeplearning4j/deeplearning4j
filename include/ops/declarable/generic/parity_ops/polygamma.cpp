//
// Created by Yurii Shyrma on 13.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/polyGamma.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(polygamma, 2, 1, false, 0, 0) {

            NDArray<T>* n = INPUT_VARIABLE(0);
            NDArray<T>* x = INPUT_VARIABLE(1);

            NDArray<T>* output   = OUTPUT_VARIABLE(0);

            if(!n->isSameShape(x))
                throw "CONFIGURABLE_OP polygamma: two input arrays must have the same shapes !";

            int arrLen = n->lengthOf();

            for(int i = 0; i < arrLen; ++i ) {

                // TODO case for n == 0 (digamma) should be of OK
                if((*n)(i) <= (T)0.)
                    throw "CONFIGURABLE_OP polygamma: all elements of n array must be > 0 !";

                if((*x)(i) <= (T)0.)
                    throw "CONFIGURABLE_OP polygamma: all elements of x array must be > 0 !";
            }

            *output = helpers::polyGamma<T>(*n, *x);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(polyGamma, polygamma);
        DECLARE_SYN(PolyGamma, polygamma);
    }
}

