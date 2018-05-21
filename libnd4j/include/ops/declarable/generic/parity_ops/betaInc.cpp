//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_betainc)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/betaInc.h>

namespace nd4j {
namespace ops  {

CONFIGURABLE_OP_IMPL(betainc, 3, 1, false, 0, 0) {

	NDArray<T>* a = INPUT_VARIABLE(0);
    NDArray<T>* b = INPUT_VARIABLE(1);
    NDArray<T>* x = INPUT_VARIABLE(2);

	NDArray<T>* output   = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(a->isSameShape(b) && a->isSameShape(x), 0, "CONFIGURABLE_OP betainc: all three input arrays must have the same shapes, bit got a=%s, b=%s and x=%s instead !", ShapeUtils<T>::shapeAsString(a).c_str(), ShapeUtils<T>::shapeAsString(b).c_str(), ShapeUtils<T>::shapeAsString(x).c_str());

    int arrLen = a->lengthOf();

    for(int i = 0; i < arrLen; ++i ) {            
        REQUIRE_TRUE((*a)(i) > (T)0.,   0, "BETAINC op: arrays a array must contain only elements > 0 !");            
        REQUIRE_TRUE((*b)(i) > (T)0.,   0, "BETAINC op: arrays b array must contain only elements > 0 !");                    
        REQUIRE_TRUE((T)0. <= (*x)(i) && (*x)(i) <= (T)1., 0, "BETAINC op: all elements of x array must be within [0, 1] range!");                
    }

    *output = helpers::betaInc<T>(*a, *b, *x);

    return ND4J_STATUS_OK;
}

DECLARE_SYN(BetaInc, betainc);
DECLARE_SYN(betaInc, betainc);


}
}

#endif