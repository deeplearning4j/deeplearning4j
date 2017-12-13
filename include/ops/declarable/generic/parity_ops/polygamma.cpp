//
// Created by Yurii Shyrma on 13.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/polyGamma.h>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
/**
   * This op calculates polygamma function psi^(n)(x). Implementation is based on serial representation written in 
   * terms of the Hurwitz zeta function: polygamma = (-1)^{n+1} * n! * zeta(n+1, x).
   * Currently the case n = 0 is not supported.
   * 
   * Input arrays: 
   *    0: n - define derivative order (n+1), type integer (however currently is implemented as float casted to integer)
   *    1: x - abscissa points where to evaluate the polygamma function, type float
   *
   * Output array: 
   *    0: values of polygamma function at corresponding x, type float
   * 
   * Two input and one output arrays have the same shape
   */      
//////////////////////////////////////////////////////////////////////////
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

