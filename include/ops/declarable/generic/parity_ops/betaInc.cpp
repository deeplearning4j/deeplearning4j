//
// Created by Yurii Shyrma on 12.12.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/betaInc.h>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
/**
   * This op calculates regularized incomplete beta integral Ix(a, b). 
   * Implementation is based on two algorithms depending on input values of a and b:
   * - when a and b are both >  maxValue (3000.), then apply Gauss-Legendre quadrature method
   * - when a and b are both <= maxValue (3000.), then apply modified Lentz’s algorithm for continued fractions
   * 
   * Input arrays: 
   *    a: define power t^{a-1}, must be > 0, type float.
   *    b: define power (1-t)^{b-1}, must be > 0, type float.
   *    x: define upper limit of integration, must be within (0 <= x <= 1) range, type float.
   *
   * Output array: 
   *    0: values of  regularized incomplete beta integral that corresponds to variable upper limit x, type float
   * 
   * Three input and one output arrays must have the same shape
   */      
//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(betainc, 3, 1, false, 0, 0) {

	NDArray<T>* a = INPUT_VARIABLE(0);
    NDArray<T>* b = INPUT_VARIABLE(1);
    NDArray<T>* x = INPUT_VARIABLE(2);

	NDArray<T>* output   = OUTPUT_VARIABLE(0);

    if(!a->isSameShape(b) || !a->isSameShape(x))
        throw "CONFIGURABLE_OP betainc: all three input arrays must have the same shapes !";

    int arrLen = a->lengthOf();

    for(int i = 0; i < arrLen; ++i ) {
        
        if((*a)(i) <= (T)0. || (*b)(i) <= (T)0.)
            throw "CONFIGURABLE_OP betainc: arrays a and b must contain elements > 0 !";
        
        if((*x)(i) < (T)0. || (*x)(i) > (T)1.)
            throw "CONFIGURABLE_OP betainc: elements of x array must be within [0, 1] range!";
    }

    *output = helpers::betaInc<T>(*a, *b, *x);

    return ND4J_STATUS_OK;
}
DECLARE_SYN(BetaInc, betainc);
DECLARE_SYN(betaInc, betainc);



}
}

