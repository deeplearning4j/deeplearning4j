//
// @author Yurii Shyrma, created on 16.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_relu6)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6, 1, 1, true, 1, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    input->template applyTransform<simdOps::RELU6<T>>(output, &T_ARG(0));
    
    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6_bp, 2, 1, true, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* gradO = INPUT_VARIABLE(1);
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);
    
    auto derivative = LAMBDA_TT(inp, grad) {
        
        if((T)0. < inp && inp < (T)6.)
            return grad;                    // derivative = 1
        else 
            return (T)0.;                   // derivative = 0
    };

    input->applyPairwiseLambda(gradO, derivative, gradI);
    
    return Status::OK();
}



}
}

#endif