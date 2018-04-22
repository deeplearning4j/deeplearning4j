//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_log_softmax)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>

namespace nd4j {
namespace ops {


CONFIGURABLE_OP_IMPL(log_softmax, 1, 1, true, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    
    const int rank = input->rankOf();
    const int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

    REQUIRE_TRUE(dim < rank, 0, "LOG_SOFTMAX OP: the value of input integer parameter (dimension) must be less than input array rank %i, but got dimension = %i instead !", rank, dim);

    if(input->isVector()) {
        
        if(rank == 1 || input->sizeAt(dim) != 1)
            helpers::logSoftMaxForVector<T>(*input, *output);
        else
            *output = 0.;
    }
    else {
        
        NDArray<T> exponents = input->template transform<simdOps::Exp<T>>();
        NDArray<T> sumAlongDim = exponents.template reduceAlongDims<simdOps::Sum<T>>({dim}, true);
        output->assign( *input - sumAlongDim.template transform<simdOps::Log<T>>() );
    }
    
    return Status::OK();
}


CONFIGURABLE_OP_IMPL(log_softmax_bp, 2, 1, true, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* gradO = INPUT_VARIABLE(1);
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);    

    const int rank = input->rankOf();
    const int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

    REQUIRE_TRUE(dim < rank, 0, "LOG_SOFTMAX_BP OP: the value of input integer parameter (dimension) must be less than input array rank %i, but got dimension = %i instead !", rank, dim);

    helpers::softmax(*input, *gradI, dim);
        
    gradI->assign( *gradO - (*gradI * *gradO).template reduceAlongDims<simdOps::Sum<T>>({dim}, true) );

    return Status::OK();
}



}
}

#endif
