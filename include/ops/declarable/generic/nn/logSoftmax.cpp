//
// Created by Yurii Shyrma on 01.02.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/softMaxForVector.h>

namespace nd4j {
namespace ops {


CONFIGURABLE_OP_IMPL(log_softmax, 1, 1, true, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    
    int rank = input->rankOf();
    int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

    REQUIRE_TRUE(dim < rank, 0, "log_softmax op: the value of input integer parameter (dimension) must be less than rank of input array !");

    if(input->isVector()) {
        
        if(rank == 1 || input->sizeAt(dim) != 1)
            helpers::logSoftMaxForVector<T>(*input, *output);
        else
            *output = 0.;
    }
    else {
        
        NDArray<T> exponents = input->template transform<simdOps::Exp<T>>();
        NDArray<T> sumAlongDim = exponents.template reduceAlongDims<simdOps::Sum<T>>({dim}, true);
        *output = *input - sumAlongDim.template transform<simdOps::Log<T>>();
    }
    
    return Status::OK();
}


CONFIGURABLE_OP_IMPL(log_softmax_bp, 2, 1, true, 0, 0) {
    
    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* epsInput = INPUT_VARIABLE(1);
    NDArray<T>* output   = OUTPUT_VARIABLE(0);    

    int rank = input->rankOf();
    int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;
    softmax<T> op;
    ResultSet<T>* results = op.execute({input}, {}, {dim});    
    output->assign(results->at(0));
    
    *output = *epsInput - (*output * *epsInput).template reduceAlongDims<simdOps::Sum<T>>({dim}, true);

    delete results;
    return Status::OK();
}


}
}
