//
// Created by raver119 on 29/10/17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_softmax)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/softMaxForVector.h>

namespace nd4j {
namespace ops {


CONFIGURABLE_OP_IMPL(softmax, 1, 1, true, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    
    int rank = input->rankOf();
    int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

    REQUIRE_TRUE(dim < rank, 0, "SOFTMAX op: the value of input integer parameter (dimension) must be less than rank of input array !");

    if(input->isVector()) {
        
        if(rank == 1 || input->sizeAt(dim) != 1)
            helpers::softMaxForVector<T>(*input, *output);
        else
            *output = 1.;
    }
    else {
        
        NDArray<T> exponents = input->template transform<simdOps::Exp<T>>();
        NDArray<T> sumAlongDim = exponents.template reduceAlongDims<simdOps::Sum<T>>({dim}, true);        
        output->assign(exponents / sumAlongDim);
    }
    
    return Status::OK();
}


CONFIGURABLE_OP_IMPL(softmax_bp, 2, 1, true, 0, 0) {
    
    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* epsInput = INPUT_VARIABLE(1);
    NDArray<T>* output   = OUTPUT_VARIABLE(0);    

    int rank = input->rankOf();
    int dim  = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;
    softmax<T> op;
    ResultSet<T>* results = op.execute({input}, {}, {dim});    
    output->assign(results->at(0));
    
    NDArray<T> sumAlongDim = (*output * *epsInput).template reduceAlongDims<simdOps::Sum<T>>({dim}, true);
    *output *= (*epsInput - sumAlongDim);

    delete results;
    return Status::OK();
}


}
}

#endif