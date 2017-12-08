//
// Created by Yurii Shyrma on 06.12.2017.
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
// Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros. The diagonal is computed as follows:
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/diag
CUSTOM_OP_IMPL(diag, 1, 1, false, 0, 0) {

	NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int inLength = input->lengthOf();    

    output->assign((T)0.);

// #pragma omp parallel for if(inLength > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < inLength; ++i)
    	(*output)(i*(inLength+1)) = (*input)(i);
    
    return ND4J_STATUS_OK;
}
DECLARE_SYN(MatrixDiag, diag);


DECLARE_SHAPE_FN(diag) {

    NDArray<T>* input = INPUT_VARIABLE(0);

    if(input->rankOf() > 3)
        throw "CUSTOM_OP diag: rank of input array must be <= 3 !";    

    return new ShapeList(ShapeUtils<T>::evalDiagShapeInfo(*input));
    
}





}
}

