//
// Created by yurii@skymind.io on 02.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <vector>


namespace nd4j {
namespace ops  {


CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
   
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    std::vector<int>* argI = block.getIArguments();
    std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), *argI);       

    auto listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);
    auto listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
       
    NDArray<T>* subArrIn  = nullptr;
    NDArray<T>* subArrOut = nullptr;    
    for(int i=0; i<listIn->size(); ++i) {               // listIn->size() = listOut->size()
        subArrIn   = listIn->at(i);
        subArrOut  = listOut->at(i);        
        helpers::reverseArray<T>(subArrIn->getBuffer(), subArrIn->getShapeInfo(), subArrOut->getBuffer(), subArrOut->getShapeInfo());
    }

    STORE_RESULT(*output);

    delete listOut;
    delete listIn;

    return ND4J_STATUS_OK;
}

}
}

#endif