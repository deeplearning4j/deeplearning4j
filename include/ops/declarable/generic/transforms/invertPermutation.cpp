//
//  // created by Yurii Shyrma on 06.12.2017
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(invert_permutation, 1, 1, false, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* output = this->getZ(block);

    if(!input->isVector())
        throw "CONFIGURABLE_OP invertPermute: input array must be vector !";
    
    std::set<T> uniqueElems;
    const int lenght = input->lengthOf();

// #pragma omp parallel for if(lenght > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < lenght; ++i) {
        T elem  = (*input)(i);
        if(!uniqueElems.insert(elem).second)
            throw "CONFIGURABLE_OP invertPermute: input array contains duplicates !";
        if(elem < (T)0. || elem > lenght - (T)1.)
            throw "CONFIGURABLE_OP invertPermute: element of input array is out of range (0, lenght-1) !";
        (*output)((int)elem) = i;
    }
    
    return ND4J_STATUS_OK;
}
        
DECLARE_SYN(InvertPermutation, invert_permutation);


}
}
