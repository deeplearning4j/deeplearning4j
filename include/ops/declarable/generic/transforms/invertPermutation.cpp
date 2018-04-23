//
// @author, Yurii Shyrma (iuriish@yahoo.com), created on 06.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_invert_permutation)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(invert_permutation, 1, 1, false, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->isVector(), 0 , "INVERT_PERMUTATION op: input array must be vector, but got shape %s instead !", ShapeUtils<T>::shapeAsString(input).c_str());
    
    helpers::invertPermutation(*input, *output);
    
    return Status::OK();
}
        
DECLARE_SYN(InvertPermutation, invert_permutation);


}
}

#endif