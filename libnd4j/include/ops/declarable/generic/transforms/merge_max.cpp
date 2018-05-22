//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mergemax)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {
    
OP_IMPL(mergemax, -1, 1, false) {
        
    REQUIRE_OK(this->validateInputDimensionsMatch(block));
        
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray<T>*> inArrs(block.width());
    
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    helpers::mergeMax(inArrs, *output);

    return Status::OK();
}

DECLARE_SYN(MergeMax, mergemax);



}
}

#endif