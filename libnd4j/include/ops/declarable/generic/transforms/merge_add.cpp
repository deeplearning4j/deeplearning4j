//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mergeadd)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

OP_IMPL(mergeadd, -1, 1, false) {
    
    REQUIRE_OK(this->validateInputDimensionsMatch(block));
        
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray<T>*> inArrs(block.width());
    
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    helpers::mergeAdd(inArrs, *output);

    return Status::OK();
}

DECLARE_SYN(mergesum, mergeadd);
DECLARE_SYN(add_n, mergeadd);
DECLARE_SYN(addn, mergeadd);
DECLARE_SYN(accumulaten, mergeadd);
DECLARE_SYN(accumulate_n, mergeadd);


}
}

#endif