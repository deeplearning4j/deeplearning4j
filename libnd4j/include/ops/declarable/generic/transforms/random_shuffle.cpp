//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 26.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_shuffle)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

OP_IMPL(random_shuffle, 1, 1, true) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    const bool isInplace = block.isInplace();
    NDArray<T>* output = isInplace ? nullptr : OUTPUT_VARIABLE(0);

    nd4j::random::RandomBuffer* rng = block.getRNG();   
    
    REQUIRE_TRUE(rng != nullptr, 0, "RANDOM_SHUFFLE op: RNG should be defined in Graph !");

    helpers::randomShuffle(*input, *output, *rng, isInplace);
    
    return Status::OK();
}


}
}

#endif