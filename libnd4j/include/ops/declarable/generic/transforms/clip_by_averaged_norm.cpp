//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyavgnorm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

CONFIGURABLE_OP_IMPL(clipbyavgnorm, 1, 1, true, 1, 0) {

    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const T clipNorm = T_ARG(0);
    const bool isInplace = block.isInplace();

    helpers::clipByAveraged(*input, *output, *block.getIArguments(), clipNorm, isInplace);

    return Status::OK();
}


}
}

#endif