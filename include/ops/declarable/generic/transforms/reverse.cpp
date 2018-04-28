//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 02.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>


namespace nd4j {
namespace ops  {


CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
       
    helpers::reverse(INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), block.getIArguments());
   
    return Status::OK();
}

}
}

#endif