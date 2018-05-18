//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_diag)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/diag.h>

namespace nd4j {
namespace ops  {

////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(diag, 1, 1, false, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    
    // input validation
    REQUIRE_TRUE(input->rankOf() <= 3, 0, "CUSTOM_OP diag: rank of input array must be <= 3 !, but got %i instead", input->rankOf());
    
    helpers::diagFunctor(input, output);    
    
    return ND4J_STATUS_OK;
}

DECLARE_SYN(MatrixDiag, diag);

////////////////////////////////////////////////////////////////////////// 
DECLARE_SHAPE_FN(diag) {
    
    const Nd4jLong* inputShapeInfo = inputShape->at(0);

    return SHAPELIST(ShapeUtils<T>::evalDiagShapeInfo(inputShapeInfo, block.workspace()));
}


}
}

#endif