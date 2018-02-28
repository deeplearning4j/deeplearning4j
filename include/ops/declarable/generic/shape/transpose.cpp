//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
namespace ops {

    //////////////////////////////////////////////////////////////////////////
    CUSTOM_OP_IMPL(transpose, 1, 1, true, 0, 0) {
        NDArray<T>* x = INPUT_VARIABLE(0);            
        if(block.isInplace()) {
            x->transposei();
            STORE_RESULT(*x);
        }
        else {
            NDArray<T>* output = OUTPUT_VARIABLE(0);
            x->transpose(*output);
            STORE_RESULT(*output);
        }
        return ND4J_STATUS_OK;
    }


    DECLARE_SHAPE_FN(transpose) {
    
    int* outputShapeInfo = ShapeUtils<T>::evalTranspShapeInfo(*INPUT_VARIABLE(0), block.workspace());
    return SHAPELIST(outputShapeInfo);
}






}
}