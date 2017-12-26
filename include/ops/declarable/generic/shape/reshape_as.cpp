//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {


    //////////////////////////////////////////////////////////////////////////
    CUSTOM_OP_IMPL(reshapeas, 2, 1, true, 0, 0) {
    
        NDArray<T> *x = INPUT_VARIABLE(0);
        NDArray<T> *y = INPUT_VARIABLE(1);

        NDArray<T>* z = OUTPUT_VARIABLE(0);
        std::vector<int> shapeNew(y->shapeOf(), y->shapeOf() + y->rankOf());
        char order = y->ordering();

        if (x->reshapei(order, shapeNew)) {
            *z = *x;
            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }

        return ND4J_STATUS_BAD_INPUT;
    }
    DECLARE_SYN(reshape_as, reshapeas);
    
    DECLARE_SHAPE_FN(reshapeas) {
    
    int* inputShapeInfo = inputShape->at(1);    
    int shapeInfoLength = inputShapeInfo[0]*2 + 4;

    int* outputShapeInfo(nullptr);
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shapeInfoLength, int); 
    memcpy(outputShapeInfo, inputShapeInfo, shapeInfoLength*sizeof(int));
    
    return new ShapeList(outputShapeInfo);
}









}

}