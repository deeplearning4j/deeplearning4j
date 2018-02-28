//
//  Created by Yurii Shyrma on 22.01.2018
//
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(eye, 1, 1, false, 0, 2) {
        
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int rank = output->rankOf();
    ResultSet<T>* arrs = NDArrayFactory<T>::allTensorsAlongDimension(output, {rank-2, rank-1});

    for(int i = 0; i < arrs->size(); ++i)
        arrs->at(i)->setIdentity();
    
    delete arrs;

    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(eye) {

    int* inShapeInfo = inputShape->at(0);
    std::vector<int> params = *block.getIArguments();
    const int size = params.size();

    int* outShapeInfo(nullptr);

    switch(size) {
        
        case 2:
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), int);
            outShapeInfo[0] = 2;
            outShapeInfo[1] = params[1];
            outShapeInfo[2] = params[1];
            break;

        case 3:
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2), int);
            outShapeInfo[0] = 2;
            outShapeInfo[1] = params[1];
            outShapeInfo[2] = params[2];
            break;

        default:
            int rank = size-1;
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);
            outShapeInfo[0] = rank;
            outShapeInfo[rank-1] = params[1];
            outShapeInfo[rank] = params[2];
            for(int i = 1; i < rank-1; ++i)
                outShapeInfo[i] = params[i+2];
            break;
    }
        
    shape::updateStrides(outShapeInfo, (char)(params[0]));
        
    return SHAPELIST(outShapeInfo);
}


}
}