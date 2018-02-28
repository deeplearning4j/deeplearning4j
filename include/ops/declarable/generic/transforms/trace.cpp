//
// Created by Yurii Shyrma on 24.01.2018.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(trace, 1, 1, false, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);
    
    const int inRank =  input->rankOf();
    REQUIRE_TRUE(inRank >= 2, 0, "CUSTOM_OP trace: the rank of input array must be >=2 !");

    ResultSet<T>* setOfSubArrs = NDArrayFactory<T>::allTensorsAlongDimension(input, {inRank-2, inRank-1});

    for(int i = 0; i < setOfSubArrs->size(); ++i)
        (*output)(i) = setOfSubArrs->at(i)->getTrace();

    delete setOfSubArrs;

    return Status::OK();
}


DECLARE_SHAPE_FN(trace) {

    int* inShapeInfo = inputShape->at(0);
    
    const int rank = inShapeInfo[0] - 2;

    int* outShapeInfo(nullptr);
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 

    outShapeInfo[0] = rank;
    for(int i=1; i <= rank; ++i)
        outShapeInfo[i] = inShapeInfo[i];

    shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));

    return SHAPELIST(outShapeInfo);
}


}
}