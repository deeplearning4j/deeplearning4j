//
// Created by GS <sgazeos@gmail.com>
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>
#include <numeric>


namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(embedding_lookup, 2, 1, false, 0, 1) {

    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param
    NDArray<T>* indeces = INPUT_VARIABLE(1); // indeces, as is
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    int indexRank = indeces->rankOf();
   
    REQUIRE_TRUE(indexRank > 0, 0, "embeded_lookup: input array of indexes can't be single scalar, the requirement is: rank > 0 !");

    int inputRank = input->rankOf();
    int lastIndDim = indeces->lengthOf();
    int partition_mode = INT_ARG(0); // partition_mode == 0 - i.e. 'mod' , 1 - 'div'
    
    nd4j::ops::gather<T> op;

    std::unique_ptr<ResultSet<T>> result(op.execute({input, indeces}, {}, {0}));
    REQUIRE_TRUE(result->status() == ND4J_STATUS_OK, 0, "embedding_lookup: cannot retrieve results from gather op.");
    REQUIRE_TRUE(result->at(0)->isSameShape(output), 0, "embedding_lookup: wrong shape of return from gather op.");
    output->assign(result->at(0));
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(embedding_lookup) {

    int* inShapeInfo = inputShape->at(0);
    int* indecesShapeInfo = inputShape->at(1);
    int inRank = shape::rank(inShapeInfo);

    int outRank = inRank; 

    int* outShapeInfo = nullptr;
    
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);
    std::vector<int> shapeInfo(outRank);

    shapeInfo[0] = indecesShapeInfo[1]; // vector - how many elements
    for (int e = 1; e < outRank; e++)
        shapeInfo[e] = shape::sizeAt(inShapeInfo, e);
    if (shape::order(inShapeInfo) == 'c')
        shape::shapeBuffer(outRank, shapeInfo.data(),  outShapeInfo);
    else
        shape::shapeBufferFortran(outRank, shapeInfo.data(),  outShapeInfo);

    return SHAPELIST(outShapeInfo);    
}




}
}