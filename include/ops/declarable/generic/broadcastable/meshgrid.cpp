//
// Created by Yurii Shyrma on 05.02.2018
//

#include <ops/declarable/CustomOperations.h>
#include <numeric>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(meshgrid, -1, -1, false, 0, 0) {

    bool swapFirst2Dims = true;
    if(block.getIArguments()->size() > 0)
        swapFirst2Dims = (bool)INT_ARG(0);

    int rank = block.width();

    if(rank == 1) {
        OUTPUT_VARIABLE(0)->assign(INPUT_VARIABLE(0));
        return Status::OK();
    }

    int* inIndices = new int[rank];
    std::iota(inIndices, inIndices + rank, 0);
    if(swapFirst2Dims && rank > 1) {
        inIndices[0] = 1;
        inIndices[1] = 0;
    }
            
    for(int i = 0; i < rank; ++i) {        
        ResultSet<T>* list = NDArrayFactory<T>::allTensorsAlongDimension(OUTPUT_VARIABLE(i), {inIndices[i]});        
        for(int j = 0; j < list->size(); ++j)
            list->at(j)->assign(INPUT_VARIABLE(i));

        delete list;
    }    

    delete []inIndices;

    return Status::OK();
}



DECLARE_SHAPE_FN(meshgrid) {

    bool swapFirst2Dims = true;
    if(block.getIArguments()->size() > 0)
        swapFirst2Dims = (bool)INT_ARG(0);
    
    int rank = block.width();
    int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);
    outShapeInfo[0] = rank;    
    for(int i = 1; i <= rank; ++i)
        outShapeInfo[i] = (int)shape::length(inputShape->at(i - 1));
    
    if(swapFirst2Dims && rank > 1)
        math::nd4j_swap<int>(outShapeInfo[1], outShapeInfo[2]);
    
    shape::updateStrides(outShapeInfo, shape::order(inputShape->at(0)));

    ShapeList* shapes = new ShapeList();
    shapes->push_back(outShapeInfo);
    
    int* tempShapeInfo = nullptr;
    for(int i = 2; i <= rank; ++i) {
        ALLOCATE(tempShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int);
        memcpy(tempShapeInfo, outShapeInfo, shape::shapeInfoByteLength(rank));
        shapes->push_back(tempShapeInfo);
    }
    
    return shapes;
}




}
}