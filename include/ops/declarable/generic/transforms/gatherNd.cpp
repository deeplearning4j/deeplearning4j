//
// Created by Yurii Shyrma on 23.01.2018.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>
#include <numeric>


namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather_nd, 2, 1, false, 0, 0) {

	NDArray<T>* input   = INPUT_VARIABLE(0);
    NDArray<T>* indices = INPUT_VARIABLE(1);
    NDArray<T>* output  = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(indices->rankOf() > 0, 0, "GATHER_ND custom operation: input array of indexes can't be single scalar, the requirement is: rank > 0 !");

    int rank0 = input->rankOf();
    int rank1 = indices->rankOf();
    int lastIndDim = indices->sizeAt(-1);

    REQUIRE_TRUE(lastIndDim <= rank0, 0, "GATHER_ND custom operation: the last dimension of indices array must be <= rank of input array !");
    
    std::vector<int> tadDims(rank0 - lastIndDim);
    std::iota(tadDims.begin(), tadDims.end(), rank0-1);
    ResultSet<T>* innerMostOut = NDArrayFactory<T>::allTensorsAlongDimension(output, tadDims); 

    ResultSet<T>* innerMost1 = NDArrayFactory<T>::allTensorsAlongDimension(indices, {rank1-1});	
    
    std::iota(tadDims.begin(), tadDims.end(), lastIndDim);
    ResultSet<T>* innerMost0 = NDArrayFactory<T>::allTensorsAlongDimension(input, tadDims);

    int* outerShapeInfo = nullptr;
    ALLOCATE(outerShapeInfo, block.getWorkspace(), shape::shapeInfoLength(lastIndDim), int);
    outerShapeInfo[0] = lastIndDim;
    for(int i = 1; i <= lastIndDim; ++i)
        outerShapeInfo[i] = input->sizeAt(i-1);
    shape::updateStrides(outerShapeInfo, input->ordering());

    int* idx = new int[lastIndDim];

    for(int i = 0; i < innerMost1->size(); ++i) {
                
        NDArray<T>* idxSubArr = innerMost1->at(i);        
        
        for(int j = 0; j < lastIndDim; ++j) {
            REQUIRE_TRUE((int)(*idxSubArr)(j) < input->sizeAt(j), 0, "GATHER_ND custom operation: wrong elements in input indices array, each element must be smaller than corresponding dimension of input array !");        
            idx[j] = (*idxSubArr)(j);
        }
                
        int currentInd0 = (int)shape::getOffset(0, shape::shapeOf(outerShapeInfo), shape::stride(outerShapeInfo), idx, lastIndDim);

        if(rank0 != lastIndDim) {
            NDArray<T>* outSubArr = innerMostOut->at(i);
            outSubArr->assign(innerMost0->at(currentInd0));
        }
        else
            (*output)(i) = ((*input)(currentInd0));
    }

    delete innerMost1;
    delete innerMost0;
    delete innerMostOut;
    delete []idx;
    RELEASE(outerShapeInfo, block.getWorkspace());
    
    return Status::OK();
}


DECLARE_SHAPE_FN(gather_nd) {

	int* inShapeInfo0 = inputShape->at(0);
    int* inShapeInfo1 = inputShape->at(1);
		
    int inRank0 = inShapeInfo0[0];
    int inRank1 = inShapeInfo1[0];
    int lastIndDim = inShapeInfo1[inRank1];

	int outRank = (inRank1 - 1) + (inRank0 - lastIndDim);

    int* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);

    outShapeInfo[0] = outRank;

    for(int i = 1; i <= inRank1-1; ++i)
        outShapeInfo[i] = inShapeInfo1[i];

    for(int i = 0; i < inRank0-lastIndDim; ++i)
        outShapeInfo[inRank1 + i] = inShapeInfo0[lastIndDim + i + 1];

	shape::updateStrides(outShapeInfo, shape::order(inShapeInfo0));

    return new ShapeList(outShapeInfo);    
}




}
}