//
// Created by yurii@skymind.io on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(stack, -1, 1, false, 0, 1) {

    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    int dim = block.getIArguments()->at(0);
    if(dim < 0)
    	dim += input->rankOf();                	

	std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {dim});	
	ArrayList<T>* list = NDArrayFactory<T>::allTensorsAlongDimension(output, dimsToExclude);		// list.size() == block.width()

	for(int i=0; i<list->size(); ++i)
		list->at(i)->assign(INPUT_VARIABLE(i));
	
	// remove unity from output shape if input arrays are vectors 
	if(input->isVector())	
	{
		std::vector<int> outShape(output->shapeOf(), output->shapeOf() + output->rankOf());		
		outShape.erase(find(outShape.begin(), outShape.end(), 1));
		output->reshapei(output->ordering(), outShape);
		if(dim != 0 && (int)block.width() == 1)			// such is implemented by tensorFlow
			output->permutei({1, 0});
		output->getShapeInfo()[output->rankOf()*2 + 2] = 1;		
	}
	

    STORE_RESULT(*output);
	delete list;
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(stack) {

	// check whether shapes of all input array are the same	
	const int inArrNum = (int) block.width();
	for (int i = 0; i < inArrNum - 1; ++i)
		if (!shape::equalsSoft(inputShape->at(i), inputShape->at(i+1)))
			throw "CUSTOM_OP stack: the shapes of input arrays are different !";
	
	// check whether input dimension is within rank range
	int* inShapeInfo = inputShape->at(0);
	int rank = inShapeInfo[0];
	int dim = block.getIArguments()->at(0);
	if(dim < 0 ) dim += rank;
	if(dim >= rank)
		throw "CUSTOM_OP stack: the input dimension is greater/equal than rank of input input arrays shapes !";

	//the rank of output ShapeInfo is larger by one compared to input ShapeInfo
	std::vector<int> outShape(inShapeInfo + 1, inShapeInfo + 1 + rank);
	// insert inArrNum at dim position of input shape to get output shape	
	outShape.insert(outShape.begin() + dim, inArrNum);
	// if input arrays are vectors remove unity from shape
	NDArray<T>* input = INPUT_VARIABLE(0);

	// evaluate output ShapeInfo
	int newRank = outShape.size();
	int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), newRank*2+4, int);
    outShapeInfo[0] = newRank;
    for(int i=1; i <= newRank; ++i)
    	outShapeInfo[i] = outShape[i-1];
	
    shape::updateStrides(outShapeInfo, input->ordering());    

    return new ShapeList(outShapeInfo);
    

}


// 1) 1х4 + 1х4 = 2х1х4 (along dim=0) = 2x4 
// 2) 1х4 + 1х4 = 1х2х4 (along dim=1) = 2x4 
// 3) 4х1 + 4х1 = 2х4x1 (along dim=0) = 2x4 
// 4) 4х1 + 4х1 = 4х2x1 (along dim=1) = 4x2 
















}
}