//
// Created by Yurii Shyrma on 16.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>
#include <numeric>


namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather, 1, 1, false, 0, 1) {

	NDArray<T>* input   = INPUT_VARIABLE(0);
	NDArray<T>* output  = OUTPUT_VARIABLE(0);

	int axis = block.getIArguments()->at(0);    
    int inputRank = input->rankOf();
	if(axis < 0)
        axis += inputRank;

	// input validation
    REQUIRE_TRUE(axis < inputRank, 0, "GATHER custom operation: input axis is out of input array inputRank !");

	if (block.width() > 1) {
		NDArray<T>* indices = INPUT_VARIABLE(1);

		for(int i = 0; i < indices->lengthOf(); ++i)
        	REQUIRE_TRUE((int)indices->getIndexedScalar(i) < input->shapeOf()[axis], 0, "GATHER custom operation: some of input indexes is larger than corresponding shape of input array !");

    
    	// first case: indices consist of only one scalar
   		if(indices->isScalar()) {
   			std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {axis});
   			shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
   			tad.createTadOnlyShapeInfo();
    		tad.createOffsets();
    		NDArray<T> tadArr(input->getBuffer() + tad.tadOffsets[(int)indices->getScalar(0)], tad.tadOnlyShapeInfo);
    		output->assign(&tadArr);
   		}
   		// second case: indices is vector
   		else if(indices->isVector()) {   	
   			ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {axis}));
   			ResultSet<T>* listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input,  ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis}));
   			for(int i = 0; i < listOut->size(); ++i)
   				listOut->at(i)->assign(listIn->at((int)indices->getIndexedScalar(i)));
   			delete listOut;
   			delete listIn;
   		}
   		// third case: indices is usual n-dim array
   		else {
   			std::vector<int> dimsOut(indices->rankOf());
	   		std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
   			std::vector<int> temp1 = ShapeUtils<T>::evalDimsToExclude(output->rankOf(), dimsOut);
	   		std::vector<int> temp2 = ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis});
   			ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, temp1);
	   		ResultSet<T>* listIn = NDArrayFactory<T>::allTensorsAlongDimension(input,  temp2 );
   			for(int i = 0; i < listOut->size(); ++i)
   				listOut->at(i)->assign(listIn->at((int)indices->getIndexedScalar(i)));
   			delete listOut;
   			delete listIn;
   		}

    	STORE_RESULT(*output);	
	} else if (block.numI() > 1) {
		
		for(int i = 1; i < block.numI(); ++i)
        	REQUIRE_TRUE(block.getIArguments()->at(i) < input->shapeOf()[axis], 0, "GATHER custom operation: some of input indexes is larger than corresponding shape of input array !");

		// we only allow scalar/vector case here
		if (block.numI() == 2) {
			// scalar case
			std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {axis});
   			shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
   			tad.createTadOnlyShapeInfo();
    		tad.createOffsets();
    		NDArray<T> tadArr(input->getBuffer() + tad.tadOffsets[block.getIArguments()->at(1)], tad.tadOnlyShapeInfo);
    		output->assign(&tadArr);
		} else {
			// vector case
			ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {axis}));
   			ResultSet<T>* listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input,  ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis}));

			// that's fine, since we know that number of iArgs matches number of elements in listOut
   			for(int i = 0; i < listOut->size(); ++i)
   				listOut->at(i)->assign(listIn->at(block.getIArguments()->at(i+1)));
   			delete listOut;
   			delete listIn;
		}
	} else {
		REQUIRE_TRUE(false, 0, "Gather: indices should be provided either as additional input array, or as IntArguments");
	}

    return Status::OK();
}


DECLARE_SHAPE_FN(gather) {

	// check shape of paddings 
	NDArray<T>* input   = INPUT_VARIABLE(0);
	int* outputShapeInfo = nullptr;

	int axis = block.getIArguments()->at(0);
	int inputRank = input->rankOf();
	if(axis < 0)
		axis += inputRank;

	if (block.width() > 1) {
		NDArray<T>* indices = INPUT_VARIABLE(1);
    
    	int indicesRank = indices->rankOf();
    	if(indices->isVector())
    		indicesRank = 1;
    	else if(indices->isScalar())
    		indicesRank = 0;

    	int outputRank = inputRank + indicesRank - 1;
    	ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), int);
    	// fill output shapeInfo
    	outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;    
    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = input->shapeOf()[i];
    	
		if(indices->isVector())
    		outputShapeInfo[shapeIdx++] = indices->lengthOf();
    	else if(!indices->isScalar())
    		for(int i = 0; i < indices->rankOf(); ++i)
    			outputShapeInfo[shapeIdx++] = indices->shapeOf()[i];

    	for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = input->shapeOf()[i];
	
    	shape::updateStrides(outputShapeInfo, input->ordering());    
	} else if (block.numI() > 1) {
		int indicesRank = block.numI() == 2 ? 0 : 1;

		int outputRank = inputRank + indicesRank - 1;
		ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), int);

		// building shape manually
		outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;    
    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = input->shapeOf()[i];

		if (block.numI() > 2)
			outputShapeInfo[shapeIdx++] = block.numI() - 1;

		for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = input->shapeOf()[i];

		shape::updateStrides(outputShapeInfo, input->ordering());    
	}

    return SHAPELIST(outputShapeInfo);
    
}





}
}