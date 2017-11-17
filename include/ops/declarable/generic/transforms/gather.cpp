//
// Created by Yurii Shyrma on 16.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <vector>


namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
// for explanations how this operation works please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/gather
CUSTOM_OP_IMPL(gather, 2, 1, false, 0, 1) {

	NDArray<T>* input   = INPUT_VARIABLE(0);
	NDArray<T>* indices = INPUT_VARIABLE(1);
	NDArray<T>* output  = OUTPUT_VARIABLE(0);

	int axis = block.getIArguments()->at(0);
    
    
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

    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(gather) {

	// check shape of paddings 
	NDArray<T>* input   = INPUT_VARIABLE(0);
	NDArray<T>* indices = INPUT_VARIABLE(1);

	int axis = block.getIArguments()->at(0);
	int inputRank = input->rankOf();

	if(axis < 0)
		axis += inputRank;
	if(axis >= inputRank)
		throw "GATHER custom operation: input axis is out of input array inputRank !";

	for(int i = 0; i < indices->lengthOf(); ++i)
		if((int)indices->getIndexedScalar(i) >= input->shapeOf()[axis])
			throw "GATHER custom operation: some of input indexes is larger than corresponding shape of input array !";
    
    int indicesRank = indices->rankOf();
    if(indices->isVector())
    	indicesRank = 1;
    else if(indices->isScalar())
    	indicesRank = 0;

    int outputRank = inputRank + indicesRank - 1;
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), outputRank*2+4, int);
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

    return new ShapeList(outputShapeInfo);
    
}





}
}