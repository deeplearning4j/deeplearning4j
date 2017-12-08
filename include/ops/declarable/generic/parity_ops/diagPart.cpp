//
// Created by Yurii Shyrma on 06.12.2017.
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
// Returns a diagonal tensor with a given diagonal values. Given a diagonal, this operation returns a tensor with the diagonal and everything else padded with zeros. The diagonal is computed as follows:
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/diag
CUSTOM_OP_IMPL(diag_part, 1, 1, false, 0, 0) {

	NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int outLen = output->lengthOf();
    const int inLen  = input->lengthOf();

    int i(0), j(0);
    while(j < outLen) {
    	
    	(*output)(j) = (*input)(i);
    	i += outLen+1;
    	++j;
    }
    
    return ND4J_STATUS_OK;
}
DECLARE_SYN(DiagPart, diag_part);


DECLARE_SHAPE_FN(diag_part) {

    NDArray<T>* input = INPUT_VARIABLE(0);

    const int inRank = input->rankOf();
    
    if(inRank != 2 &&  inRank != 4 && inRank != 6)
        throw "CUSTOM_OP diag_part: input array must have even rank <= 6 !";    

    for(int i = 0; i < inRank-1; ++i)
    	if(input->sizeAt(i) != input->sizeAt(i+1))
    		throw "CUSTOM_OP diag_part: dimensions of input array must be equal !";    

    int* outShapeInfo = nullptr;
	
	int outRank = inRank/2;
	if(outRank == 1)
		outRank += 1;
	
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), int);
	
	outShapeInfo[0] = outRank;
	for(int i = 0; i < outRank; ++i)
		outShapeInfo[i+1] = input->sizeAt(i);

	if(inRank/2 == 1)
		outShapeInfo[1] = 1;

	shape::updateStrides(outShapeInfo, input->ordering());

    return new ShapeList(outShapeInfo);
    
}





}
}

