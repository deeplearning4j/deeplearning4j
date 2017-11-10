//
// Created by yurii@skymind.io on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <vector>
#include <numeric>

namespace nd4j {
    namespace ops {

// declare auxiliary function which serves for recursion purpose
template<typename T>
void recursiveLoop(const int mode, Block<T>& block, NDArray<T>* input, const NDArray<T>* paddings, NDArray<T>* output, std::vector<int> dimensions, int dim, int inIdx, int outIdx);


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(pad, 2, 1, false, 0, 1) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    NDArray<T>* output   = OUTPUT_VARIABLE(0);
    std::vector<int>* argI = block.getIArguments();

	// CONSTANT->0, REFLECT->1, SYMMETRIC->2
    if(argI->at(0) < 0 || argI->at(0) > 2)
    	throw "CUSTOM_OP pad: unknown padding mode, there are only three possible legal values -> 0,1,2 !";

	std::vector<int> dimensions(input->rankOf());	
    std::iota(dimensions.begin(), dimensions.end(), 0);   			// fill with 0, 1, ... rank-1
    
	recursiveLoop(argI->at(0), block, input, paddings, output, dimensions, 0, 0, 0);

    STORE_RESULT(*output);
	
    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(pad) {

	// check shape of paddings 
	NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    int rank =  input->rankOf();    

	if (paddings->rankOf() != 2 || paddings->shapeOf()[0] != rank || paddings->shapeOf()[1] != 2)
		throw "CUSTOM_OP pad: wrong shape of input paddings !";

	std::vector<int>* argI = block.getIArguments();

	// in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements 
	// REFLECT case
	if(argI->at(0) == 1)				
		for(int dim=0; dim < rank; ++dim)
			if(!(paddings->getScalar(dim,0) <= (input->shapeOf()[dim]-1) && paddings->getScalar(dim,1) <= (input->shapeOf()[dim]-1)))
				throw "CUSTOM_OP pad: wrong shape of input paddings for REFLECT mode !";
	// SYMMETRIC case
	if(argI->at(0) == 2)				
	for(int dim=0; dim < rank; ++dim)
		if(!(paddings->getScalar(dim,0) <= input->shapeOf()[dim] && paddings->getScalar(dim,1) <= input->shapeOf()[dim]))
			throw "CUSTOM_OP pad: wrong shape of input paddings for SYMMETRIC mode !";
	
	int* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), rank*2+4, int);
    outShapeInfo[0] = rank;
    for(int i=1; i <= rank; ++i)
    	outShapeInfo[i] = input->shapeOf()[i-1] + paddings->getScalar(i-1,0) + paddings->getScalar(i-1,1);
	
    shape::updateStrides(outShapeInfo, input->ordering());    

    return new ShapeList(outShapeInfo);
    
}



////////////////////////////////////////////////////////////////////////
template<typename T>
void recursiveLoop(const int mode, Block<T>& block, NDArray<T>* input, const NDArray<T>* paddings, NDArray<T>* output, std::vector<int> dimensions, int dim, int inIdx, int outIdx ) {   // initial values of inIdx,outIdx,dim must be equal to zero
	
	int leftOffset;
	// dimensions are array of input dimensions, it is sorted by increasing order
	// every time at the beginning we erase first element from it (not good idea to use vector for this purpose, but luckily it is small enough)
	// then we use this array for tads building, every time while recursion the number of built tads becomes bigger 
	dimensions.erase(dimensions.begin());    	
    // build tad basing on output array, also create auxiliary arrays pointing on required output array ranges
    shape::TAD tadOut(output->getShapeInfo(), dimensions.data(), dimensions.size());
    tadOut.createTadOnlyShapeInfo();
    tadOut.createOffsets();
    NDArray<T> subArrOut(output->getBuffer(), tadOut.tadOnlyShapeInfo, block.getWorkspace());
    NDArray<T> subArr(output->getBuffer(), tadOut.tadOnlyShapeInfo, block.getWorkspace());
	// build tad basing on input array, also create auxiliary array pointing on required input array range
    shape::TAD tadIn(input->getShapeInfo(), dimensions.data(), dimensions.size());
    tadIn.createTadOnlyShapeInfo();
    tadIn.createOffsets();
	NDArray<T> subArrIn(input->getBuffer(), tadIn.tadOnlyShapeInfo, block.getWorkspace());
	// these indices take into account recursion and always point to actual tads numbers
	outIdx = outIdx*output->shapeOf()[dim+1];
	inIdx  = inIdx*input->shapeOf()[dim+1];
	// current input tad number, we add to it unity in a loop
    int k = -1;
    // loop through current dimension
    for(int i = 0; i < output->shapeOf()[dim]; ++i) {
    	// corresponds to outer range (relevant indices are absent in input)						
		if(i < (int)paddings->getScalar(dim,0) || i >= (input->shapeOf()[dim] + (int)paddings->getScalar(dim,0))) 			
			continue;
		// increase input tads number
		++k;
		// recursion condition allows for the fact that tad can't reduce to scalar
		if(dim < input->rankOf()-2)
			recursiveLoop(mode, block, input, paddings, output, dimensions, dim+1, inIdx + k, outIdx + i);
		else {
   			// shift buffers pointers to actual element position
   			subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + i]);
   			subArrIn.setBuffer (input->getBuffer()  + tadIn.tadOffsets[inIdx + i - (int)paddings->getScalar(dim,0)]);		    			   			
   			leftOffset = (int)paddings->getScalar(dim+1,0);
   			// most inner loop, corresponds to last dim = rank-1
   			switch (mode) {
 				case 0:				// CONSTANT mode	   				
					for(int j = 0; j < subArrOut.lengthOf(); ++j) 					
						if(j < leftOffset || j >= (subArrIn.lengthOf() + leftOffset) )					// firstly fill with zeros outer ranges
							subArrOut.putIndexedScalar(j, (T)0.);
						else
							subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(j - leftOffset));	// fill middle with elements of input array
					break;

				case 1:				// REFLECT mode					
   					for(int j = 1;  j <= leftOffset; ++j) 												// fill firstly left side 
   						subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j));   					
					for(int j = 0; j < subArrIn.lengthOf(); ++j) 										// fill middle
						subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));					
					for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)		// fill right side
						subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j - 1));
					break;

				case 2:				// SYMMETRIC mode				
   					for(int j = 1;  j <= leftOffset; ++j) 												// fill firstly left side 
   						subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j-1));			   					
					for(int j = 0; j < subArrIn.lengthOf(); ++j) 										// fill middle
						subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));					
					for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)		// fill right side
						subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j));		
					break;
			}
		}	
	}	

	// populate sub-array formed previously 
	leftOffset = (int)paddings->getScalar(dim,0);		
	switch (mode) {
		case 0:			// CONSTANT mode
			for(int j = 1;  j <= leftOffset; ++j) {														// fill left side with zeros
				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
   				subArrOut.assign((T)0.);
   			}
   			for(int j = (output->shapeOf()[dim] - leftOffset); j < output->shapeOf()[dim]; ++j) {		// fill left side with zeros
   				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + j]);
   				subArrOut.assign((T)0.);
			}	
			break;

		case 1:			// REFLECT mode	
			for(int j = 1;  j <= leftOffset; ++j) {														// fill left side 
   				subArr.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + leftOffset + j]);
   				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
   				subArrOut.assign(&subArr);
   			}   			
			for(int j = (output->shapeOf()[dim] - leftOffset); j < output->shapeOf()[dim]; ++j) {		// fill right side
				subArr.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + output->shapeOf()[dim] + leftOffset - 1 - j]);
   				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + j]);
   				subArrOut.assign(&subArr);				
			}	
			break;

		case 2:			// SYMMETRIC mode	
			for(int j = 1;  j <= leftOffset; ++j) {														// fill left side
   				subArr.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + leftOffset + j - 1]);
   				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
   				subArrOut.assign(&subArr);
   			}   		
			for(int j = (output->shapeOf()[dim] - leftOffset); j < output->shapeOf()[dim]; ++j) {		// fill right side
				subArr.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + output->shapeOf()[dim] + leftOffset - j]);
   				subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[outIdx + j]);
   				subArrOut.assign(&subArr);		
   			}
   			break;
   	}
}


////////////////////////////////////////////////////////////////////////
// recursive loop for CONSTANT mode
// template<typename T>
// void recursiveLoop0(Block<T>& block, NDArray<T>* input, const NDArray<T>* paddings, NDArray<T>* output, std::vector<int>& dimensions, int dim ) {   // initial values of inIdx,outIdx,dim have to be zero
	
// 	dimensions.erase(dimensions.begin());	
// 	shape::TAD tadOut(output->getShapeInfo(), dimensions.data(), dimensions.size());
//     tadOut.createTadOnlyShapeInfo();
//     tadOut.createOffsets();
//     NDArray<T> subArrOut(output->getBuffer(), tadOut.tadOnlyShapeInfo, block.getWorkspace());
    
//     shape::TAD* tadIn = nullptr;
//     NDArray<T>* subArrIn = nullptr;
//     if(dim == input->rankOf()-2) {
//     	tadIn = new  shape::TAD(input->getShapeInfo(), dimensions.data(), dimensions.size());
//     	tadIn->createTadOnlyShapeInfo();
//     	tadIn->createOffsets();
//     	subArrIn = new NDArray<T>(input->getBuffer(), tadIn->tadOnlyShapeInfo, block.getWorkspace());
//     }

//     for(int i = 0; i < output->shapeOf()[dim]; ++i) {		
// 		subArrOut.setBuffer(output->getBuffer() + tadOut.tadOffsets[i]);	
// 		if(i < (int)paddings->getScalar(dim,0) || i >= (input->shapeOf()[dim] + (int)paddings->getScalar(dim,0))) 			// corresponds to outer range (relevant indices are absent in input)						
// 			subArrOut.assign((T)0.);					
// 		else {			
// 			if(dim < input->rankOf()-2)
// 				recursiveLoop0(block, input, paddings, output, dimensions, dim+1);
// 			else {		// now we are on next to last dim = rank-2								
//     			subArrIn->setBuffer(input->getBuffer() + tadIn->tadOffsets[i - (int)paddings->getScalar(dim,0)]);		    			
// 				// most inner loop, corresponds to last dim = rank-1
// 				for(int j=0; j < output->shapeOf()[dim+1]; ++j) 					
// 					if(j < (int)paddings->getScalar(dim+1,0) || j >= (input->shapeOf()[dim+1] + (int)paddings->getScalar(dim+1,0))) 
// 						subArrOut.putIndexedScalar(j, (T)0.);											
// 					else 															
// 						subArrOut.putIndexedScalar(j, subArrIn->getIndexedScalar(j - (int)paddings->getScalar(dim+1,0)));					
// 			}
// 		}
// 	}
// 	delete tadIn;
// 	delete subArrIn;			
// }


}
}