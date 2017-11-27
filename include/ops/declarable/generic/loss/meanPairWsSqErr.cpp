//
// Created by Yurii Shyrma on 24.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <numeric>
#include <iostream>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(meanPairWsSqErr, 3, 1, false, 0, 0) {

  	NDArray<T>* predictions = INPUT_VARIABLE(0);
    NDArray<T>* weights     = INPUT_VARIABLE(1);
    NDArray<T>* labels      = INPUT_VARIABLE(2);
    NDArray<T>* output      = OUTPUT_VARIABLE(0);
    
	// perform weights broadcasting/tile to labels if needed	
	NDArray<T>* weightsBroad = weights;	
	if(!weights->isScalar() && !weights->isSameShape(predictions)) {
		// evaluate repeat dimensions for tile operation
		std::vector<int> reps;
		for(int i = 0; i < labels->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray<T>(weights->tile(reps));
	}	
	
	NDArray<T> diffs = *predictions - *labels;
	std::vector<int> reductionIdx(diffs.rankOf()-1);
	std::iota(reductionIdx.begin(), reductionIdx.end(), 1);
	NDArray<T> sumSqrsDiffPerBatch = (diffs*diffs).template reduceAlongDims<simdOps::Sum<T>>(reductionIdx, true);

	NDArray<T> numOfNonZeroWeights(sumSqrsDiffPerBatch.getShapeInfo(), block.getWorkspace());
	if(weights->isScalar()) {
		if((*weights)(0) != (T)0.)
			numOfNonZeroWeights.assign((T)(labels->lengthOf()/labels->sizeAt(0)));
	}
	else {
		int sizeAtRestDims =  weightsBroad->lengthOf()/weightsBroad->sizeAt(0);
		for(int i = 0; i < numOfNonZeroWeights.lengthOf(); ++i)
			for(int j = 0; j < sizeAtRestDims; ++j)
				if((*weightsBroad)(i*sizeAtRestDims + j) != (T)0.)
					++numOfNonZeroWeights(i);
	}
	
	sumSqrsDiffPerBatch.template applyPairwiseTransform<simdOps::SafeDivide<T>>(&numOfNonZeroWeights, nullptr);	

	NDArray<T> sumDiff = diffs.template reduceAlongDims<simdOps::Sum<T>>(reductionIdx, true);	
	NDArray<T> nonZerosSquared = numOfNonZeroWeights*numOfNonZeroWeights;	
	(sumDiff*sumDiff).template applyPairwiseTransform<simdOps::SafeDivide<T>>(&nonZerosSquared, &sumDiff, nullptr);		
	
	NDArray<T> weightedLosses = (sumSqrsDiffPerBatch - sumDiff)*(T)2.;

    // multiply weightedLosses on weights
 	if(weights->isScalar())
 		weightedLosses *= (*weights)(0);
 	else
 		weightedLosses *= (*weights); 	
 		
	if(numOfNonZeroWeights.template reduceNumber<simdOps::Sum<T>>() == (T)0.)
		(*output)(0) = (T)0.;
	else
		(*output)(0) = weightedLosses.template reduceNumber<simdOps::Sum<T>>();


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(meanPairWsSqErr) {

	// labels and predictions must have the same shapes 
	NDArray<T>* predictions = INPUT_VARIABLE(0);
    NDArray<T>* weights     = INPUT_VARIABLE(1);
    NDArray<T>* labels      = INPUT_VARIABLE(2);

    if(!labels->isSameShape(predictions))
    	throw "CUSTOM_OP loss function meanPairWsSqErr: labels and predictions arrays have different shapes!";
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    if(!weights->isScalar() && weights->rankOf() != labels->rankOf())
    	throw "CUSTOM_OP loss function meanPairWsSqErr: weights array must have the same rank as labels array!";
    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)
        	if (weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1)
            	throw "CUSTOM_OP loss function meanPairWsSqErr: shapes of weights array is not broadcastable to labels shape!";

    int* outShapeInfo = nullptr;
    // output is scalar
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2) /*rank=2*/, int);
    outShapeInfo[0] = 2;
    outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    outShapeInfo[5] = 0;
    outShapeInfo[6] = 1;
    outShapeInfo[7] = 99;

    return new ShapeList(outShapeInfo);    

}

// INT_ARG(0) - reduction mode











}
}